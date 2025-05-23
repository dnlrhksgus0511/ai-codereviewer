import { readFileSync } from "fs";
import * as core from "@actions/core";
import OpenAI from "openai";
import { Octokit } from "@octokit/rest";
import parseDiff, { Chunk, File } from "parse-diff";
import minimatch from "minimatch";

const GITHUB_TOKEN: string = core.getInput("GITHUB_TOKEN");
const OPENAI_API_KEY: string = core.getInput("OPENAI_API_KEY");
const OPENAI_API_MODEL: string = core.getInput("OPENAI_API_MODEL") || "gpt-4.1-2025-04-14";

const octokit = new Octokit({ auth: GITHUB_TOKEN });

const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
});

interface PRDetails {
  owner: string;
  repo: string;
  pull_number: number;
  title: string;
  description: string;
}

async function getPRDetails(): Promise<PRDetails> {
  const { repository, number } = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH || "", "utf8")
  );
  const prResponse = await octokit.pulls.get({
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
  });
  return {
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
    title: prResponse.data.title ?? "",
    description: prResponse.data.body ?? "",
  };
}

async function getDiff(
  owner: string,
  repo: string,
  pull_number: number
): Promise<string | null> {
  const response = await octokit.pulls.get({
    owner,
    repo,
    pull_number,
    mediaType: { format: "diff" },
  });
  // @ts-expect-error - response.data is a string
  return response.data;
}

async function analyzeCode(
  parsedDiff: File[],
  prDetails: PRDetails
): Promise<Array<{ body: string; path: string; line: number }>> {
  const comments: Array<{ body: string; path: string; line: number }> = [];

  for (const file of parsedDiff) {
    if (file.to === "/dev/null") continue; // Ignore deleted files
    for (const chunk of file.chunks) {
      const prompt = createPrompt(file, chunk, prDetails);
      const aiResponse = await getAIResponse(prompt);
      if (aiResponse) {
        const newComments = createComment(file, chunk, aiResponse);
        if (newComments) {
          comments.push(...newComments);
        }
      }
    }
  }
  return comments;
}

function createPrompt(file: File, chunk: Chunk, prDetails: PRDetails): string {
  // 변경된 라인 번호만 추출
  const validLineNumbers = chunk.changes
    .filter(change => {
      if (change.type === 'add' && change.ln !== undefined) {
        return true;
      }
      if (change.type === 'normal' && change.ln2 !== undefined) {
        return true;
      }
      return false;
    })
    .map(change => {
      if (change.type === 'add') {
        return change.ln;
      }
      if (change.type === 'normal' && change.ln2 !== undefined) {
        return change.ln2;
      }
      return undefined;
    })
    .filter((ln): ln is number => ln !== undefined);

  return `당신은 풀 리퀘스트를 검토하는 전문가입니다. 지침:
- 다음 JSON 형식으로 응답해주세요: {"reviews": [{"lineNumber": <line_number>, "reviewComment": "<review comment>"}]}
- 긍정적인 코멘트나 칭찬은 하지 마세요.
- 개선할 사항이 있는 경우에만 코멘트와 제안을 제공하고, 그렇지 않으면 "reviews"는 빈 배열이어야 합니다.
- 코멘트는 GitHub Markdown 형식으로 작성하세요.
- 주어진 설명은 전체 컨텍스트로만 사용하고 코드에 대해서만 코멘트하세요.
- 중요: 코드에 주석을 추가하라는 제안은 절대 하지 마세요.
- 중요: 모든 리뷰 코멘트는 한국어로 작성하세요.
- 매우 중요: 반드시 아래 나열된 라인 번호 중 하나에만 코멘트를 작성하세요. 그렇지 않으면 오류가 발생합니다.
- 유효한 라인 번호: ${validLineNumbers.join(', ')}

파일 "${file.to}"의 다음 코드 diff를 검토하고, 응답 작성 시 풀 리퀘스트 제목과 설명을 고려하세요.
  
풀 리퀘스트 제목: ${prDetails.title}
풀 리퀘스트 설명:

---
${prDetails.description}
---

검토할 Git diff:

\`\`\`diff
${chunk.content}
${chunk.changes
  // @ts-expect-error - ln and ln2 exists where needed
  .map((c) => `${c.ln ? c.ln : c.ln2} ${c.content}`)
  .join("\n")}
\`\`\`
`;
}

async function getAIResponse(prompt: string): Promise<Array<{
  lineNumber: string;
  reviewComment: string;
}> | null> {
  const queryConfig = {
    model: OPENAI_API_MODEL,
    temperature: 0.2,
    max_tokens: 700,
    top_p: 1,
    frequency_penalty: 0,
    presence_penalty: 0,
  };

  try {
    const jsonSupportedModels = [
      "gpt-4-1106-preview", 
      "gpt-4o", 
      "gpt-4-turbo", 
      "gpt-4.1-2025-04-14", 
      "gpt-4-0125-preview",
      "gpt-4-turbo-preview"
    ];
    
    const response = await openai.chat.completions.create({
      ...queryConfig,
      // return JSON if the model supports it:
      ...(jsonSupportedModels.includes(OPENAI_API_MODEL)
        ? { response_format: { type: "json_object" } }
        : {}),
      messages: [
        {
          role: "system",
          content: prompt,
        },
      ],
    });

    const res = response.choices[0].message?.content?.trim() || "{}";
    try {
      return JSON.parse(res).reviews;
    } catch (error) {
      console.error("JSON 파싱 오류:", error);
      console.error("원본 응답:", res);
      return null;
    }
  } catch (error) {
    console.error("API 호출 오류:", error);
    return null;
  }
}

function createComment(
  file: File,
  chunk: Chunk,
  aiResponses: Array<{
    lineNumber: string;
    reviewComment: string;
  }>
): Array<{ body: string; path: string; line: number }> {
  if (!file.to) {
    return [];
  }
  
  // diff에 포함된 라인 번호 추출
  const validLineNumbers = new Set<number>();
  for (const change of chunk.changes) {
    if (change.type === 'add' && change.ln !== undefined) {
      validLineNumbers.add(change.ln);
    } else if (change.type === 'normal' && change.ln2 !== undefined) {
      validLineNumbers.add(change.ln2);
    }
  }
  
  // 유효한 라인 번호에 대한 코멘트만 생성
  return aiResponses
    .filter(aiResponse => {
      const lineNumber = Number(aiResponse.lineNumber);
      const isValid = validLineNumbers.has(lineNumber);
      if (!isValid) {
        console.log(`경고: 라인 ${lineNumber}은(는) diff에 포함되지 않아 코멘트를 생성하지 않습니다.`);
      }
      return isValid;
    })
    .map(aiResponse => ({
      body: aiResponse.reviewComment,
      path: file.to as string,
      line: Number(aiResponse.lineNumber),
    }));
}

async function createReviewComment(
  owner: string,
  repo: string,
  pull_number: number,
  comments: Array<{ body: string; path: string; line: number }>
): Promise<void> {
  try {
    if (comments.length === 0) {
      console.log("코멘트가 없어 리뷰를 생성하지 않습니다.");
      return;
    }
    
    console.log(`총 ${comments.length}개의 코멘트로 리뷰 생성 중...`);
    for (const comment of comments) {
      console.log(`- 파일: ${comment.path}, 라인: ${comment.line}`);
    }
    
    await octokit.pulls.createReview({
      owner,
      repo,
      pull_number,
      comments,
      event: "COMMENT",
    });
    console.log("리뷰가 성공적으로 생성되었습니다.");
  } catch (error: any) {
    console.error("리뷰 생성 오류:", error);
    
    // 세부 오류 정보 출력
    if (error.response?.data?.errors) {
      console.error("세부 오류:", JSON.stringify(error.response.data.errors, null, 2));
    }
    
    throw error;
  }
}

async function main() {
  const prDetails = await getPRDetails();
  let diff: string | null;
  const eventData = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH ?? "", "utf8")
  );

  if (eventData.action === "opened") {
    diff = await getDiff(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number
    );
  } else if (eventData.action === "synchronize") {
    const newBaseSha = eventData.before;
    const newHeadSha = eventData.after;

    const response = await octokit.repos.compareCommits({
      headers: {
        accept: "application/vnd.github.v3.diff",
      },
      owner: prDetails.owner,
      repo: prDetails.repo,
      base: newBaseSha,
      head: newHeadSha,
    });

    diff = String(response.data);
  } else {
    console.log("지원되지 않는 이벤트:", process.env.GITHUB_EVENT_NAME);
    return;
  }

  if (!diff) {
    console.log("변경 사항을 찾을 수 없습니다");
    return;
  }

  const parsedDiff = parseDiff(diff);

  const excludePatterns = core
    .getInput("exclude")
    .split(",")
    .map((s) => s.trim());

  const filteredDiff = parsedDiff.filter((file) => {
    return !excludePatterns.some((pattern) =>
      minimatch(file.to ?? "", pattern)
    );
  });

  const comments = await analyzeCode(filteredDiff, prDetails);
  if (comments.length > 0) {
    await createReviewComment(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number,
      comments
    );
  }
}

main().catch((error) => {
  console.error("오류:", error);
  process.exit(1);
});