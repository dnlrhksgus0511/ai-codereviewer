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
  
  // Generate PR summary first
  const prSummary = await generatePRSummary(parsedDiff, prDetails);
  
  // Find a suitable line for the summary comment
  if (prSummary && parsedDiff.length > 0) {
    // Find the first valid line in the diff to attach the summary to
    let summaryTarget = null;
    
    // Look for an added line in the first file
    for (const file of parsedDiff) {
      if (file.to) {
        for (const chunk of file.chunks) {
          for (const change of chunk.changes) {
            if (change.type === 'add' && change.ln !== undefined) {
              summaryTarget = {
                path: file.to,
                line: change.ln
              };
              break;
            }
          }
          if (summaryTarget) break;
        }
      }
      if (summaryTarget) break;
    }
    
    // Only add the summary if we found a valid line to attach it to
    if (summaryTarget) {
      comments.push({
        body: prSummary,
        path: summaryTarget.path,
        line: summaryTarget.line,
      });
    }
  }

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

async function generatePRSummary(
  parsedDiff: File[],
  prDetails: PRDetails
): Promise<string> {
  // Create a more detailed summary of the code changes
  const detailedChanges = parsedDiff.map(file => {
    const path = file.to || file.from || '';
    const changeType = file.to ? (file.from ? '수정됨' : '추가됨') : '삭제됨';
    const changesCount = file.chunks.reduce((sum, chunk) => sum + chunk.changes.length, 0);
    
    // Extract the actual code changes for better context
    // Limit to a reasonable number of changes to avoid token limits
    const codeChanges = file.chunks.flatMap(chunk => 
      chunk.changes
        .filter(change => change.type === 'add' || change.type === 'del')
        .slice(0, 10) // Limit to 10 changes per file
        .map(change => `${change.type === 'add' ? '+' : '-'} ${change.content.trim()}`)
    ).join('\n');
    
    return `파일: ${path} (${changeType}, ${changesCount}개 라인 변경)
주요 코드 변경:
\`\`\`
${codeChanges}
${file.chunks.length > 0 && file.chunks[0].changes.length > 10 ? '... (더 많은 변경사항 있음)' : ''}
\`\`\``;
  }).join('\n\n');

  const prompt = `당신은 코드 변경사항을 분석하고 요약하는 전문가입니다. 아래 PR(Pull Request)의 코드 변경을 분석하여 다음을 포함한 요약을 한국어로 작성해주세요:

1. 어떤 기능이 추가되었는지
2. 어떤 부분이 수정되었는지
3. 아키텍처나 성능에 영향을 미치는 중요한 변경사항
4. 코드만 보고 유추할 수 있는 PR의 목적

코드 변경을 보고 최대한 정확하게 유추해주세요. PR 제목과 설명은 참고용으로만 사용하고, 실제 코드 변경을 중심으로 분석해주세요.

Pull request 제목: ${prDetails.title}
Pull request 설명:

---
${prDetails.description}
---

상세 코드 변경사항:
${detailedChanges}

응답 형식:
{
  "summary": "PR에 대한 기술적 요약 (한국어)",
  "changes": ["주요 변경사항 1", "주요 변경사항 2", "..."]
}`;

  try {
    const response = await openai.chat.completions.create({
      model: OPENAI_API_MODEL,
      temperature: 0.3,
      max_tokens: 700,
      ...(OPENAI_API_MODEL === "gpt-4-1106-preview" || OPENAI_API_MODEL === "gpt-4o" || OPENAI_API_MODEL === "gpt-4-turbo"
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
    const summaryData = JSON.parse(res);
    
    // Format the changes as bullet points
    const changesList = summaryData.changes?.length 
      ? "\n\n### 주요 변경사항\n" + summaryData.changes.map((change: string) => `- ${change}`).join("\n")
      : "";
    
    return `## 🔍 PR 요약
    
${summaryData.summary}${changesList}

---
`;
  } catch (error) {
    console.error("PR 요약 생성 오류:", error);
    return ""; // Return empty string on error
  }
}

function createPrompt(file: File, chunk: Chunk, prDetails: PRDetails): string {
  return `Your task is to review pull requests. Instructions:
- Provide the response in following JSON format:  {"reviews": [{"lineNumber":  <line_number>, "reviewComment": "<review comment>", "severity": <severity_score>}]}
- Do not give positive comments or compliments.
- Provide comments and suggestions ONLY if there is something to improve, otherwise "reviews" should be an empty array.
- Write the comment in GitHub Markdown format.
- Use the given description only for the overall context and only comment the code.
- IMPORTANT: NEVER suggest adding comments to the code.
- IMPORTANT: Write all review comments in Korean language.
- IMPORTANT: Include a severity score (1-5) for each issue where:
  - 1: Minor style suggestion that can be ignored
  - 2: Minor issue that should be fixed but not critical
  - 3: Moderate issue that should be addressed
  - 4: Significant issue that could lead to bugs or maintenance problems
  - 5: Critical issue that must be fixed (security vulnerability, performance issue, etc.)

Review the following code diff in the file "${
    file.to
  }" and take the pull request title and description into account when writing the response.
  
Pull request title: ${prDetails.title}
Pull request description:

---
${prDetails.description}
---

Git diff to review:

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
  severity: number;
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
    const response = await openai.chat.completions.create({
      ...queryConfig,
      // return JSON if the model supports it:
      ...(OPENAI_API_MODEL === "gpt-4-1106-preview" || OPENAI_API_MODEL === "gpt-4o" || OPENAI_API_MODEL === "gpt-4-turbo"
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
    return JSON.parse(res).reviews;
  } catch (error) {
    console.error("오류:", error);
    return null;
  }
}

function createComment(
  file: File,
  chunk: Chunk,
  aiResponses: Array<{
    lineNumber: string;
    reviewComment: string;
    severity: number;
  }>
): Array<{ body: string; path: string; line: number }> {
  return aiResponses.flatMap((aiResponse) => {
    if (!file.to) {
      return [];
    }
    
    // Convert lineNumber to a number
    const lineNum = Number(aiResponse.lineNumber);
    
    // Verify that this line number exists in the diff chunk
    const lineExists = chunk.changes.some(change => {
      if (change.type === 'add') {
        // For added lines, check ln property
        return change.ln === lineNum;
      } else if (change.type === 'normal') {
        // For normal (context) lines, check ln2 property (which is the new file line number)
        return change.ln2 === lineNum;
      }
      // Ignore deleted lines for comments
      return false;
    });
    
    // Skip this comment if the line isn't part of the diff
    if (!lineExists) {
      console.log(`Skipping comment for line ${lineNum} in ${file.to} as it's not part of the diff`);
      return [];
    }
    
    const severityLabel = getSeverityLabel(aiResponse.severity);
    const body = `**심각도: ${aiResponse.severity}/5** - ${severityLabel}\n\n${aiResponse.reviewComment}`;
    
    return {
      body,
      path: file.to,
      line: lineNum,
    };
  });
}

function getSeverityLabel(severity: number): string {
  switch (severity) {
    case 1:
      return "🟢 무시 가능한 minor 이슈";
    case 2:
      return "🟡 중요도 낮음";
    case 3:
      return "🟠 중요도 중간";
    case 4:
      return "🔴 중요 이슈";
    case 5:
      return "⛔ 심각한 문제 - 반드시 수정 필요";
    default:
      return "";
  }
}

async function createReviewComment(
  owner: string,
  repo: string,
  pull_number: number,
  comments: Array<{ body: string; path: string; line: number }>
): Promise<void> {
  await octokit.pulls.createReview({
    owner,
    repo,
    pull_number,
    comments,
    event: "COMMENT",
  });
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
