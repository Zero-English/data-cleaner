import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";
import { join } from "node:path";
import { readFile, writeFile } from "jsonfile";
import { quizTypes, quizClasses, bulkQuizQuestionSchema } from "../schemas/quiz.schema.js";

dotenv.config({ path: join(import.meta.dirname, "..", ".env") });

const PROMPT = `
ROLE:
You are a highly accurate educational quiz content enrichment agent.

INPUT:
You will receive a JSON array of valid quiz questions. Every question already contains a correct answer and its options.

OBJECTIVE:
Return a JSON array with EVERY input question fully intact, plus an "explanation" field added to each question that clearly explains why the given answer is correct.

STRICT OUTPUT RULES:
1. Output ONLY a valid JSON array.
2. Do NOT output Markdown.
3. Do NOT wrap the JSON in \`\`\`json.
4. Do NOT include explanations, comments, notes, or any text outside the JSON array.
5. Every object must follow the required schema exactly.
6. Use double quotes for all JSON keys and string values.
7. Never return JavaScript syntax such as \`export const\`, single quotes, or trailing commas.
8. Return exactly one object per input question, in the same order as the input.

ALLOWED VALUES:

quizTypes:
${JSON.stringify(quizTypes)}

quizClasses:
${JSON.stringify(quizClasses)}

difficultyLevels:
["EASY", "MEDIUM", "HARD"]

EXPLANATION RULES:

1. Keep quizType, questionText, options, difficultyLevel, and answer EXACTLY as provided. Do NOT change the correct answer, do NOT rename or reorder options, do NOT reword the question.
2. Add "explanation": a clear, concise, student-friendly explanation of 1–3 sentences explaining WHY the given answer is correct. Reference the specific grammar rule, idiom/phrase meaning, or concept being tested.
3. For translation questions (ENGLISH_TO_BANGLA / BANGLA_TO_ENGLISH), keep the explanation brief (1–2 sentences) and explain the translation.
4. The explanation must be consistent with the provided answer and must never contradict it.
5. For questions whose answer or meaning is in Bangla, you may write the explanation in Bangla if that is clearer, otherwise write it in English.
6. Do not add, remove, or modify the provided options.
7. If the input question contains an obvious, unambiguous typo (e.g. OCR artifact), you may fix only that typo, but never change the meaning, the answer, or the options for the intended text.
8. You can add a class only if a question doesn't have it.

VALIDATION SCHEMA:

{
  "quizType": "<one of quizTypes>",
  "questionText": "<question>",
  "options": ["<option 1>", "<option 2>", "..."],
  "difficultyLevel": "<EASY | MEDIUM | HARD>",
  "answer": "<exact correct option text>",
  "class": "<exact the class that the question has.>",
  "explanation": "<concise explanation of why the answer is correct>"
}

FINAL CHECK BEFORE OUTPUT:
- Is the output valid JSON?
- Is it an array?
- Does every object contain exactly the required fields, including "explanation"?
- Is every quizType allowed and unchanged?
- Is every difficultyLevel uppercase and allowed?
- Is class exactly ["Diploma"]?
- Is the answer unchanged and still one of the provided options?
- Is the explanation clear, correct, and consistent with the answer?
- Did you output nothing except the JSON array?

EXPECTED OUTPUT FORMAT:
[{
  "quizType": "IDIOMS_AND_PHRASES",
  "questionText": "Complete the sentence: \"Smoking _____ health.\"",
  "options": ["tells upon", "takes after", "turns down", "searches for"],
  "difficultyLevel": "EASY",
  "answer": "tells upon",
  "class": ["Diploma"],
  "explanation": "\"Tell upon\" means to have a harmful effect on — কোনো কিছুর ওপর ক্ষতিকর প্রভাব ফেলা। তাই, \"Smoking tells upon health\" অর্থ ধূমপান স্বাস্থ্যের ওপর ক্ষতিকর প্রভাব ফেলে। <br>\"Takes after\" means resembles — কারও মতো হওয়া; <br>\"turns down\" means rejects/refuses or reduces — প্রত্যাখ্যান করা বা কমানো; <br>\"searches for\" means looks for — খোঁজা।"
}, {
  "quizType": "PREPOSITIONS",
  "questionText": "I was admitted _____ the room.",
  "options": ["to", "into", "in", "with"],
  "difficultyLevel": "MEDIUM",
  "answer": "into",
  "class": ["Diploma"],
  "explanation": "\"Into\" is used to show movement from outside to inside — বাইরে থেকে ভেতরে প্রবেশ বোঝাতে \"into\" ব্যবহৃত হয়। তাই, \"I was admitted into the room\" অর্থ আমাকে কক্ষে প্রবেশ করতে দেওয়া হয়েছিল। <br>\"To\" সাধারণত কোনো কিছুর দিকে নির্দেশ করে; </br>\"in\" অবস্থান বোঝায়, প্রবেশ নয়; <br>আর \"with\" সঙ্গে বা সম্পর্ক বোঝায়।"
}]
`

const MODEL = "gemini-3.6-flash";
const QUESTIONS_PER_BATCH = 30;
const MAX_JSON_RETRIES = 0;
const MAX_QUOTA_RETRIES = 0;
const RETRY_MAX_WAIT_MS = 60 * 1000;

const SOURCE_PATH = join(import.meta.dirname, "..", "un_processed_json", "quizzes.json");
const OUTPUT_PATH = join(import.meta.dirname, "..", "jsons", "quiz.json");

class DailyQuotaError extends Error {}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function chunk(items, size) {
  const out = [];
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));
  return out;
}

function keyOf(item) {
  return item.questionText.trim().toLowerCase();
}

function cleanJsonText(text) {
  return (text || "")
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/, "");
}

function parseRetrySeconds(error) {
  try {
    const body = JSON.parse(error.message);
    for (const detail of body?.error?.details ?? []) {
      if (typeof detail?.retryDelay === "string") {
        const seconds = parseFloat(detail.retryDelay.replace(/s$/i, ""));
        if (Number.isFinite(seconds)) return seconds;
      }
    }
  } catch {}
  return null;
}

function isDailyQuotaError(error) {
  return (
    error instanceof DailyQuotaError ||
    /PerDayPerProjectPerModel|RequestsPerDay|per_day/i.test(error?.message ?? "")
  );
}

async function generateContentWithQuotaRetry(ai, { model, contents, config }) {
  for (let attempt = 0; attempt < MAX_QUOTA_RETRIES; attempt++) {
    try {
      return await ai.models.generateContent({ model, contents, config });
    } catch (error) {
      const isQuotaExhausted = /RESOURCE_EXHAUSTED|"code":\s*429/i.test(error?.message ?? "");
      if (!isQuotaExhausted) throw error;
      if (isDailyQuotaError(error)) throw new DailyQuotaError("Daily generate_content quota exhausted.");
      const seconds = parseRetrySeconds(error);
      const waitMs = seconds == null
        ? RETRY_MAX_WAIT_MS
        : Math.min(seconds * 1000, RETRY_MAX_WAIT_MS);
      console.warn(`Rate limited; retrying in ${Math.round(waitMs / 1000)}s...`);
      await delay(waitMs);
    }
  }
  throw new Error("Rate-limit retries exhausted");
}

async function generateJson(ai, contents) {
  const RETRY_NUDGE = "\n\nYour previous output was not valid JSON. Output ONLY a plain JSON array. No markdown, no code fences, no extra text.";
  for (let attempt = 0; attempt <= MAX_JSON_RETRIES; attempt++) {
    const response = await generateContentWithQuotaRetry(ai, {
      model: MODEL,
      contents: attempt === 0
        ? contents
        : [...contents, { text: RETRY_NUDGE }],
      config: { responseMimeType: "application/json" },
    });
    try {
      return JSON.parse(cleanJsonText(response.text));
    } catch {
      if (attempt === MAX_JSON_RETRIES) throw new Error(`AI output was not valid JSON after ${MAX_JSON_RETRIES + 1} attempts`);
    }
  }
}

function dedupeByText(items) {
  const seen = new Set();
  return items.filter((item) => {
    const key = keyOf(item);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

async function main() {
  const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
  const ai = new GoogleGenAI({
    apiKey: GEMINI_API_KEY,
  });

  let pending = await readFile(SOURCE_PATH);
  if (!Array.isArray(pending)) throw new Error("un_processed_json/quizzes.json must contain a JSON array of quiz questions");

  let jsonData = [];
  try {
    jsonData = await readFile(OUTPUT_PATH);
    if (!Array.isArray(jsonData)) jsonData = [];
  } catch {
    jsonData = [];
  }

  const existingKeys = new Set(jsonData.map((q) => keyOf(q)));
  pending = pending.filter((q) => !existingKeys.has(keyOf(q)));

  const batches = chunk(pending, QUESTIONS_PER_BATCH);
  console.log(`${pending.length} questions queued → ~${batches.length} request(s) (${QUESTIONS_PER_BATCH} questions/batch)`);

  for (const batch of batches) {
    const batchLabel = `batch ${batch.length} questions`;
    try {
      const parsed = await generateJson(ai, [
        {
          role: "user",
          parts: [{ text: `${PROMPT}\n\nINPUT JSON:\n${JSON.stringify(batch)}` }],
        },
      ]);

      if (!Array.isArray(parsed)) throw new Error("AI output was not an array");

      const valid = [];
      for (const item of parsed) {
        const result = bulkQuizQuestionSchema.safeParse(item);
        if (result.success) valid.push({ ...result.data, class: ["Diploma"] });
      }
      if (valid.length !== parsed.length) {
        console.warn(`[${batchLabel}] ${parsed.length - valid.length}/${parsed.length} questions failed validation; keeping ${valid.length}`);
      }

      const fresh = dedupeByText([...jsonData, ...valid]);
      const added = fresh.length - jsonData.length;
      jsonData = fresh;

      await writeFile(OUTPUT_PATH, jsonData);

      const processedKeys = new Set(valid.map((q) => keyOf(q)));
      const remaining = pending.filter((q) => !processedKeys.has(keyOf(q)));
      if (remaining.length !== pending.length) {
        pending = remaining;
        await writeFile(SOURCE_PATH, pending);
      }

      console.log(`[${batchLabel}] enriched ${valid.length} questions (${added} new)`);
    } catch (error) {
      if (isDailyQuotaError(error)) {
        console.error("Daily quota exhausted — stopping. Remaining questions stay in un_processed_json/quizzes.json; resume after ~midnight Pacific time.");
        break;
      }
      console.error(`[${batchLabel}] FAILED (keeping questions in un_processed_json/quizzes.json for retry): ${error.message}`);
    }
  }

  console.log(`Done. quiz.json now has ${jsonData.length} questions with explanations.`);
}

main();