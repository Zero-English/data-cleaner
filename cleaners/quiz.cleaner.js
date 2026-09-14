import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";
import { readdir, rename } from "node:fs/promises";
import { join } from "node:path";
import { readFile, writeFile } from "jsonfile";
import { quizTypes, quizClasses, bulkQuizQuestionSchema } from "../schemas/quiz.schema.js";

dotenv.config({ path: join(import.meta.dirname, "..", ".env") });

const PROMPT = `
ROLE: 
You are a highly accurate educational data extraction agent.

INPUT: 
You will receive one or more images containing questions from a Diploma English Guide Book.

OBJECTIVE:
Extract every valid quiz question visible in ANY of the provided images and return the results as a single JSON array.

STRICT OUTPUT RULES:
1. Output ONLY a valid JSON array.
2. Do NOT output Markdown.
3. Do NOT wrap the JSON in \`\`\`json.
4. Do NOT include explanations, comments, notes, or any text outside the JSON array.
5. Every object must follow the required schema exactly.
6. Use double quotes for all JSON keys and string values.
7. Never return JavaScript syntax such as \`export const\`, single quotes, or trailing commas.

ALLOWED VALUES:

quizTypes:
${JSON.stringify(quizTypes)}

quizClasses:
${JSON.stringify(quizClasses)}

difficultyLevels:
["EASY", "MEDIUM", "HARD"]

EXTRACTION RULES:

1. Extract ALL complete questions that are clearly visible across ALL of the provided images. Aim to extract as many questions as possible; err on the side of including grammatically-solvable questions.
2. Correct obvious OCR errors when the intended English text is unambiguous.
3. Do NOT create questions from headings, instructions, examples, explanations, or answer keys unless they are clearly presented as quiz questions.
4. If a question has multiple-choice options, extract every visible option.
5. The "answer" field must contain the correct option text, not the option number or letter.
6. Determine the correct answer by applying standard English grammar rules (e.g., right form of verb, transformation, parts of speech, sentence types). If no answer key is present in the image, still provide the answer when you are confident it follows from the question itself. Exclude a question only if the correct answer is genuinely ambiguous or the text is unreadable.
7. If fewer than 2 options are visible, supply plausible, distinct options derived from the grammar concept being tested. Construct options only when you are confident they are reasonable; if options truly cannot be constructed, exclude the question.
8. If the question is incomplete or unreadable, exclude it.
8. Do not duplicate the same question.
9. Keep punctuation where it is meaningful.
10. Remove unnecessary numbering such as "1.", "2.", "(a)", etc. from questionText unless it is part of the actual question.
11. Every question must have:
    class: ["Diploma"]
12. Do not infer a different class from the image. Always use ["Diploma"].
13. quizType must be one of the values from quizTypes. Never invent a new quizType.
14. difficultyLevel must be exactly one of:
    "EASY", "MEDIUM", "HARD"
15. If the image contains an answer key, use it to determine the answer when it clearly corresponds to the question.
16. If no answer key is available, determine the answer from the question itself using the grammar rules above. Be confident, but do not guess randomly.
17. Never use information from outside the provided image to invent missing question content.
18. Prefer extraction over exclusion: only drop a question when it is broken beyond reasonable repair.

QUESTION TYPE:
Assign the most appropriate quizType from the provided quizTypes list based on the actual question.

VALIDATION SCHEMA:

{
  "quizType": "<one of quizTypes>",
  "questionText": "<question>",
  "options": ["<option 1>", "<option 2>", "..."],
  "difficultyLevel": "<EASY | MEDIUM | HARD>",
  "answer": "<exact correct option text>",
  "class": ["Diploma"]
}

FINAL CHECK BEFORE OUTPUT:
- Is the output valid JSON?
- Is it an array?
- Does every object contain exactly the required fields?
- Is every quizType allowed?
- Is every difficultyLevel uppercase and allowed?
- Is class exactly ["Diploma"]?
- Does every question have at least 2 options?
- Is every answer one of the provided options?
- Did you avoid inventing unreadable information?
- Did you output nothing except the JSON array?

EXPECTED OUTPUT FORMAT:
[{
  "quizType": "Parts of Speech",
  "questionText": "\"Kamal determined to do well in the exam.\" which part of speech is \"well\"?",
  "options": ["Adjective", "Adverb", "Verb", "Preposition"],
  "difficultyLevel": "MEDIUM",
  "answer": "Adverb",
  "class": ["Diploma"]
}, {
  "quizType": "Transformation",
  "questionText": "\"It is very useful.\" what is the Negative form of the sentence?",
  "options": ["It is not very useful.", "It is not useful.", "It is useless.", "It isn't useless at all."],
  "difficultyLevel": "EASY",
  "answer": "It isn't useless at all.",
  "class": ["Diploma"]
}, {
  "quizType": "Sentence",
  "questionText": "\"May Allah bless you.\" what type of sentence is this?",
  "options": ["Optative", "Imperative", "Exclamatory", "Assertive"],
  "difficultyLevel": "HARD",
  "answer": "Optative",
  "class": ["Diploma"]
}, {
    "quizType": "Preposition",
    "questionText": "The man is addicted _____ New Market.",
    "options": ["to", "into", "at", "for"],
    "difficultyLevel": "EASY",
    "answer": "to",
    "class": ["Diploma"]
},
  {
    "quizType": "Preposition",
    "questionText": "I was admitted _____ the room.",
    "options": ["to", "into", "in", "with"],
    "difficultyLevel": "MEDIUM",
    "answer": "into",
    "class": ["Diploma"]
},
  {
    "quizType": "Preposition",
    "questionText": "Karim aimed his gun _____ the dove.",
    "options": ["at", "to", "on", "against"],
    "difficultyLevel": "EASY",
    "answer": "at",
    "class": ["Diploma"]
},
  {
    "quizType": "Preposition",
    "questionText": "He is angry _____ me for speaking against him.",
    "options": ["with", "at", "to", "for"],
    "difficultyLevel": "EASY",
    "answer": "with",
    "class": ["Diploma"]
  },
  {
    "quizType": "Preposition",
    "questionText": "The train has arrived _____ the station in time.",
    "options": ["at", "in", "to", "on"],
    "difficultyLevel": "EASY",
    "answer": "at",
    "class": ["Diploma"]
  }, {
    "quizType": "Sentence Pattern",
    "questionText": "What is the structural pattern of the sentence \"She looks nice\"?",
    "options": [
      "Sub (NP) + Vi",
      "Sub (NP) + Be-verb + Obj (NP)",
      "Sub (NP) + LV + Adj",
      "Sub (NP) + Be-verb + Adj"
    ],
    "difficultyLevel": "MEDIUM",
    "answer": "Sub (NP) + LV + Adj",
    "class": ["Diploma"]
  },
  {
    "quizType": "Parts of Speech",
    "questionText": "\"The pen is blue.\" Which part of speech is \"blue\"?",
    "options": [
      "Noun",
      "Adjective",
      "Verb",
      "Adverb"
    ],
    "difficultyLevel": "EASY",
    "answer": "Adjective",
    "class": ["Diploma"]
  },
  {
    "quizType": "Sentence Structure",
    "questionText": "In the sentence \"Birds fly.\", what type of verb is \"fly\"?",
    "options": [
      "Transitive Verb",
      "Intransitive Verb",
      "Linking Verb",
      "Auxiliary Verb"
    ],
    "difficultyLevel": "HARD",
    "answer": "Intransitive Verb",
    "class": ["Diploma"]
  }, {
    "quizType": "Sentence Structure",
    "questionText": "Which sentence follows the pattern \"Sub (NP) + Vi\"?",
    "options": [
      "The sun rises.",
      "We are students.",
      "She is beautiful.",
      "It seems easy."
    ],
    "difficultyLevel": "EASY",
    "answer": "The sun rises.",
    "class": ["Diploma"]
  },
  {
    "quizType": "Sentence Structure",
    "questionText": "Which sentence follows the pattern \"Sub (NP) + Be-verb + Obj (NP)\"?",
    "options": [
      "They cry.",
      "They are players.",
      "She is kind.",
      "He went mad."
    ],
    "difficultyLevel": "MEDIUM",
    "answer": "They are players.",
    "class": ["Diploma"]
  },
  {
    "quizType": "Sentence Structure",
    "questionText": "Which sentence follows the pattern \"Sub (NP) + LV + Adj\"?",
    "options": [
      "He sleeps.",
      "He is a doctor.",
      "I am afraid.",
      "It seems easy."
    ],
    "difficultyLevel": "HARD",
    "answer": "It seems easy.",
    "class": ["Diploma"]
}, {
    "quizType": "Sentence Pattern",
    "questionText": "What is the sentence pattern of \"She went to the MD herself\"?",
    "options": [
      "Sub (NP) + Transitive verb + Obj (NP)",
      "Sub (NP) + Tr.v + obj(NP) + Obj(Adj)",
      "Sub (NP) + Tr.v + obj(NP) + obj (Pronoun)",
      "Sub (NP) + LV + Adj"
    ],
    "difficultyLevel": "MEDIUM",
    "answer": "Sub (NP) + Tr.v + obj(NP) + obj (Pronoun)",
    "class": ["Diploma"]
  },
  {
    "quizType": "Idioms and Phrases",
    "questionText": "What is the meaning of the idiom \"A black sheep\"?",
    "options": [
      "An unexpected danger",
      "A disreputable or unworthy person",
      "A person who works hard",
      "A rich person"
    ],
    "difficultyLevel": "EASY",
    "answer": "A disreputable or unworthy person",
    "class": ["Diploma"]
  },
  {
    "quizType": "Idioms and Phrases",
    "questionText": "Complete the sentence: \"Don't roam about _____.\"",
    "options": [
      "to and fro",
      "at large",
      "up and down",
      "by and by"
    ],
    "difficultyLevel": "EASY",
    "answer": "to and fro",
    "class": ["Diploma"]
  },
  {
    "quizType": "Idioms and Phrases",
    "questionText": "Which idiom best completes the sentence: \"He fought _____ against his poverty.\"",
    "options": [
      "step by step",
      "tooth and nail",
      "at a loss",
      "well and good"
    ],
    "difficultyLevel": "MEDIUM",
    "answer": "tooth and nail",
    "class": ["Diploma"]
  },
  {
    "quizType": "Sentence Structure",
    "questionText": "Which sentence follows the pattern \"Sub (NP) + Tr.v + obj(NP) + Obj(Adj)\"?",
    "options": [
      "He eats rice.",
      "I saw the king himself.",
      "They torn the letter open.",
      "He met the teacher himself."
    ],
    "difficultyLevel": "HARD",
    "answer": "They torn the letter open.",
    "class": ["Diploma"]
  },
  {
    "quizType": "Idioms and Phrases",
    "questionText": "Which sentence correctly demonstrates the usage of \"At daggers drawn\"?",
    "options": [
      "The president and the secretary of the club are at daggers drawn.",
      "He was taken to task for negligence of duty.",
      "Human life is full of weal and woe.",
      "He is a victim of circumstances."
    ],
    "difficultyLevel": "HARD",
    "answer": "The president and the secretary of the club are at daggers drawn.",
    "class": ["Diploma"]
  }]
`

const MODEL = "gemini-3.6-flash";
const IMAGES_PER_BATCH = 3;
const MAX_JSON_RETRIES = 0;
const MAX_QUOTA_RETRIES = 5;
const RETRY_MAX_WAIT_MS = 60 * 1000;

class DailyQuotaError extends Error {}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function chunk(items, size) {
  const out = [];
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));
  return out;
}

function mimeTypeFor(file) {
  const ext = file.split(".").pop().toLowerCase();
  if (ext === "png") return "image/png";
  if (ext === "webp") return "image/webp";
  return "image/jpeg";
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
      config: { responseMimeType: "application/json"},
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
    const key = item.questionText.trim().toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

async function uploadAndWait(ai, imagePath, image) {
  const uploadedFile = await ai.files.upload({
    file: imagePath,
    config: { mimeType: mimeTypeFor(image) },
  });

  let state = (await ai.files.get({ name: uploadedFile.name })).state;
  while (state !== "ACTIVE") {
    state = (await ai.files.get({ name: uploadedFile.name })).state;
    if (state === "FAILED") throw new Error("File upload failed");
    await delay(2000);
  }
  return uploadedFile;
}

async function main() {
  const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
  const ai = new GoogleGenAI({
      apiKey: GEMINI_API_KEY,
  });

  const images = await readdir(join(import.meta.dirname, "..", "quiz_images"));

  let jsonData = [];
  try {
    jsonData = await readFile(join(import.meta.dirname, "..", "jsons", "quiz.json"));
  } catch {
    await writeFile(join(import.meta.dirname, "..", "jsons", "quiz.json"), jsonData);
  }

  const batches = chunk(images, IMAGES_PER_BATCH);
  console.log(`${images.length} images queued → ~${batches.length} request(s) (${IMAGES_PER_BATCH} images/batch)`);

  for (const batch of batches) {
    const batchLabel = batch.length === 1 ? batch[0] : `batch ${batch.length} images`;
    try {
      const parts = [];
      for (let i = 0; i < batch.length; i++) {
        const image = batch[i];
        const uploadedFile = await uploadAndWait(ai, join(import.meta.dirname, "..", "quiz_images", image), image);
        parts.push({ fileData: { fileUri: uploadedFile.uri, mimeType: uploadedFile.mimeType } });
        console.log(`  [${image}] uploaded (${i + 1}/${batch.length})`);
      }

      const parsed = await generateJson(ai, [
        {
          role: "user",
          parts: [...parts, { text: PROMPT }],
        },
      ]);

      if (!Array.isArray(parsed)) throw new Error("AI output was not an array");

      const valid = parsed.filter((item) => bulkQuizQuestionSchema.safeParse(item).success);
      if (valid.length !== parsed.length) {
        console.warn(`[${batchLabel}] ${parsed.length - valid.length}/${parsed.length} questions failed validation; keeping ${valid.length}`);
      }
      const fresh = dedupeByText([...jsonData, ...valid]);
      const added = fresh.length - jsonData.length;
      jsonData = fresh;

      await writeFile(join(import.meta.dirname, "..", "jsons", "quiz.json"), jsonData);

      for (const image of batch) {
        await rename(
          join(import.meta.dirname, "..", "quiz_images", image),
          join(import.meta.dirname, "..", "uploaded_images", image)
        );
      }
      console.log(`[${batchLabel}] extracted ${valid.length} questions (${added} new)`);
    } catch (error) {
      if (isDailyQuotaError(error)) {
        console.error("Daily quota exhausted — stopping. Remaining images stay in quiz_images/; resume after ~midnight Pacific time.");
        break;
      }
      console.error(`[${batchLabel}] FAILED (keeping images in quiz_images/ for retry): ${error.message}`);
    }
  }

  console.log(`Done. quiz.json now has ${jsonData.length} questions.`);
}

main();
