import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";
import { readdir } from "node:fs/promises";
import { join } from "node:path";
import { readFile, writeFile } from "jsonfile";
import { quizTypes, quizClasses } from "../schemas/quiz.schema";

dotenv.config({ path: join(import.meta.dirname, "..", ".env") });

const PROMPT = `
ROLE: 
You are a highly accurate educational data extraction agent.

INPUT: 
You will receive an image containing questions from a Class 6 English Guide Book.

OBJECTIVE:
Extract every valid quiz question visible in the image and return the results as a JSON array.

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

1. Extract ALL complete questions that are clearly visible in the image.
2. Preserve the original meaning and wording of each question.
3. Correct obvious OCR errors when the intended English text is unambiguous.
4. Do NOT invent, complete, or reconstruct text that is not reasonably readable.
5. Do NOT create questions from headings, instructions, examples, explanations, or answer keys unless they are clearly presented as quiz questions.
6. If a question has multiple-choice options, extract every visible option.
7. Preserve the order of the options as they appear in the image.
8. The "answer" field must contain the correct option text, not the option number or letter.
9. Do NOT invent an answer. If the correct answer cannot be determined reliably, exclude that question.
10. If fewer than 2 valid options are available, exclude the question.
11. If the question is incomplete or unreadable, exclude it.
12. Do not duplicate the same question.
13. Keep punctuation where it is meaningful.
14. Remove unnecessary numbering such as "1.", "2.", "(a)", etc. from questionText unless it is part of the actual question.
15. Every question must have:
    class: ["Class6"]
16. Do not infer a different class from the image. Always use ["Class6"].
17. quizType must be one of the values from quizTypes. Never invent a new quizType.
18. difficultyLevel must be exactly one of:
    "EASY", "MEDIUM", "HARD"
19. Choose difficulty based on the reasoning/grammar complexity of the question:
    - EASY: direct recall or simple identification
    - MEDIUM: requires some grammatical understanding or transformation
    - HARD: requires multiple grammatical concepts, nuanced reasoning, or more complex transformation
20. If the image contains an answer key, use it to determine the answer when it clearly corresponds to the question.
21. If no answer key is available, determine the answer only when you are confident.
22. Never use information from outside the provided image to invent missing question content.

QUESTION TYPE:
Assign the most appropriate quizType from the provided quizTypes list based on the actual question.

VALIDATION SCHEMA:

{
  "quizType": "<one of quizTypes>",
  "questionText": "<question>",
  "options": ["<option 1>", "<option 2>", "..."],
  "difficultyLevel": "<EASY | MEDIUM | HARD>",
  "answer": "<exact correct option text>",
  "class": ["Class6"]
}

FINAL CHECK BEFORE OUTPUT:
- Is the output valid JSON?
- Is it an array?
- Does every object contain exactly the required fields?
- Is every quizType allowed?
- Is every difficultyLevel uppercase and allowed?
- Is class exactly ["Class6"]?
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
  "class": ["Class6"]
}, {
  "quizType": "Transformation",
  "questionText": "\"It is very useful.\" what is the Negative form of the sentence?",
  "options": ["It is not very useful.", "It is not useful.", "It is useless.", "It isn't useless at all."],
  "difficultyLevel": "EASY",
  "answer": "It isn't useless at all.",
  "class": ["Class6"]
}, {
  "quizType": "Sentence",
  "questionText": "\"May Allah bless you.\" what type of sentence is this?",
  "options": ["Optative", "Imperative", "Exclamatory", "Assertive"],
  "difficultyLevel": "HARD",
  "answer": "Optative",
  "class": ["Class6"]
}]
`

async function main() {
  const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
  const ai = new GoogleGenAI({
      apiKey: GEMINI_API_KEY,
  });

  const images = await readdir("../quiz_images");

  for(const image of images){
    const uploadedFile = await ai.files.upload({
    file: `../quiz_images/${image}`,
    config: { mimeType: "image/jpeg" },
  });

  let state = (await ai.files.get({ name: uploadedFile.name })).state;
  while (state !== "ACTIVE") {
    state = (await ai.files.get({ name: uploadedFile.name })).state;
    if (state === "FAILED") throw new Error("File upload failed");
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }

  const jsonData = await readFile("../jsons/quiz.json");
  const response = await ai.models.generateContent({
    model: 'gemini-3.6-flash',
    contents: [
      {
        role: "user",
        parts: [
          { fileData: { fileUri: uploadedFile.uri, mimeType: uploadedFile.mimeType } },
          { text: PROMPT },
        ],
      },
    ],
  });
  jsonData.push(response.text);
  await writeFile("../jsons/quiz.json", jsonData);
  }
}

main();
