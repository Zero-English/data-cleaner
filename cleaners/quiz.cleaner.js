import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";
import { readdir } from "node:fs/promises";
import { join } from "node:path";

dotenv.config({ path: join(import.meta.dirname, "..", ".env") });

async function main() {
  const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
  const ai = new GoogleGenAI({
      apiKey: GEMINI_API_KEY,
  });
  
  const images = await readdir("../quiz_images");
  console.log(images);
  console.log(`Image File: `, images[0])
  console.log("\n\n")

  const uploadedFile = await ai.files.upload({
    file: `../quiz_images/${images[0]}`,
    config: { mimeType: "image/jpeg" },
  });

  let state = (await ai.files.get({ name: uploadedFile.name })).state;
  while (state !== "ACTIVE") {
    state = (await ai.files.get({ name: uploadedFile.name })).state;
    if (state === "FAILED") throw new Error("File upload failed");
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }

  const response = await ai.models.generateContent({
    model: 'gemini-3.6-flash',
    contents: [
      {
        role: "user",
        parts: [
          { fileData: { fileUri: uploadedFile.uri, mimeType: uploadedFile.mimeType } },
          { text: 'Can you view the contents of the uploaded image?' },
        ],
      },
    ],
  });

  console.log(response.text);
}

main();