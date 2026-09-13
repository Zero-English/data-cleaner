import { GoogleGenAI } from "@google/genai";
import "dotenv/config";
import { readdir } from "node:fs";

async function imageProcessor() {
  const images = await readdir("../quiz_images");
  console.log(images);
}

async function main() {
  const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
  const ai = new GoogleGenAI({
      apiKey: GEMINI_API_KEY,
  });
  const response = await ai.models.generateContent({
    model: 'gemini-3.6-flash',
    contents: 'Can you help Extract me some questions in JSON format from an Image?',
  });

  console.log(response.text);
}

imageProcessor()
// main();