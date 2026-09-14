import { z } from "zod";

export const quizTypes = [
    "ENGLISH_TO_BANGLA",
    "BANGLA_TO_ENGLISH",
    "SYNONYMS",
    "ANTONYMS",
    "MIXED",
    "IDIOMS_AND_PHRASES",
    "PREPOSITIONS",
    "TRUE_FALSE",
    "Parts of Speech",
    "Singular-Plural",
    "Auxiliary Verb",
    "Define in English",
    "Spelling",
    "Right Form of Verb",
    "Transformation",
    "Punctuation",
    "Sentence",
    "Basic Grammar",
    "Fill in the blanks",
    "Sentence Structure"
]

export const quizClasses = [
  "PrePrimary",
  "Class1",
  "Class2",
  "Class3",
  "Class4",
  "Class5",
  "Class6",
  "Class7",
  "Class8",
  "SSC",
  "HSC",
  "IELTS",
  "TOEFL",
  "University",
  "Masters",
  "Diploma",
  "BCS",
  "JOB",
]

export const difficultyLevels = ["EASY", "MEDIUM", "HARD"];

export const quizClassEnumSchema = z.enum(quizClasses);
export const quizTypeEnumSchema = z.enum(quizTypes);
export const difficultyLevelEnumSchema = z.enum(difficultyLevels);


export const bulkQuizQuestionSchema = z.object({
  quizType: quizTypeEnumSchema,
  questionText: z.string().trim().min(1, "Question text is required"),
  options: z.array(z.string().min(1)).min(2, "At least 2 options are required"),
  difficultyLevel: difficultyLevelEnumSchema,
  answer: z.string().trim().min(1, "Answer is required"),
  class: z.array(quizClassEnumSchema).default([]),
});