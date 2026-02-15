import { createOpenAI } from "@ai-sdk/openai";

const baseURL = process.env.AI_GATEWAY_BASE_URL ?? "https://gateway.ai.vercel.com/v1";
const apiKey = process.env.AI_GATEWAY_API_KEY;

export const gateway = createOpenAI({
  baseURL,
  apiKey: apiKey ?? "missing-key",
});

export const gatewayModel = process.env.AI_GATEWAY_MODEL ?? "openai/gpt-4.1-mini";
