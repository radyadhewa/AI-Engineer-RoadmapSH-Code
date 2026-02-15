import { generateText } from "ai";
import { NextResponse } from "next/server";
import { z } from "zod";
import { gateway, gatewayModel } from "@/lib/ai/gateway";
import { buildAugmentedPrompt } from "@/lib/rag/augmentation";
import { retrieveRelevantChunks } from "@/lib/rag/retrieval";

const payloadSchema = z.object({
  question: z.string().min(1),
});

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    if (!process.env.AI_GATEWAY_API_KEY) {
      return NextResponse.json(
        {
          ok: false,
          error: "AI_GATEWAY_API_KEY is missing. Add it to your .env file.",
        },
        { status: 400 },
      );
    }

    const body = await request.json();
    const { question } = payloadSchema.parse(body);

    const chunks = await retrieveRelevantChunks(question);
    const prompt = buildAugmentedPrompt(question, chunks);

    const result = await generateText({
      model: gateway(gatewayModel),
      prompt,
      temperature: 0.2,
    });

    return NextResponse.json({
      ok: true,
      answer: result.text,
      sources: chunks,
    });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        error: error instanceof Error ? error.message : "Unknown error while generating response",
      },
      { status: 400 },
    );
  }
}
