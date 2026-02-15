import { NextResponse } from "next/server";
import { z } from "zod";
import { indexDocument } from "@/lib/rag/indexing";

const payloadSchema = z.object({
  title: z.string().min(1),
  content: z.string().min(1),
});

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const payload = payloadSchema.parse(body);

    const result = await indexDocument(payload);

    return NextResponse.json({
      ok: true,
      ...result,
    });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        error: error instanceof Error ? error.message : "Unknown error while indexing",
      },
      { status: 400 },
    );
  }
}
