import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json({
    ok: true,
    message: "RAG learning API is ready.",
    timestamp: new Date().toISOString(),
  });
}
