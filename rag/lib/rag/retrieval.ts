import { desc, ilike } from "drizzle-orm";
import { db } from "@/lib/db/client";
import { chunks } from "@/lib/db/schema";

export const retrieveRelevantChunks = async (question: string, limit = 5) => {
  const safeQuestion = question.trim();

  if (!safeQuestion) {
    return [];
  }

  const keywordResults = await db
    .select({
      id: chunks.id,
      content: chunks.content,
    })
    .from(chunks)
    .where(ilike(chunks.content, `%${safeQuestion}%`))
    .orderBy(desc(chunks.id))
    .limit(limit);

  if (keywordResults.length > 0) {
    return keywordResults;
  }

  return db
    .select({
      id: chunks.id,
      content: chunks.content,
    })
    .from(chunks)
    .orderBy(desc(chunks.id))
    .limit(limit);
};
