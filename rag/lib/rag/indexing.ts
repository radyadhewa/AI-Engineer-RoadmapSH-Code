import { db } from "@/lib/db/client";
import { chunks, documents } from "@/lib/db/schema";
import { splitIntoChunks } from "@/lib/rag/chunking";
import { createLearningEmbedding } from "@/lib/rag/embeddings";

type IndexDocumentInput = {
  title: string;
  content: string;
};

export const indexDocument = async ({ title, content }: IndexDocumentInput) => {
  const [document] = await db
    .insert(documents)
    .values({
      title,
      content,
    })
    .returning({ id: documents.id });

  const textChunks = splitIntoChunks(content);

  for (const chunkText of textChunks) {
    const embedding = await createLearningEmbedding(chunkText);

    await db.insert(chunks).values({
      documentId: document.id,
      content: chunkText,
      embedding,
    });
  }

  return {
    documentId: document.id,
    chunkCount: textChunks.length,
  };
};
