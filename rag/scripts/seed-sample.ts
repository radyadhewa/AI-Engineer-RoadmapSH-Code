import "dotenv/config";
import { indexDocument } from "@/lib/rag/indexing";

const sampleDoc = {
  title: "RAG fundamentals",
  content:
    "Retrieval-Augmented Generation combines document retrieval with language model generation. " +
    "The indexing phase transforms source text into chunks and embeddings. " +
    "The retrieval phase fetches relevant chunks at query time. " +
    "The augmentation phase injects those chunks into the model prompt. " +
    "The generation phase creates the final answer grounded on context.",
};

const main = async () => {
  const result = await indexDocument(sampleDoc);
  console.log("Seed complete:", result);
};

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
