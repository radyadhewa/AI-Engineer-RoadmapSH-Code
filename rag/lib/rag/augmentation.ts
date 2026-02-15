type RetrievedChunk = {
  id: number;
  content: string;
};

export const buildAugmentedPrompt = (question: string, retrievedChunks: RetrievedChunk[]) => {
  const context = retrievedChunks
    .map((chunk, index) => `[Context ${index + 1}] ${chunk.content}`)
    .join("\n\n");

  return [
    "You are a helpful assistant.",
    "Answer the question using only the provided context when possible.",
    "If context is insufficient, clearly say that context is insufficient.",
    "",
    `Question: ${question}`,
    "",
    "Context:",
    context || "(No context found)",
  ].join("\n");
};
