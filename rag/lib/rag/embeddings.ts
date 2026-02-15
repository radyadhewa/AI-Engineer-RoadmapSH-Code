const EMBEDDING_DIMENSIONS = 1536;

const deterministicValue = (source: string, index: number): number => {
  const code = source.charCodeAt(index % source.length) || 0;
  return ((code + index * 31) % 1000) / 1000;
};

export const createLearningEmbedding = async (text: string): Promise<number[]> => {
  const input = text.trim();

  if (!input) {
    return Array.from({ length: EMBEDDING_DIMENSIONS }, () => 0);
  }

  return Array.from({ length: EMBEDDING_DIMENSIONS }, (_, index) => deterministicValue(input, index));
};
