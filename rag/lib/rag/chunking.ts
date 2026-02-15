export const splitIntoChunks = (text: string, chunkSize = 500, overlap = 80): string[] => {
  if (!text.trim()) {
    return [];
  }

  const chunks: string[] = [];
  let start = 0;

  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    const chunk = text.slice(start, end).trim();

    if (chunk.length > 0) {
      chunks.push(chunk);
    }

    if (end === text.length) {
      break;
    }

    start = Math.max(end - overlap, start + 1);
  }

  return chunks;
};
