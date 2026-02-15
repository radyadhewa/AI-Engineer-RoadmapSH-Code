# 01 - Indexing

Goal: convert raw text into chunks and store them with embeddings.

Key files:

- `lib/rag/chunking.ts`
- `lib/rag/embeddings.ts`
- `lib/rag/indexing.ts`
- `scripts/seed-sample.ts`

Study steps:

1. Read chunking logic and adjust `chunkSize` / `overlap`.
2. See how each chunk gets an embedding.
3. Observe insert flow into `documents` + `chunks` tables.
4. Run `npm run seed` and inspect records.
