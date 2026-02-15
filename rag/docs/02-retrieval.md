# 02 - Retrieval

Goal: find chunks relevant to the user question.

Key file:

- `lib/rag/retrieval.ts`

Current behavior:

- Uses text-match search (`ILIKE`) first.
- Falls back to latest chunks if no match.

Next improvement ideas:

1. Replace keyword search with pgvector similarity search.
2. Add metadata filters (source, date, topic).
3. Add reranking before prompt construction.
