# RAG Learning Starter (Next.js 14 + AI SDK + Drizzle + Postgres/pgvector)

This folder is a **learning-first RAG project** using your requested stack:

- Next.js 14 (App Router) + TypeScript
- AI SDK
- Vercel AI Gateway
- Drizzle ORM
- Postgres + pgvector
- shadcn-ui + TailwindCSS

It is intentionally simple and focused on understanding concepts, **not** production hardening.

## What to learn where

- **Indexing**: `lib/rag/indexing.ts`, `lib/rag/chunking.ts`, `scripts/seed-sample.ts`
- **Retrieval**: `lib/rag/retrieval.ts`
- **Augmentation**: `lib/rag/augmentation.ts`
- **Generation**: `app/api/chat/route.ts`, `lib/ai/gateway.ts`
- **Data schema**: `lib/db/schema.ts`, `drizzle.config.ts`

## Quick start

1. Install dependencies (already done if you cloned this folder as-is):

	```bash
	npm install
	```

2. Configure environment:

	```bash
	cp .env.example .env
	```

	Fill in:
	- `DATABASE_URL`
	- `AI_GATEWAY_API_KEY`

3. Prepare database + extension:

	```bash
	npm run db:setup
	npm run db:generate
	npm run db:migrate
	```

4. Seed sample content:

	```bash
	npm run seed
	```

5. Run app:

	```bash
	npm run dev
	```

## API endpoints for experiments

- `GET /api/health`
- `POST /api/index` with body:

  ```json
  {
	 "title": "My doc",
	 "content": "Your text content here"
  }
  ```

- `POST /api/chat` with body:

  ```json
  {
	 "question": "What does the indexed document say about indexing?"
  }
  ```

## Notes

- Retrieval in this starter uses a simple text-match fallback so the flow is easy to understand.
- You can evolve `lib/rag/retrieval.ts` to true vector similarity search once you are comfortable.
