import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export default function Home() {
  return (
    <main className="container mx-auto max-w-5xl px-6 py-10 space-y-8">
      <section className="space-y-3">
        <h1 className="text-3xl font-bold tracking-tight">RAG Learning Starter</h1>
        <p className="text-muted-foreground">
          This project is intentionally built for learning, not as a production-ready app.
          Use it to study the RAG pipeline end-to-end: indexing, retrieval, augmentation, and generation.
        </p>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>1) Indexing</CardTitle>
            <CardDescription>Chunk content and store embeddings in Postgres/pgvector.</CardDescription>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground space-y-2">
            <p>Code: <code>lib/rag/indexing.ts</code>, <code>lib/rag/chunking.ts</code></p>
            <p>DB: <code>lib/db/schema.ts</code>, <code>scripts/setup-db.ts</code></p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>2) Retrieval</CardTitle>
            <CardDescription>Fetch relevant chunks from your indexed knowledge base.</CardDescription>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground space-y-2">
            <p>Code: <code>lib/rag/retrieval.ts</code></p>
            <p>Try: <code>POST /api/chat</code> with a question in JSON.</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>3) Augmentation</CardTitle>
            <CardDescription>Build prompts that include retrieved context safely.</CardDescription>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground space-y-2">
            <p>Code: <code>lib/rag/augmentation.ts</code></p>
            <p>Route integration: <code>app/api/chat/route.ts</code></p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>4) Generation</CardTitle>
            <CardDescription>Call models via AI SDK + Vercel AI Gateway.</CardDescription>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground space-y-2">
            <p>Code: <code>lib/ai/gateway.ts</code>, <code>app/api/chat/route.ts</code></p>
            <p>Model/provider config from environment variables.</p>
          </CardContent>
        </Card>
      </section>

      <section className="flex flex-wrap gap-3">
        <Button asChild>
          <Link href="/api/health">Check API Health</Link>
        </Button>
        <Button asChild variant="secondary">
          <Link href="https://sdk.vercel.ai/docs" target="_blank">AI SDK Docs</Link>
        </Button>
      </section>
    </main>
  );
}
