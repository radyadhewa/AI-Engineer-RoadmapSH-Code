import "dotenv/config";
import { sql } from "drizzle-orm";
import { db } from "@/lib/db/client";

const main = async () => {
  await db.execute(sql`CREATE EXTENSION IF NOT EXISTS vector;`);

  console.log("pgvector extension is ready.");
  console.log("Next: run `npm run db:generate` and `npm run db:migrate`.");
};

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
