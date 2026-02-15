import { relations } from "drizzle-orm";
import { integer, pgTable, serial, text, timestamp, vector } from "drizzle-orm/pg-core";

export const documents = pgTable("documents", {
  id: serial("id").primaryKey(),
  title: text("title").notNull(),
  content: text("content").notNull(),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
});

export const chunks = pgTable("chunks", {
  id: serial("id").primaryKey(),
  documentId: integer("document_id")
    .notNull()
    .references(() => documents.id, { onDelete: "cascade" }),
  content: text("content").notNull(),
  embedding: vector("embedding", { dimensions: 1536 }).notNull(),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
});

export const documentsRelations = relations(documents, ({ many }) => ({
  chunks: many(chunks),
}));

export const chunksRelations = relations(chunks, ({ one }) => ({
  document: one(documents, {
    fields: [chunks.documentId],
    references: [documents.id],
  }),
}));
