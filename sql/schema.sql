-- GI guidelines RAG schema.
-- Embedding dimension is 1024 to match Voyage AI voyage-3-large.
-- If you switch to OpenAI text-embedding-3-large, change vector(1024) -> vector(3072)
-- and rebuild the HNSW index.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            SERIAL PRIMARY KEY,
    society       TEXT NOT NULL,            -- e.g. AGA, ACG, ASGE, AASLD
    title         TEXT NOT NULL,
    year          INT  NOT NULL,
    topic         TEXT,                     -- free-form clinical topic tag
    doi           TEXT,
    source_url    TEXT,
    pdf_path      TEXT,                     -- path on local disk, relative to repo root
    ingested_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id                 SERIAL PRIMARY KEY,
    document_id        INT REFERENCES documents(id) ON DELETE CASCADE,
    section_title      TEXT,
    recommendation_id  TEXT,                -- e.g. "Recommendation 3.2", "Statement 4"
    grade_evidence     TEXT,                -- low / moderate / high (or GRADE phrasing)
    grade_strength     TEXT,                -- strong / conditional / weak
    page_start         INT,
    page_end           INT,
    text               TEXT NOT NULL,
    embedding          vector(1024),
    token_count        INT
);

-- Dense semantic retrieval (HNSW with cosine distance).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Document-scoped lookups (per-society filters, etc.).
CREATE INDEX IF NOT EXISTS chunks_document_id_idx
    ON chunks (document_id);

-- BM25-ish keyword search via Postgres full-text (used in Phase 3 hybrid retrieval).
CREATE INDEX IF NOT EXISTS chunks_text_fts_idx
    ON chunks USING gin (to_tsvector('english', text));

-- Idempotency: prevents the embed step from inserting the same chunk twice
-- when build_index.py is re-run. (document_id, section_title, page_start)
-- is the natural key we de-dupe on in src/ingest/embed.py.
CREATE UNIQUE INDEX IF NOT EXISTS chunks_idempotent_idx
    ON chunks (document_id, COALESCE(section_title, ''), COALESCE(page_start, 0), md5(text));
