"""Batch embed chunks with Voyage (or OpenAI fallback) and load into pgvector.

Idempotency
-----------
The unique index ``chunks_idempotent_idx`` on
``(document_id, COALESCE(section_title,''), COALESCE(page_start,0), md5(text))``
means re-running the pipeline on the same PDF will not duplicate rows. We use
``ON CONFLICT DO NOTHING`` on insert so re-runs are cheap and safe.

Embedding provider selection
----------------------------
- If ``VOYAGE_API_KEY`` is set we use ``voyage-3-large`` (1024-dim).
- Otherwise we fall back to OpenAI ``text-embedding-3-large`` (3072-dim).
  The schema is hard-coded to 1024-dim, so the OpenAI fallback ALSO requires
  changing ``vector(1024)`` -> ``vector(3072)`` in ``sql/schema.sql`` and
  rebuilding the HNSW index. The fallback exists for portability; the project
  default is Voyage.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import psycopg

from src.ingest.chunk import Chunk

logger = logging.getLogger(__name__)


# --- Embedding providers ----------------------------------------------------


@dataclass
class EmbedderConfig:
    provider: str  # "voyage" | "openai"
    model: str
    batch_size: int = 100
    expected_dim: int = 1024


class Embedder:
    """Thin abstraction over Voyage / OpenAI embedding APIs."""

    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        if cfg.provider == "voyage":
            import voyageai  # local import to keep cold start light

            key = os.environ.get("VOYAGE_API_KEY")
            if not key:
                raise RuntimeError(
                    "VOYAGE_API_KEY is not set. Either export it or switch the "
                    "embedding.provider in config.yaml to 'openai'."
                )
            self._client = voyageai.Client(api_key=key)
        elif cfg.provider == "openai":
            import openai

            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            self._client = openai.OpenAI(api_key=key)
        else:
            raise ValueError(f"Unknown embedding provider: {cfg.provider}")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.cfg.provider == "voyage":
            # input_type="document" tells Voyage to use the doc-side encoder
            # (different from the query-side encoder used at retrieval time).
            resp = self._client.embed(
                texts,
                model=self.cfg.model,
                input_type="document",
            )
            return resp.embeddings
        # openai
        resp = self._client.embeddings.create(model=self.cfg.model, input=texts)
        return [d.embedding for d in resp.data]


def make_embedder_from_config(config: dict) -> Embedder:
    """Build an Embedder from the parsed config.yaml dict."""
    emb_cfg = config.get("embedding", {})
    provider = emb_cfg.get("provider", "voyage")
    if provider == "voyage":
        model = emb_cfg.get("voyage_model", "voyage-3-large")
        expected_dim = 1024
    else:
        model = emb_cfg.get("openai_model", "text-embedding-3-large")
        expected_dim = 3072
    return Embedder(
        EmbedderConfig(
            provider=provider,
            model=model,
            batch_size=int(emb_cfg.get("batch_size", 100)),
            expected_dim=expected_dim,
        )
    )


# --- DB I/O -----------------------------------------------------------------


def upsert_document(
    conn: psycopg.Connection,
    society: str,
    title: str,
    year: int,
    topic: Optional[str] = None,
    doi: Optional[str] = None,
    source_url: Optional[str] = None,
    pdf_path: Optional[str] = None,
) -> int:
    """Insert a document row and return its id.

    A document is uniquely identified for our purposes by (society, title, year).
    If a row with the same triple already exists we return its id (no duplicate).
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM documents WHERE society = %s AND title = %s AND year = %s",
            (society, title, year),
        )
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute(
            """
            INSERT INTO documents (society, title, year, topic, doi, source_url, pdf_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (society, title, year, topic, doi, source_url, pdf_path),
        )
        new_id = cur.fetchone()[0]
    conn.commit()
    return new_id


def embed_and_insert_chunks(
    conn: psycopg.Connection,
    document_id: int,
    chunks: list[Chunk],
    embedder: Embedder,
) -> int:
    """Embed chunks in batches and insert into the chunks table.

    Returns the number of newly inserted rows (excludes idempotent skips).
    """
    if not chunks:
        return 0

    # Register the pgvector adapter so we can pass Python lists as vector params.
    from pgvector.psycopg import register_vector
    register_vector(conn)

    inserted = 0
    bs = embedder.cfg.batch_size
    total = len(chunks)
    for i in range(0, total, bs):
        batch = chunks[i : i + bs]
        t0 = time.time()
        embeddings = embedder.embed_batch([c.text for c in batch])

        # Sanity check: embedding dim must match schema.
        if embeddings and len(embeddings[0]) != embedder.cfg.expected_dim:
            raise RuntimeError(
                f"Embedding dim {len(embeddings[0])} != expected "
                f"{embedder.cfg.expected_dim}. Did you switch providers without "
                f"updating sql/schema.sql?"
            )

        rows = [
            (
                document_id,
                c.section_title,
                c.recommendation_id,
                c.grade_evidence,
                c.grade_strength,
                c.page_start,
                c.page_end,
                c.text,
                emb,
                c.token_count,
                c.element_type,
                c.table_html,
                c.figure_image_path,
            )
            for c, emb in zip(batch, embeddings)
        ]

        batch_inserted = 0
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (
                    document_id, section_title, recommendation_id,
                    grade_evidence, grade_strength,
                    page_start, page_end, text, embedding, token_count,
                    element_type, table_html, figure_image_path
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT DO NOTHING
                """,
                rows,
            )
            # rowcount is per-statement after executemany in psycopg3:
            # it returns the cumulative number of inserted rows, with -1 if unknown.
            if cur.rowcount and cur.rowcount > 0:
                batch_inserted = cur.rowcount
        conn.commit()
        inserted += batch_inserted

        elapsed = time.time() - t0
        logger.info(
            "  embedded+inserted batch %d-%d / %d (%.1fs, %d new rows)",
            i + 1, min(i + bs, total), total, elapsed, batch_inserted,
        )

    return inserted
