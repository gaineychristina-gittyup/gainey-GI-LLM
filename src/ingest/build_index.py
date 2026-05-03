"""End-to-end ingestion orchestrator: parse → chunk → embed → load.

CLI usage
---------

Bulk mode (full corpus):
    python -m src.ingest.build_index --csv data/corpus_metadata.csv

Single-file mode (good for first sanity check):
    python -m src.ingest.build_index \
        --pdf data/raw_pdfs/AGA/AGA_2025_Gastroparesis_Staller.pdf \
        --society AGA --title "AGA Clinical Practice Update on Gastroparesis" \
        --year 2025 --topic gastroparesis

The CSV format mirrors the columns in ``data/corpus_metadata.csv``:
    society,title,year,topic,doi,source_url,pdf_path

Per-document stats (chunks, tokens, time) are printed to stdout.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psycopg
import yaml
from dotenv import load_dotenv

from src.ingest.chunk import chunk_elements
from src.ingest.embed import (
    embed_and_insert_chunks,
    make_embedder_from_config,
    upsert_document,
)
from src.ingest.parse_pdfs import parse_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


@dataclass
class DocumentRow:
    society: str
    title: str
    year: int
    topic: Optional[str]
    doi: Optional[str]
    source_url: Optional[str]
    pdf_path: str


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_corpus_csv(csv_path: Path) -> list[DocumentRow]:
    rows: list[DocumentRow] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r.get("pdf_path"):
                continue
            rows.append(
                DocumentRow(
                    society=r["society"].strip(),
                    title=r["title"].strip(),
                    year=int(r["year"]),
                    topic=(r.get("topic") or None) and r["topic"].strip() or None,
                    doi=(r.get("doi") or None) and r["doi"].strip() or None,
                    source_url=(r.get("source_url") or None) and r["source_url"].strip() or None,
                    pdf_path=r["pdf_path"].strip(),
                )
            )
    return rows


def ingest_one(
    conn: psycopg.Connection,
    doc: DocumentRow,
    config: dict,
    embedder,
) -> dict:
    """Run parse → chunk → embed → insert for a single document.

    Returns a stats dict for logging.
    """
    pdf_path = (REPO_ROOT / doc.pdf_path).resolve()
    if not pdf_path.exists():
        # Allow callers to pass absolute paths too.
        alt = Path(doc.pdf_path)
        if alt.exists():
            pdf_path = alt.resolve()
        else:
            raise FileNotFoundError(f"PDF not found: {doc.pdf_path}")

    t0 = time.time()
    parsing_cfg = config.get("parsing", {})
    elements = parse_pdf(
        pdf_path,
        strategy=parsing_cfg.get("strategy", "hi_res"),
        infer_table_structure=parsing_cfg.get("infer_table_structure", True),
        fallback_after_seconds=int(parsing_cfg.get("fallback_to_fast_after_seconds", 600)),
    )
    t_parse = time.time() - t0

    chunking_cfg = config.get("chunking", {})
    chunks = chunk_elements(
        elements,
        target_tokens=int(chunking_cfg.get("target_tokens", 500)),
        min_tokens=int(chunking_cfg.get("min_tokens", 200)),
        max_tokens=int(chunking_cfg.get("max_tokens", 800)),
        overlap_tokens=int(chunking_cfg.get("overlap_tokens", 50)),
        tokenizer=chunking_cfg.get("tokenizer", "cl100k_base"),
    )
    t_chunk = time.time() - t0 - t_parse

    document_id = upsert_document(
        conn,
        society=doc.society,
        title=doc.title,
        year=doc.year,
        topic=doc.topic,
        doi=doc.doi,
        source_url=doc.source_url,
        pdf_path=doc.pdf_path,
    )

    inserted = embed_and_insert_chunks(conn, document_id, chunks, embedder)
    t_total = time.time() - t0

    total_tokens = sum(c.token_count for c in chunks)
    rec_with_id = sum(1 for c in chunks if c.recommendation_id)
    return {
        "document_id": document_id,
        "pdf": pdf_path.name,
        "elements": len(elements),
        "chunks": len(chunks),
        "inserted": inserted,
        "recommendation_chunks": rec_with_id,
        "total_tokens": total_tokens,
        "parse_s": round(t_parse, 1),
        "chunk_s": round(t_chunk, 1),
        "total_s": round(t_total, 1),
    }


def get_db_connection() -> psycopg.Connection:
    url = os.environ.get("DATABASE_URL", "postgresql://gi:gi@localhost:5432/gi_guidelines")
    return psycopg.connect(url)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build the GI guidelines RAG index.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="Path to corpus_metadata.csv")
    src.add_argument("--pdf", type=Path, help="Path to a single PDF (single-file mode)")

    # Single-file metadata (only used with --pdf)
    parser.add_argument("--society", help="Society code (AGA, ACG, ASGE, AASLD)")
    parser.add_argument("--title", help="Guideline title")
    parser.add_argument("--year", type=int, help="Publication year")
    parser.add_argument("--topic", default=None)
    parser.add_argument("--doi", default=None)
    parser.add_argument("--source-url", default=None)

    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    embedder = make_embedder_from_config(config)

    if args.pdf:
        for required in ("society", "title", "year"):
            if getattr(args, required) is None:
                parser.error(f"--{required} is required with --pdf")
        rows = [
            DocumentRow(
                society=args.society,
                title=args.title,
                year=args.year,
                topic=args.topic,
                doi=args.doi,
                source_url=args.source_url,
                pdf_path=str(args.pdf),
            )
        ]
    else:
        rows = load_corpus_csv(args.csv)
        logger.info("Loaded %d documents from %s", len(rows), args.csv)

    conn = get_db_connection()
    grand_chunks = 0
    grand_inserted = 0
    try:
        for doc in rows:
            logger.info("Ingesting: %s — %s (%d)", doc.society, doc.title, doc.year)
            try:
                stats = ingest_one(conn, doc, config, embedder)
            except Exception as e:
                logger.exception("FAILED to ingest %s: %s", doc.pdf_path, e)
                continue
            grand_chunks += stats["chunks"]
            grand_inserted += stats["inserted"]
            logger.info(
                "  done: %d chunks (%d inserted, %d new), %d tokens, "
                "%d recs, parse=%ss chunk=%ss total=%ss",
                stats["chunks"], stats["inserted"], stats["inserted"],
                stats["total_tokens"], stats["recommendation_chunks"],
                stats["parse_s"], stats["chunk_s"], stats["total_s"],
            )

        # Final summary line — matches the format requested by the user.
        if len(rows) == 1:
            r = rows[0]
            print(
                f"Ingested {grand_chunks} chunks from {Path(r.pdf_path).name} "
                f"into documents+chunks tables."
            )
        else:
            print(
                f"Ingested {grand_chunks} chunks from {len(rows)} documents "
                f"into documents+chunks tables."
            )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
