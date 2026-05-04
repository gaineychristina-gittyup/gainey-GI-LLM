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
import concurrent.futures
import csv
import logging
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psycopg
import yaml
from dotenv import load_dotenv

from src.ingest.chunk import Chunk, chunk_elements
from src.ingest.embed import (
    embed_and_insert_chunks,
    make_embedder_from_config,
    upsert_document,
)
from src.ingest.extract_figures import crop_figure
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


FIGURES_DIR = REPO_ROOT / "data" / "parsed" / "figures"


def _resolve_pdf_path(pdf_path_str: str) -> Path:
    pdf_path = (REPO_ROOT / pdf_path_str).resolve()
    if pdf_path.exists():
        return pdf_path
    alt = Path(pdf_path_str)
    if alt.exists():
        return alt.resolve()
    raise FileNotFoundError(f"PDF not found: {pdf_path_str}")


def parse_and_chunk(doc: DocumentRow, config: dict) -> tuple[DocumentRow, list[Chunk], int, float, float]:
    """Worker entry point: parse PDF + chunk it.

    Runs in a ProcessPoolExecutor worker so the heavy parse step can run in
    parallel across documents. Returns (doc, chunks, n_elements, parse_s,
    chunk_s) so the main process can extract figures, embed, and insert.
    """
    pdf_path = _resolve_pdf_path(doc.pdf_path)

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
    return doc, chunks, len(elements), t_parse, t_chunk


def _materialize_figures(
    pdf_path: Path,
    document_id: int,
    chunks: list[Chunk],
) -> int:
    """For each figure_caption chunk, crop the figure region into a PNG and
    populate ``chunk.figure_image_path``. Returns the number of figures saved.

    Skips silently if no figure_caption chunks are present. Per-document
    figure counter ``n`` increments across the whole document so two figures
    on the same page get distinct filenames.
    """
    fig_chunks = [c for c in chunks if c.element_type == "figure_caption"]
    if not fig_chunks:
        return 0
    saved = 0
    n = 0
    for c in fig_chunks:
        n += 1
        page = c.page_start or 1
        rel_path = f"data/parsed/figures/{document_id}_p{page}_fig{n}.png"
        out_path = REPO_ROOT / rel_path
        layout_w, layout_h = (c.figure_layout_size or (None, None))
        ok = crop_figure(
            pdf_path=pdf_path,
            page_number=page,
            bbox_layout=c.figure_bbox,
            layout_width=layout_w,
            layout_height=layout_h,
            output_path=out_path,
        )
        if ok:
            c.figure_image_path = rel_path
            saved += 1
    return saved


def ingest_parsed(
    conn: psycopg.Connection,
    doc: DocumentRow,
    chunks: list[Chunk],
    n_elements: int,
    t_parse: float,
    t_chunk: float,
    embedder,
) -> dict:
    """Main-process tail: figure extraction + upsert + embed + insert."""
    pdf_path = _resolve_pdf_path(doc.pdf_path)
    t0 = time.time()

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

    n_figures_saved = _materialize_figures(pdf_path, document_id, chunks)

    inserted = embed_and_insert_chunks(conn, document_id, chunks, embedder)
    t_tail = time.time() - t0
    t_total = t_parse + t_chunk + t_tail

    type_counts = Counter(c.element_type for c in chunks)
    return {
        "document_id": document_id,
        "pdf": pdf_path.name,
        "elements": n_elements,
        "chunks": len(chunks),
        "inserted": inserted,
        "type_counts": dict(type_counts),
        "figures_saved": n_figures_saved,
        "total_tokens": sum(c.token_count for c in chunks),
        "parse_s": round(t_parse, 1),
        "chunk_s": round(t_chunk, 1),
        "tail_s": round(t_tail, 1),
        "total_s": round(t_total, 1),
    }


def ingest_one(
    conn: psycopg.Connection,
    doc: DocumentRow,
    config: dict,
    embedder,
) -> dict:
    """Serial single-doc ingest used by --pdf mode and --workers=1 fallback."""
    doc, chunks, n_elements, t_parse, t_chunk = parse_and_chunk(doc, config)
    return ingest_parsed(conn, doc, chunks, n_elements, t_parse, t_chunk, embedder)


def get_db_connection() -> psycopg.Connection:
    url = os.environ.get("DATABASE_URL", "postgresql://gi:gi@localhost:5432/gi_guidelines")
    return psycopg.connect(url)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=True)
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
    parser.add_argument(
        "--workers", type=int, default=None,
        help=(
            "Parallel parse workers. Each worker loads detectron2 + a torch "
            "model so memory per worker is ~1.5–2 GB. Default: "
            "min(cpu_count // 2, 4). Set to 1 to disable parallelism."
        ),
    )
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

    if args.workers is None:
        args.workers = min((os.cpu_count() or 2) // 2, 4)
    args.workers = max(1, args.workers)
    if len(rows) > 1 and args.workers > 1:
        logger.info("Parallel parse with %d workers", args.workers)

    conn = get_db_connection()
    grand_chunks = 0
    grand_inserted = 0
    grand_type_counts: Counter = Counter()
    grand_figures = 0
    try:
        if len(rows) == 1 or args.workers == 1:
            iterator = _serial_iter(rows, config)
        else:
            iterator = _parallel_iter(rows, config, args.workers)

        for parse_result in iterator:
            if parse_result is None:
                continue
            doc, chunks, n_elements, t_parse, t_chunk = parse_result
            logger.info(
                "Loading: %s — %s (%d)  [parse=%.1fs chunk=%.1fs, %d chunks]",
                doc.society, doc.title, doc.year, t_parse, t_chunk, len(chunks),
            )
            try:
                stats = ingest_parsed(
                    conn, doc, chunks, n_elements, t_parse, t_chunk, embedder
                )
            except Exception as e:
                logger.exception("FAILED to load %s: %s", doc.pdf_path, e)
                continue
            grand_chunks += stats["chunks"]
            grand_inserted += stats["inserted"]
            grand_type_counts.update(stats["type_counts"])
            grand_figures += stats["figures_saved"]
            tc = stats["type_counts"]
            logger.info(
                "  done: %d chunks (%d new) [prose=%d, rec=%d, table=%d, "
                "figure=%d (%d saved), key_concept=%d], %d tokens, "
                "tail=%ss total=%ss",
                stats["chunks"], stats["inserted"],
                tc.get("prose", 0), tc.get("recommendation", 0),
                tc.get("table", 0), tc.get("figure_caption", 0),
                stats["figures_saved"], tc.get("key_concept", 0),
                stats["total_tokens"], stats["tail_s"], stats["total_s"],
            )

        # Final summary — typed counts called out per Phase 2 spec.
        target = (
            Path(rows[0].pdf_path).name if len(rows) == 1
            else f"{len(rows)} documents"
        )
        print(f"Ingested {grand_chunks} chunks from {target} into documents+chunks tables.")
        print(f"  - {grand_type_counts.get('prose', 0)} prose")
        print(f"  - {grand_type_counts.get('recommendation', 0)} recommendations")
        print(f"  - {grand_type_counts.get('table', 0)} tables (HTML preserved)")
        print(
            f"  - {grand_type_counts.get('figure_caption', 0)} figure captions "
            f"({grand_figures} images saved to data/parsed/figures/)"
        )
        print(f"  - {grand_type_counts.get('key_concept', 0)} key concepts")
    finally:
        conn.close()
    return 0


def _serial_iter(rows, config):
    for doc in rows:
        try:
            yield parse_and_chunk(doc, config)
        except Exception as e:
            logger.exception("FAILED to parse %s: %s", doc.pdf_path, e)
            yield None


def _parallel_iter(rows, config, workers):
    """Submit all docs to a process pool and yield results as they complete.

    We use 'spawn' explicitly because unstructured + detectron2 + torch are
    not fork-safe on macOS. Each worker imports unstructured lazily on first
    parse_pdf call, which costs ~3-5s of warm-up amortized across the workload.
    """
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx
    ) as pool:
        futures = {pool.submit(parse_and_chunk, doc, config): doc for doc in rows}
        for fut in concurrent.futures.as_completed(futures):
            doc = futures[fut]
            try:
                yield fut.result()
            except Exception as e:
                logger.exception("FAILED to parse %s: %s", doc.pdf_path, e)
                yield None


if __name__ == "__main__":
    sys.exit(main())
