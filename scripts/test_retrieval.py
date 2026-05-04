#!/usr/bin/env python
"""Interactive retrieval probe — no LLM involved.

  python scripts/test_retrieval.py "what's the recommended H. pylori regimen for a penicillin-allergic patient?"

Defaults to hybrid retrieval (dense via pgvector + BM25 via Postgres FTS
fused with RRF) plus Cohere rerank when COHERE_API_KEY is set. Both steps
are configurable in ``config.yaml`` under ``retrieval:``.

Filter flags map directly to :func:`src.retrieve.retrieve` ``filters=``:

  --society ACG --year-min 2023 --topic "H. pylori"
  --element-types recommendation,table

The filter list arguments accept comma-separated values or repeated flags.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _csv_list(s: str | None) -> list[str] | None:
    if not s:
        return None
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def format_result(idx: int, row: dict) -> str:
    """Per-result block matching the spec output format."""
    score = row.get("relevance_score")
    score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
    head = f"--- Result {idx} (score: {score_str}) ---"

    society = row.get("society") or "?"
    year = row.get("year") or "?"
    title = (row.get("title") or "").strip()
    citation = f"[{society} {year}, \"{title}\"]"

    page_start, page_end = row.get("page_start"), row.get("page_end")
    if page_start and page_end and page_start != page_end:
        page = f"Page {page_start}-{page_end}"
    elif page_start:
        page = f"Page {page_start}"
    else:
        page = "Page unknown"

    rec_id = row.get("recommendation_id")
    page_line = f"{page}{', ' + rec_id if rec_id else ''}"

    grade_bits = []
    if (gs := row.get("grade_strength")):
        grade_bits.append(f"{gs.title()} recommendation")
    if (ge := row.get("grade_evidence")):
        grade_bits.append(f"{ge.replace('_',' ')}-quality evidence")
    grade_line = f"GRADE: {'; '.join(grade_bits)}" if grade_bits else None

    etype = row.get("element_type") or "prose"
    type_line = f"Element type: {etype}"

    text = (row.get("text") or "").strip()
    body = textwrap.fill(text, width=92)

    extra = []
    if etype == "table" and row.get("table_html"):
        extra.append(f"  table_html: {len(row['table_html'])} chars stored")
    if etype == "figure_caption" and row.get("figure_image_path"):
        extra.append(f"  figure_image_path: {row['figure_image_path']}")

    parts = [head, citation, page_line]
    if grade_line:
        parts.append(grade_line)
    parts.append(type_line)
    parts.append("")
    parts.append(body)
    if extra:
        parts.append("")
        parts.extend(extra)
    parts.append("")
    parts.append("---")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", nargs="+", help="Natural-language clinical question.")
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Number of results to return (default from config.yaml retrieval.top_k).",
    )
    parser.add_argument(
        "--no-rerank", action="store_true",
        help="Skip the reranker; return RRF-fused order directly.",
    )

    # Filter flags — map to src.retrieve.retrieve(filters={...})
    parser.add_argument("--society", type=str, default=None,
                        help="Comma-separated list, e.g. 'ACG,AGA'.")
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=None)
    parser.add_argument("--topic", type=str, default=None,
                        help="Comma-separated topic substrings (ILIKE).")
    parser.add_argument("--element-types", type=str, default=None,
                        help="Comma list: prose,recommendation,table,figure_caption,key_concept")
    args = parser.parse_args(argv)

    # Late import so --help works even if DB / API keys are misconfigured.
    from src.retrieve import retrieve

    query = " ".join(args.query)
    filters: dict = {}
    if (s := _csv_list(args.society)):
        filters["society"] = [x.upper() for x in s]
    if args.year_min is not None:
        filters["year_min"] = args.year_min
    if args.year_max is not None:
        filters["year_max"] = args.year_max
    if (t := _csv_list(args.topic)):
        filters["topic"] = t
    if (e := _csv_list(args.element_types)):
        filters["element_types"] = e

    print(f"Q: {query}")
    if filters:
        print(f"Filters: {filters}")
    print()

    rows = retrieve(
        query,
        filters=filters or None,
        top_k=args.top_k or 6,
        rerank=not args.no_rerank,
    )

    if not rows:
        print("(no results — check filters / corpus state)")
        return 0

    for i, row in enumerate(rows, start=1):
        print(format_result(i, row))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
