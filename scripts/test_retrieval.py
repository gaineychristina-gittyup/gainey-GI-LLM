#!/usr/bin/env python
"""Ad-hoc retrieval probe over the GI guidelines corpus.

  python scripts/test_retrieval.py "what's the recommended surveillance interval for low-grade dysplasia in Barrett's?"

Defaults to hybrid retrieval (dense via pgvector + BM25 via Postgres FTS,
combined with Reciprocal Rank Fusion). If COHERE_API_KEY is set, also
reranks the fused candidates with Cohere rerank-english-v3.0. Pass
--no-hybrid to use dense-only or --no-rerank to skip the reranker.

Knobs come from config.yaml ``retrieval:``; CLI flags override them.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Optional

import psycopg
import yaml
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def get_db_connection() -> psycopg.Connection:
    url = os.environ.get("DATABASE_URL", "postgresql://gi:gi@localhost:5432/gi_guidelines")
    return psycopg.connect(url)


def embed_query(text: str, model: str) -> list[float]:
    """Embed the query with Voyage. Uses input_type='query' (different
    encoder than the document side, per Voyage docs)."""
    import voyageai
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    resp = client.embed([text], model=model, input_type="query")
    return resp.embeddings[0]


def dense_search(conn, query_emb: list[float], k: int) -> list[tuple]:
    """Return [(chunk_id, score, ...)] ordered by ascending cosine distance."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, 1.0 - (c.embedding <=> %s::vector) AS score,
                   c.document_id, d.society, d.year, d.title,
                   c.section_title, c.recommendation_id,
                   c.grade_evidence, c.grade_strength,
                   c.element_type, c.page_start, c.page_end,
                   c.text
              FROM chunks c
              JOIN documents d ON d.id = c.document_id
             WHERE c.embedding IS NOT NULL
             ORDER BY c.embedding <=> %s::vector
             LIMIT %s
            """,
            (query_emb, query_emb, k),
        )
        return cur.fetchall()


def bm25_search(conn, query: str, k: int) -> list[tuple]:
    """Postgres full-text ranking. Same row shape as dense_search."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id,
                   ts_rank_cd(to_tsvector('english', c.text),
                              plainto_tsquery('english', %s)) AS score,
                   c.document_id, d.society, d.year, d.title,
                   c.section_title, c.recommendation_id,
                   c.grade_evidence, c.grade_strength,
                   c.element_type, c.page_start, c.page_end,
                   c.text
              FROM chunks c
              JOIN documents d ON d.id = c.document_id
             WHERE to_tsvector('english', c.text) @@ plainto_tsquery('english', %s)
             ORDER BY score DESC
             LIMIT %s
            """,
            (query, query, k),
        )
        return cur.fetchall()


def rrf_fuse(dense: list[tuple], bm25: list[tuple], k_rrf: int = 60) -> list[tuple]:
    """Reciprocal Rank Fusion: score = Σ 1/(k_rrf + rank).

    Returns rows in fused order. Each row is the original row from
    whichever list ranked it — we de-dup on chunk_id.
    """
    fused: dict[int, dict] = {}
    for rank, row in enumerate(dense, start=1):
        cid = row[0]
        fused.setdefault(cid, {"row": row, "score": 0.0})
        fused[cid]["score"] += 1.0 / (k_rrf + rank)
    for rank, row in enumerate(bm25, start=1):
        cid = row[0]
        fused.setdefault(cid, {"row": row, "score": 0.0})
        fused[cid]["score"] += 1.0 / (k_rrf + rank)
    ordered = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    return [
        # Replace the score column (index 1) with the fused score.
        (entry["row"][0], entry["score"], *entry["row"][2:])
        for entry in ordered
    ]


def cohere_rerank(query: str, rows: list[tuple], top_n: int, model: str) -> list[tuple]:
    """Rerank fused candidates with Cohere; returns top_n."""
    import cohere
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        return rows[:top_n]
    client = cohere.ClientV2(api_key=api_key)
    docs = [r[-1] for r in rows]  # text is the last column
    resp = client.rerank(query=query, documents=docs, model=model, top_n=top_n)
    return [
        (rows[r.index][0], float(r.relevance_score), *rows[r.index][2:])
        for r in resp.results
    ]


def format_result(row: tuple, idx: int) -> str:
    (cid, score, doc_id, society, year, title,
     section, rec_id, grade_ev, grade_str,
     element_type, page_start, page_end, text) = row

    head_bits = [f"[{idx}]", f"score={score:.3f}", f"{society} {year}"]
    if element_type and element_type != "prose":
        head_bits.append(f"({element_type})")
    if rec_id:
        head_bits.append(rec_id)
    if grade_str or grade_ev:
        g = " / ".join(x for x in [grade_str, grade_ev] if x)
        head_bits.append(f"GRADE: {g}")
    page = (
        f"p{page_start}" if page_start == page_end or page_end is None
        else f"pp{page_start}-{page_end}"
    ) if page_start else "p?"
    head_bits.append(page)

    head = "  ".join(head_bits)
    title_line = f"      {title[:100]}{'…' if len(title) > 100 else ''}"
    section_line = f"      § {section[:100]}" if section else ""
    snippet = textwrap.shorten(
        text.replace("\n", " "), width=380, placeholder=" …"
    )
    body = textwrap.indent(textwrap.fill(snippet, width=92), "      ")
    parts = [head, title_line]
    if section_line:
        parts.append(section_line)
    parts.append(body)
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="+", help="Natural-language clinical question.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--k", type=int, default=None, help="Override top_k for both stages.")
    parser.add_argument("--no-hybrid", action="store_true", help="Dense-only retrieval (skip BM25+RRF).")
    parser.add_argument("--no-rerank", action="store_true", help="Skip Cohere rerank even if key is set.")
    parser.add_argument("--society", default=None, help="Filter to a single society (AGA/ACG/ASGE/AASLD).")
    args = parser.parse_args(argv)

    query = " ".join(args.query)
    cfg = yaml.safe_load(open(args.config))
    rcfg = cfg.get("retrieval", {})
    ecfg = cfg.get("embedding", {})

    top_dense = args.k or int(rcfg.get("top_k_dense", 20))
    top_bm25 = args.k or int(rcfg.get("top_k_bm25", 20))
    top_n = args.k or int(rcfg.get("rerank_top_n", 8))
    voyage_model = ecfg.get("voyage_model", "voyage-3-large")
    cohere_model = rcfg.get("cohere_model", "rerank-english-v3.0")

    print(f"Q: {query}\n")

    query_emb = embed_query(query, voyage_model)
    conn = get_db_connection()
    register_vector(conn)
    try:
        dense = dense_search(conn, query_emb, top_dense)
        if args.no_hybrid:
            candidates = dense
            mode = "dense"
        else:
            bm25 = bm25_search(conn, query, top_bm25)
            candidates = rrf_fuse(dense, bm25)
            mode = f"hybrid (dense={len(dense)}, bm25={len(bm25)})"

        if args.society:
            soc = args.society.upper()
            candidates = [r for r in candidates if r[3] == soc]

        # Take a generous head of fused candidates into the reranker.
        head = candidates[: max(top_n * 3, top_n)]
        if not args.no_rerank and os.environ.get("COHERE_API_KEY"):
            final = cohere_rerank(query, head, top_n, cohere_model)
            mode += " + Cohere rerank"
        else:
            final = head[:top_n]
            if not args.no_rerank and not os.environ.get("COHERE_API_KEY"):
                mode += " (no rerank: COHERE_API_KEY unset)"

        print(f"Retrieval mode: {mode}\n")
        for i, row in enumerate(final, start=1):
            print(format_result(row, i))
            print()
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
