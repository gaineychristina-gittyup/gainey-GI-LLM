"""Hybrid retrieval: pgvector cosine + Postgres FTS BM25, fused with RRF.

Filters apply *in the SQL WHERE clause* on both branches so good candidates
are not lost to post-hoc filtering. Each branch returns ``n_dense`` /
``n_bm25`` rows; Reciprocal Rank Fusion combines them and we keep the top
``n_fused``.

Returned rows are dicts with the chunk text plus the document metadata
needed for citation. The reranker reads these directly.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterable

import psycopg
from pgvector.psycopg import register_vector

from src.retrieve.embedder import embed_query

logger = logging.getLogger(__name__)

# Columns selected from chunks/documents — kept in one place so the dense
# and BM25 branches return identical row shapes.
_SELECT_COLS = """
    c.id              AS chunk_id,
    c.document_id     AS document_id,
    d.society         AS society,
    d.year            AS year,
    d.title           AS title,
    d.topic           AS topic,
    d.doi             AS doi,
    d.source_url      AS source_url,
    c.section_title   AS section_title,
    c.recommendation_id AS recommendation_id,
    c.grade_evidence  AS grade_evidence,
    c.grade_strength  AS grade_strength,
    c.element_type    AS element_type,
    c.page_start      AS page_start,
    c.page_end        AS page_end,
    c.token_count     AS token_count,
    c.text            AS text,
    c.table_html      AS table_html,
    c.figure_image_path AS figure_image_path
"""


# Typed chunks that we always want a fair shot at the candidate pool.
# 'figure_caption' is intentionally excluded — figure captions are useful when
# the user explicitly asks for an algorithm/diagram, but they tend to crowd
# out actual recommendation text on factual questions.
_BOOSTED_TYPES = ("table", "recommendation", "key_concept")


def hybrid_search(
    query: str,
    *,
    filters: dict[str, Any] | None = None,
    n_dense: int = 60,
    n_bm25: int = 60,
    n_fused: int = 20,
    rrf_k: int = 60,
    boost_typed: bool = True,
    n_typed_dense: int = 20,
    n_typed_bm25: int = 20,
) -> list[dict[str, Any]]:
    """Run dense + BM25 in parallel and return the RRF-fused top ``n_fused``.

    ``filters`` recognized keys are documented on :func:`src.retrieve.retrieve`.

    When ``boost_typed`` is True and the caller hasn't already constrained
    ``element_types``, we run an additional dense+BM25 pass restricted to
    table / recommendation / key_concept chunks and union those into the
    RRF as two additional ranked lists. This protects retrieval from cases
    where the table-row text ("col1 | col2 | ...") under-scores a rationale
    paragraph in the open pool — empirically the bug behind the LGD probe.
    """
    filters = filters or {}
    where_sql, where_params = _filter_clause(filters)

    qvec = embed_query(query)
    do_boost = boost_typed and not filters.get("element_types")

    branches: list[list[dict]] = []
    with _db() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            branches.append(_dense(cur, qvec, where_sql, where_params, n_dense))
            branches.append(_bm25(cur, query, where_sql, where_params, n_bm25))

            if do_boost:
                typed_filters = {**filters, "element_types": list(_BOOSTED_TYPES)}
                t_where, t_params = _filter_clause(typed_filters)
                branches.append(_dense(cur, qvec, t_where, t_params, n_typed_dense))
                branches.append(_bm25(cur, query, t_where, t_params, n_typed_bm25))

    fused = _rrf(*branches, k=rrf_k)
    return fused[:n_fused]


# --- branches ---------------------------------------------------------------


def _dense(cur, qvec, where_sql: str, where_params: tuple, n: int) -> list[dict]:
    sql = f"""
        SELECT {_SELECT_COLS},
               1.0 - (c.embedding <=> %s::vector) AS dense_score
          FROM chunks c
          JOIN documents d ON d.id = c.document_id
         WHERE c.embedding IS NOT NULL
           {where_sql}
         ORDER BY c.embedding <=> %s::vector
         LIMIT %s
    """
    params = (qvec, *where_params, qvec, n)
    cur.execute(sql, params)
    return _to_dicts(cur)


def _bm25(cur, query: str, where_sql: str, where_params: tuple, n: int) -> list[dict]:
    sql = f"""
        SELECT {_SELECT_COLS},
               ts_rank_cd(to_tsvector('english', c.text),
                          plainto_tsquery('english', %s)) AS bm25_score
          FROM chunks c
          JOIN documents d ON d.id = c.document_id
         WHERE to_tsvector('english', c.text) @@ plainto_tsquery('english', %s)
           {where_sql}
         ORDER BY bm25_score DESC
         LIMIT %s
    """
    params = (query, query, *where_params, n)
    cur.execute(sql, params)
    return _to_dicts(cur)


# --- fusion -----------------------------------------------------------------


def _rrf(*lists: list[dict], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion over an arbitrary number of ranked lists.

    rrf_score = Σ_lists 1/(k + rank_in_list). A chunk that appears across
    multiple lists accumulates score from each. We carry forward dense_score
    and bm25_score from whichever branch surfaced them so downstream
    components can inspect the per-branch contribution.
    """
    fused: dict[int, dict[str, Any]] = {}
    for ranked in lists:
        for rank, row in enumerate(ranked, start=1):
            cid = row["chunk_id"]
            if cid not in fused:
                fused[cid] = dict(row)
                fused[cid]["rrf_score"] = 0.0
            else:
                # Carry per-branch scores forward when subsequent lists touch
                # the same chunk.
                if "dense_score" in row and "dense_score" not in fused[cid]:
                    fused[cid]["dense_score"] = row["dense_score"]
                if "bm25_score" in row and "bm25_score" not in fused[cid]:
                    fused[cid]["bm25_score"] = row["bm25_score"]
            fused[cid]["rrf_score"] += 1.0 / (k + rank)
    return sorted(fused.values(), key=lambda r: r["rrf_score"], reverse=True)


# --- filters ----------------------------------------------------------------


def _filter_clause(filters: dict[str, Any]) -> tuple[str, tuple]:
    """Build a ``AND ...`` fragment + parameters tuple from the filter dict.

    Filters apply to both dense and BM25 branches. ``topic`` uses ILIKE
    against the document slug so callers can pass a substring (e.g.
    "barrett" matches both "barretts" and "barretts-esophagus").
    """
    parts: list[str] = []
    params: list[Any] = []

    if (society := filters.get("society")):
        parts.append("AND d.society = ANY(%s)")
        params.append(_as_list(society))

    if (year_min := filters.get("year_min")) is not None:
        parts.append("AND d.year >= %s")
        params.append(int(year_min))

    if (year_max := filters.get("year_max")) is not None:
        parts.append("AND d.year <= %s")
        params.append(int(year_max))

    if (topic := filters.get("topic")):
        # ILIKE ANY(...) using % wrapping so "h. pylori" hits "helicobacter-pylori".
        wrapped = [f"%{_slugify(t)}%" for t in _as_list(topic)]
        parts.append(
            "AND EXISTS (SELECT 1 FROM unnest(%s::text[]) AS pat WHERE d.topic ILIKE pat)"
        )
        params.append(wrapped)

    if (etypes := filters.get("element_types")):
        parts.append("AND c.element_type = ANY(%s)")
        params.append(_as_list(etypes))

    return ("\n           ".join(parts), tuple(params))


def _slugify(t: str) -> str:
    """Best-effort: lowercase, drop punctuation/spaces -> hyphens. We don't
    enforce hyphen *boundaries* because ILIKE with %x% gets us substring
    semantics regardless."""
    return (
        t.lower()
        .replace(".", "")
        .replace(",", "")
        .replace("'", "")
        .replace(" ", "-")
        .strip("-")
    )


# --- helpers ----------------------------------------------------------------


def _to_dicts(cur) -> list[dict]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _as_list(x: Any) -> list:
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]


@contextmanager
def _db() -> Iterable[psycopg.Connection]:
    url = os.environ.get("DATABASE_URL", "postgresql://gi:gi@localhost:5432/gi_guidelines")
    conn = psycopg.connect(url)
    try:
        yield conn
    finally:
        conn.close()
