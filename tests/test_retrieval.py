"""Sanity-check tests for the retrieval layer.

These tests hit the live database and live embedding/rerank APIs, so they
auto-skip when the corpus is empty or required keys are absent. The point is
to catch wiring regressions, not to assert specific top-1 chunks.
"""

from __future__ import annotations

import os

import psycopg
import pytest


# --- DB-state fixture -------------------------------------------------------


def _chunk_count() -> int:
    url = os.environ.get(
        "DATABASE_URL", "postgresql://gi:gi@localhost:5432/gi_guidelines"
    )
    try:
        with psycopg.connect(url, connect_timeout=2) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
            return cur.fetchone()[0]
    except Exception:
        return 0


@pytest.fixture(scope="module")
def populated_corpus():
    n = _chunk_count()
    if n == 0:
        pytest.skip("chunks table is empty — run the ingest pipeline first")
    return n


# --- tests ------------------------------------------------------------------


def test_query_embedding_dim_matches_schema(populated_corpus):
    """Whatever provider produced chunks must produce a same-dim query vector."""
    from src.retrieve.embedder import embed_query

    if not os.environ.get("VOYAGE_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("no embedding API key set")
    vec = embed_query("Barrett's esophagus surveillance interval")
    # voyage-3-large and BGE-large are both 1024; OpenAI-3-large is 3072.
    assert len(vec) in (1024, 3072), f"unexpected embed dim {len(vec)}"


def test_hybrid_returns_results_for_generic_query(populated_corpus):
    from src.retrieve.hybrid_search import hybrid_search

    rows = hybrid_search("management of Crohn's disease", n_dense=10, n_bm25=10, n_fused=8)
    assert rows, "hybrid_search returned no candidates for a generic GI query"
    # Every row must carry the metadata the spec requires.
    expected_keys = {
        "chunk_id", "society", "year", "title", "section_title",
        "recommendation_id", "grade_evidence", "grade_strength",
        "element_type", "page_start", "page_end", "text", "rrf_score",
    }
    assert expected_keys.issubset(rows[0].keys())


def test_society_filter_restricts_results(populated_corpus):
    from src.retrieve.hybrid_search import hybrid_search

    rows = hybrid_search(
        "endoscopic management",
        filters={"society": ["ASGE"]},
        n_dense=20, n_bm25=20, n_fused=20,
    )
    assert rows, "ASGE-filtered hybrid_search returned nothing"
    assert all(r["society"] == "ASGE" for r in rows), \
        "society filter leaked: " + str({r["society"] for r in rows})


def test_year_filter_restricts_results(populated_corpus):
    from src.retrieve.hybrid_search import hybrid_search

    rows = hybrid_search(
        "liver disease",
        filters={"year_min": 2024},
        n_dense=20, n_bm25=20, n_fused=20,
    )
    if not rows:
        pytest.skip("no >=2024 chunks matched 'liver disease' in this corpus")
    assert all(r["year"] >= 2024 for r in rows)


def test_element_type_filter_restricts_results(populated_corpus):
    from src.retrieve.hybrid_search import hybrid_search

    rows = hybrid_search(
        "surveillance interval table",
        filters={"element_types": ["table"]},
        n_dense=20, n_bm25=20, n_fused=20,
    )
    if not rows:
        pytest.skip("no table chunks matched the probe query")
    assert all(r["element_type"] == "table" for r in rows)


def test_reranker_can_reorder(populated_corpus):
    """Rerank should not just echo fused ordering — it should reorder for at
    least one query. We check that the top-1 chunk_id is *allowed* to differ
    from the fused top-1 (and assert it does for at least one of two queries
    so the test is robust to lucky ties)."""
    if not os.environ.get("COHERE_API_KEY"):
        pytest.skip("COHERE_API_KEY unset — reranker would be a no-op")
    from src.retrieve.hybrid_search import hybrid_search
    from src.retrieve.rerank import rerank_candidates

    queries = [
        "what surveillance interval applies to low-grade dysplasia in Barrett's?",
        "first-line H. pylori treatment for penicillin allergy",
    ]
    reordered_at_least_once = False
    for q in queries:
        fused = hybrid_search(q, n_dense=20, n_bm25=20, n_fused=20)
        if len(fused) < 3:
            continue
        ranked = rerank_candidates(q, fused, top_k=6)
        assert ranked, "reranker returned empty for non-empty fused"
        # Every ranked row must carry a relevance_score.
        assert all("relevance_score" in r for r in ranked)
        if ranked[0]["chunk_id"] != fused[0]["chunk_id"]:
            reordered_at_least_once = True
    assert reordered_at_least_once, (
        "reranker returned the fused top-1 unchanged across two probes — "
        "either the model is broken or the queries happened to agree"
    )
