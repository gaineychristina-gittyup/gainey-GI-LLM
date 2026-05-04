"""Hybrid retrieval over the GI guidelines corpus.

Public API
----------

The single stable entry point is :func:`retrieve`. Phase 4 (generation) is
expected to import only this; nothing else here is part of the contract.

    from src.retrieve import retrieve

    results = retrieve(
        "what's the recommended H. pylori regimen for a penicillin-allergic patient?",
        filters={"society": ["ACG"], "year_min": 2023},
        top_k=6,
    )

Each ``results`` element is a dict with the chunk text plus the document
metadata needed for citation (society, year, title, page span,
recommendation_id, GRADE, element_type, table_html, figure_image_path).
"""

from __future__ import annotations

from typing import Any, Optional

from src.retrieve.hybrid_search import hybrid_search
from src.retrieve.rerank import rerank_candidates

__all__ = ["retrieve"]


def retrieve(
    query: str,
    filters: Optional[dict[str, Any]] = None,
    top_k: int = 6,
    *,
    rerank: bool = True,
    n_dense: int = 60,
    n_bm25: int = 60,
    n_fused: int = 20,
    boost_typed: bool = True,
) -> list[dict[str, Any]]:
    """Run hybrid retrieval and return the top_k chunks with metadata.

    Parameters
    ----------
    query
        Natural-language clinical question.
    filters
        Optional dict applied as a WHERE clause *before* retrieval (so
        candidates are not lost to post-hoc filtering). Recognized keys:
        ``society`` (list[str]), ``year_min``/``year_max`` (int),
        ``topic`` (list[str], matched ILIKE so substring slugs work),
        ``element_types`` (list[str]).
    top_k
        Final number of chunks to return after reranking.
    rerank
        Whether to call the reranker. When False, returns the top_k chunks
        from RRF-fused ranking directly.
    n_dense, n_bm25, n_fused
        Pool sizes — n_dense candidates from the vector index, n_bm25 from
        the FTS index, RRF-fused down to n_fused before reranking.
    boost_typed
        When True, run an additional dense+BM25 pass restricted to typed
        chunks (table/recommendation/key_concept) and union those into RRF.
        Suppressed automatically if ``filters['element_types']`` is set.
    """
    fused = hybrid_search(
        query,
        filters=filters or {},
        n_dense=n_dense,
        n_bm25=n_bm25,
        n_fused=n_fused,
        boost_typed=boost_typed,
    )
    if rerank:
        return rerank_candidates(query, fused, top_k=top_k)
    return [_attach_score(row, row.get("rrf_score")) for row in fused[:top_k]]


def _attach_score(row: dict[str, Any], score: float | None) -> dict[str, Any]:
    out = dict(row)
    out["relevance_score"] = score
    return out
