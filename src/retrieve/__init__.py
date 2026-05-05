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

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

from src.retrieve.hybrid_search import hybrid_search
from src.retrieve.rerank import rerank_candidates

logger = logging.getLogger(__name__)

__all__ = ["retrieve"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def _load_config(path: str = str(_DEFAULT_CONFIG)) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


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
    expand_query: Optional[bool] = None,
    n_query_variants: Optional[int] = None,
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
    expand_query
        When True, ask Haiku to generate paraphrased query variants and
        run each as additional dense+BM25 branches in the RRF pool. None
        (default) reads ``retrieval.query_expansion.enabled`` from
        config.yaml.
    n_query_variants
        How many variants to generate when expansion is on. None reads
        ``retrieval.query_expansion.n_variants`` from config (default 3).
    """
    cfg = _load_config().get("retrieval", {}).get("query_expansion", {}) or {}
    if expand_query is None:
        expand_query = bool(cfg.get("enabled", False))
    if n_query_variants is None:
        n_query_variants = int(cfg.get("n_variants", 3))

    extra: list[str] = []
    if expand_query and n_query_variants > 0:
        try:
            from src.retrieve.query_expand import expanded_queries
            qs = expanded_queries(query, n=n_query_variants)
            extra = qs[1:]  # drop the original; hybrid_search adds it back
        except Exception as e:
            logger.warning("query expansion skipped: %s", e)

    fused = hybrid_search(
        query,
        filters=filters or {},
        n_dense=n_dense,
        n_bm25=n_bm25,
        n_fused=n_fused,
        boost_typed=boost_typed,
        extra_queries=extra or None,
    )
    if rerank:
        return rerank_candidates(query, fused, top_k=top_k)
    return [_attach_score(row, row.get("rrf_score")) for row in fused[:top_k]]


def _attach_score(row: dict[str, Any], score: float | None) -> dict[str, Any]:
    out = dict(row)
    out["relevance_score"] = score
    return out
