"""Rerank fused candidates from :func:`hybrid_search`.

Provider is selected via ``config.yaml`` ``retrieval.reranker_provider``.

* ``cohere`` (default) — Cohere Rerank v3, requires ``COHERE_API_KEY``. Fast,
  high quality, no local model download.
* ``bge`` — local ``BAAI/bge-reranker-v2-m3`` cross-encoder via
  ``sentence-transformers``. Stub raises with an install hint until the
  model + lib are pulled in.

Both branches return the top_k chunks as dicts with ``relevance_score`` set.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"

Reranker = Callable[[str, list[dict[str, Any]], int], list[dict[str, Any]]]


@lru_cache(maxsize=1)
def _load_config(path: str = str(DEFAULT_CONFIG)) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Rerank ``candidates`` and return the top ``top_k`` with relevance_score.

    ``top_k`` precedence: explicit caller arg > ``config.yaml``
    ``retrieval.top_k`` > 6. (The previous version inverted that and made
    ``--top-k`` from the CLI a no-op.)

    After the model picks its top_k, we apply a *typed-chunk floor*:
    the top fused ``table`` and top fused ``recommendation`` chunks are
    guaranteed a slot in the final output, replacing the lowest-scored
    rerank pick if needed. This counteracts Cohere v3's tendency to
    underweight cell-flattened table text — empirically the table chunk
    holding the answer to "what's the surveillance interval for LGD" was
    fused-rank #1 but rerank-rank #11 because its text reads as noisy.
    Disable via ``retrieval.typed_floor: false`` in config.yaml.
    """
    if not candidates:
        return []
    cfg = _load_config().get("retrieval", {})
    provider = cfg.get("reranker_provider", "cohere")
    if top_k is None:
        top_k = int(cfg.get("top_k", 6))
    ranked = _get_reranker(provider)(query, candidates, top_k)
    return _apply_floors(
        candidates, ranked, top_k,
        typed_types=tuple(cfg.get("typed_floor_types", ("table", "recommendation"))),
        typed_floor_enabled=bool(cfg.get("typed_floor", True)),
        society_floor_enabled=bool(cfg.get("society_floor", True)),
        society_floor_min=int(cfg.get("society_floor_min_societies", 2)),
        society_floor_pool_size=int(cfg.get("society_floor_pool_size", 20)),
        society_floor_min_rrf_ratio=float(
            cfg.get("society_floor_min_rrf_ratio", 0.5)
        ),
    )


def _score(c: dict[str, Any]) -> float:
    """Pick the most discriminative score available on a candidate.

    Cohere relevance score is preferred when present — its dynamic range
    cleanly separates on-topic (~0.99) from off-topic (~0.05) chunks. RRF
    is the fallback for callers using a non-Cohere reranker (BGE branch
    doesn't yet annotate candidates with relevance scores).
    """
    if "cohere_score" in c:
        return float(c["cohere_score"])
    return float(c.get("rrf_score") or 0.0)


def _apply_floors(
    candidates: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    top_k: int,
    *,
    typed_types: tuple[str, ...],
    typed_floor_enabled: bool,
    society_floor_enabled: bool,
    society_floor_min: int,
    society_floor_pool_size: int,
    society_floor_min_rrf_ratio: float = 0.5,
) -> list[dict[str, Any]]:
    """Apply the typed-chunk floor and the society-diversity floor.

    Both floors collect "promotion" rows that are guaranteed a slot in the
    final output. Promotions are then spliced in at the tail as a single
    operation, so a later promotion can't evict an earlier one (the bug
    that surfaced when the typed-floor used per-iteration list mutation).

    Typed floor: ensures the top fused chunk of each element_type in
    ``typed_types`` is present.

    Society-diversity floor: when the fused top ``society_floor_pool_size``
    spans ``society_floor_min`` or more societies, ensures each of those
    societies contributes at least one chunk. Cohere v3 has a measurable
    bias toward whichever society's writing reads most like a direct
    answer to the query, so for cross-society questions we'd otherwise
    lose the dissenting-society chunk to the reranker every time.

    The ``society_floor_min_rrf_ratio`` gate (default 0.5) prevents the
    floor from promoting a chunk whose RRF score is far below the top
    fused candidate. Without it, a question like "differential for SEL"
    can pull a tangentially-related AGA Gastric-Polyps recommendation
    via the floor even when no on-topic AGA chunk is in the head — the
    typed-floor "must promote per society" mandate has no other AGA
    typed chunk to choose from. With ratio 0.5 (typical defensible
    default), the promoted chunk has to be within a factor of 2 of the
    top fused score, which keeps off-topic injections out.
    """
    out = list(ranked)
    seen = {r["chunk_id"] for r in out}
    promotions: list[dict[str, Any]] = []

    def already_covers(predicate) -> bool:
        return any(predicate(r) for r in out + promotions)

    # Same score-ratio gate applies to BOTH floors. Without it, the
    # typed floor will promote whatever the top-fused chunk of a given
    # type is — even when its cohere relevance is near zero. AGA Gastric
    # Polyps BPA 13 had RRF rank #2 on a SEL question because of token
    # overlap on "gastric"/"polyp"/"differential", but cohere-scored
    # it at 0.01 — without the gate the typed-floor promoted it anyway.
    top_score = max(_score(c) for c in candidates) if candidates else 0.0
    min_score = society_floor_min_rrf_ratio * top_score

    if typed_floor_enabled:
        for etype in typed_types:
            if already_covers(lambda r: r.get("element_type") == etype):
                continue
            promoted = next(
                (c for c in candidates
                 if c.get("element_type") == etype and c["chunk_id"] not in seen),
                None,
            )
            if promoted is None:
                continue
            promoted_score = _score(promoted)
            if min_score > 0 and promoted_score < min_score:
                continue  # off-topic; don't burn a slot
            p = dict(promoted)
            p["relevance_score"] = promoted_score
            p["promoted_by"] = "typed_floor"
            promotions.append(p)
            seen.add(p["chunk_id"])

    if society_floor_enabled:
        # Look at the head of fused candidates (not all of them) so a
        # society with only deep-rank chunks doesn't claim a slot. The
        # same min_score gate computed above applies here too.
        head = candidates[: max(society_floor_pool_size, top_k)]
        societies_in_head = []
        for c in head:
            soc = c.get("society")
            if soc and soc not in societies_in_head:
                societies_in_head.append(soc)
        typed_set = set(typed_types)
        if len(societies_in_head) >= society_floor_min:
            for soc in societies_in_head:
                # Does this society have any TYPED chunk (table /
                # recommendation / etc.) in the fused head? If not, no
                # promotion target — skip.
                soc_typed_in_head = [
                    c for c in head
                    if c.get("society") == soc and c.get("element_type") in typed_set
                ]
                if not soc_typed_in_head:
                    # Fall back to any chunk for this society, since there's
                    # nothing structured to promote.
                    if already_covers(lambda r, s=soc: r.get("society") == s):
                        continue
                    promoted = next(
                        (c for c in head
                         if c.get("society") == soc and c["chunk_id"] not in seen),
                        None,
                    )
                else:
                    # Society has typed content available. Require a typed
                    # chunk specifically — covers Cohere's "AGA wins all rerank
                    # slots; ACG only surfaces as prose context" pattern.
                    if already_covers(
                        lambda r, s=soc, ts=typed_set:
                        r.get("society") == s and r.get("element_type") in ts
                    ):
                        continue
                    promoted = next(
                        (c for c in soc_typed_in_head
                         if c["chunk_id"] not in seen),
                        None,
                    )
                if promoted is None:
                    continue
                # Score-ratio gate: skip promotion when the candidate's
                # score is far below the top — keeps tangential off-topic
                # chunks (e.g. AGA Gastric-Polyps BPA on a gastroparesis
                # question) from claiming a slot. Uses Cohere relevance
                # when available (sharp discrimination), RRF as a fallback.
                promoted_score = _score(promoted)
                if min_score > 0 and promoted_score < min_score:
                    continue
                p = dict(promoted)
                p["relevance_score"] = promoted_score
                p["promoted_by"] = "society_floor"
                promotions.append(p)
                seen.add(p["chunk_id"])

    if not promotions:
        return out

    # Splice in one shot: keep the top (top_k - len(promotions)) reranker
    # picks, then append the promotions. Bounded to top_k slots total.
    n_keep = max(0, top_k - len(promotions))
    return out[:n_keep] + promotions


@lru_cache(maxsize=4)
def _get_reranker(provider: str) -> Reranker:
    if provider == "cohere":
        return _make_cohere_reranker()
    if provider == "bge":
        return _make_bge_reranker()
    raise ValueError(
        f"Unknown reranker provider {provider!r}. "
        "Set retrieval.reranker_provider in config.yaml to 'cohere' or 'bge'."
    )


# --- providers --------------------------------------------------------------


def _make_cohere_reranker() -> Reranker:
    import cohere

    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "COHERE_API_KEY is not set. Either export it, or switch "
            "retrieval.reranker_provider to 'bge' in config.yaml."
        )
    cfg = _load_config().get("retrieval", {})
    model = cfg.get("cohere_model", "rerank-english-v3.0")
    client = cohere.ClientV2(api_key=api_key)

    def rerank(query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        docs = [r["text"] for r in candidates]
        # Request scores for ALL candidates (Cohere bills per document, not
        # per request, so this is the same cost as top_k=top_k). The full
        # score map lets the floor logic gate promotions on Cohere relevance
        # rather than RRF — Cohere's dynamic range is much wider (~0.05 for
        # off-topic vs ~0.99 for on-topic), giving a real signal for
        # "is this chunk close to the question?".
        resp = client.rerank(
            query=query, documents=docs, model=model, top_n=len(docs),
        )
        # Side-channel: annotate candidates with their Cohere score so
        # downstream code (in particular _apply_floors) can use it.
        for r in resp.results:
            candidates[r.index]["cohere_score"] = float(r.relevance_score)
        out = []
        for r in resp.results[:top_k]:
            row = dict(candidates[r.index])
            row["relevance_score"] = float(r.relevance_score)
            out.append(row)
        return out

    return rerank


def _make_bge_reranker() -> Reranker:
    """Local BGE cross-encoder reranker. Stub: lib + model are not installed
    by default. The ``raise`` below tells the user how to enable it.

    When wiring this up: ``CrossEncoder`` from sentence-transformers takes
    a list of (query, document) pairs and returns relevance logits. Sort
    by score desc and return the top_k candidates with score attached.
    """
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "BGE reranker requires sentence-transformers and pulls "
            "BAAI/bge-reranker-v2-m3 (~600MB) on first use. To enable: "
            "  pip install 'sentence-transformers~=3.0' "
            "Then set retrieval.reranker_provider to 'bge' in config.yaml."
        ) from e

    cfg = _load_config().get("retrieval", {})
    model_name = cfg.get("bge_reranker_model", "BAAI/bge-reranker-v2-m3")
    logger.info("Loading BGE reranker %s (first run downloads ~600MB)...", model_name)
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name)

    def rerank(query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        pairs = [(query, r["text"]) for r in candidates]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        out = []
        for score, row in ranked[:top_k]:
            r = dict(row)
            r["relevance_score"] = float(score)
            out.append(r)
        return out

    return rerank
