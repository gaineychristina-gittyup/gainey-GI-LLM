"""Intent-aware second-pass reranker (Haiku).

Cohere v3 ranks candidates well for keyword/structural similarity but is
weak on clinical-intent disambiguation. On the decompensated-cirrhosis
primary-prophylaxis probe, the dACLD chunk that directly answers the
question scored 0.117 while compensated-cACLD chunks scored 0.99 —
because the cACLD prose reads more like "a guideline answer to a varices
question" than the denser dACLD paragraph does, even though the question
explicitly says "decompensated."

This module runs Haiku as a SECOND pass on top of the Cohere ranking. It
sees the user's question and the top-N Cohere candidates and produces an
intent-aware re-ordering. Cohere is still the precision filter (we trust
it to drop genuinely off-topic chunks); Haiku decides which of the
on-topic chunks most directly answers the user's intent.

Failure mode: any error returns the input ranking unchanged. The caller
falls back to Cohere's order.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

LLM_RERANK_MODEL = "claude-haiku-4-5-20251001"

LLM_RERANK_SYSTEM = """You re-rank candidate clinical-guideline chunks by how directly they answer a clinical question.

The candidates have already been pre-filtered by a keyword/embedding reranker — they are all topically related. Your job is to read the question's CLINICAL INTENT (population, scenario, sub-question) and pick the chunks that most directly address it.

Score each candidate 0-10:
- 10: directly answers the question — right population, right scenario, right intervention
- 8: same population/scenario, slight variation (e.g., the right drug class but a different specific drug)
- 6: closely related — same intervention but a different population (e.g., compensated cirrhosis when the question is about decompensated; pediatric when the question is adult)
- 4: tangentially related background (e.g., epidemiology, mechanism, definitions)
- 2: same disease area but different sub-question
- 0: off-topic for this specific question

Important: don't reward keyword overlap or guideline-formal language alone — what matters is whether the chunk's CONTENT addresses the question's CLINICAL SCENARIO. A dense paragraph about decompensated cirrhosis primary prophylaxis is more relevant to a decompensated-cirrhosis primary-prophylaxis question than a numbered guidance statement about compensated cirrhosis screening, even though the latter reads more like "a guideline answer."

Output format: one line per candidate, "<id>: <score>", in original order. No commentary, no preamble, nothing else.
Example output:
1: 6
2: 9
3: 4
"""


_client_cache: dict = {}


def _get_client():
    if "client" not in _client_cache:
        from anthropic import Anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set; cannot LLM-rerank.")
        _client_cache["client"] = Anthropic(api_key=api_key)
    return _client_cache["client"]


def _candidate_summary(c: dict[str, Any], snippet_chars: int = 600) -> str:
    soc = c.get("society") or "?"
    yr = c.get("year") or "?"
    title = (c.get("title") or "")[:90]
    etype = c.get("element_type") or "prose"
    text = (c.get("text") or "").strip().replace("\n", " ")
    text = text[:snippet_chars] + ("..." if len(text) > snippet_chars else "")
    return f"{soc} {yr} ({etype}): {title}\n  {text}"


def llm_rerank_intent(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    top_k: int = 6,
    n_candidates: int = 12,
    model: str = LLM_RERANK_MODEL,
) -> Optional[list[dict[str, Any]]]:
    """Re-order ``candidates`` by clinical intent using Haiku.

    Picks the top ``n_candidates`` by ``cohere_score`` (or whatever score
    is on the candidate), sends them to Haiku with ``query``, and returns
    the top ``top_k`` chunks in Haiku's intent-ranked order.

    Returns None on any failure — the caller should fall back to the
    input order.
    """
    if not candidates or n_candidates <= 0:
        return None

    def _score(c: dict[str, Any]) -> float:
        for k in ("cohere_score", "relevance_score", "rrf_score"):
            if k in c and c[k] is not None:
                return float(c[k])
        return 0.0

    pool = sorted(candidates, key=_score, reverse=True)[:n_candidates]
    if len(pool) <= 1:
        return None

    summaries = "\n\n".join(
        f"[{i + 1}] {_candidate_summary(c)}" for i, c in enumerate(pool)
    )
    user_msg = f"QUESTION: {query.strip()}\n\nCANDIDATES:\n\n{summaries}"

    try:
        resp = _get_client().messages.create(
            model=model,
            max_tokens=400,
            temperature=0.0,
            system=LLM_RERANK_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text if resp.content else ""
    except Exception as e:
        logger.warning("LLM rerank failed: %s", e)
        return None

    scores: dict[int, float] = {}
    for line in text.splitlines():
        m = re.match(r"\s*\[?(\d+)\]?\s*[:\-]\s*([\d.]+)", line)
        if m:
            idx = int(m.group(1))
            try:
                scores[idx] = float(m.group(2))
            except ValueError:
                continue
    if not scores:
        logger.warning("LLM rerank returned no parseable scores; falling back")
        return None

    indexed = list(enumerate(pool, start=1))
    indexed.sort(
        key=lambda pair: (scores.get(pair[0], -1.0), -_score(pair[1])),
        reverse=True,
    )
    out = []
    for orig_idx, c in indexed[:top_k]:
        row = dict(c)
        row["llm_intent_score"] = scores.get(orig_idx)
        row["relevance_score"] = scores.get(orig_idx, _score(c))
        out.append(row)
    return out
