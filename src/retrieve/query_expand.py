"""LLM-based query expansion for clinical retrieval.

The Voyage embeddings can't disambiguate clinical intent buried in negations
("NO history of bleeding" → primary prophylaxis) or implicit context
("decompensated cirrhosis with varices on endoscopy" → dACLD primary
prophylaxis vs acute variceal hemorrhage). A small Haiku call rewrites the
question into 2-4 retrieval-friendly variants that make that intent explicit.

The original query is always preserved; variants are run as additional
dense+BM25 branches in hybrid_search and fused via RRF, so a successful
original-query retrieval is never displaced by a bad rewrite — it's
augmented.

Failure mode: on any Haiku error (no API key, network blip, malformed
response) we return ``[]`` so the caller falls back to single-query
retrieval. Expansion is best-effort; never load-bearing.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

EXPANSION_MODEL = "claude-haiku-4-5-20251001"

EXPANSION_SYSTEM = """You rewrite clinical questions for a GI guidelines RAG system to improve retrieval recall.

Given a clinician's natural-language question, generate {n} concise alternative phrasings that:
- Make implicit clinical intent explicit. Examples:
  * "no history of bleeding" / "never bled" → "primary prophylaxis", "varices that have not bled"
  * "actively bleeding" / hemodynamic instability → "acute variceal hemorrhage", "AVH management"
  * "decompensated cirrhosis" → also include "dACLD", "advanced chronic liver disease"
  * "compensated cirrhosis" → also include "cACLD", "CSPH"
  * "what's recommended" / "what should I do" → "guideline recommendation", "GRADE rating"
- Add precise medical synonyms and acronyms used in society guidelines (NSBB, EVL, TIPS, ERCP, ADR, BRTO, PCAB, etc.)
- Reframe negations into positive equivalents (embeddings handle negation poorly)
- Use the formal phrasing of clinical practice guidelines, not casual paraphrase

Do NOT:
- Invent diagnoses, drugs, dosing, or assumptions not implied by the original
- Add demographic details (age, sex) not in the original
- Repeat the original verbatim
- Number or bullet the output
- Add commentary, explanation, or quotation marks

Output exactly {n} alternative phrasings, one per line, nothing else."""


_client_cache: dict = {}


def _get_client():
    if "client" not in _client_cache:
        from anthropic import Anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set; cannot expand queries.")
        _client_cache["client"] = Anthropic(api_key=api_key)
    return _client_cache["client"]


@lru_cache(maxsize=256)
def expand_query(query: str, n: int = 3, model: str = EXPANSION_MODEL) -> tuple[str, ...]:
    """Return up to ``n`` alternate phrasings of ``query``.

    Cached so repeat calls (e.g. eval re-runs) skip the LLM round-trip.
    Returns an empty tuple if expansion fails — callers should fall back
    to single-query retrieval.
    """
    if n <= 0 or not query.strip():
        return ()
    try:
        resp = _get_client().messages.create(
            model=model,
            max_tokens=400,
            temperature=0.2,
            system=EXPANSION_SYSTEM.format(n=n),
            messages=[{"role": "user", "content": query}],
        )
        text = resp.content[0].text if resp.content else ""
    except Exception as e:
        logger.warning("query expansion failed: %s", e)
        return ()

    variants = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # Strip leading bullets / numbering the model might add despite the prompt.
        for prefix in ("- ", "* ", "• "):
            if s.startswith(prefix):
                s = s[len(prefix):]
                break
        if s and len(s) > 4 and s.lower() != query.strip().lower():
            variants.append(s)
        if len(variants) >= n:
            break
    return tuple(variants)


def expanded_queries(query: str, n: int = 3) -> list[str]:
    """Return ``[query, *variants]`` deduped, preserving order."""
    out = [query.strip()]
    seen = {query.strip().lower()}
    for v in expand_query(query, n=n):
        key = v.lower()
        if key not in seen:
            out.append(v)
            seen.add(key)
    return out
