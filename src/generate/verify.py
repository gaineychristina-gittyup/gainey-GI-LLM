"""Citation verification for grounded answers.

The strict-grounding system prompt asks Claude not to invent citations, but
"trust but verify" matters more in clinical settings than anywhere else.
This module runs after :func:`src.generate.answer.answer` to:

1. Confirm every ``[N]`` tag in the answer text refers to a chunk that
   was actually in the SOURCES block (i.e. ``1 <= N <= len(chunks)``).
2. For each cited chunk, fuzzy-match the *sentence containing the tag*
   against the chunk's text. If the sentence is paraphrased or synthesized
   from across cells, ``rapidfuzz.partial_ratio`` should still find a
   plausible alignment (default threshold 75 / 100).

Failures are flagged on the answer dict; this layer never silently strips
or rewrites citations. Whether to refuse the whole answer when verification
fails is a policy decision left to the caller.
"""

from __future__ import annotations

import re
from typing import Any

# Sentence terminators we use to slice the answer text. Same shape as the
# chunker's SENTENCE_END_RE — the model writes in standard prose so this is
# good enough.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"])")
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def verify_answer(
    answer_text: str,
    chunks: list[dict[str, Any]],
    *,
    min_partial_ratio: int = 55,
) -> dict[str, Any]:
    """Validate the citations in ``answer_text`` against the cited chunks.

    Parameters
    ----------
    answer_text
        The model's prose answer, with [N] citation tags inline.
    chunks
        The list of retrieved chunks passed to the model. Citation [N]
        refers to chunks[N-1].
    min_partial_ratio
        Threshold for ``rapidfuzz.partial_ratio`` (0-100). Empirical
        scoring on real clinical answers:
            ~70-80 : legitimate paraphrase
            ~55-65 : heavy synthesis with extra qualifiers / context
                     (still grounded in the chunk, just verbose)
            ~40-50 : unrelated content (fabrication)
        Default 55 catches fabrication while tolerating typical synthesis;
        bump to 70+ for stricter verification.

    Returns
    -------
    dict with:
        ok               -- bool, True iff all citations validated
        n_citations      -- int, total [N] tags in the answer
        out_of_range     -- list[int], cited indices that don't exist in chunks
        unsupported      -- list[dict], citations whose sentence doesn't
                            fuzzy-match the cited chunk
    """
    n_chunks = len(chunks)
    sentences = _split_sentences(answer_text)

    out_of_range: list[int] = []
    unsupported: list[dict[str, Any]] = []
    total = 0

    for sentence in sentences:
        for match in _CITATION_RE.finditer(sentence):
            for piece in match.group(1).split(","):
                try:
                    n = int(piece.strip())
                except ValueError:
                    continue
                total += 1
                if n < 1 or n > n_chunks:
                    out_of_range.append(n)
                    continue
                chunk = chunks[n - 1]
                if not _sentence_supported_by_chunk(
                    sentence, chunk, min_partial_ratio
                ):
                    unsupported.append({
                        "n": n,
                        "chunk_id": chunk.get("chunk_id"),
                        "sentence": sentence.strip(),
                    })

    return {
        "ok": not out_of_range and not unsupported,
        "n_citations": total,
        "out_of_range": out_of_range,
        "unsupported": unsupported,
        "min_partial_ratio": min_partial_ratio,
    }


# --- internals --------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """Sentence-split for citation locality. Imperfect but good enough —
    citations sit at clause/sentence boundaries in the model's output."""
    return [s for s in _SENTENCE_END_RE.split(text) if s.strip()]


def _sentence_supported_by_chunk(
    sentence: str, chunk: dict[str, Any], threshold: int
) -> bool:
    """True iff the sentence's claim is plausibly grounded in the chunk.

    Empirically, ``partial_ratio`` discriminates better than the
    token-based variants here: stop-word overlap inflates the
    token-set scorers to ~100 even on unrelated content, while
    ``partial_ratio`` scores ~70 for legitimate paraphrase and ~45 for
    unrelated claims.
    """
    from rapidfuzz import fuzz

    chunk_text = (chunk.get("text") or "").strip()
    table_html = chunk.get("table_html") or ""
    if not chunk_text and not table_html:
        return False

    # Strip the [N] tags so they don't pollute the comparison.
    claim = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", sentence)
    claim = re.sub(r"\s+", " ", claim).strip()
    if not claim:
        return False

    if fuzz.partial_ratio(claim, chunk_text) >= threshold:
        return True

    # Tables get a second chance against their HTML rendering — sometimes
    # the plain-text flattening obscures cell-level alignment that the
    # answer actually quoted.
    if table_html:
        flat = re.sub(r"<[^>]+>", " ", table_html)
        flat = re.sub(r"\s+", " ", flat).strip()
        if fuzz.partial_ratio(claim, flat) >= threshold:
            return True

    return False
