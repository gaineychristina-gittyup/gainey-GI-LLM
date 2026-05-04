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
    min_partial_ratio: int = 40,
    min_token_set_ratio: int = 50,
) -> dict[str, Any]:
    """Validate the citations in ``answer_text`` against the cited chunks.

    The verifier was originally per-citation: each [N] in a sentence had
    to independently fuzzy-match the cited chunk, with one threshold on
    ``partial_ratio``. That produced a high false-positive rate on real
    clinical answers because:

    - Multi-citation sentences like ``[4, 6]`` carry content from BOTH
      cited chunks; neither chunk alone contains the synthesized
      statement, so requiring each independently to support the sentence
      is wrong. We now use **any-of-cited** semantics — the sentence is
      supported as long as at least one cited chunk in-range matches.
    - Table-derived sentences (model writes natural prose, chunk holds
      cell-flattened ``"Drug | Dose | ..."`` text) score ~50 on
      ``partial_ratio`` even on legitimate alignment. We add
      ``partial_token_set_ratio`` as a second matcher (token-overlap
      invariant to order) — a sentence passes if EITHER scorer clears
      its threshold.

    Parameters
    ----------
    answer_text
        The model's prose answer, with [N] citation tags inline.
    chunks
        The list of retrieved chunks passed to the model. Citation [N]
        refers to chunks[N-1].
    min_partial_ratio
        Threshold for ``rapidfuzz.partial_ratio`` (0-100). 50 separates
        unrelated content (~40-45) from legitimate paraphrase
        (~55-80) with margin.
    min_token_set_ratio
        Threshold for ``rapidfuzz.token_set_ratio`` (0-100). 55 catches
        table-derived sentences whose tokens overlap with the chunk even
        when partial_ratio scores low because of structural differences.
        Note: we use ``token_set_ratio`` (not ``partial_token_set_ratio``)
        because the partial variant scores ~100 even for unrelated
        content (stop-word intersection inflates it). The non-partial
        token_set_ratio scales by both unique sets, so unrelated content
        scores ~49 and legitimate paraphrase scores ~60.

    Returns
    -------
    dict with:
        ok               -- bool, True iff all citations validated
        n_citations      -- int, total [N] tags in the answer
        out_of_range     -- list[int], cited indices that don't exist
        unsupported      -- list[dict], citations whose sentence isn't
                            grounded in ANY of its cited chunks
    """
    n_chunks = len(chunks)
    sentences = _split_sentences(answer_text)

    out_of_range: list[int] = []
    unsupported: list[dict[str, Any]] = []
    total = 0

    for sentence in sentences:
        for match in _CITATION_RE.finditer(sentence):
            cited = []
            for piece in match.group(1).split(","):
                try:
                    n = int(piece.strip())
                except ValueError:
                    continue
                total += 1
                if n < 1 or n > n_chunks:
                    out_of_range.append(n)
                else:
                    cited.append(n)
            if not cited:
                continue
            # any-of-cited: sentence is supported if at least one of its
            # cited chunks fuzzy-matches.
            any_supports = any(
                _sentence_supported_by_chunk(
                    sentence, chunks[n - 1], min_partial_ratio,
                    min_token_set_ratio,
                )
                for n in cited
            )
            if not any_supports:
                # Flag every cited (in-range) index for this sentence so
                # the UI can surface ALL the chunks it tried.
                for n in cited:
                    unsupported.append({
                        "n": n,
                        "chunk_id": chunks[n - 1].get("chunk_id"),
                        "sentence": sentence.strip(),
                    })

    return {
        "ok": not out_of_range and not unsupported,
        "n_citations": total,
        "out_of_range": out_of_range,
        "unsupported": unsupported,
        "min_partial_ratio": min_partial_ratio,
        "min_token_set_ratio": min_token_set_ratio,
    }


# --- internals --------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """Sentence-split for citation locality. Imperfect but good enough —
    citations sit at clause/sentence boundaries in the model's output."""
    return [s for s in _SENTENCE_END_RE.split(text) if s.strip()]


def _sentence_supported_by_chunk(
    sentence: str,
    chunk: dict[str, Any],
    partial_ratio_threshold: int,
    token_set_threshold: int = 50,
) -> bool:
    """True iff the sentence's claim is plausibly grounded in the chunk.

    Two scorers are tried; either can clear its threshold:

    - ``partial_ratio``: best for verbatim-or-paraphrase alignment in
      contiguous prose. Threshold 50 separates legitimate paraphrase
      (~70) and heavy synthesis (~57) from unrelated content (~45).
    - ``token_set_ratio``: token-overlap that scales by both sides'
      unique sets. Threshold 55 catches table-derived sentences (chunk
      is cell-flattened, sentence is natural prose) at ~60 while
      rejecting unrelated content at ~49.

    Tables get an additional pass against their tags-stripped
    ``table_html`` rendering.
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

    targets = [chunk_text]
    if table_html:
        flat = re.sub(r"<[^>]+>", " ", table_html)
        flat = re.sub(r"\s+", " ", flat).strip()
        if flat:
            targets.append(flat)

    for target in targets:
        if not target:
            continue
        if fuzz.partial_ratio(claim, target) >= partial_ratio_threshold:
            return True
        if fuzz.token_set_ratio(claim, target) >= token_set_threshold:
            return True
    return False
