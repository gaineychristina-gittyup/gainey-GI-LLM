"""Section-aware chunking for clinical guidelines.

Chunking rules (in plain English so they're easy to tune):

  1. Walk the parsed elements in document order. ``Title`` elements start a new
     section. Everything until the next ``Title`` belongs to that section.

  2. Within a section, accumulate text until we hit ~``target_tokens``. When the
     accumulator goes over, flush a chunk and start the next one with an
     overlap window of ``overlap_tokens`` tokens (so context bleeds across
     boundaries — important for retrieval).

  3. We never split inside a recommendation block. A "recommendation block"
     is detected by ``RECOMMENDATION_RE`` — patterns like ``Recommendation 3.2``
     or ``Statement 4`` or a ListItem starting with a GRADE phrase. If a
     recommendation block would push us past ``max_tokens``, we flush the
     in-progress chunk *first*, then emit the recommendation as its own chunk.

  4. We never split mid-sentence. The forced-split path (when a section is
     longer than ``max_tokens`` and has no recommendation boundaries) walks
     back to the most recent sentence terminator before flushing.

  5. After all chunks are produced, any chunk shorter than ``min_tokens`` is
     merged with its previous neighbor — short orphan chunks (titles by
     themselves, single bullet points) hurt retrieval quality.

  6. For each chunk, we extract:
       - ``recommendation_id`` (e.g. "Recommendation 3.2", "Statement 4")
       - ``grade_evidence`` (low / moderate / high)
       - ``grade_strength`` (strong / conditional / weak)
     via regex. GRADE language is reasonably standardized across societies.

The chunker is the highest-stakes component in the ingestion pipeline because
it's where signal is preserved or destroyed before embedding. Tune
``target_tokens`` / ``overlap_tokens`` first if retrieval is noisy.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

import tiktoken

from src.ingest.parse_pdfs import ParsedElement

logger = logging.getLogger(__name__)


# --- Regexes for recommendation / GRADE extraction --------------------------
# Matched case-insensitively. The (?P<...>...) groups let us pull out IDs.

RECOMMENDATION_RE = re.compile(
    r"\b(?P<id>"
    r"(?:Recommendation|Statement|Best\s+Practice\s+Advice|BPA|Quality\s+Indicator)"
    r"\s*\d+(?:\.\d+)?[A-Za-z]?"
    r")\b",
    re.IGNORECASE,
)

# GRADE strength: "strong" or "conditional"/"weak" recommendation.
GRADE_STRENGTH_RE = re.compile(
    r"\b(?P<strength>strong|conditional|weak)\s+recommendation\b",
    re.IGNORECASE,
)

# GRADE evidence quality: low / moderate / high (sometimes "very low").
GRADE_EVIDENCE_RE = re.compile(
    r"\b(?P<evidence>very\s+low|low|moderate|high)[-\s]+(?:quality|certainty)\s+evidence\b",
    re.IGNORECASE,
)

# Sentence-terminator boundary used when forced to split inside a paragraph.
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


@dataclass
class Chunk:
    text: str
    section_title: Optional[str] = None
    recommendation_id: Optional[str] = None
    grade_evidence: Optional[str] = None
    grade_strength: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


# --- Tokenizer (cached) -----------------------------------------------------
# We use tiktoken's cl100k_base when available (matches GPT-4 / Claude well
# enough for budgeting). If tiktoken can't load its encoding (offline env, no
# cache), we fall back to a simple whitespace-based approximation so the
# chunker remains usable without internet access. The approximation
# underestimates token counts by ~25% on prose, which is acceptable for
# chunking budgets but means token_count rows in the DB are best-effort.
_ENCODER = None
_ENCODER_FAILED = False


class _WhitespaceEncoder:
    """Fallback ~tokenizer used when tiktoken can't fetch its encoding."""

    @staticmethod
    def encode(text: str) -> list[int]:
        # Approximate: 1 token per word, with extra tokens for punctuation runs.
        # Good enough for chunk-size budgeting; not exact.
        words = text.split()
        return list(range(len(words) + max(0, text.count(",") + text.count(";"))))

    @staticmethod
    def decode(ids: list[int]) -> str:  # pragma: no cover - only used for overlap tails
        return ""  # Fallback path can't reconstruct text; overlap will degrade gracefully.


def _get_encoder(name: str = "cl100k_base"):
    global _ENCODER, _ENCODER_FAILED
    if _ENCODER is not None:
        return _ENCODER
    if _ENCODER_FAILED:
        return _WhitespaceEncoder()
    try:
        _ENCODER = tiktoken.get_encoding(name)
        return _ENCODER
    except Exception as e:
        logger.warning(
            "tiktoken could not load encoding %r (%s); falling back to "
            "whitespace tokenizer. Token counts will be approximate.",
            name, e,
        )
        _ENCODER_FAILED = True
        return _WhitespaceEncoder()


def count_tokens(text: str, tokenizer: str = "cl100k_base") -> int:
    return len(_get_encoder(tokenizer).encode(text))


# --- Public API -------------------------------------------------------------


def chunk_elements(
    elements: Iterable[ParsedElement],
    target_tokens: int = 500,
    min_tokens: int = 200,
    max_tokens: int = 800,
    overlap_tokens: int = 50,
    tokenizer: str = "cl100k_base",
) -> list[Chunk]:
    """Section-aware chunker. See module docstring for the full ruleset."""
    sections = _group_into_sections(list(elements))
    chunks: list[Chunk] = []
    for section_title, section_elems in sections:
        chunks.extend(
            _chunk_section(
                section_title=section_title,
                section_elems=section_elems,
                target_tokens=target_tokens,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                tokenizer=tokenizer,
            )
        )

    # Merge orphans
    chunks = _merge_short_chunks(chunks, min_tokens=min_tokens, tokenizer=tokenizer)

    # Annotate with extracted metadata
    for c in chunks:
        _extract_grade_metadata(c)

    return chunks


# --- Internals --------------------------------------------------------------


def _group_into_sections(
    elements: list[ParsedElement],
) -> list[tuple[Optional[str], list[ParsedElement]]]:
    """Partition the element stream by ``Title`` boundaries.

    Returns a list of (section_title, [elements]) tuples in document order.
    Anything before the first Title goes under ``section_title=None``.
    """
    sections: list[tuple[Optional[str], list[ParsedElement]]] = []
    current_title: Optional[str] = None
    current: list[ParsedElement] = []

    for el in elements:
        if el.category == "Title":
            if current:
                sections.append((current_title, current))
                current = []
            current_title = el.text
            # The title itself is included in its own section so retrieval
            # can match on heading text.
            current.append(el)
        else:
            current.append(el)

    if current:
        sections.append((current_title, current))
    return sections


def _is_recommendation_block(elem: ParsedElement) -> bool:
    """Heuristic: does this element start a recommendation we shouldn't split?"""
    head = elem.text[:200]
    if RECOMMENDATION_RE.search(head):
        return True
    # ListItems that lead with GRADE phrasing also count.
    if elem.category == "ListItem" and GRADE_STRENGTH_RE.search(head):
        return True
    return False


def _chunk_section(
    section_title: Optional[str],
    section_elems: list[ParsedElement],
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
    tokenizer: str,
) -> list[Chunk]:
    """Chunk a single section into one or more Chunks."""
    chunks: list[Chunk] = []

    buf_texts: list[str] = []
    buf_tokens = 0
    buf_pages: list[int] = []

    def flush() -> Optional[Chunk]:
        nonlocal buf_texts, buf_tokens, buf_pages
        if not buf_texts:
            return None
        text = "\n\n".join(buf_texts).strip()
        if not text:
            buf_texts, buf_tokens, buf_pages = [], 0, []
            return None
        chunk = Chunk(
            text=text,
            section_title=section_title,
            page_start=min(buf_pages) if buf_pages else None,
            page_end=max(buf_pages) if buf_pages else None,
            token_count=count_tokens(text, tokenizer),
        )
        chunks.append(chunk)
        # Reset with overlap from tail of just-flushed text.
        if overlap_tokens > 0:
            tail = _tail_by_tokens(text, overlap_tokens, tokenizer)
            buf_texts = [tail] if tail else []
            buf_tokens = count_tokens(tail, tokenizer) if tail else 0
        else:
            buf_texts, buf_tokens = [], 0
        buf_pages = []
        return chunk

    for el in section_elems:
        el_tokens = count_tokens(el.text, tokenizer)
        is_rec = _is_recommendation_block(el)

        # If this is a recommendation that would overflow, flush first so the
        # recommendation lives in its own chunk (or its own pair of chunks if
        # it's >max_tokens by itself).
        if is_rec and (buf_tokens + el_tokens > target_tokens) and buf_tokens > 0:
            flush()

        # Recommendation blocks that fit go straight in.
        if is_rec and el_tokens <= max_tokens:
            buf_texts.append(el.text)
            buf_tokens += el_tokens
            if el.page_number is not None:
                buf_pages.append(el.page_number)
            # After a recommendation, flush so it's the dominant content of
            # its chunk; the next element starts fresh (with overlap).
            if buf_tokens >= target_tokens:
                flush()
            continue

        # Recommendation blocks longer than max_tokens are split on sentence
        # boundaries but kept as their own chunks (no merging with neighbors).
        if is_rec and el_tokens > max_tokens:
            if buf_texts:
                flush()
            for piece in _split_long_text(el.text, target_tokens, max_tokens, tokenizer):
                chunks.append(
                    Chunk(
                        text=piece,
                        section_title=section_title,
                        page_start=el.page_number,
                        page_end=el.page_number,
                        token_count=count_tokens(piece, tokenizer),
                    )
                )
            continue

        # Regular element: would adding it overflow?
        if buf_tokens + el_tokens > max_tokens and buf_texts:
            flush()

        # If this single element is itself larger than max_tokens, split it.
        if el_tokens > max_tokens:
            if buf_texts:
                flush()
            for piece in _split_long_text(el.text, target_tokens, max_tokens, tokenizer):
                buf_texts.append(piece)
                buf_tokens += count_tokens(piece, tokenizer)
                if el.page_number is not None:
                    buf_pages.append(el.page_number)
                if buf_tokens >= target_tokens:
                    flush()
            continue

        buf_texts.append(el.text)
        buf_tokens += el_tokens
        if el.page_number is not None:
            buf_pages.append(el.page_number)

        if buf_tokens >= target_tokens:
            flush()

    flush()
    return chunks


def _split_long_text(
    text: str, target_tokens: int, max_tokens: int, tokenizer: str
) -> list[str]:
    """Split a text that's too long into sentence-bounded pieces near target_tokens.

    Walks sentences greedily; never breaks mid-sentence.
    """
    sentences = SENTENCE_END_RE.split(text)
    pieces: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        st = count_tokens(sent, tokenizer)
        if cur_tokens + st > max_tokens and cur:
            pieces.append(" ".join(cur))
            cur, cur_tokens = [sent], st
        else:
            cur.append(sent)
            cur_tokens += st
            if cur_tokens >= target_tokens:
                pieces.append(" ".join(cur))
                cur, cur_tokens = [], 0
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _tail_by_tokens(text: str, n_tokens: int, tokenizer: str) -> str:
    """Return the trailing ``n_tokens`` of text, sliced on token boundaries.

    Uses tiktoken when available; falls back to a word-based tail when not.
    """
    enc = _get_encoder(tokenizer)
    if isinstance(enc, _WhitespaceEncoder):
        words = text.split()
        if len(words) <= n_tokens:
            return text
        return " ".join(words[-n_tokens:])
    ids = enc.encode(text)
    if len(ids) <= n_tokens:
        return text
    return enc.decode(ids[-n_tokens:])


def _merge_short_chunks(
    chunks: list[Chunk], min_tokens: int, tokenizer: str
) -> list[Chunk]:
    """Merge any chunk shorter than min_tokens into a same-section neighbor.

    Pass 1 (backward): for each non-first chunk, if it's short, merge into the
    previous chunk (preferred direction so context flows forward).
    Pass 2 (forward): if the FIRST chunk is short (typically a lone Title),
    merge it forward into the next chunk in the same section.

    Recommendation chunks are kept even when short — they're high-signal.
    """
    if not chunks:
        return chunks
    merged: list[Chunk] = [chunks[0]]
    for c in chunks[1:]:
        prev = merged[-1]
        is_rec = bool(RECOMMENDATION_RE.search(c.text[:200]))
        same_section = c.section_title == prev.section_title
        if (
            c.token_count < min_tokens
            and same_section
            and not is_rec
        ):
            prev.text = prev.text.rstrip() + "\n\n" + c.text.lstrip()
            prev.token_count = count_tokens(prev.text, tokenizer)
            if c.page_end is not None:
                prev.page_end = max(prev.page_end or c.page_end, c.page_end)
            if c.page_start is not None and prev.page_start is None:
                prev.page_start = c.page_start
        else:
            merged.append(c)

    # Pass 2: leading-orphan handling.
    if len(merged) >= 2 and merged[0].token_count < min_tokens:
        first, second = merged[0], merged[1]
        is_first_rec = bool(RECOMMENDATION_RE.search(first.text[:200]))
        if first.section_title == second.section_title and not is_first_rec:
            second.text = first.text.rstrip() + "\n\n" + second.text.lstrip()
            second.token_count = count_tokens(second.text, tokenizer)
            if first.page_start is not None:
                second.page_start = min(
                    first.page_start, second.page_start or first.page_start
                )
            merged = merged[1:]

    return merged


def _extract_grade_metadata(chunk: Chunk) -> None:
    """Populate recommendation_id, grade_evidence, grade_strength on the chunk."""
    rec = RECOMMENDATION_RE.search(chunk.text)
    if rec:
        chunk.recommendation_id = rec.group("id").strip()

    strength = GRADE_STRENGTH_RE.search(chunk.text)
    if strength:
        chunk.grade_strength = strength.group("strength").lower()

    evidence = GRADE_EVIDENCE_RE.search(chunk.text)
    if evidence:
        chunk.grade_evidence = evidence.group("evidence").lower().replace("  ", " ")
