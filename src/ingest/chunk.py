"""Section-aware chunking for clinical guidelines.

Chunking rules (in plain English so they're easy to tune):

  1. Walk the parsed elements in document order. ``Title`` elements start a new
     section. Everything until the next ``Title`` belongs to that section.

  2. Within a section, accumulate prose text until we hit ~``target_tokens``.
     When the accumulator goes over, flush a prose chunk and start the next one
     with an overlap window of ``overlap_tokens`` tokens (so context bleeds
     across boundaries — important for retrieval).

  3. **Typed chunks are never merged with prose.** Tables, figure captions,
     recommendations, and key concepts each become standalone chunks regardless
     of length. They carry an ``element_type`` field on the Chunk so the UI and
     retrieval layer can render or filter them differently.

       - element_type='table': one chunk per Table element. ``table_html``
         carries unstructured's HTML rendering; ``text`` is the plain-text
         flattening (which is what we embed).
       - element_type='figure_caption': one chunk per FigureCaption element.
         ``figure_image_path`` is set later by build_index.py once the figure
         crop is materialized to disk.
       - element_type='recommendation': one chunk per element that matches
         RECOMMENDATION_RE (Recommendation N, Statement N, BPA N, Quality
         Indicator N).
       - element_type='key_concept': one chunk per element that matches
         KEY_CONCEPT_RE (ACG-style "Key Concept N" labels).
       - element_type='prose': everything else.

  4. We never split mid-sentence. The forced-split path (when a section is
     longer than ``max_tokens``) walks back to the most recent sentence
     terminator before flushing.

  5. After all chunks are produced, any *prose* chunk shorter than ``min_tokens``
     is merged with its previous prose neighbor in the same section. Typed
     chunks (table / figure_caption / recommendation / key_concept) are kept
     even when short — they're high-signal and per spec must not be merged.

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

# ACG-style "Key Concept N" labels. ACG guidelines pair their numbered
# Recommendations (which carry GRADE) with Key Concepts (which don't), so we
# track them separately even though they look like recommendations textually.
KEY_CONCEPT_RE = re.compile(
    r"\b(?P<id>Key\s+Concept\s*\d+(?:\.\d+)?[A-Za-z]?)\b",
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

# Real figure captions in society guidelines start with "Figure N." or "Fig. N.".
# unstructured over-fires FigureCaption on small inline icons / pull-quote
# graphics / margin furniture (e.g. ASGE Cholangitis 2021 emitted 82 figure
# captions). Demote anything not matching this shape to prose.
FIGURE_CAPTION_RE = re.compile(
    r"^\s*(?:Figure|Fig\.?|FIGURE)\s+\d+[A-Za-z]?\s*[.:]",
)

# Single-character "Title" elements that come from vertical-margin runs
# ("G U I D E L I N E S" letters along the page edge) — each one creates
# a fake section under _group_into_sections, defeating short-chunk merging.
# We drop them; legitimate Roman-numeral / chapter-letter headings are 3+
# chars or include punctuation.
_TITLE_GARBAGE_RE = re.compile(r"^[A-Z]{1,2}$")


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
    # Phase 2: typed chunks
    element_type: str = "prose"   # 'prose'|'table'|'figure_caption'|'recommendation'|'key_concept'
    table_html: Optional[str] = None          # only for element_type='table'
    figure_image_path: Optional[str] = None   # only for element_type='figure_caption'
    figure_bbox: Optional[tuple[float, float, float, float]] = None  # passthrough for cropper
    figure_layout_size: Optional[tuple[float, float]] = None         # (layout_w, layout_h)
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


def _is_garbage_title(text: str) -> bool:
    """Detect single/double-letter "Title" elements that are vertical-margin
    glyph artifacts (e.g. the letters of 'G U I D E L I N E S' running down
    the page edge), not real section headings. Each one would otherwise open
    a new section that the same-section merge can't bridge.
    """
    s = text.strip()
    return bool(_TITLE_GARBAGE_RE.match(s))


def _group_into_sections(
    elements: list[ParsedElement],
) -> list[tuple[Optional[str], list[ParsedElement]]]:
    """Partition the element stream by ``Title`` boundaries.

    Returns a list of (section_title, [elements]) tuples in document order.
    Anything before the first Title goes under ``section_title=None``.
    Single-character / 2-uppercase-letter Title elements are skipped (treated
    as if they didn't exist) — see :func:`_is_garbage_title`.
    """
    sections: list[tuple[Optional[str], list[ParsedElement]]] = []
    current_title: Optional[str] = None
    current: list[ParsedElement] = []

    for el in elements:
        if el.category == "Title" and not _is_garbage_title(el.text):
            if current:
                sections.append((current_title, current))
                current = []
            current_title = el.text
            # The title itself is included in its own section so retrieval
            # can match on heading text.
            current.append(el)
        else:
            # Garbage titles fall through to here too — they get appended as
            # ordinary elements so any tokens they contain aren't lost, but
            # they do NOT open a new section.
            current.append(el)

    if current:
        sections.append((current_title, current))
    return sections


def _classify_element(elem: ParsedElement) -> str:
    """Return one of 'table'|'figure_caption'|'recommendation'|'key_concept'|'prose'.

    Order matters: a table-of-recommendations is a Table element first; a
    Key Concept block is detected before Recommendation because some societies
    (notably ACG) put both kinds of labeled blocks side-by-side.

    FigureCaption / Image elements only earn the 'figure_caption' label when
    their text actually opens with "Figure N." or "Fig N.". Without this
    filter, unstructured over-fires Image/FigureCaption on small inline icons,
    table separators, and pull-quote graphics — the ASGE Cholangitis 2021
    PDF produced 82 figure captions before this filter, ~6 after.
    """
    if elem.category == "Table":
        return "table"
    if elem.category in ("FigureCaption", "Image"):
        if FIGURE_CAPTION_RE.match(elem.text or ""):
            return "figure_caption"
        # Spurious figure detection — treat as prose so any text content
        # is still embedded but doesn't claim a figure_caption slot.
        return "prose"
    head = elem.text[:200]
    if KEY_CONCEPT_RE.search(head):
        return "key_concept"
    if RECOMMENDATION_RE.search(head):
        return "recommendation"
    if elem.category == "ListItem" and GRADE_STRENGTH_RE.search(head):
        return "recommendation"
    return "prose"


_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_TH_RE = re.compile(r"<th[^>]*>(.*?)</th>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
# "1." / "2a." / "10." at the start of a row, when the table is a
# recommendations table — AGA pharma guidelines use this layout.
_NUMBERED_ROW_RE = re.compile(r"^(?P<num>\d+[a-zA-Z]?)\s*\.\s*\S")
# Headers that signal "this whole table is recommendations" — used to gate
# the numbered-row extraction so we don't false-positive on data tables that
# happen to have numbered rows ("1. study", etc.).
_REC_HEADER_RE = re.compile(
    # Stem-match (no trailing \b) so "recommendation"/"recommendations"/
    # "statements" all hit. Leading \b only.
    r"\b(recommend|statement|best\s+practice|key\s+concept|bpa|quality\s+indicator)",
    re.IGNORECASE,
)


def _table_is_recommendations(table_html: str) -> bool:
    """True if any <th> in the table mentions recommendation/statement/etc."""
    for th_html in _TH_RE.findall(table_html):
        text = _TAG_RE.sub(" ", th_html)
        if _REC_HEADER_RE.search(text):
            return True
    return False


def _extract_table_row_recommendations(
    table_html: Optional[str],
    page_number: Optional[int],
    section_title: Optional[str],
    tokenizer: str,
) -> list[Chunk]:
    """For tables that contain a "Recommendation N" / "Statement N" / "BPA N"
    / "Quality Indicator N" / "Key Concept N" labeled row, emit each such row
    as its own typed chunk so it's individually retrievable and carries
    GRADE metadata.

    The original table chunk is kept as well — it's the parent for UI
    rendering. The row chunks live alongside it in the chunks table.

    AGA pharmacological-management guidelines (IBS-D, IBS-C, UC-pharm)
    layout their recommendations as table rows where the rec text is
    labeled with a bare number ("1.", "2a.", "2b.", "3.") rather than the
    literal word "Recommendation". For those tables we detect a
    "recommendations" column header in <th>, and when present accept
    numbered-row leads as recommendations, synthesizing a recommendation_id.
    GRADE strength/evidence on each row are then filled in by
    :func:`_extract_grade_metadata` during the final annotation pass.
    """
    if not table_html:
        return []
    is_rec_table = _table_is_recommendations(table_html)
    out: list[Chunk] = []
    for row_html in _TR_RE.findall(table_html):
        text = _TAG_RE.sub(" ", row_html)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        head = text[:200]

        etype: Optional[str] = None
        synthetic_rec_id: Optional[str] = None
        if KEY_CONCEPT_RE.search(head):
            etype = "key_concept"
        elif RECOMMENDATION_RE.search(head):
            etype = "recommendation"
        elif is_rec_table:
            m = _NUMBERED_ROW_RE.match(head)
            if m:
                etype = "recommendation"
                synthetic_rec_id = f"Recommendation {m.group('num')}"

        if etype is None:
            continue

        chunk = Chunk(
            text=text,
            section_title=section_title,
            page_start=page_number,
            page_end=page_number,
            token_count=count_tokens(text, tokenizer),
            element_type=etype,
        )
        if synthetic_rec_id:
            chunk.recommendation_id = synthetic_rec_id
        out.append(chunk)
    return out


def _table_to_plain_text(html: Optional[str], fallback_text: str) -> str:
    """Render an unstructured Table HTML payload to plain text for embedding.

    The fallback is the element's own ``str(...)`` text. We try a tiny HTML
    flattener first because unstructured's str(Table) is sometimes a flat
    space-joined run that mangles row boundaries.
    """
    if not html:
        return fallback_text.strip()
    # Inject row/cell separators before stripping tags so the embedded text
    # preserves cell boundaries — important for retrieval over tables.
    cleaned = (
        html.replace("</tr>", "</tr>\n")
        .replace("</td>", " | </td>")
        .replace("</th>", " | </th>")
    )
    text = re.sub(r"<[^>]+>", "", cleaned)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # Trim trailing " | " left over from the last cell on each row.
    text = re.sub(r"\s*\|\s*$", "", text, flags=re.MULTILINE)
    return text or fallback_text.strip()


def _chunk_section(
    section_title: Optional[str],
    section_elems: list[ParsedElement],
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
    tokenizer: str,
) -> list[Chunk]:
    """Chunk a single section. Typed elements emit their own standalone chunks;
    prose accumulates into windowed prose chunks with overlap between them.
    """
    chunks: list[Chunk] = []

    buf_texts: list[str] = []
    buf_tokens = 0
    buf_pages: list[int] = []

    def flush_prose() -> None:
        """Flush the accumulated prose buffer as one element_type='prose' chunk."""
        nonlocal buf_texts, buf_tokens, buf_pages
        if not buf_texts:
            return
        text = "\n\n".join(buf_texts).strip()
        if not text:
            buf_texts, buf_tokens, buf_pages = [], 0, []
            return
        chunks.append(
            Chunk(
                text=text,
                section_title=section_title,
                page_start=min(buf_pages) if buf_pages else None,
                page_end=max(buf_pages) if buf_pages else None,
                token_count=count_tokens(text, tokenizer),
                element_type="prose",
            )
        )
        # Reset with overlap from tail of just-flushed text.
        if overlap_tokens > 0:
            tail = _tail_by_tokens(text, overlap_tokens, tokenizer)
            buf_texts = [tail] if tail else []
            buf_tokens = count_tokens(tail, tokenizer) if tail else 0
        else:
            buf_texts, buf_tokens = [], 0
        buf_pages = []

    for el in section_elems:
        kind = _classify_element(el)

        # Typed elements always interrupt prose accumulation and emit their own
        # standalone chunk (or pair of chunks for over-long recommendations).
        if kind == "table":
            flush_prose()
            text_for_embedding = _table_to_plain_text(el.table_html, el.text)
            chunks.append(
                Chunk(
                    text=text_for_embedding,
                    section_title=section_title,
                    page_start=el.page_number,
                    page_end=el.page_number,
                    token_count=count_tokens(text_for_embedding, tokenizer),
                    element_type="table",
                    table_html=el.table_html,
                )
            )
            # If the table contains "Recommendation N" / "Statement N" / "BPA N"
            # / "Key Concept N" rows, also emit each row as a typed chunk so
            # individual recs are retrievable with their own GRADE metadata.
            chunks.extend(
                _extract_table_row_recommendations(
                    el.table_html, el.page_number, section_title, tokenizer,
                )
            )
            continue

        if kind == "figure_caption":
            flush_prose()
            layout = el.metadata.get("coordinates", {}) or {}
            layout_size = None
            lw, lh = layout.get("layout_width"), layout.get("layout_height")
            if lw and lh:
                layout_size = (float(lw), float(lh))
            chunks.append(
                Chunk(
                    text=el.text,
                    section_title=section_title,
                    page_start=el.page_number,
                    page_end=el.page_number,
                    token_count=count_tokens(el.text, tokenizer),
                    element_type="figure_caption",
                    figure_bbox=el.bbox,
                    figure_layout_size=layout_size,
                )
            )
            continue

        if kind in ("recommendation", "key_concept"):
            flush_prose()
            el_tokens = count_tokens(el.text, tokenizer)
            if el_tokens <= max_tokens:
                chunks.append(
                    Chunk(
                        text=el.text,
                        section_title=section_title,
                        page_start=el.page_number,
                        page_end=el.page_number,
                        token_count=el_tokens,
                        element_type=kind,
                    )
                )
            else:
                # Long recommendations split on sentence boundaries; each piece
                # keeps element_type so it isn't merged with neighbors later.
                for piece in _split_long_text(el.text, target_tokens, max_tokens, tokenizer):
                    chunks.append(
                        Chunk(
                            text=piece,
                            section_title=section_title,
                            page_start=el.page_number,
                            page_end=el.page_number,
                            token_count=count_tokens(piece, tokenizer),
                            element_type=kind,
                        )
                    )
            continue

        # ---- prose path ----
        el_tokens = count_tokens(el.text, tokenizer)

        # Would adding this element overflow?
        if buf_tokens + el_tokens > max_tokens and buf_texts:
            flush_prose()

        if el_tokens > max_tokens:
            if buf_texts:
                flush_prose()
            for piece in _split_long_text(el.text, target_tokens, max_tokens, tokenizer):
                buf_texts.append(piece)
                buf_tokens += count_tokens(piece, tokenizer)
                if el.page_number is not None:
                    buf_pages.append(el.page_number)
                if buf_tokens >= target_tokens:
                    flush_prose()
            continue

        buf_texts.append(el.text)
        buf_tokens += el_tokens
        if el.page_number is not None:
            buf_pages.append(el.page_number)

        if buf_tokens >= target_tokens:
            flush_prose()

    flush_prose()
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
    """Merge any short *prose* chunk into a same-section prose neighbor.

    Pass 1 (backward): for each non-first prose chunk, if it's short, merge
    into the previous prose chunk (preferred direction so context flows
    forward).
    Pass 2 (forward): if the FIRST chunk is a short prose chunk (typically a
    lone Title), merge it forward into the next prose chunk in the same
    section.

    Typed chunks (table / figure_caption / recommendation / key_concept) are
    NEVER merged — per Phase 2 spec, those stand alone regardless of length.
    """
    if not chunks:
        return chunks
    merged: list[Chunk] = [chunks[0]]
    for c in chunks[1:]:
        prev = merged[-1]
        same_section = c.section_title == prev.section_title
        # Both must be prose for a merge — protects all typed chunks in either slot.
        both_prose = c.element_type == "prose" and prev.element_type == "prose"
        if (
            c.token_count < min_tokens
            and same_section
            and both_prose
        ):
            prev.text = prev.text.rstrip() + "\n\n" + c.text.lstrip()
            prev.token_count = count_tokens(prev.text, tokenizer)
            if c.page_end is not None:
                prev.page_end = max(prev.page_end or c.page_end, c.page_end)
            if c.page_start is not None and prev.page_start is None:
                prev.page_start = c.page_start
        else:
            merged.append(c)

    # Pass 2: leading-orphan handling — only when both first and second are prose.
    if (
        len(merged) >= 2
        and merged[0].element_type == "prose"
        and merged[1].element_type == "prose"
        and merged[0].token_count < min_tokens
        and merged[0].section_title == merged[1].section_title
    ):
        first, second = merged[0], merged[1]
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
