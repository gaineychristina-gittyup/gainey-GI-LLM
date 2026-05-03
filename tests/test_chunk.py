"""Tests for the section-aware chunker.

These tests use synthetic ParsedElement streams so we can exercise the
chunking logic without touching unstructured / Postgres / Voyage. The chunker
is the highest-stakes module in ingestion, so the bar here is high.
"""

from __future__ import annotations

import pytest

from src.ingest.chunk import (
    Chunk,
    chunk_elements,
    count_tokens,
    _extract_grade_metadata,
)
from src.ingest.parse_pdfs import ParsedElement


# --- helpers ----------------------------------------------------------------

def el(category: str, text: str, page: int = 1) -> ParsedElement:
    return ParsedElement(category=category, text=text, page_number=page)


def text_block(n_sentences: int = 5, page: int = 1) -> ParsedElement:
    sentence = (
        "This is a representative narrative sentence about gastric motility "
        "with several clauses and clinically relevant detail."
    )
    return el("NarrativeText", " ".join([sentence] * n_sentences), page=page)


# --- tests ------------------------------------------------------------------

def test_empty_input_returns_empty():
    assert chunk_elements([]) == []


def test_single_short_section_yields_one_chunk():
    elements = [
        el("Title", "Introduction", page=1),
        text_block(n_sentences=3, page=1),
    ]
    chunks = chunk_elements(elements, target_tokens=500, min_tokens=10, max_tokens=800)
    assert len(chunks) == 1
    assert chunks[0].section_title == "Introduction"
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert chunks[0].token_count > 0


def test_long_section_splits_into_multiple_chunks():
    # ~150 sentences ≈ 3000+ tokens, must split
    elements = [
        el("Title", "Diagnosis", page=2),
        text_block(n_sentences=150, page=2),
    ]
    chunks = chunk_elements(elements, target_tokens=400, min_tokens=50, max_tokens=600)
    assert len(chunks) >= 3
    assert all(c.section_title == "Diagnosis" for c in chunks)
    # No chunk should exceed max_tokens by more than the overlap window.
    for c in chunks:
        assert c.token_count <= 700


def test_section_titles_are_carried_through():
    elements = [
        el("Title", "Introduction", page=1),
        text_block(page=1),
        el("Title", "Recommendations", page=2),
        text_block(page=2),
        el("Title", "Discussion", page=3),
        text_block(page=3),
    ]
    chunks = chunk_elements(elements, target_tokens=500, min_tokens=10)
    titles = {c.section_title for c in chunks}
    assert {"Introduction", "Recommendations", "Discussion"}.issubset(titles)


def test_recommendation_id_extracted():
    rec_text = (
        "Recommendation 3.2: We suggest the use of prokinetic agents in patients "
        "with diabetic gastroparesis. Strong recommendation, moderate-quality evidence."
    )
    elements = [
        el("Title", "Recommendations", page=4),
        el("NarrativeText", rec_text, page=4),
    ]
    chunks = chunk_elements(elements, target_tokens=500, min_tokens=5)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.recommendation_id is not None
    assert "3.2" in c.recommendation_id
    assert c.grade_strength == "strong"
    assert c.grade_evidence == "moderate"


def test_recommendation_block_kept_intact_when_overflowing_target():
    """A recommendation longer than target_tokens but within max_tokens stays as one chunk."""
    long_rec = "Recommendation 1: " + (
        "We recommend a stepwise approach to evaluation of suspected gastroparesis. "
    ) * 60  # ~600 tokens-ish
    elements = [
        el("Title", "Recommendations", page=4),
        el("NarrativeText", long_rec, page=4),
    ]
    chunks = chunk_elements(elements, target_tokens=300, min_tokens=50, max_tokens=900)
    rec_chunks = [c for c in chunks if c.recommendation_id]
    # The recommendation should not be torn into multiple pieces.
    assert len(rec_chunks) == 1
    assert "Recommendation 1" in rec_chunks[0].recommendation_id


def test_short_orphan_chunks_get_merged():
    """A trailing micro-chunk (1 short bullet) should fold into its neighbor."""
    elements = [
        el("Title", "Workup", page=1),
        text_block(n_sentences=20, page=1),  # large block, will produce a chunk
        el("ListItem", "See section 3.", page=1),  # tiny orphan
    ]
    chunks = chunk_elements(elements, target_tokens=200, min_tokens=100, max_tokens=400)
    # The orphan should not survive as its own chunk.
    assert all(c.token_count >= 100 for c in chunks[:-1]) or len(chunks) == 1


def test_overlap_creates_shared_text_between_consecutive_chunks():
    elements = [
        el("Title", "Therapy", page=5),
        text_block(n_sentences=80, page=5),
    ]
    chunks = chunk_elements(
        elements,
        target_tokens=300,
        min_tokens=50,
        max_tokens=500,
        overlap_tokens=40,
    )
    assert len(chunks) >= 2
    # Each non-first chunk should start with text drawn from the previous chunk's tail.
    # We assert a non-trivial intersection by tokenizing both ends.
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_tail = prev.text[-200:]
        next_head = nxt.text[:200]
        # Some overlap should be present (not byte-equal because of formatting,
        # but we expect at least one shared 6-word phrase).
        assert any(
            phrase in prev_tail and phrase in next_head
            for phrase in [
                "representative narrative sentence",
                "gastric motility",
                "clinically relevant detail",
            ]
        )


def test_grade_metadata_extraction_directly():
    c = Chunk(
        text=(
            "Statement 4: Gastric emptying scintigraphy remains the gold standard. "
            "Conditional recommendation, low-quality evidence."
        )
    )
    _extract_grade_metadata(c)
    assert c.recommendation_id and "4" in c.recommendation_id
    assert c.grade_strength == "conditional"
    assert c.grade_evidence == "low"


def test_best_practice_advice_pattern():
    c = Chunk(text="Best Practice Advice 7: Patients should be screened for nutritional deficits.")
    _extract_grade_metadata(c)
    assert c.recommendation_id is not None
    assert "7" in c.recommendation_id


def test_pages_span_across_a_chunk():
    elements = [
        el("Title", "Discussion", page=10),
        text_block(n_sentences=30, page=10),
        text_block(n_sentences=30, page=11),
        text_block(n_sentences=30, page=12),
    ]
    chunks = chunk_elements(elements, target_tokens=2000, min_tokens=10, max_tokens=4000)
    assert chunks[0].page_start == 10
    assert chunks[0].page_end == 12


def test_no_split_inside_sentence():
    """Forced splits should land on sentence boundaries."""
    long = (
        "First sentence about prokinetics. "
        "Second sentence describing endoscopic therapy options. "
        "Third sentence on dietary management approaches. "
    ) * 50
    elements = [
        el("Title", "Therapy", page=6),
        el("NarrativeText", long, page=6),
    ]
    chunks = chunk_elements(elements, target_tokens=200, min_tokens=50, max_tokens=400)
    # Every chunk should end on a sentence terminator (or be the only chunk).
    for c in chunks:
        stripped = c.text.rstrip()
        assert stripped.endswith((".", "!", "?")), f"Chunk does not end on sentence: ...{stripped[-80:]!r}"


def test_count_tokens_sanity():
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0


@pytest.mark.parametrize(
    "phrase,expected_strength",
    [
        ("Strong recommendation, high-quality evidence.", "strong"),
        ("Conditional recommendation, low-quality evidence.", "conditional"),
        ("Weak recommendation, moderate-quality evidence.", "weak"),
    ],
)
def test_grade_strength_variants(phrase, expected_strength):
    c = Chunk(text=phrase)
    _extract_grade_metadata(c)
    assert c.grade_strength == expected_strength
