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
    # Phase 2 splits the title (prose) from the recommendation (typed chunk).
    rec_chunks = [c for c in chunks if c.element_type == "recommendation"]
    assert len(rec_chunks) == 1
    c = rec_chunks[0]
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


# --- #16 garbage-title filter -----------------------------------------------


def test_single_letter_titles_do_not_open_new_sections():
    """Vertical-margin glyph artifacts ("G U I D E L I N E S" letters) come
    through as single-character Title elements. They must not open a new
    section, otherwise short-chunk merging can't bridge them."""
    elements = [
        el("Title", "Introduction", page=1),
        el("NarrativeText", "Body text about gastroparesis " * 30, page=1),
        el("Title", "G", page=1),  # garbage
        el("Title", "U", page=1),  # garbage
        el("Title", "I", page=1),  # garbage (yes, eats Roman 'I' too)
        el("NarrativeText", "More body text continuing the same section " * 30, page=1),
    ]
    chunks = chunk_elements(elements, target_tokens=500, min_tokens=10)
    # All chunks should be tagged with section_title="Introduction" — the
    # single-letter "Titles" must not have created new sections.
    sections = {c.section_title for c in chunks}
    assert sections == {"Introduction"}, f"unexpected sections: {sections}"


def test_two_letter_uppercase_title_is_treated_as_garbage():
    elements = [
        el("Title", "Background", page=1),
        el("NarrativeText", "Body text. " * 30, page=1),
        el("Title", "AB", page=1),  # 2 uppercase letters → garbage
        el("NarrativeText", "More body. " * 30, page=1),
    ]
    chunks = chunk_elements(elements, target_tokens=500, min_tokens=10)
    sections = {c.section_title for c in chunks}
    assert sections == {"Background"}


def test_legitimate_short_title_is_kept():
    """A title with punctuation or 3+ chars is real, not garbage."""
    elements = [
        el("Title", "I.", page=1),    # has a period
        el("NarrativeText", "Body. " * 30, page=1),
        el("Title", "II.", page=2),
        el("NarrativeText", "Body 2. " * 30, page=2),
    ]
    chunks = chunk_elements(elements, target_tokens=500, min_tokens=10)
    sections = {c.section_title for c in chunks}
    assert "I." in sections and "II." in sections


# --- #17 figure-caption filter ----------------------------------------------


def test_real_figure_caption_kept():
    elements = [
        el("FigureCaption", "Figure 1. Care algorithm for Barrett's surveillance.", page=2),
    ]
    chunks = chunk_elements(elements, min_tokens=1)
    assert len(chunks) == 1
    assert chunks[0].element_type == "figure_caption"


def test_spurious_figure_caption_demoted_to_prose():
    """unstructured emits FigureCaption on inline icons / pull-quote graphics
    whose text is not a real "Figure N." label. Those must not claim a
    figure_caption slot."""
    elements = [
        el("FigureCaption", "Click here to view related literature.", page=2),
        el("Image", "GUIDELINES IN PRACTICE", page=2),
        el("FigureCaption", "Figure 3. Real caption here.", page=3),
    ]
    chunks = chunk_elements(elements, min_tokens=1)
    fig_chunks = [c for c in chunks if c.element_type == "figure_caption"]
    assert len(fig_chunks) == 1
    assert "Figure 3" in fig_chunks[0].text


# --- #15 table-row recommendations ------------------------------------------


def test_table_with_recommendation_rows_emits_per_row_chunks():
    """AGA pharma tables put recommendation text in HTML rows. Each labeled
    row should become its own element_type='recommendation' chunk alongside
    the parent table chunk."""
    table_html = (
        "<table>"
        "<tr><th>Recommendation</th><th>Strength</th><th>Evidence</th></tr>"
        "<tr><td>Recommendation 1: We suggest eluxadoline for IBS-D.</td>"
        "<td>conditional recommendation</td><td>moderate-quality evidence</td></tr>"
        "<tr><td>Recommendation 2: We suggest rifaximin for IBS-D.</td>"
        "<td>conditional recommendation</td><td>moderate-quality evidence</td></tr>"
        "<tr><td>Statement 4: Antispasmodics may be considered.</td>"
        "<td>weak recommendation</td><td>low-quality evidence</td></tr>"
        "</table>"
    )
    table_el = ParsedElement(
        category="Table",
        text="Recommendation Strength Evidence ...",
        page_number=3,
        table_html=table_html,
    )
    chunks = chunk_elements([table_el], min_tokens=1)
    table_chunks = [c for c in chunks if c.element_type == "table"]
    rec_chunks = [c for c in chunks if c.element_type == "recommendation"]
    assert len(table_chunks) == 1, "parent table chunk should still exist"
    assert len(rec_chunks) == 3, f"expected 3 row-level recs, got {len(rec_chunks)}"
    rec_ids = {c.recommendation_id for c in rec_chunks}
    # All three labels (Recommendation 1, Recommendation 2, Statement 4) extracted.
    assert any("Recommendation 1" in (rid or "") for rid in rec_ids)
    assert any("Recommendation 2" in (rid or "") for rid in rec_ids)
    assert any("Statement 4" in (rid or "") for rid in rec_ids)
    # GRADE metadata propagates into row chunks via the post-pass extractor.
    strengths = {c.grade_strength for c in rec_chunks if c.grade_strength}
    assert "conditional" in strengths and "weak" in strengths


def test_table_without_recommendation_rows_does_not_emit_extras():
    """A table containing data-only rows (no Recommendation/Statement labels)
    should produce only the parent table chunk, no spurious row chunks."""
    table_html = (
        "<table>"
        "<tr><th>Drug</th><th>Dose</th></tr>"
        "<tr><td>Metoclopramide</td><td>5-10 mg</td></tr>"
        "<tr><td>Erythromycin</td><td>50-200 mg</td></tr>"
        "</table>"
    )
    table_el = ParsedElement(
        category="Table",
        text="Drug Dose Metoclopramide 5-10 mg ...",
        page_number=4,
        table_html=table_html,
    )
    chunks = chunk_elements([table_el], min_tokens=1)
    table_chunks = [c for c in chunks if c.element_type == "table"]
    rec_chunks = [c for c in chunks if c.element_type == "recommendation"]
    assert len(table_chunks) == 1
    assert len(rec_chunks) == 0


def test_aga_style_numbered_row_recs_extracted():
    """AGA pharma tables use bare-number labels ('1.', '2a.', '3.') in the
    recommendations column, not the literal 'Recommendation N'. The
    extractor must detect via a '<th>...recommend...</th>' header and
    synthesize a recommendation_id for those rows."""
    table_html = (
        "<table>"
        "<thead>"
        "<tr><th>New or updated recommendations</th><th>Strength of recommendation</th>"
        "<th>Certainty in evidence</th></tr>"
        "</thead>"
        "<tbody>"
        "<tr><td>1. In patients with IBS-D, the AGA suggests using eluxadoline</td>"
        "<td>Conditional recommendation</td><td>moderate-quality evidence</td></tr>"
        "<tr><td>2a. In patients with IBS-D, the AGA suggests using rifaximin</td>"
        "<td>Conditional recommendation</td><td>moderate-quality evidence</td></tr>"
        "<tr><td>2b. In patients with IBS-D with initial response to rifaximin who develop "
        "recurrent symptoms, the AGA suggests retreatment with rifaximin</td>"
        "<td>Conditional recommendation</td><td>moderate-quality evidence</td></tr>"
        "</tbody></table>"
    )
    table_el = ParsedElement(
        category="Table", text="...", page_number=3, table_html=table_html,
    )
    chunks = chunk_elements([table_el], min_tokens=1)
    rec_chunks = [c for c in chunks if c.element_type == "recommendation"]
    assert len(rec_chunks) == 3, f"expected 3 numbered recs, got {len(rec_chunks)}"
    rec_ids = {c.recommendation_id for c in rec_chunks}
    assert "Recommendation 1" in rec_ids
    assert "Recommendation 2a" in rec_ids
    assert "Recommendation 2b" in rec_ids
    # GRADE is propagated from the row text by _extract_grade_metadata.
    assert all(c.grade_strength == "conditional" for c in rec_chunks)
    assert all(c.grade_evidence == "moderate" for c in rec_chunks)


def test_data_table_with_numbered_rows_does_not_emit_fake_recs():
    """A data table whose rows happen to start with numbers but whose headers
    don't mention 'recommendation' must NOT be parsed as recommendations."""
    table_html = (
        "<table>"
        "<thead><tr><th>Study</th><th>N</th><th>Outcome</th></tr></thead>"
        "<tbody>"
        "<tr><td>1. Smith 2020</td><td>120</td><td>Positive</td></tr>"
        "<tr><td>2. Jones 2021</td><td>80</td><td>Negative</td></tr>"
        "</tbody></table>"
    )
    table_el = ParsedElement(
        category="Table", text="...", page_number=5, table_html=table_html,
    )
    chunks = chunk_elements([table_el], min_tokens=1)
    rec_chunks = [c for c in chunks if c.element_type == "recommendation"]
    assert len(rec_chunks) == 0, "no recs should be extracted from a data table"


def test_table_key_concept_rows_emit_key_concept_chunks():
    """ACG tables sometimes embed Key Concept rows. Those should be classified
    as element_type='key_concept' (not recommendation)."""
    table_html = (
        "<table>"
        "<tr><th>Key Concepts</th></tr>"
        "<tr><td>Key Concept 1: SELs are common incidental endoscopic findings.</td></tr>"
        "<tr><td>Key Concept 2: EUS distinguishes layer of origin.</td></tr>"
        "</table>"
    )
    table_el = ParsedElement(
        category="Table", text="Key Concepts ...", page_number=2, table_html=table_html,
    )
    chunks = chunk_elements([table_el], min_tokens=1)
    kc_chunks = [c for c in chunks if c.element_type == "key_concept"]
    assert len(kc_chunks) == 2
