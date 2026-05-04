"""Tests for the Streamlit rendering helpers in :mod:`src.ui.components`.

These tests don't boot Streamlit — they exercise the pure-string helpers so
we can pin down behavior that's easy to regress (e.g. the inline citation
renderer used to replace every newline with <br/>, which silently broke
Markdown bullet lists and tables in the answer).
"""

from __future__ import annotations

from src.ui.components import (
    derive_doc_type,
    derive_lead_author,
    format_citation,
    format_grade_label,
    render_inline_citations,
)


# --- citation rendering ----------------------------------------------------


def test_inline_citation_becomes_anchor():
    out = render_inline_citations("BQT preferred [1].")
    assert 'href="#cite-1"' in out
    assert ">[1]<" in out


def test_grouped_inline_citations_split():
    out = render_inline_citations("Both societies agree [1, 3].")
    assert 'href="#cite-1"' in out
    assert 'href="#cite-3"' in out
    # The two badges should be space-separated, not concatenated.
    assert "</a> <a" in out


def test_newlines_are_preserved_for_markdown():
    """Regression: an earlier version replaced every \\n with <br/>, which
    flattened bulleted lists and pipe-table rows so Streamlit's Markdown
    renderer gave up. Newlines must now pass through unchanged."""
    out = render_inline_citations("- item one [1]\n- item two [2]\n")
    assert "<br/>" not in out
    assert "\n- item two " in out


def test_pipe_table_passes_through():
    md = "| Society | First-line |\n| --- | --- |\n| ACG | BQT [1] |\n"
    out = render_inline_citations(md)
    assert "<br/>" not in out
    assert "| --- | --- |\n" in out
    assert "| ACG | BQT " in out


def test_paragraph_breaks_preserved():
    out = render_inline_citations("Para one [1].\n\nPara two [2].")
    assert "\n\n" in out
    assert "<br/>" not in out


def test_text_without_citations_unchanged():
    out = render_inline_citations("Plain prose with no citations.")
    assert out == "Plain prose with no citations."


# --- doc_type / GRADE / author helpers (unchanged behavior, pinned here) ---


def test_doc_type_classifier():
    assert derive_doc_type("ACG Clinical Guideline: H. pylori") == "Guideline"
    assert derive_doc_type("AASLD Practice Guidance on HCC") == "Guidance"
    assert derive_doc_type("ASGE Guideline on Minimum Staffing") == "Standards"
    assert derive_doc_type("Quality Indicators for ERCP") == "Standards"
    assert derive_doc_type("Clinical Practice Update on IBS-D") == "Other"
    assert derive_doc_type(None) == "Other"


def test_format_grade_label_handles_partials():
    assert format_grade_label("strong", "moderate") == (
        "Strong recommendation; moderate-quality evidence"
    )
    assert format_grade_label("conditional", None) == "Conditional recommendation"
    assert format_grade_label(None, "low") == "low-quality evidence"
    assert format_grade_label(None, None) == ""


def test_lead_author_from_filename():
    assert derive_lead_author("data/pdfs/ACG_2024_Hpylori_Chey.pdf") == "Chey"
    assert derive_lead_author("data/pdfs/AGA_2025_Gastroparesis_Staller.pdf") == "Staller"
    assert derive_lead_author("toofewunderscores.pdf") == ""
    assert derive_lead_author(None) == ""


def test_format_citation_compact():
    c = {
        "society": "ACG", "year": 2024,
        "pdf_path": "data/pdfs/ACG_2024_Hpylori_Chey.pdf",
        "recommendation_id": "Rec 7",
        "page_start": 1742, "page_end": 1745,
    }
    s = format_citation(c)
    assert "ACG 2024" in s and "Chey" in s and "Rec 7" in s and "pp. 1742-1745" in s
