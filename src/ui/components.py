"""Reusable rendering components for the Phase-5 Streamlit UI.

Keeping these out of ``streamlit_app.py`` so the main app file reads as a
layout/orchestration file, not a wall of HTML.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = REPO_ROOT / "data" / "corpus_snapshot.txt"
FIGURES_DIR = REPO_ROOT / "data" / "parsed" / "figures"

API_BASE = os.environ.get("GI_API_BASE", "http://localhost:8000")

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


# --- header / footer / corpus snapshot -------------------------------------


def read_corpus_snapshot() -> str:
    try:
        return SNAPSHOT_PATH.read_text().strip()
    except FileNotFoundError:
        return "unknown"


def render_corpus_snapshot_header(title: str, subtitle: str) -> None:
    """Page header — title, subtitle, snapshot date, disclaimer."""
    st.title(title)
    st.markdown(
        f"<p style='color:#5b6772; margin-top:-0.5rem;'>{subtitle}</p>",
        unsafe_allow_html=True,
    )
    snapshot = read_corpus_snapshot()
    st.markdown(
        f"<p style='font-size:0.85rem; color:#5b6772; margin-bottom:0.25rem;'>"
        f"<b>Corpus last updated:</b> {snapshot}"
        f"</p>",
        unsafe_allow_html=True,
    )
    st.info(
        "For research and reference. Not a substitute for clinical judgment.",
        icon=None,
    )


def render_footer() -> None:
    snapshot = read_corpus_snapshot()
    st.markdown("---")
    st.markdown(
        f"<p style='font-size:0.8rem; color:#7c8794;'>"
        f"Sources: AGA, ACG, ASGE, AASLD published guidelines. "
        f"Last refreshed {snapshot}. Built at UNC for research use."
        f"</p>",
        unsafe_allow_html=True,
    )


# --- citation / GRADE labels -----------------------------------------------


def format_grade_label(
    grade_strength: str | None, grade_evidence: str | None
) -> str:
    parts = []
    if grade_strength:
        parts.append(f"{grade_strength.title()} recommendation")
    if grade_evidence:
        parts.append(f"{grade_evidence.replace('_', ' ').replace('-', ' ')}-quality evidence")
    return "; ".join(parts)


def derive_doc_type(title: str | None) -> str:
    """Title-pattern classifier matching the SQL doc_type filter in
    src/retrieve/hybrid_search.py. Standards-style titles often also
    contain the word 'guideline' (e.g. 'ASGE guideline on minimum
    staffing'), so we check Standards keywords FIRST."""
    t = (title or "").lower()
    if any(k in t for k in (
        "quality indicator", "standards", "reprocessing", "minimum staffing"
    )):
        return "Standards"
    if "guidance" in t:
        return "Guidance"
    if "guideline" in t:
        return "Guideline"
    return "Other"


def derive_lead_author(pdf_path: str | None) -> str:
    """Pull the lead author from the filename — by convention the last
    underscore-separated token before the .pdf extension. Examples:
        ACG_2024_Hpylori_Chey.pdf  -> "Chey"
        AGA_2025_Gastroparesis_Staller.pdf -> "Staller"
        ASGE_2023_PostErcpPancreatitis_Buxbaum.pdf -> "Buxbaum"
    """
    if not pdf_path:
        return ""
    stem = Path(pdf_path).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return ""
    return parts[-1]


def format_citation(c: dict[str, Any]) -> str:
    """Compact one-line citation header — used in passage cards and the
    citations panel."""
    bits = [f"{c.get('society') or '?'} {c.get('year') or '?'}"]
    author = derive_lead_author(c.get("pdf_path") or "")
    if not author and c.get("title"):
        # Fallback: pull from "...by Smith et al."-ish patterns
        m = re.search(r"by\s+([A-Z][a-z]+)", c.get("title") or "")
        if m:
            author = m.group(1)
    if author:
        bits.append(author)
    if c.get("recommendation_id"):
        bits.append(c["recommendation_id"])
    page = ""
    if c.get("page_start") and c.get("page_end") and c["page_start"] != c["page_end"]:
        page = f"pp. {c['page_start']}-{c['page_end']}"
    elif c.get("page_start"):
        page = f"p. {c['page_start']}"
    if page:
        bits.append(page)
    return ", ".join(bits)


# --- inline citations in answer text ----------------------------------------


def render_inline_citations(answer_text: str) -> str:
    """Render [N] tags in the answer as small clickable anchors.

    Streamlit's markdown can't easily handle button-callbacks inside HTML,
    so we use plain `#passage-N` anchors that scroll the browser to the
    matching ``<div id="passage-N">`` rendered later by
    :func:`render_passage_card`. This is light-weight and works without
    any JS we'd have to ship.
    """
    def repl(m: re.Match) -> str:
        nums = [n.strip() for n in m.group(1).split(",")]
        badges = []
        for n in nums:
            badges.append(
                f'<a href="#passage-{n}" '
                f'style="background:#e3f2fd;color:#0b69c7;'
                f'padding:1px 7px;border-radius:10px;text-decoration:none;'
                f'font-size:0.85em;border:1px solid #cfe1f5;margin:0 2px;'
                f'font-weight:500;">[{n}]</a>'
            )
        return " ".join(badges)
    out = _CITATION_RE.sub(repl, answer_text)
    return out.replace("\n", "<br/>")


# --- citation validation badge ---------------------------------------------


def render_citation_badge(verification: dict[str, Any] | None) -> None:
    """Green/yellow badge showing how many citations validated.

    The verification dict is whatever ``src.generate.verify.verify_answer``
    produced — see that module for the shape.
    """
    if not verification:
        return
    n = int(verification.get("n_citations") or 0)
    if n == 0:
        return  # nothing to validate (e.g., refusal)
    bad = (
        len(verification.get("out_of_range") or [])
        + len(verification.get("unsupported") or [])
    )
    good = max(0, n - bad)
    if bad == 0:
        st.success(f"✓ {good}/{n} citations verified against retrieved sources")
    else:
        details = []
        if verification.get("out_of_range"):
            details.append(
                f"{len(verification['out_of_range'])} out-of-range "
                f"(citation index doesn't exist)"
            )
        if verification.get("unsupported"):
            details.append(
                f"{len(verification['unsupported'])} unsupported "
                f"(sentence may not be grounded in cited passage)"
            )
        st.warning(
            f"⚠ {good}/{n} citations verified — {bad} flagged for review "
            f"({', '.join(details)}). The model may have paraphrased "
            f"heavily or, less likely, hallucinated. Review the source "
            f"passages below before relying on those claims."
        )


# --- passage card -----------------------------------------------------------


def render_passage_card(
    chunk: dict[str, Any],
    rank: int,
    was_cited: bool,
    *,
    api_base: str = API_BASE,
) -> None:
    """Render one retrieved chunk as an expandable card.

    ``rank`` is the 1-based position used for the ``#passage-N`` anchor
    so the inline citation badges in the answer can scroll here.
    Cited cards get a colored left border to make them visually
    discoverable in the long passage list.
    """
    border_color = "#3b82f6" if was_cited else "#e5e7eb"
    fill_color = "#eff6ff" if was_cited else "transparent"
    icon = "🔖" if was_cited else "📄"

    soc = chunk.get("society") or "?"
    year = chunk.get("year") or "?"
    author = derive_lead_author(chunk.get("pdf_path") or "")
    rec_id = chunk.get("recommendation_id") or ""
    grade = format_grade_label(
        chunk.get("grade_strength"), chunk.get("grade_evidence")
    )
    score = chunk.get("relevance_score")
    score_str = f" · score {score:.3f}" if isinstance(score, (int, float)) else ""

    head_bits = [f"**[{rank}] {soc} {year}**"]
    if author:
        head_bits.append(author)
    if rec_id:
        head_bits.append(f"`{rec_id}`")
    if (et := chunk.get("element_type")) and et != "prose":
        head_bits.append(f"`{et}`")
    page = ""
    if chunk.get("page_start") and chunk.get("page_end") and chunk["page_start"] != chunk["page_end"]:
        page = f"pp. {chunk['page_start']}-{chunk['page_end']}"
    elif chunk.get("page_start"):
        page = f"p. {chunk['page_start']}"
    if page:
        head_bits.append(page)
    head = "  ·  ".join(head_bits) + score_str

    # Anchor target so [N] in the answer can scroll here.
    st.markdown(
        f'<div id="passage-{rank}" '
        f'style="border-left:4px solid {border_color}; padding:0.5rem 0.75rem; '
        f'background:{fill_color}; border-radius:4px; margin-top:0.25rem;">'
        f'<div style="font-size:0.95rem;">{icon} {head}</div>',
        unsafe_allow_html=True,
    )
    title = (chunk.get("title") or "").strip()
    if title:
        st.markdown(
            f'<div style="font-size:0.85rem; color:#5b6772; margin-top:0.25rem;">{title}</div>',
            unsafe_allow_html=True,
        )
    if grade:
        st.markdown(
            f'<div style="font-size:0.85rem; color:#3b82f6; margin-top:0.25rem;">'
            f'GRADE: {grade}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Show source passage", expanded=was_cited):
        et = chunk.get("element_type") or "prose"
        if et == "table" and chunk.get("table_html"):
            # Streamlit forbids nested expanders, so we use tabs instead to
            # offer both the rendered HTML view and the plain-text view (what
            # the embedder actually saw).
            tab_render, tab_plain = st.tabs(
                ["Rendered table", "Plain-text (what was embedded)"]
            )
            with tab_render:
                st.markdown(chunk["table_html"], unsafe_allow_html=True)
            with tab_plain:
                st.text(chunk.get("text") or "")
        elif et == "figure_caption" and chunk.get("figure_image_path"):
            rel = chunk["figure_image_path"]
            local_path = REPO_ROOT / rel
            if local_path.exists():
                st.image(str(local_path), caption=chunk.get("text") or "")
            else:
                # Fall back to API serve (lets the UI work when run from
                # a different working directory).
                stripped = rel.removeprefix("data/parsed/figures/")
                api_url = f"{api_base}/sources/figure/{stripped}"
                try:
                    st.image(api_url, caption=chunk.get("text") or "")
                except Exception:
                    st.info(
                        f"(Figure image not available on disk at `{rel}`. "
                        f"Caption text shown only.)"
                    )
                    st.markdown(chunk.get("text") or "(no caption)")
        else:
            text = chunk.get("text") or "(no text in chunk)"
            st.markdown(text)

        # Source PDF link, with sensible fallback to source_url
        st.markdown("---")
        link_cols = st.columns([1, 1])
        with link_cols[0]:
            doc_id = chunk.get("document_id")
            if doc_id is not None:
                st.link_button(
                    "📄 Open source PDF",
                    f"{api_base}/sources/document/{doc_id}/pdf",
                    use_container_width=True,
                )
            elif chunk.get("source_url"):
                st.link_button(
                    "🌐 Open source URL",
                    chunk["source_url"],
                    use_container_width=True,
                )
        with link_cols[1]:
            if chunk.get("doi"):
                st.markdown(
                    f"<div style='text-align:right; font-size:0.85rem; "
                    f"color:#5b6772; padding-top:0.5rem;'>"
                    f"DOI: {chunk['doi']}</div>",
                    unsafe_allow_html=True,
                )
