"""GaineyGuidelines — Streamlit UI for the GI guidelines RAG system.

Run with:
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import re

import streamlit as st

from src.ui.retriever import Citation, answer

st.set_page_config(
    page_title="GaineyGuidelines",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


_CSS = """
<style>
:root {
    --gg-accent: #0d6b5e;
    --gg-accent-soft: #e6f2f0;
    --gg-ink: #15212b;
    --gg-muted: #5b6770;
    --gg-border: #e3e8ec;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 920px;
}

.gg-header {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    border-bottom: 1px solid var(--gg-border);
    padding-bottom: 0.75rem;
    margin-bottom: 1.5rem;
}
.gg-header h1 {
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--gg-ink);
    margin: 0;
    letter-spacing: -0.01em;
}
.gg-header .gg-tag {
    font-size: 0.75rem;
    color: var(--gg-accent);
    background: var(--gg-accent-soft);
    padding: 2px 8px;
    border-radius: 999px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.gg-subtitle {
    color: var(--gg-muted);
    font-size: 0.92rem;
    margin: 0;
}

.gg-answer {
    background: #ffffff;
    border: 1px solid var(--gg-border);
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    line-height: 1.6;
    font-size: 1.02rem;
    color: var(--gg-ink);
    box-shadow: 0 1px 2px rgba(20, 33, 43, 0.04);
}
.gg-answer-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--gg-muted);
    margin-bottom: 0.4rem;
    font-weight: 600;
}

.gg-citations-row {
    margin-top: 1rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
}

/* Style the inline citation buttons (Streamlit renders them as <button>) */
.gg-citations-row div[data-testid="stHorizontalBlock"] button,
.gg-cite-buttons button {
    background: var(--gg-accent-soft) !important;
    color: var(--gg-accent) !important;
    border: 1px solid transparent !important;
    border-radius: 999px !important;
    padding: 2px 10px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    min-height: 0 !important;
    line-height: 1.4 !important;
    transition: background 120ms ease, border-color 120ms ease;
}
.gg-cite-buttons button:hover {
    background: #d3eae5 !important;
    border-color: var(--gg-accent) !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #fafbfc;
    border-right: 1px solid var(--gg-border);
}
.gg-side-empty {
    color: var(--gg-muted);
    font-size: 0.9rem;
    line-height: 1.5;
    padding: 0.5rem 0;
}
.gg-side-card {
    background: #ffffff;
    border: 1px solid var(--gg-border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-top: 0.4rem;
}
.gg-side-eyebrow {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--gg-accent);
    font-weight: 700;
    margin-bottom: 0.35rem;
}
.gg-side-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--gg-ink);
    margin: 0 0 0.35rem 0;
    line-height: 1.35;
}
.gg-side-meta {
    font-size: 0.82rem;
    color: var(--gg-muted);
    margin-bottom: 0.65rem;
}
.gg-side-quote {
    font-size: 0.92rem;
    color: var(--gg-ink);
    line-height: 1.5;
    border-left: 3px solid var(--gg-accent);
    padding: 0.1rem 0 0.1rem 0.7rem;
    margin: 0.35rem 0 0.6rem 0;
}
.gg-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: 0.5rem;
}
.gg-chip {
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 999px;
    background: var(--gg-accent-soft);
    color: var(--gg-accent);
    font-weight: 600;
}
.gg-chip-neutral {
    background: #eef1f3;
    color: var(--gg-muted);
}

footer, [data-testid="stStatusWidget"] { display: none; }
</style>
"""


def _render_header() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="gg-header">
            <h1>GaineyGuidelines</h1>
            <span class="gg-tag">Research preview</span>
        </div>
        <p class="gg-subtitle">
            Ask a question about GI clinical guidelines (AGA, ACG, ASGE, AASLD).
            Every claim is grounded in a society guideline — click a citation to
            inspect the source passage in the sidebar.
        </p>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(citations_by_n: dict[int, Citation]) -> None:
    with st.sidebar:
        st.markdown(
            '<div class="gg-side-eyebrow">Citation</div>',
            unsafe_allow_html=True,
        )
        selected = st.session_state.get("selected_citation")
        if selected is None or selected not in citations_by_n:
            st.markdown(
                '<div class="gg-side-empty">'
                "Click any [n] citation in the answer to see the underlying "
                "guideline passage here."
                "</div>",
                unsafe_allow_html=True,
            )
            return

        c = citations_by_n[selected]
        pages = ""
        if c.page_start is not None:
            pages = (
                f"p. {c.page_start}"
                if c.page_end in (None, c.page_start)
                else f"pp. {c.page_start}–{c.page_end}"
            )
        meta_bits = [f"{c.society} • {c.year}"]
        if c.recommendation_id:
            meta_bits.append(c.recommendation_id)
        if pages:
            meta_bits.append(pages)

        chips_html = ""
        if c.grade_strength:
            chips_html += (
                f'<span class="gg-chip">Strength: {c.grade_strength}</span>'
            )
        if c.grade_evidence:
            chips_html += (
                f'<span class="gg-chip gg-chip-neutral">'
                f"Evidence: {c.grade_evidence}</span>"
            )

        st.markdown(
            f"""
            <div class="gg-side-card">
                <div class="gg-side-eyebrow">Reference [{c.n}]</div>
                <div class="gg-side-title">{c.title}</div>
                <div class="gg-side-meta">{" &nbsp;·&nbsp; ".join(meta_bits)}</div>
                <div class="gg-side-quote">{c.text}</div>
                <div class="gg-chip-row">{chips_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Clear selection", key="clear_citation"):
            st.session_state["selected_citation"] = None
            st.rerun()


def _render_answer(answer_text: str, citations: list[Citation]) -> None:
    """Render the answer with inline numbered citation buttons.

    Streamlit can't put real buttons inside Markdown, so we split the answer
    around [n] tokens and render text segments + citation buttons in sequence
    using `st.columns` so they sit on the same line.
    """
    by_n = {c.n: c for c in citations}
    parts = re.split(r"(\[\d+\])", answer_text)

    st.markdown(
        '<div class="gg-answer-label">Answer</div>'
        '<div class="gg-answer">',
        unsafe_allow_html=True,
    )

    # Render the prose. We render text as Markdown and citation tokens as
    # small buttons inline-ish using a flex container we control with CSS.
    # Streamlit doesn't easily support inline buttons in markdown, so we
    # display the prose with citation tokens kept as visible [n] markers,
    # then offer a clickable row of citation chips immediately below.
    rendered = "".join(parts)
    st.markdown(rendered, unsafe_allow_html=False)
    st.markdown("</div>", unsafe_allow_html=True)

    if citations:
        st.markdown(
            '<div class="gg-cite-buttons" '
            'style="margin-top:0.85rem;display:flex;flex-wrap:wrap;gap:0.4rem;">',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(citations))
        for col, c in zip(cols, citations):
            with col:
                if st.button(
                    f"[{c.n}] {c.short_label}",
                    key=f"cite-{c.n}",
                    use_container_width=True,
                ):
                    st.session_state["selected_citation"] = c.n
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    if "selected_citation" not in st.session_state:
        st.session_state["selected_citation"] = None
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    _render_header()

    with st.form("ask", clear_on_submit=False):
        question = st.text_area(
            "Your question",
            placeholder="e.g. What is the first-line workup for suspected gastroparesis?",
            height=90,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask GaineyGuidelines", type="primary")

    if submitted and question.strip():
        answer_text, citations = answer(question)
        st.session_state["last_result"] = (answer_text, citations)
        st.session_state["selected_citation"] = None

    result = st.session_state["last_result"]
    if result:
        answer_text, citations = result
        _render_answer(answer_text, citations)
        _render_sidebar({c.n: c for c in citations})
    else:
        _render_sidebar({})


if __name__ == "__main__":
    main()
