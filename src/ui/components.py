"""Reusable rendering components for the Phase-5 Streamlit UI.

Keeping these out of ``streamlit_app.py`` so the main app file reads as a
layout/orchestration file, not a wall of HTML.
"""

from __future__ import annotations

import html as html_module
import os
import re
from pathlib import Path
from typing import Any

import streamlit as st

# Sentence-ish terminators used to slice the answer for the
# "Cited claim" preview. The verifier's regex only catches
# `period + space + capital letter`, but the model frequently
# answers in bulleted lists where items begin with `\n- ` and the
# bullet text doesn't start capital. Splitting there keeps each
# claim short and focused, so the highlight match has a chance.
_SENTENCE_END_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z(\"])"  # ordinary terminator
    r"|"
    r"\n\s*(?=[-*•]\s)"            # newline → bullet
    r"|"
    r"\n{2,}",                     # blank line (paragraph break)
)

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


def inject_global_styles() -> None:
    """Inject the GaineyGuidelines visual theme.

    Streamlit's default chrome is functional but visually noisy (loud
    primary color, default sans-serif, hard borders). A small CSS layer
    quiets it down: a refined font stack, softer surfaces, a calmer
    palette built around slate + a single teal accent, and rounded
    cards. Idempotent — calling it multiple times is harmless.
    """
    st.markdown(
        """
        <style>
        :root {
            --gg-bg: #fafbfc;
            --gg-surface: #ffffff;
            --gg-border: #e5e9ef;
            --gg-text: #1f2937;
            --gg-muted: #5b6772;
            --gg-accent: #0f766e;
            --gg-accent-soft: #e6f4f1;
            --gg-cite-bg: #eef4ff;
            --gg-cite-border: #c7d8f2;
            --gg-cite-text: #1d4ed8;
        }
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Inter",
                "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }
        .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }
        h1, h2, h3 { color: var(--gg-text); letter-spacing: -0.01em; }
        h1 { font-weight: 700; }
        .stButton > button[kind="primary"] {
            background: var(--gg-accent);
            border: 1px solid var(--gg-accent);
            border-radius: 8px;
            font-weight: 600;
        }
        .stButton > button[kind="primary"]:hover {
            background: #0b5e58; border-color: #0b5e58;
        }
        .stButton > button:not([kind="primary"]) {
            border-radius: 8px;
            border: 1px solid var(--gg-border);
            background: var(--gg-surface);
            color: var(--gg-text);
            font-size: 0.85rem;
        }
        .stButton > button:not([kind="primary"]):hover {
            border-color: var(--gg-accent);
            color: var(--gg-accent);
        }
        section[data-testid="stSidebar"] {
            background: #f5f7fa;
            border-right: 1px solid var(--gg-border);
        }
        /* Repaint Streamlit's red primary chrome (multiselect tags,
           slider track/thumb, spinner) in our teal accent so the page
           reads as one coherent palette. */
        [data-baseweb="tag"] {
            background-color: var(--gg-accent) !important;
        }
        [data-baseweb="slider"] [role="slider"] {
            background-color: var(--gg-accent) !important;
            border-color: var(--gg-accent) !important;
        }
        [data-baseweb="slider"] div[style*="rgb(255"] {
            background-color: var(--gg-accent) !important;
        }
        .stSlider [data-testid="stTickBarMin"],
        .stSlider [data-testid="stTickBarMax"] { color: var(--gg-muted); }
        /* Form text input borders */
        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color: var(--gg-accent) !important;
            box-shadow: 0 0 0 1px var(--gg-accent) !important;
        }
        .gg-brand {
            display: flex; align-items: center; gap: 0.6rem;
            margin-bottom: 0.25rem;
        }
        .gg-logo {
            display: inline-flex; align-items: center; justify-content: center;
            width: 38px; height: 38px; border-radius: 9px;
            background: linear-gradient(135deg, #0f766e 0%, #0ea5a4 100%);
            color: white; font-weight: 700; font-size: 1rem;
            letter-spacing: -0.02em;
        }
        .gg-title {
            font-size: 1.7rem; font-weight: 700; line-height: 1.1;
            color: var(--gg-text);
        }
        .gg-subtitle {
            color: var(--gg-muted); font-size: 0.95rem; margin-top: 0.1rem;
        }
        .gg-meta {
            font-size: 0.8rem; color: var(--gg-muted); margin: 0.5rem 0 0.75rem;
        }
        .gg-disclaimer {
            background: var(--gg-accent-soft);
            color: #114b46;
            border: 1px solid #c7e3df;
            border-radius: 8px;
            padding: 0.55rem 0.9rem;
            font-size: 0.85rem;
            margin: 0.5rem 0 1rem;
        }
        .gg-pane {
            background: var(--gg-surface);
            border: 1px solid var(--gg-border);
            border-radius: 10px;
            padding: 1rem 1.1rem;
            margin-top: 0.5rem;
        }
        .gg-pane-empty {
            color: var(--gg-muted); font-size: 0.9rem;
            text-align: center; padding: 2rem 0.5rem;
        }
        .gg-pane h4 {
            margin: 0 0 0.4rem; font-size: 1rem; color: var(--gg-text);
        }
        .gg-cite-link {
            background: var(--gg-cite-bg);
            color: var(--gg-cite-text);
            padding: 1px 8px;
            border-radius: 10px;
            text-decoration: none;
            font-size: 0.85em;
            border: 1px solid var(--gg-cite-border);
            margin: 0 2px;
            font-weight: 600;
        }
        .gg-cite-link:hover {
            background: #dde9fb;
            text-decoration: none;
        }
        .gg-card {
            border: 1px solid var(--gg-border);
            border-radius: 10px;
            background: var(--gg-surface);
            padding: 0.7rem 0.9rem;
            margin-top: 0.5rem;
            transition: border-color 0.15s ease;
        }
        .gg-card.cited {
            border-left: 4px solid var(--gg-accent);
            background: var(--gg-accent-soft);
        }
        .gg-card-head { font-size: 0.95rem; color: var(--gg-text); }
        .gg-card-title {
            font-size: 0.85rem; color: var(--gg-muted); margin-top: 0.25rem;
        }
        .gg-card-grade {
            font-size: 0.8rem; color: var(--gg-accent); margin-top: 0.25rem;
            font-weight: 600;
        }

        /* Source-pane "show only the clicked citation" — pure CSS via
           the :target pseudo-class. The badge link `#cite-N` makes the
           div with id=`cite-N` the URL target; siblings stay hidden.
           :has() flips the empty-state off when any cite is targeted. */
        .gg-source-stack { margin-top: 0.5rem; position: relative; }
        .gg-empty-state {
            background: var(--gg-surface);
            border: 1px solid var(--gg-border);
            border-radius: 10px;
            padding: 2.2rem 1rem;
            text-align: center;
            color: var(--gg-muted);
            font-size: 0.9rem;
        }
        .gg-cite-target {
            display: none;
            background: var(--gg-surface);
            border: 1px solid var(--gg-border);
            border-radius: 10px;
            padding: 1rem 1.1rem;
            line-height: 1.55;
        }
        .gg-cite-target:target { display: block; }
        .gg-source-stack:has(.gg-cite-target:target) .gg-empty-state {
            display: none;
        }
        .gg-cite-card-head {
            font-size: 1rem; color: var(--gg-text); font-weight: 500;
        }
        .gg-cite-card-head b { font-weight: 700; }
        .gg-cite-card-meta {
            font-size: 0.85rem; color: var(--gg-muted);
            margin-top: 0.25rem;
        }
        .gg-cite-card-grade {
            font-size: 0.8rem; color: var(--gg-accent);
            margin-top: 0.25rem; font-weight: 600;
        }
        .gg-claim-box {
            background: var(--gg-accent-soft);
            border-left: 3px solid var(--gg-accent);
            border-radius: 6px;
            padding: 0.55rem 0.75rem;
            margin: 0.7rem 0 0.5rem;
        }
        .gg-claim-label {
            font-size: 0.7rem; color: var(--gg-accent);
            font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.05em; margin-bottom: 0.2rem;
        }
        .gg-claim-text {
            font-size: 0.88rem; color: var(--gg-text);
        }
        .gg-claim-list {
            margin: 0.2rem 0 0 1rem; padding: 0;
            font-size: 0.88rem; color: var(--gg-text);
        }
        .gg-claim-list li { margin-bottom: 0.2rem; }
        .gg-source-text {
            margin-top: 0.6rem; font-size: 0.9rem; color: var(--gg-text);
            white-space: pre-wrap;
        }
        .gg-source-text mark.gg-highlight {
            background: #fff7c2;
            color: var(--gg-text);
            padding: 0 2px;
            border-radius: 2px;
            box-decoration-break: clone;
        }
        .gg-source-text table {
            border-collapse: collapse; font-size: 0.83rem;
            margin: 0.4rem 0; width: 100%;
        }
        .gg-source-text th, .gg-source-text td {
            border: 1px solid var(--gg-border);
            padding: 0.3rem 0.5rem; vertical-align: top;
        }
        .gg-source-figure { margin: 0.5rem 0; }
        .gg-source-figure img {
            max-width: 100%; height: auto; border-radius: 6px;
            border: 1px solid var(--gg-border);
        }
        .gg-source-figure figcaption {
            font-size: 0.85rem; color: var(--gg-muted); margin-top: 0.3rem;
        }
        .gg-pdf-link {
            display: inline-block;
            margin-top: 0.9rem;
            padding: 0.4rem 0.8rem;
            border: 1px solid var(--gg-border);
            border-radius: 6px;
            text-decoration: none;
            color: var(--gg-accent);
            font-size: 0.85rem;
            font-weight: 600;
            background: var(--gg-surface);
        }
        .gg-pdf-link:hover {
            background: var(--gg-accent-soft);
            border-color: var(--gg-accent);
            text-decoration: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_corpus_snapshot_header(title: str, subtitle: str) -> None:
    """Page header — brand, title, subtitle, snapshot date, disclaimer."""
    inject_global_styles()
    snapshot = read_corpus_snapshot()
    initials = "".join(w[0] for w in title.split()[:2]).upper() or "GG"
    st.markdown(
        f"""
        <div class="gg-brand">
            <span class="gg-logo">{initials}</span>
            <div>
                <div class="gg-title">{title}</div>
                <div class="gg-subtitle">{subtitle}</div>
            </div>
        </div>
        <div class="gg-meta">Corpus last updated: <b>{snapshot}</b></div>
        <div class="gg-disclaimer">
            For research and reference. Not a substitute for clinical judgment.
        </div>
        """,
        unsafe_allow_html=True,
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


# --- inline citations in answer text ----------------------------------------


def render_inline_citations(answer_text: str) -> str:
    """Render [N] tags in the answer as clickable badges.

    Each badge points at ``#cite-N`` — the id of the corresponding
    focused-citation div in the source pane. Clicking switches the
    pane via the CSS ``:target`` selector, so only the cited source
    is visible at a time; no Streamlit rerun is involved.

    Newlines pass through unchanged so Markdown lists and pipe tables
    render correctly. An earlier version replaced "\\n" with "<br/>"
    which silently flattened bullet lists and table separators — the
    system prompt now instructs Claude to emit Markdown, so preserving
    the structure here is load-bearing.
    """
    def repl(m: re.Match) -> str:
        nums = [n.strip() for n in m.group(1).split(",")]
        badges = [
            f'<a class="gg-cite-link" href="#cite-{n}">[{n}]</a>'
            for n in nums
        ]
        return " ".join(badges)
    return _CITATION_RE.sub(repl, answer_text)


def format_citation(c: dict[str, Any]) -> str:
    """Compact one-line citation header — society, year, lead author,
    recommendation ID, and page range. Used by tests and any caller
    that wants a short single-line label for a chunk."""
    bits = [f"{c.get('society') or '?'} {c.get('year') or '?'}"]
    author = derive_lead_author(c.get("pdf_path") or "")
    if not author and c.get("title"):
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


def _build_citing_sentences(answer_text: str) -> dict[int, list[str]]:
    """Return ``{chunk_index: [sentence_text, ...]}`` — for each chunk N,
    the answer sentences that include ``[N]``. The pane uses these as
    the "Cited claim" preview alongside the source text, so the user
    sees what the citation is supporting before reading the chunk.
    """
    citing: dict[int, list[str]] = {}
    for sentence in _SENTENCE_END_RE.split(answer_text):
        sentence = sentence.strip()
        if not sentence:
            continue
        for m in _CITATION_RE.finditer(sentence):
            for piece in m.group(1).split(","):
                try:
                    n = int(piece.strip())
                except ValueError:
                    continue
                citing.setdefault(n, []).append(sentence)
    return citing


def _highlight_span(
    claim: str, text: str, *, min_score: int = 50,
) -> tuple[int, int] | None:
    """Best-matching span of ``claim`` within ``text`` using rapidfuzz
    alignment. Returns ``(start, end)`` indices into ``text`` or
    ``None`` if no span clears ``min_score``.

    ``rapidfuzz.fuzz.partial_ratio_alignment`` gives back a
    ``ScoreAlignment(score, src_start, src_end, dest_start, dest_end)``
    where ``dest`` refers to the second positional argument. Matches
    the verifier's scoring intent — high partial_ratio means the claim
    is genuinely paraphrased from this region.
    """
    if not claim or not text:
        return None
    try:
        from rapidfuzz import fuzz
        # Normalize the claim before matching: drop [N] citation tags,
        # leading bullet markers (`- ` / `* ` / `• `), and Markdown
        # emphasis (`**bold**`, `*italic*`). The chunk text has none of
        # these, and they crater partial_ratio if left in.
        cleaned = _CITATION_RE.sub("", claim)
        cleaned = re.sub(r"^\s*[-*•]\s+", "", cleaned)
        cleaned = cleaned.replace("**", "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return None
        alignment = fuzz.partial_ratio_alignment(cleaned, text)
    except Exception:
        return None
    if alignment is None or alignment.score < min_score:
        return None
    if alignment.dest_start >= alignment.dest_end:
        return None
    return alignment.dest_start, alignment.dest_end


def _highlight_text(claims: list[str], text: str) -> str:
    """HTML-escape ``text`` and wrap each claim's matched span in a
    ``<mark>``. Overlapping spans are merged so the highlight reads as
    one contiguous yellow band rather than nested wrappers.
    """
    if not text:
        return ""
    spans: list[tuple[int, int]] = []
    for claim in claims:
        s = _highlight_span(claim, text)
        if s:
            spans.append(s)
    if not spans:
        return html_module.escape(text)
    spans.sort()
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out: list[str] = []
    last = 0
    for s, e in merged:
        out.append(html_module.escape(text[last:s]))
        out.append(
            f'<mark class="gg-highlight">'
            f'{html_module.escape(text[s:e])}</mark>'
        )
        last = e
    out.append(html_module.escape(text[last:]))
    return "".join(out)


def _render_chunk_focused_html(
    chunk: dict[str, Any],
    rank: int,
    claims: list[str],
    api_base: str,
) -> str:
    """One focused-citation card, returned as a single HTML string.

    Emits the cited claim(s) at the top, then the source text with
    fuzzy-matched spans wrapped in ``<mark>``. The wrapper div has
    ``id="cite-N"`` so the CSS ``:target`` rule shows it only when
    the URL hash is ``#cite-N``.
    """
    soc = chunk.get("society") or "?"
    year = chunk.get("year") or "?"
    author = derive_lead_author(chunk.get("pdf_path") or "")
    rec_id = chunk.get("recommendation_id") or ""
    title = (chunk.get("title") or "").strip()
    grade = format_grade_label(
        chunk.get("grade_strength"), chunk.get("grade_evidence")
    )
    page = ""
    if (chunk.get("page_start") and chunk.get("page_end")
            and chunk["page_start"] != chunk["page_end"]):
        page = f"pp. {chunk['page_start']}-{chunk['page_end']}"
    elif chunk.get("page_start"):
        page = f"p. {chunk['page_start']}"

    head_bits = [f"<b>[{rank}] {html_module.escape(str(soc))} "
                 f"{html_module.escape(str(year))}</b>"]
    if author:
        head_bits.append(html_module.escape(author))
    if rec_id:
        head_bits.append(f"<code>{html_module.escape(rec_id)}</code>")
    if page:
        head_bits.append(html_module.escape(page))
    head = " · ".join(head_bits)

    title_html = (
        f'<div class="gg-cite-card-meta">{html_module.escape(title)}</div>'
        if title else ""
    )
    grade_html = (
        f'<div class="gg-cite-card-grade">'
        f'GRADE: {html_module.escape(grade)}</div>'
        if grade else ""
    )

    if not claims:
        claim_html = ""
    elif len(claims) == 1:
        claim_html = (
            f'<div class="gg-claim-box">'
            f'<div class="gg-claim-label">Cited claim</div>'
            f'<div class="gg-claim-text">'
            f'{html_module.escape(claims[0])}</div></div>'
        )
    else:
        items = "".join(
            f"<li>{html_module.escape(c)}</li>" for c in claims
        )
        claim_html = (
            f'<div class="gg-claim-box">'
            f'<div class="gg-claim-label">'
            f'Cited claims ({len(claims)})</div>'
            f'<ul class="gg-claim-list">{items}</ul></div>'
        )

    et = chunk.get("element_type") or "prose"
    if et == "table" and chunk.get("table_html"):
        # Table HTML is already structured; embed verbatim. We don't
        # try to highlight inside cell HTML because partial_ratio
        # against tag-soup gives noisy spans.
        body_html = (
            f'<div class="gg-source-text">{chunk["table_html"]}</div>'
        )
    elif et == "figure_caption" and chunk.get("figure_image_path"):
        rel = chunk["figure_image_path"]
        stripped = rel.removeprefix("data/parsed/figures/")
        img_url = f"{api_base}/sources/figure/{stripped}"
        caption = chunk.get("text") or ""
        body_html = (
            f'<figure class="gg-source-figure">'
            f'<img src="{html_module.escape(img_url)}" alt="figure" />'
            f'<figcaption>{_highlight_text(claims, caption)}</figcaption>'
            f'</figure>'
        )
    else:
        text = chunk.get("text") or ""
        body_html = (
            f'<div class="gg-source-text">'
            f'{_highlight_text(claims, text)}'
            f'</div>'
        )

    footer_html = ""
    doc_id = chunk.get("document_id")
    if doc_id is not None:
        url = f"{api_base}/sources/document/{doc_id}/pdf"
        footer_html = (
            f'<a class="gg-pdf-link" href="{html_module.escape(url)}" '
            f'target="_blank" rel="noopener">📄 Open source PDF</a>'
        )
    elif chunk.get("source_url"):
        footer_html = (
            f'<a class="gg-pdf-link" '
            f'href="{html_module.escape(chunk["source_url"])}" '
            f'target="_blank" rel="noopener">🌐 Open source URL</a>'
        )

    return (
        f'<div id="cite-{rank}" class="gg-cite-target">'
        f'<div class="gg-cite-card-head">{head}</div>'
        f'{title_html}{grade_html}{claim_html}{body_html}{footer_html}'
        f'</div>'
    )


def render_source_pane(
    chunks: list[dict[str, Any]],
    cited_indices: set[int],
    answer_text: str,
    elapsed: float | None,
    *,
    api_base: str = API_BASE,
) -> None:
    """Right-side pane: shows ONE focused source at a time.

    All chunks render as ``id="cite-N"`` divs that are hidden by
    default. Clicking ``[N]`` in the answer changes the URL hash to
    ``#cite-N`` and CSS ``:target`` reveals just that one. The cited
    answer-sentence is shown on top of each card as the "Cited
    claim", and the matching span inside the source text is wrapped
    in ``<mark>`` so the user sees exactly what supports the claim.
    """
    st.markdown("#### 📑 Source viewer")
    if not chunks:
        st.markdown(
            '<div class="gg-empty-state">'
            'Ask a question to see retrieved sources here. '
            'Each <span class="gg-cite-link" '
            'style="cursor:default;">[N]</span> in the answer will '
            'open the cited passage in this pane.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    n_cited = len(cited_indices)
    elapsed_str = f" · answered in {elapsed:.1f}s" if elapsed else ""
    st.caption(
        f"{len(chunks)} passages retrieved · {n_cited} cited{elapsed_str}. "
        f"Click any [N] in the answer to view the cited source."
    )

    citing_map = _build_citing_sentences(answer_text or "")

    parts = ['<div class="gg-source-stack">']
    parts.append(
        '<div class="gg-empty-state">'
        'Click a <span class="gg-cite-link" style="cursor:default;">'
        '[N]</span> citation in the answer to view the cited source.'
        '</div>'
    )
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            _render_chunk_focused_html(
                chunk, rank=i, claims=citing_map.get(i, []),
                api_base=api_base,
            )
        )
    parts.append('</div>')
    st.markdown("\n".join(parts), unsafe_allow_html=True)


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


