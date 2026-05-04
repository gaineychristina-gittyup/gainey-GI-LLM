"""Streamlit clinician UI for the GI Guidelines RAG.

Run with::

    streamlit run src/ui/app.py

Talks to the FastAPI backend at ``API_BASE`` (default http://localhost:8000).
The two share the same DB; the UI never queries Postgres directly.

Layout:
    Sidebar (left)    — retrieval filters, generation knobs, conversation
                        history, backend health.
    Main pane (left)  — current question form + streaming answer with
                        clickable [N] badges + a citation list with
                        "View source" buttons per citation.
    Source pane (right) — the *source viewer*: when a citation is selected,
                          shows the cited chunk's full content rendered
                          appropriately (prose, HTML table, figure image)
                          plus a deep-link to the source PDF.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
import streamlit as st

API_BASE = os.environ.get("GI_API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="GI Guidelines RAG", layout="wide",
    initial_sidebar_state="expanded",
)


# --- session state ---------------------------------------------------------


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("conversation_id", None)
    ss.setdefault("turns", [])  # list of {question, answer, citations, ...}
    ss.setdefault("filter_society", [])
    ss.setdefault("filter_year_range", (2019, 2026))
    ss.setdefault("filter_element_types", [])
    ss.setdefault("top_k", 6)
    ss.setdefault("rerank", True)
    # Source-viewer state — (turn_index, citation_n) tuple, or None
    ss.setdefault("selected_citation", None)


_init_state()


# --- helpers ---------------------------------------------------------------


_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _build_filters() -> dict[str, Any]:
    f: dict[str, Any] = {}
    if st.session_state.filter_society:
        f["society"] = st.session_state.filter_society
    yr_min, yr_max = st.session_state.filter_year_range
    if yr_min > 2019:
        f["year_min"] = yr_min
    if yr_max < 2026:
        f["year_max"] = yr_max
    if st.session_state.filter_element_types:
        f["element_types"] = st.session_state.filter_element_types
    return f


def _ensure_conversation() -> int:
    if st.session_state.conversation_id is not None:
        return st.session_state.conversation_id
    r = requests.post(f"{API_BASE}/conversations", json={}, timeout=10)
    r.raise_for_status()
    cid = r.json()["id"]
    st.session_state.conversation_id = cid
    return cid


def _highlight_citations_html(text: str) -> str:
    """Render [N] tags as small badges. Wrap newlines as <br/>."""
    def repl(m: re.Match) -> str:
        nums = [n.strip() for n in m.group(1).split(",")]
        badges = " ".join(
            f'<span style="background:#e3f2fd;color:#0b69c7;padding:1px 6px;'
            f'border-radius:10px;font-size:0.85em;border:1px solid #cfe1f5;'
            f'margin:0 2px;">[{n}]</span>'
            for n in nums
        )
        return badges
    out = _CITATION_RE.sub(repl, text)
    return out.replace("\n", "<br/>")


def _post_stream(query: str, filters: dict, top_k: int, rerank: bool, conv_id: int):
    """Yield SSE events from /answer with stream=true."""
    payload = {
        "query": query,
        "filters": filters or None,
        "top_k": top_k,
        "rerank": rerank,
        "stream": True,
        "conversation_id": conv_id,
        "save": True,
    }
    with requests.post(
        f"{API_BASE}/answer", json=payload, stream=True, timeout=300,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                line = line[len("data: "):]
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _select_citation(turn_idx: int, n: int) -> None:
    st.session_state.selected_citation = (turn_idx, n)


def _clear_citation() -> None:
    st.session_state.selected_citation = None


def _find_citation(turn_idx: int, n: int) -> dict[str, Any] | None:
    if turn_idx < 0 or turn_idx >= len(st.session_state.turns):
        return None
    citations = st.session_state.turns[turn_idx].get("citations") or []
    for c in citations:
        if c.get("n") == n:
            return c
    return None


def _hydrate_citation(c: dict[str, Any]) -> dict[str, Any]:
    """If a citation lacks the rich source fields (e.g. it came from the
    history endpoint which only persisted the trimmed shape), fetch the
    chunk metadata to fill them in."""
    if c.get("text") and (c.get("table_html") is not None or c.get("element_type") != "table"):
        return c
    cid = c.get("chunk_id")
    if not cid:
        return c
    try:
        meta = requests.get(f"{API_BASE}/sources/chunk/{cid}", timeout=5).json()
    except requests.RequestException:
        return c
    enriched = dict(c)
    for k in ("text", "table_html", "figure_image_path", "document_id",
              "section_title", "doi", "source_url"):
        if not enriched.get(k) and meta.get(k):
            enriched[k] = meta[k]
    return enriched


# --- sidebar (filters + history) -------------------------------------------


with st.sidebar:
    st.header("Retrieval filters")
    society = st.multiselect(
        "Society", options=["AGA", "ACG", "ASGE", "AASLD"],
        default=st.session_state.filter_society, key="filter_society",
    )
    yr = st.slider(
        "Year range", min_value=2019, max_value=2026,
        value=st.session_state.filter_year_range, key="filter_year_range",
    )
    et = st.multiselect(
        "Element types",
        options=["prose", "recommendation", "table", "figure_caption", "key_concept"],
        default=st.session_state.filter_element_types,
        key="filter_element_types",
    )

    st.divider()
    st.header("Generation")
    top_k = st.slider("top_k", 3, 12, st.session_state.top_k, key="top_k")
    rerank = st.checkbox(
        "Cohere rerank + floors", value=st.session_state.rerank, key="rerank",
    )

    st.divider()
    st.header("Conversation")
    if st.button("New conversation", type="secondary"):
        st.session_state.conversation_id = None
        st.session_state.turns = []
        st.session_state.selected_citation = None
        st.rerun()

    # Health badge
    try:
        h = requests.get(f"{API_BASE}/health", timeout=2).json()
        ok = h.get("db_ok") and h.get("anthropic_key_set")
        chip = "🟢" if ok else "🔴"
        st.caption(f"{chip} backend · chunks={h.get('chunk_count', '?')}")
    except Exception:
        st.caption("🔴 backend unreachable")

    # History list
    try:
        convs = requests.get(f"{API_BASE}/conversations", timeout=5).json().get(
            "conversations", []
        )
    except Exception:
        convs = []
    if convs:
        st.caption("Recent")
        for c in convs[:15]:
            label = (c.get("title") or f"#{c['id']}")[:60]
            if st.button(f"#{c['id']} {label}", key=f"hist_{c['id']}",
                         use_container_width=True):
                conv = requests.get(
                    f"{API_BASE}/conversations/{c['id']}", timeout=5,
                ).json()
                st.session_state.conversation_id = conv["meta"]["id"]
                st.session_state.turns = conv["turns"]
                st.session_state.selected_citation = None
                st.rerun()


# --- main layout: answer pane | source pane --------------------------------


st.title("GI Guidelines RAG")
st.caption(
    "Ask questions grounded in 80 society guidelines (AGA / ACG / ASGE / AASLD). "
    "Every answer cites its sources; off-corpus questions are refused. Click "
    "a citation's **View source** button to see the underlying passage."
)

answer_col, source_col = st.columns([3, 2], gap="medium")


# ---- answer pane ----------------------------------------------------------

with answer_col:
    # Render prior turns first so the new question lands at the bottom.
    for turn_idx, turn in enumerate(st.session_state.turns):
        with st.container(border=True):
            st.markdown(f"**Q:** {turn['question']}")
            if turn.get("refused"):
                st.warning("Off-corpus refusal")
            st.markdown(
                _highlight_citations_html(turn.get("answer", "")),
                unsafe_allow_html=True,
            )
            cits = turn.get("citations") or []
            if cits:
                st.caption(f"Citations ({len(cits)})")
                for c in cits:
                    n = c.get("n", "?")
                    bits = [f"**[{n}]**", f"{c.get('society') or '?'} {c.get('year') or '?'}"]
                    if c.get("recommendation_id"):
                        bits.append(c["recommendation_id"])
                    if (gs := c.get("grade_strength")) or (ge := c.get("grade_evidence")):
                        grade = " / ".join(x for x in [gs, ge] if x)
                        bits.append(f"GRADE: {grade}")
                    if (et := c.get("element_type")) and et != "prose":
                        bits.append(f"`{et}`")
                    page = ""
                    if c.get("page_start") and c.get("page_end") and c["page_start"] != c["page_end"]:
                        page = f"pp {c['page_start']}-{c['page_end']}"
                    elif c.get("page_start"):
                        page = f"p {c['page_start']}"
                    if page:
                        bits.append(page)
                    line_cols = st.columns([5, 1])
                    with line_cols[0]:
                        st.markdown(" · ".join(bits))
                        title = (c.get("title") or "").strip()
                        if title:
                            st.caption(title[:120])
                    with line_cols[1]:
                        st.button(
                            "View source",
                            key=f"view_{turn_idx}_{n}",
                            on_click=_select_citation,
                            args=(turn_idx, n),
                            type="secondary",
                            use_container_width=True,
                        )

    # Question form lands at the bottom of the answer column.
    with st.form(key="ask_form", clear_on_submit=False):
        q = st.text_area(
            "Question", height=100,
            placeholder="e.g. What's the recommended H. pylori regimen for a "
                        "penicillin-allergic patient?",
            key="question_input",
        )
        submitted = st.form_submit_button("Ask", type="primary")


# ---- source pane (right) --------------------------------------------------


def _render_source_pane() -> None:
    sel = st.session_state.selected_citation
    if sel is None:
        st.info(
            "Click **View source** on any citation to see the cited passage "
            "here — full text for prose, structured HTML for tables, and the "
            "extracted figure image for figure captions."
        )
        return

    turn_idx, n = sel
    c = _find_citation(turn_idx, n)
    if c is None:
        st.warning(f"Citation [{n}] not found.")
        st.button("Close", on_click=_clear_citation)
        return

    c = _hydrate_citation(c)

    header_cols = st.columns([5, 1])
    with header_cols[0]:
        st.subheader(f"Source [{n}]")
    with header_cols[1]:
        st.button("✕", on_click=_clear_citation, key=f"close_{turn_idx}_{n}")

    # Citation metadata block
    meta_lines = []
    soc_year = f"**{c.get('society') or '?'} {c.get('year') or '?'}**"
    meta_lines.append(soc_year)
    title = (c.get("title") or "").strip()
    if title:
        meta_lines.append(title)
    detail_bits = []
    if c.get("recommendation_id"):
        detail_bits.append(c["recommendation_id"])
    if (gs := c.get("grade_strength")) or (ge := c.get("grade_evidence")):
        grade = " / ".join(x for x in [gs, ge] if x)
        detail_bits.append(f"GRADE: {grade}")
    page = ""
    if c.get("page_start") and c.get("page_end") and c["page_start"] != c["page_end"]:
        page = f"pp {c['page_start']}-{c['page_end']}"
    elif c.get("page_start"):
        page = f"p {c['page_start']}"
    if page:
        detail_bits.append(page)
    et = c.get("element_type") or "prose"
    detail_bits.append(f"`{et}`")
    if c.get("section_title"):
        detail_bits.append(f"§ {c['section_title'][:60]}")
    meta_lines.append(" · ".join(detail_bits))
    st.markdown("  \n".join(meta_lines))

    # Source-content rendering, dispatched by element_type
    st.divider()
    if et == "table" and c.get("table_html"):
        st.markdown(c["table_html"], unsafe_allow_html=True)
        with st.expander("Plain-text rendering (what the embedder saw)"):
            st.text(c.get("text") or "")
    elif et == "figure_caption" and c.get("figure_image_path"):
        # The path is repo-relative; strip the data/parsed/figures/ prefix
        # before handing to the API.
        rel = c["figure_image_path"]
        if rel.startswith("data/parsed/figures/"):
            rel = rel[len("data/parsed/figures/"):]
        img_url = f"{API_BASE}/sources/figure/{rel}"
        st.image(img_url, caption=c.get("text") or "(no caption)")
        st.caption(f"Image saved to `{c['figure_image_path']}`")
    else:
        # prose / recommendation / key_concept — render full text
        text = c.get("text") or "(no text in chunk)"
        st.markdown(text)

    # Always show a deep-link to the source PDF when we know which doc.
    st.divider()
    doc_id = c.get("document_id")
    if doc_id is not None:
        url = f"{API_BASE}/sources/document/{doc_id}/pdf"
        st.link_button("📄 Open source PDF", url, use_container_width=True)
    elif c.get("source_url"):
        st.link_button("🌐 Open source URL", c["source_url"], use_container_width=True)
    else:
        st.caption("Source PDF unavailable.")
    if c.get("doi"):
        st.caption(f"DOI: {c['doi']}")


with source_col:
    _render_source_pane()


# --- ask handler -----------------------------------------------------------


if submitted and q.strip():
    conv_id = _ensure_conversation()
    filters = _build_filters()
    with answer_col:
        answer_box = st.empty()
    full = ""
    final: dict[str, Any] | None = None
    err: str | None = None

    with st.spinner("Retrieving and generating..."):
        for ev in _post_stream(q.strip(), filters, st.session_state.top_k,
                                st.session_state.rerank, conv_id):
            if ev["type"] == "token":
                full += ev["text"]
                with answer_col:
                    answer_box.markdown(
                        _highlight_citations_html(full), unsafe_allow_html=True,
                    )
            elif ev["type"] == "done":
                final = ev["result"]
            elif ev["type"] == "error":
                err = ev["message"]
                break

    if err:
        st.error(f"Generation failed: {err}")
    elif final is not None:
        st.session_state.turns.append({
            "question": q.strip(),
            "answer": final.get("answer", ""),
            "citations": final.get("citations") or [],
            "chunks": final.get("chunks") or [],
            "refused": final.get("refused"),
            "verification": final.get("verification") or {},
            "usage": final.get("usage") or {},
        })
        st.rerun()
