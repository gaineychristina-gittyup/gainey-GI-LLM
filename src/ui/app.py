"""Streamlit clinician UI for the GI Guidelines RAG.

Run with::

    streamlit run src/ui/app.py

Talks to the FastAPI backend at ``API_BASE`` (default http://localhost:8000).
The two share the same DB; the UI never queries Postgres directly.

Layout:
    Left column  — filters (society, year, element type, top_k), conversation
                   history sidebar.
    Right column — current question + streaming answer + citation cards with
                   source-PDF links.
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
    ss.setdefault("pending_query", None)
    ss.setdefault("filter_society", [])
    ss.setdefault("filter_year_range", (2019, 2026))
    ss.setdefault("filter_element_types", [])
    ss.setdefault("top_k", 6)
    ss.setdefault("rerank", True)


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
    """Render [N] tags as small badges. Wrap newlines as <br/>.
    The badges are HTML anchors (#cite-N) so clicking scrolls to the card."""
    def repl(m: re.Match) -> str:
        nums = [n.strip() for n in m.group(1).split(",")]
        badges = " ".join(
            f'<a href="#cite-{n}" '
            f'style="background:#e3f2fd;color:#0b69c7;padding:1px 6px;'
            f'border-radius:10px;text-decoration:none;font-size:0.85em;'
            f'border:1px solid #cfe1f5;margin:0 2px;">[{n}]</a>'
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


def _render_citation_card(c: dict[str, Any]) -> None:
    """Inside an expander already; renders one citation card."""
    bits = []
    if c.get("recommendation_id"):
        bits.append(f"**{c['recommendation_id']}**")
    if c.get("grade_strength") or c.get("grade_evidence"):
        grade = " / ".join(
            x for x in [c.get("grade_strength"), c.get("grade_evidence")] if x
        )
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
    if bits:
        st.markdown(" · ".join(bits))

    title = (c.get("title") or "").strip()
    if title:
        st.caption(title)

    # Show full chunk text (citation-card readers read it).
    full_chunk = _full_chunk_text(c)
    if full_chunk:
        st.text_area(
            "chunk text", value=full_chunk, height=160,
            key=f"chunk_text_{c.get('n')}_{c.get('chunk_id', '')}",
            label_visibility="collapsed",
        )

    # PDF link uses document_id when we have it.
    doc_id = c.get("document_id")
    if doc_id is None:
        # Fallback: fetch via chunk metadata endpoint.
        cid = c.get("chunk_id")
        if cid:
            try:
                meta = requests.get(
                    f"{API_BASE}/sources/chunk/{cid}", timeout=5,
                ).json()
                doc_id = meta.get("document_id")
                # Also enrich the card with chunk text if we didn't have it.
                if not full_chunk and meta.get("text"):
                    st.text_area(
                        "chunk text", value=meta["text"], height=160,
                        key=f"chunk_text_meta_{cid}", label_visibility="collapsed",
                    )
            except requests.RequestException:
                pass
    if doc_id is not None:
        url = f"{API_BASE}/sources/document/{doc_id}/pdf"
        st.markdown(f"[Open source PDF]({url})")


def _full_chunk_text(c: dict[str, Any]) -> str:
    """Citations are sometimes the abbreviated record; only chunk dicts in
    the SSE done-event carry the full text."""
    return c.get("text") or ""


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
                st.rerun()


# --- main pane -------------------------------------------------------------


st.title("GI Guidelines RAG")
st.caption(
    "Ask questions grounded in 80 society guidelines (AGA / ACG / ASGE / AASLD). "
    "Every answer cites its sources; off-corpus questions are refused."
)

# Render prior turns first so the new question lands at the bottom.
for turn in st.session_state.turns:
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
            with st.expander(f"Citations ({len(cits)})"):
                for c in cits:
                    n = c.get("n", "?")
                    st.markdown(f"<a id='cite-{n}'></a> [{n}]",
                                unsafe_allow_html=True)
                    _render_citation_card(c)
                    st.divider()


with st.form(key="ask_form", clear_on_submit=False):
    q = st.text_area(
        "Question", height=100,
        placeholder="e.g. What's the recommended H. pylori regimen for a "
                    "penicillin-allergic patient?",
        key="question_input",
    )
    submitted = st.form_submit_button("Ask", type="primary")


if submitted and q.strip():
    conv_id = _ensure_conversation()
    filters = _build_filters()
    answer_box = st.empty()
    full = ""
    final: dict[str, Any] | None = None
    err: str | None = None

    with st.spinner("Retrieving and generating..."):
        for ev in _post_stream(q.strip(), filters, st.session_state.top_k,
                                st.session_state.rerank, conv_id):
            if ev["type"] == "token":
                full += ev["text"]
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
