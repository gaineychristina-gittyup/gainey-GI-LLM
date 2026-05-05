"""GI Guidelines Assistant — Phase 5 clinician UI.

Run with::

    bash scripts/run_ui.sh
    # or directly:
    streamlit run src/ui/streamlit_app.py --server.port 8501

Backend lives at ``GI_API_BASE`` (default http://localhost:8000), brought
up separately by ``uvicorn src.api.app:app``.

Layout (top to bottom):
    Header              — title + subtitle + corpus snapshot date + disclaimer
    Sidebar (filters)   — society / year / topic / doc-type / top_k / debug
    Question form       — text input + Ask + 4-5 example-question buttons
    Answer area         — answer text with clickable [N] citations,
                          verification badge, refusal banner if applicable
    Passage cards       — one per retrieved chunk; cited cards highlighted
                          with a left-border. Tables render inline as HTML;
                          figures render as images.
    Footer              — provenance + snapshot date + UNC attribution

Each query is independent — no conversation memory in Phase 5. The Q&A is
appended to logs/qa_log.jsonl for offline review.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


# set_page_config MUST be the first Streamlit call, once per session.
# When using st.navigation, calling set_page_config from inside a page
# render function fails with StreamlitSetPageConfigMustBeFirstCommandError
# because the navigation chrome has already rendered. So we set it here,
# at module top, before any other Streamlit work.
st.set_page_config(
    page_title="GI Guidelines Assistant", layout="wide",
    initial_sidebar_state="expanded",
)


# Bring repo root onto sys.path so `from src.ui.* import` works whether
# Streamlit is launched from the repo root or anywhere else.
import sys
sys.path.insert(0, str(REPO_ROOT))

from src.ui import about_page
from src.ui.components import (  # noqa: E402
    derive_doc_type,
    format_grade_label,
    read_corpus_snapshot,
    render_citation_badge,
    render_corpus_snapshot_header,
    render_footer,
    render_inline_citations,
    render_passage_card,
)
from src.ui.phi import phi_warning_message, screen_for_phi  # noqa: E402

API_BASE = os.environ.get("GI_API_BASE", "http://localhost:8000")
CONFIG_PATH = REPO_ROOT / "config.yaml"
QA_LOG_PATH = REPO_ROOT / "logs" / "qa_log.jsonl"
QA_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# --- config ---------------------------------------------------------------


@st.cache_resource
def load_ui_config() -> dict[str, Any]:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("ui") or {}


# --- topic list ------------------------------------------------------------


@st.cache_data(ttl=300)
def fetch_topics() -> list[str]:
    """Pull the distinct topic slugs from the documents table — used to
    populate the sidebar topic multiselect. Cached for 5 minutes."""
    import psycopg
    url = os.environ.get(
        "DATABASE_URL", "postgresql://gi:gi@localhost:5432/gi_guidelines"
    )
    try:
        with psycopg.connect(url, connect_timeout=2) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT topic FROM documents WHERE topic IS NOT NULL "
                "ORDER BY topic"
            )
            return [row[0] for row in cur.fetchall()]
    except Exception:
        return []


# --- backend calls ---------------------------------------------------------


def call_answer(
    query: str,
    filters: dict[str, Any],
    top_k: int,
    *,
    history: list[dict[str, str]] | None = None,
    conversation_id: int | None = None,
) -> dict[str, Any] | None:
    """POST /answer (non-streaming). Returns the answer dict or None on
    failure, with the failure already surfaced to the user."""
    payload = {
        "query": query,
        "filters": filters or None,
        "top_k": top_k,
        "rerank": True,
        "stream": False,
        "save": conversation_id is not None,
        "conversation_id": conversation_id,
        "history": history or None,
    }
    try:
        r = requests.post(f"{API_BASE}/answer", json=payload, timeout=300)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        st.error(
            f"The backend returned an error ({e.response.status_code}). "
            "If this persists, check that the FastAPI process is healthy "
            f"at {API_BASE}/health."
        )
        return None
    except requests.RequestException as e:
        st.error(
            f"Could not reach the backend at {API_BASE}. Is `uvicorn "
            f"src.api.app:app` running? ({type(e).__name__})"
        )
        return None


def create_conversation(title: str | None = None) -> int | None:
    """POST /conversations to start a server-side conversation. Returns the
    new id, or None on failure (we degrade gracefully — the UI still works
    without a server-side row, just without persistence across reloads)."""
    try:
        r = requests.post(
            f"{API_BASE}/conversations",
            json={"title": title},
            timeout=10,
        )
        r.raise_for_status()
        return int(r.json().get("id"))
    except requests.RequestException:
        return None


def submit_feedback(payload: dict[str, Any]) -> dict[str, Any] | None:
    """POST /feedback. Returns the triage record on success, None on error
    (already surfaced via st.error)."""
    try:
        r = requests.post(f"{API_BASE}/feedback", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        st.error(
            f"Couldn't submit the report (HTTP {e.response.status_code}). "
            "Please try again, or copy your description and email it."
        )
        return None
    except requests.RequestException as e:
        st.error(
            f"Couldn't reach the backend at {API_BASE} to submit the report "
            f"({type(e).__name__}). Your description is preserved in the form."
        )
        return None


# --- logging ---------------------------------------------------------------


def append_qa_log(record: dict[str, Any]) -> None:
    """Append one Q&A turn to logs/qa_log.jsonl. Best-effort — never
    raises; logging failure shouldn't block the UI."""
    try:
        with QA_LOG_PATH.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass


def log_phi_block(query_len: int, hits: list) -> None:
    """Log a PHI block for review. We DO NOT log the actual query content —
    just length + which patterns triggered."""
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "phi_blocked",
        "query_len": query_len,
        "patterns": sorted({h.pattern_name for h in hits}),
    }
    append_qa_log(rec)


# --- main page rendering ---------------------------------------------------


EXAMPLE_QUESTIONS = [
    "What's the recommended endoscopic surveillance interval for low-grade "
    "dysplasia in Barrett's esophagus, and what's the GRADE?",
    "What's the first-line pharmacologic treatment for diabetic gastroparesis, "
    "and do AGA and ACG agree?",
    "What's the differential diagnosis for a gastric subepithelial lesion?",
    "How do I assess severity in acute pancreatitis at presentation?",
    "What does ASGE recommend for prevention of post-ERCP pancreatitis "
    "in high-risk patients?",
]


def _render_turn(turn: dict[str, Any], turn_idx: int, *, show_prompt: bool) -> None:
    """Render one turn (question header, answer, citation badge, passages)."""
    question = turn.get("question", "")
    result = turn.get("result") or {}
    elapsed = turn.get("elapsed_s")

    st.markdown(f"#### Q{turn_idx + 1}.  {question}")
    if result.get("refused"):
        st.warning(
            "**The corpus didn't cover this question directly.** "
            "Try broadening filters (more societies / years), rephrasing "
            "with different terms, or asking about a related published "
            "topic."
        )
    st.markdown(
        render_inline_citations(result.get("answer", "")),
        unsafe_allow_html=True,
    )
    render_citation_badge(result.get("verification") or {})

    chunks = result.get("chunks") or []
    cited_indices = {c.get("n") for c in (result.get("citations") or [])}
    if chunks:
        with st.expander(
            f"Retrieved passages — {len(chunks)} retrieved, "
            f"{len(cited_indices)} cited"
            + (f" · {elapsed:.1f}s" if elapsed else ""),
            expanded=False,
        ):
            for i, chunk in enumerate(chunks, start=1):
                render_passage_card(
                    chunk, rank=i, was_cited=(i in cited_indices),
                )

    if show_prompt:
        with st.expander("Debug — SOURCES block", expanded=False):
            from src.generate.prompt import build_user_message
            st.code(build_user_message(question, chunks), language="markdown")


def _render_feedback_form(
    *,
    conversation_id: int | None,
    last_turn: dict[str, Any] | None,
) -> None:
    """Sidebar/expander form letting the user submit an error report.

    Pre-fills hidden context (last question, last answer, retrieved chunk
    ids, verification flags, filter state) so the developer triaging the
    report has the same view the user did.
    """
    with st.expander("🐞 Report a problem", expanded=False):
        st.caption(
            "Tell us what went wrong. The most recent question and answer "
            "are attached automatically so we can reproduce the issue."
        )
        with st.form("feedback_form", clear_on_submit=True):
            category = st.selectbox(
                "What's the issue?",
                options=[
                    ("wrong_answer", "Answer is wrong or misleading"),
                    ("missing_source",
                     "A guideline I expected wasn't cited or retrieved"),
                    ("wrongly_refused",
                     "It refused but the corpus does cover this"),
                    ("ui_bug", "UI / display bug"),
                    ("other", "Something else"),
                ],
                format_func=lambda x: x[1],
            )
            description = st.text_area(
                "Describe what went wrong",
                height=120,
                placeholder=(
                    "e.g. The answer cited only ACG, but the AGA 2024 "
                    "gastroparesis guideline has a Recommendation 4 that "
                    "directly addresses metoclopramide and should have "
                    "been retrieved."
                ),
            )
            contact = st.text_input(
                "Email (optional — only if you want a follow-up)",
                placeholder="you@example.com",
            )
            attach_last = st.checkbox(
                "Attach the most recent Q&A to this report",
                value=True,
                disabled=last_turn is None,
                help=(
                    "When checked, the question, answer, retrieved chunk "
                    "ids, citation list, and active filters are sent with "
                    "the report so the team can reproduce the issue."
                ),
            )
            submitted = st.form_submit_button("Submit report", type="primary")

        if submitted:
            if not description.strip():
                st.error("Please describe what went wrong.")
                return

            payload: dict[str, Any] = {
                "category": category[0],
                "description": description.strip(),
                "contact": contact.strip() or None,
                "conversation_id": conversation_id,
            }
            if attach_last and last_turn is not None:
                result = last_turn.get("result") or {}
                chunk_ids = [
                    c.get("chunk_id")
                    for c in (result.get("chunks") or [])
                ]
                payload["question"] = last_turn.get("question")
                payload["answer"] = result.get("answer")
                payload["context"] = {
                    "filters": last_turn.get("filters") or {},
                    "top_k": last_turn.get("top_k"),
                    "chunk_ids": chunk_ids,
                    "citations": result.get("citations") or [],
                    "verification": result.get("verification") or {},
                    "model": result.get("model"),
                    "refused": bool(result.get("refused")),
                    "elapsed_s": last_turn.get("elapsed_s"),
                }

            with st.spinner("Submitting report..."):
                triage = submit_feedback(payload)
            if triage is None:
                return

            append_qa_log({
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "feedback",
                "report_id": triage.get("report_id"),
                "category": triage.get("category"),
                "conversation_id": conversation_id,
            })

            st.success(
                f"Report **{triage.get('report_id')}** received. "
                f"{triage.get('triage_hint', '')}"
            )


def render_main_page() -> None:
    cfg = load_ui_config()
    title = cfg.get("title", "GI Guidelines Assistant")
    subtitle = cfg.get(
        "subtitle",
        "Grounded answers from AGA, ACG, ASGE, and AASLD clinical guidelines",
    )
    default_top_k = int(cfg.get("default_top_k", 6))
    show_debug = bool(cfg.get("show_debug_toggle", True))
    phi_enabled = bool((cfg.get("phi_check") or {}).get("enabled", True))

    render_corpus_snapshot_header(title, subtitle)

    # --- session state ----
    if "turns" not in st.session_state:
        st.session_state.turns = []  # list[dict]: {question, result, filters, top_k, elapsed_s}
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    # --- sidebar filters ----
    with st.sidebar:
        st.subheader("Conversation")
        n_turns = len(st.session_state.turns)
        if n_turns:
            st.caption(
                f"{n_turns} turn{'s' if n_turns != 1 else ''} in this thread"
                + (
                    f" · id #{st.session_state.conversation_id}"
                    if st.session_state.conversation_id else ""
                )
            )
            if st.button("🗑 Start new conversation", use_container_width=True):
                st.session_state.turns = []
                st.session_state.conversation_id = None
                st.rerun()
        else:
            st.caption("No turns yet — ask a question to begin.")

        st.markdown("---")
        st.subheader("Filters")
        society_sel = st.multiselect(
            "Society", options=["AGA", "ACG", "ASGE", "AASLD"],
            default=["AGA", "ACG", "ASGE", "AASLD"],
            help="Limit retrieval to chunks from these societies.",
        )
        year_min = st.slider(
            "Earliest year", min_value=2019, max_value=2026, value=2020,
            help="Drop guidelines published before this year.",
        )
        topic_options = fetch_topics()
        topic_sel = st.multiselect(
            "Topic", options=topic_options, default=[],
            help="Topic slugs derived from the corpus metadata. Empty = all.",
        )
        doc_type_sel = st.multiselect(
            "Document type",
            options=["Guideline", "Guidance", "Standards", "Other"],
            default=["Guideline", "Guidance", "Standards", "Other"],
            help=(
                "'Guideline' = numbered society guidelines with GRADE; "
                "'Guidance' = AASLD's narrative practice-guidance documents; "
                "'Standards' = quality indicators / reprocessing / staffing; "
                "'Other' = clinical practice updates / expert reviews / summaries."
            ),
        )
        top_k = st.slider(
            "Passages to retrieve", min_value=3, max_value=10,
            value=default_top_k,
            help="More passages = broader context but more noise.",
        )
        show_prompt = False
        if show_debug:
            show_prompt = st.toggle(
                "Show full prompt sent to Claude", value=False,
                help="Debug view: dumps the SOURCES block and system prompt.",
            )

        st.markdown("---")
        _render_feedback_form(
            conversation_id=st.session_state.conversation_id,
            last_turn=(
                st.session_state.turns[-1] if st.session_state.turns else None
            ),
        )

        st.markdown("---")
        st.caption(
            f"Backend: `{API_BASE}` · "
            f"Snapshot: {read_corpus_snapshot()}"
        )

    # --- main area: input form ----
    is_followup = bool(st.session_state.turns)
    st.subheader("Follow-up question" if is_followup else "Ask a question")

    if is_followup:
        st.caption(
            "Your follow-up will be answered with the prior turns as context. "
            "Click **Start new conversation** in the sidebar to reset."
        )

    with st.form("question_form", clear_on_submit=True):
        q = st.text_area(
            "Clinical question",
            height=100,
            placeholder=(
                "e.g. What about for a penicillin-allergic patient?"
                if is_followup else
                "e.g. What's the recommended H. pylori regimen for a "
                "penicillin-allergic patient?"
            ),
            label_visibility="collapsed",
        )
        col_submit, _ = st.columns([1, 5])
        with col_submit:
            submitted = st.form_submit_button(
                "Ask follow-up" if is_followup else "Ask",
                type="primary",
                use_container_width=True,
            )

    # Example-question chips only on the first turn
    if not is_followup:
        st.caption("Try one of these:")
        eq_cols = st.columns(len(EXAMPLE_QUESTIONS))
        selected_example = None
        for i, eq in enumerate(EXAMPLE_QUESTIONS):
            with eq_cols[i]:
                short = eq.split(",")[0].split("?")[0]
                if len(short) > 60:
                    short = short[:57] + "…"
                if st.button(short, key=f"eg_{i}", use_container_width=True):
                    selected_example = eq

        if selected_example:
            st.session_state["pending_example"] = selected_example
            st.rerun()
        pending = st.session_state.pop("pending_example", None)
        if pending and not submitted:
            q = pending
            submitted = True

    # --- handle submission ----
    if submitted and q.strip():
        question = q.strip()

        if phi_enabled:
            hits = screen_for_phi(question)
            if hits:
                st.warning(phi_warning_message(hits))
                log_phi_block(len(question), hits)
                return

        filters: dict[str, Any] = {}
        if society_sel and len(society_sel) < 4:
            filters["society"] = society_sel
        if year_min > 2019:
            filters["year_min"] = year_min
        if topic_sel:
            filters["topic"] = topic_sel
        if doc_type_sel and len(doc_type_sel) < 4:
            filters["doc_type"] = doc_type_sel

        # Lazily create the server-side conversation row on the first turn.
        if st.session_state.conversation_id is None:
            st.session_state.conversation_id = create_conversation(
                title=question[:80]
            )

        history_payload = [
            {"question": t["question"], "answer": (t.get("result") or {}).get("answer", "")}
            for t in st.session_state.turns
        ]

        with st.spinner("Retrieving and generating..."):
            t0 = time.time()
            result = call_answer(
                question, filters, top_k,
                history=history_payload or None,
                conversation_id=st.session_state.conversation_id,
            )
            elapsed = time.time() - t0
        if result is None:
            return

        st.session_state.turns.append({
            "question": question,
            "result": result,
            "filters": filters,
            "top_k": top_k,
            "elapsed_s": round(elapsed, 1),
        })

        append_qa_log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "qa",
            "conversation_id": st.session_state.conversation_id,
            "turn_index": len(st.session_state.turns) - 1,
            "question": question,
            "answer": result.get("answer", ""),
            "refused": bool(result.get("refused")),
            "n_chunks": len(result.get("chunks") or []),
            "n_citations": len(result.get("citations") or []),
            "verification": result.get("verification") or {},
            "model": result.get("model"),
            "filters": filters,
            "elapsed_s": round(elapsed, 1),
        })

        st.rerun()

    # --- conversation thread (rendered every run) ----
    if st.session_state.turns:
        st.markdown("---")
        st.markdown("### Conversation")
        for i, turn in enumerate(st.session_state.turns):
            _render_turn(turn, i, show_prompt=show_prompt)
            if i < len(st.session_state.turns) - 1:
                st.markdown("---")

    render_footer()


# --- multi-page setup -------------------------------------------------------


def main() -> None:
    # default=True triggers an internal ``_default`` attribute access on
    # streamlit 1.40.x that crashes; positional ordering already makes the
    # first page the default, so we just rely on that.
    pg_main = st.Page(render_main_page, title="Ask", icon="🔎")
    pg_about = st.Page(about_page.render, title="About", icon="ℹ")
    nav = st.navigation([pg_main, pg_about])
    nav.run()


main()
