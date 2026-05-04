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
    query: str, filters: dict[str, Any], top_k: int,
) -> dict[str, Any] | None:
    """POST /answer (non-streaming). Returns the answer dict or None on
    failure, with the failure already surfaced to the user."""
    payload = {
        "query": query,
        "filters": filters or None,
        "top_k": top_k,
        "rerank": True,
        "stream": False,
        "save": False,  # Phase 5 doesn't use server-side conversation history
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

    st.set_page_config(
        page_title=title, layout="wide", initial_sidebar_state="expanded",
    )

    render_corpus_snapshot_header(title, subtitle)

    # --- sidebar filters ----
    with st.sidebar:
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
        st.caption(
            f"Backend: `{API_BASE}` · "
            f"Snapshot: {read_corpus_snapshot()}"
        )

    # --- main area ----
    st.subheader("Ask a question")
    with st.form("question_form", clear_on_submit=False):
        q = st.text_area(
            "Clinical question",
            height=100,
            placeholder="e.g. What's the recommended H. pylori regimen for a "
                        "penicillin-allergic patient?",
            label_visibility="collapsed",
        )
        col_submit, _ = st.columns([1, 5])
        with col_submit:
            submitted = st.form_submit_button("Ask", type="primary",
                                               use_container_width=True)

    # Example-question chips below the form
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

    # If an example was clicked, store and rerun so the form picks it up.
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

        # 1) PHI screen
        if phi_enabled:
            hits = screen_for_phi(question)
            if hits:
                st.warning(phi_warning_message(hits))
                log_phi_block(len(question), hits)
                return

        # 2) Build filters
        filters: dict[str, Any] = {}
        if society_sel and len(society_sel) < 4:
            filters["society"] = society_sel
        if year_min > 2019:
            filters["year_min"] = year_min
        if topic_sel:
            filters["topic"] = topic_sel
        if doc_type_sel and len(doc_type_sel) < 4:
            filters["doc_type"] = doc_type_sel

        # 3) Call backend
        with st.spinner("Retrieving and generating..."):
            t0 = time.time()
            result = call_answer(question, filters, top_k)
            elapsed = time.time() - t0
        if result is None:
            return

        # 4) Log to JSONL (without ballooning record size)
        append_qa_log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "qa",
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

        # 5) Render answer
        st.markdown("### Answer")
        if result.get("refused"):
            st.warning(
                "**The corpus didn't cover this question directly.** "
                "Try broadening filters (more societies / years), rephrasing "
                "with different terms, or asking about a related published "
                "topic. Some questions — pediatric GI, surgical decision-"
                "making, very recent guidelines — are out of scope by design."
            )
        st.markdown(
            render_inline_citations(result.get("answer", "")),
            unsafe_allow_html=True,
        )
        render_citation_badge(result.get("verification") or {})

        # 6) Render retrieved passages
        chunks = result.get("chunks") or []
        cited_indices = {c.get("n") for c in (result.get("citations") or [])}
        if chunks:
            st.markdown("### Retrieved passages")
            st.caption(
                f"{len(chunks)} passages retrieved · "
                f"{len(cited_indices)} cited in the answer above · "
                f"answer generated in {elapsed:.1f}s"
            )
            for i, chunk in enumerate(chunks, start=1):
                render_passage_card(
                    chunk, rank=i, was_cited=(i in cited_indices),
                )

        # 7) Optional debug view
        if show_prompt:
            with st.expander("Debug — SOURCES block sent to Claude", expanded=False):
                # Reconstruct what the model saw using the same formatter
                # the backend uses.
                from src.generate.prompt import build_user_message
                st.code(
                    build_user_message(question, chunks),
                    language="markdown",
                )

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
