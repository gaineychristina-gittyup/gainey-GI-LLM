"""GaineyGuidelines — Phase 5 clinician UI.

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
    page_title="GaineyGuidelines", layout="wide",
    initial_sidebar_state="expanded",
)


# Bring repo root onto sys.path so `from src.ui.* import` works whether
# Streamlit is launched from the repo root or anywhere else.
import sys
sys.path.insert(0, str(REPO_ROOT))

from src.ui import about_page
from src.ui.components import (  # noqa: E402
    read_corpus_snapshot,
    render_citation_badge,
    render_corpus_snapshot_header,
    render_footer,
    render_inline_citations,
    render_source_pane,
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


def read_qa_history(limit: int = 25) -> list[dict[str, Any]]:
    """Return the most recent QA entries from qa_log.jsonl, newest first,
    deduped by question text. Best-effort — returns [] on any read error."""
    if not QA_LOG_PATH.exists():
        return []
    try:
        with QA_LOG_PATH.open() as f:
            lines = f.readlines()
    except OSError:
        return []
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") != "qa":
            continue
        q = (rec.get("question") or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(rec)
        if len(out) >= limit:
            break
    return out


def _format_history_ts(ts: str) -> str:
    """Render a log timestamp as a short relative label (e.g. '3h ago')."""
    try:
        dt = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return ""
    now = datetime.now(timezone.utc)
    delta = now - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    if secs < 86400 * 7:
        return f"{secs // 86400}d ago"
    return dt.strftime("%Y-%m-%d")


def render_saved_history(rec: dict[str, Any]) -> None:
    """Render a previously logged Q&A from qa_log.jsonl in the main column.

    Retrieved passages aren't logged, so the [N] anchors in the answer
    won't reveal a source card — we surface that caveat in a banner.
    """
    q = (rec.get("question") or "").strip()
    answer = rec.get("answer") or ""
    rel = _format_history_ts(rec.get("ts") or "")

    head_col, clear_col = st.columns([6, 1])
    with head_col:
        st.markdown(
            f"### Saved answer · _{rel}_" if rel else "### Saved answer"
        )
    with clear_col:
        if st.button("Clear", key="hist_clear", use_container_width=True):
            st.session_state.pop("viewing_history", None)
            st.rerun()

    st.markdown("**Question**")
    st.markdown(f"> {q}")

    if rec.get("refused"):
        st.warning(
            "**The corpus didn't cover this question directly.** "
            "Ask again with broader filters or a related topic."
        )

    st.markdown("**Answer**")
    st.markdown(render_inline_citations(answer), unsafe_allow_html=True)
    render_citation_badge(rec.get("verification") or {})

    n_chunks = rec.get("n_chunks") or 0
    n_cites = rec.get("n_citations") or 0
    elapsed = rec.get("elapsed_s")
    model = rec.get("model") or "?"
    bits = [
        f"{n_chunks} passages retrieved",
        f"{n_cites} cited",
        f"model: `{model}`",
    ]
    if elapsed is not None:
        bits.append(f"answered in {elapsed}s")
    st.caption(" · ".join(bits))

    filters = rec.get("filters") or {}
    if filters:
        with st.expander("Filters used", expanded=False):
            st.json(filters)

    st.info(
        "Retrieved passages aren't preserved for prior questions, so the "
        "[N] tags above are inert. Ask the same question again to re-run "
        "retrieval and see the source pane on the right."
    )


# --- main page rendering ---------------------------------------------------


EXAMPLE_QUESTIONS = [
    ("Barrett's surveillance interval",
     "What's the recommended endoscopic surveillance interval for low-grade "
     "dysplasia in Barrett's esophagus, and what's the GRADE?"),
    ("First-line gastroparesis treatment",
     "What's the first-line pharmacologic treatment for diabetic gastroparesis, "
     "and do AGA and ACG agree?"),
    ("Acute pancreatitis severity",
     "How do I assess severity in acute pancreatitis at presentation?"),
    ("Post-ERCP pancreatitis prevention",
     "What does ASGE recommend for prevention of post-ERCP pancreatitis "
     "in high-risk patients?"),
]


def render_main_page() -> None:
    cfg = load_ui_config()
    title = cfg.get("title", "GaineyGuidelines")
    subtitle = cfg.get(
        "subtitle",
        "Grounded answers from AGA, ACG, ASGE, and AASLD clinical guidelines",
    )
    default_top_k = int(cfg.get("default_top_k", 6))
    show_debug = bool(cfg.get("show_debug_toggle", True))
    phi_enabled = bool((cfg.get("phi_check") or {}).get("enabled", True))

    render_corpus_snapshot_header(title, subtitle)

    # --- sidebar: prior conversations + filters ----
    with st.sidebar:
        st.subheader("Prior conversations")
        history = read_qa_history(limit=25)
        if not history:
            st.caption(
                "No prior questions yet. Ask one and it'll show up here."
            )
        else:
            st.caption(f"{len(history)} recent · click to restore")
            for i, rec in enumerate(history):
                q_text = (rec.get("question") or "").strip()
                short = q_text if len(q_text) <= 70 else q_text[:67] + "…"
                rel = _format_history_ts(rec.get("ts") or "")
                label = f"{short}\n\n_{rel}_" if rel else short
                if st.button(
                    label, key=f"hist_{i}", use_container_width=True,
                    help=q_text,
                ):
                    st.session_state["viewing_history"] = rec
                    st.rerun()

        st.markdown("---")
        with st.expander("Filters", expanded=False):
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

    # Two-column layout: question + answer on the left, source viewer
    # pane (all retrieved chunks) on the right. Each chunk in the pane
    # gets ``id="passage-N"``; the [N] badges in the answer are plain
    # in-page anchors that scroll the page so the matching passage is
    # in view in the right pane. No session-state, no query params — a
    # citation click is just a browser-native scroll.
    col_main, col_pane = st.columns([3, 2], gap="large")

    # Run the full submission pipeline up-front so the result drives
    # both columns. Do NOT use early `return` before both columns and
    # the footer have rendered, otherwise the right pane disappears.
    result: dict[str, Any] | None = None
    elapsed: float | None = None
    chunks: list[dict[str, Any]] = []
    cited_indices: set[int] = set()
    phi_msg: str | None = None

    with col_main:
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
                submitted = st.form_submit_button(
                    "Ask", type="primary", use_container_width=True,
                )

        st.caption("Try one of these:")
        selected_example = None
        for row_start in range(0, len(EXAMPLE_QUESTIONS), 2):
            row_cols = st.columns(2)
            for j, (label, eq) in enumerate(
                EXAMPLE_QUESTIONS[row_start:row_start + 2]
            ):
                with row_cols[j]:
                    if st.button(
                        label, key=f"eg_{row_start + j}",
                        use_container_width=True,
                    ):
                        selected_example = eq

        # Example-button click acts as a same-tick form submission with
        # the predefined query. Streamlit fires us a rerun on click, and
        # on that rerun ``selected_example`` is set when we re-enter the
        # button loop above — so this branch is reached on the same run
        # the user actually expects to see the answer on.
        if selected_example and not submitted:
            q = selected_example
            submitted = True

        # If the user clicked a sidebar history item AND isn't kicking off
        # a fresh query, render the saved Q&A here instead of the form's
        # answer area. A new submission clears the history view below.
        saved = st.session_state.get("viewing_history")
        if saved and not (submitted and q.strip()):
            render_saved_history(saved)

        if submitted and q.strip():
            # Clear any saved-history view on a fresh submission so the
            # new answer takes the column.
            st.session_state.pop("viewing_history", None)
            question = q.strip()

            if phi_enabled:
                hits = screen_for_phi(question)
                if hits:
                    phi_msg = phi_warning_message(hits)
                    log_phi_block(len(question), hits)

            if phi_msg is None:
                filters: dict[str, Any] = {}
                if society_sel and len(society_sel) < 4:
                    filters["society"] = society_sel
                if year_min > 2019:
                    filters["year_min"] = year_min
                if topic_sel:
                    filters["topic"] = topic_sel
                if doc_type_sel and len(doc_type_sel) < 4:
                    filters["doc_type"] = doc_type_sel

                with st.spinner("Retrieving and generating..."):
                    t0 = time.time()
                    result = call_answer(question, filters, top_k)
                    elapsed = time.time() - t0
                if result is not None:
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

        if phi_msg:
            st.warning(phi_msg)

        if result:
            st.markdown("### Answer")
            if result.get("refused"):
                st.warning(
                    "**The corpus didn't cover this question directly.** "
                    "Try broadening filters (more societies / years), "
                    "rephrasing with different terms, or asking about a "
                    "related published topic. Some questions — pediatric "
                    "GI, surgical decision-making, very recent guidelines — "
                    "are out of scope by design."
                )
            st.markdown(
                render_inline_citations(result.get("answer", "")),
                unsafe_allow_html=True,
            )
            render_citation_badge(result.get("verification") or {})
            chunks = result.get("chunks") or []
            cited_indices = {
                c.get("n") for c in (result.get("citations") or [])
            }

            if show_prompt:
                with st.expander(
                    "Debug — SOURCES block sent to Claude", expanded=False,
                ):
                    from src.generate.prompt import build_user_message
                    st.code(
                        build_user_message(question, chunks),
                        language="markdown",
                    )

    with col_pane:
        # Single-citation focused source viewer. Hidden cards for
        # every chunk; the CSS :target rule reveals just the one
        # whose id matches the URL hash (#cite-N), driven by the
        # [N] badges in the answer.
        render_source_pane(
            chunks=chunks,
            cited_indices=cited_indices,
            answer_text=(result or {}).get("answer", "") if result else "",
            elapsed=elapsed,
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
