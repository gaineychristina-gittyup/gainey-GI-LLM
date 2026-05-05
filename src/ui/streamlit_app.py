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

import hmac
import html as html_module
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
    query: str,
    filters: dict[str, Any],
    top_k: int,
    *,
    history: list[dict[str, str]] | None = None,
    conversation_id: int | None = None,
) -> dict[str, Any] | None:
    """POST /answer (non-streaming). Returns the answer dict or None on
    failure, with the failure already surfaced to the user.

    ``history`` is a list of prior ``{question, answer}`` turns from this
    conversation; the backend threads them as Claude messages so follow-up
    questions can reference earlier context. When ``conversation_id`` is
    set, the backend also persists each turn server-side."""
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
    """POST /conversations to start a server-side thread. Returns the new
    id, or None if the backend isn't reachable (we degrade gracefully —
    the in-session thread still works without a persisted row)."""
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
    """POST /feedback. Returns the triage record on success."""
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
            f"Couldn't reach the backend at {API_BASE} to submit the "
            f"report ({type(e).__name__}). Your description is preserved "
            "in the form."
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


def _render_feedback_form(
    *,
    conversation_id: int | None,
    last_turn: dict[str, Any] | None,
) -> None:
    """Sidebar form letting the user submit an error report.

    Pre-fills hidden context (last question, last answer, retrieved chunk
    ids, verification flags, filter state) so the developer triaging the
    report has the same view the reporter did. Each report comes back
    with a tracking id (FB-XXXXXXXXXX) and a category-specific triage
    hint summarising what the team will do with it.
    """
    with st.expander("🐞 Report a problem", expanded=False):
        st.caption(
            "Tell us what went wrong. The most recent question and answer "
            "are attached so we can reproduce the issue."
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
                    "gastroparesis guideline has a Recommendation that "
                    "should have been retrieved."
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
                    "ids, citation list, and active filters are sent so "
                    "the team can reproduce the issue."
                ),
            )
            submitted = st.form_submit_button("Submit report", type="primary")

        if not submitted:
            return
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
                c.get("chunk_id") for c in (result.get("chunks") or [])
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


def _render_prior_turn(turn: dict[str, Any], turn_idx: int) -> None:
    """Render one prior turn in the active conversation thread.

    The latest turn is rendered separately by the main flow (with the
    full source pane in col_pane). Prior turns get a compact
    expander-based view so the column doesn't grow unbounded.
    """
    question = turn.get("question") or ""
    result = turn.get("result") or {}

    st.markdown(f"**Q{turn_idx + 1}.** {question}")
    if result.get("refused"):
        st.warning(
            "**The corpus didn't cover this question directly.** "
            "Try broadening filters or rephrasing.",
            icon="⚠",
        )
    st.markdown(
        render_inline_citations(result.get("answer", "")),
        unsafe_allow_html=True,
    )
    n_chunks = len(result.get("chunks") or [])
    n_cited = len(result.get("citations") or [])
    elapsed = turn.get("elapsed_s")
    bits = [f"{n_chunks} retrieved", f"{n_cited} cited"]
    if elapsed is not None:
        bits.append(f"{elapsed}s")
    st.caption(" · ".join(bits))


# --- main page rendering ---------------------------------------------------


EXAMPLE_QUESTIONS = [
    "Should patients with a history of diverticulitis avoid NSAIDS and aspirin?",
    "For patients with pouchitis who start vedolizumab, should I stop antibiotics?",
    "Barrett's esophagus with low-grade dysplasia — how do I discuss endoscopic "
    "eradication therapy vs surveillance with a hesitant patient?",
    "What's the first-line pharmacologic treatment for diabetic gastroparesis, "
    "and do AGA and ACG agree?",
    "What does ASGE recommend for prevention of post-ERCP pancreatitis "
    "in high-risk patients?",
]


def _bucket_history_by_age(history: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group history records into time-bucket sections (Today / Yesterday /
    This week / Earlier), preserving the input order within each bucket.
    Returns only non-empty buckets, in chronological order."""
    today: list[dict[str, Any]] = []
    yesterday: list[dict[str, Any]] = []
    this_week: list[dict[str, Any]] = []
    earlier: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    today_date = now.date()
    for rec in history:
        try:
            ts = datetime.fromisoformat(rec.get("ts") or "")
        except (TypeError, ValueError):
            earlier.append(rec)
            continue
        days = (today_date - ts.date()).days
        if days <= 0:
            today.append(rec)
        elif days == 1:
            yesterday.append(rec)
        elif days < 7:
            this_week.append(rec)
        else:
            earlier.append(rec)
    sections = [
        ("Today", today),
        ("Yesterday", yesterday),
        ("This week", this_week),
        ("Earlier", earlier),
    ]
    return [(name, items) for name, items in sections if items]


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

    # --- session state for the active conversation thread ----
    if "turns" not in st.session_state:
        st.session_state.turns = []  # list[{question, result, filters, top_k, elapsed_s}]
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    # --- sidebar: conversations + filters ----
    history = read_qa_history(limit=50)
    # Handle clicks on sidebar-conversation and suggested-question links
    # via query params. Streamlit's st.button has emotion-cache CSS
    # specificity that overrides our styling, so we render those lists as
    # plain HTML <a> links that navigate to ?open_history=N or ?ask=N
    # — the script re-runs on click and we route the intent here.
    qp = st.query_params
    if "open_history" in qp:
        try:
            idx = int(qp["open_history"])
        except (TypeError, ValueError):
            idx = -1
        if 0 <= idx < len(history):
            st.session_state["viewing_history"] = history[idx]
        del qp["open_history"]
        st.rerun()
    if "ask" in qp:
        try:
            idx = int(qp["ask"])
        except (TypeError, ValueError):
            idx = -1
        if 0 <= idx < len(EXAMPLE_QUESTIONS):
            st.session_state["pending_query"] = EXAMPLE_QUESTIONS[idx]
            st.session_state.pop("viewing_history", None)
        del qp["ask"]
        st.rerun()

    with st.sidebar:
        # Active thread controls — only shown when a thread is in progress.
        if st.session_state.turns:
            n = len(st.session_state.turns)
            st.markdown(
                f"#### Active thread · {n} turn{'s' if n != 1 else ''}"
            )
            if st.button(
                "🗑 Start new conversation",
                use_container_width=True,
                key="start_new_conv",
            ):
                st.session_state.turns = []
                st.session_state.conversation_id = None
                st.session_state.pop("viewing_history", None)
                st.rerun()
            st.markdown("---")

        st.markdown("#### Conversations")
        if not history:
            st.caption(
                "No prior questions yet. Ask one and it'll show up here."
            )
        else:
            sections = _bucket_history_by_age(history)
            global_idx = 0
            html_parts: list[str] = ["<div class='gg-conv-list'>"]
            for section_name, items in sections:
                html_parts.append(
                    f"<div class='gg-conv-bucket'>{section_name}</div>"
                )
                for rec in items:
                    q_text = (rec.get("question") or "").strip()
                    safe_text = html_module.escape(q_text)
                    rel = html_module.escape(
                        _format_history_ts(rec.get("ts") or "")
                    )
                    title_budget = max(18, 28 - len(rel))
                    short = (
                        safe_text if len(safe_text) <= title_budget
                        else safe_text[:title_budget - 1] + "…"
                    )
                    html_parts.append(
                        f"<a class='gg-conv-item' "
                        f"href='?open_history={global_idx}' "
                        f"title='{safe_text}' target='_self'>"
                        f"<span class='gg-conv-icon'>💬</span>"
                        f"<span class='gg-conv-title'>{short}</span>"
                        f"<span class='gg-conv-time'>{rel}</span>"
                        f"</a>"
                    )
                    global_idx += 1
            html_parts.append("</div>")
            st.markdown("\n".join(html_parts), unsafe_allow_html=True)

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
        is_followup = bool(st.session_state.turns)

        # Render any prior turns at the top of col_main so the user can
        # see the thread they're following up on. The latest turn's
        # answer renders below the form (next block).
        if is_followup:
            st.markdown("### Conversation")
            for i, turn in enumerate(st.session_state.turns):
                _render_prior_turn(turn, i)
                st.markdown("---")
            st.subheader("Follow-up question")
            st.caption(
                "Your follow-up will be answered with the prior turns as "
                "context. Click **Start new conversation** in the sidebar "
                "to reset."
            )
        else:
            st.subheader("Ask a question")

        with st.form("question_form", clear_on_submit=is_followup):
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

        # If the query-param handler at the top of the page stashed a
        # suggested question for us, treat it as if the user typed and
        # hit Ask on this run.
        pending = st.session_state.pop("pending_query", None)
        if pending and not submitted:
            q = pending
            submitted = True

        # If the user clicked a sidebar history item AND isn't kicking off
        # a fresh query, render the saved Q&A here instead of the form's
        # answer area. A new submission clears the history view below.
        saved = st.session_state.get("viewing_history")
        if saved and not (submitted and q.strip()):
            render_saved_history(saved)

        # Auto-collapse the Suggested Questions panel when there's an
        # answer or saved view to show, so the answer stays above the
        # fold. Render as raw HTML <a> links (not st.button) to avoid
        # Streamlit's emotion-cache CSS specificity battles.
        will_show_answer = (
            (submitted and q.strip()) or bool(saved)
        )
        sugg_html = ["<div class='gg-sugg-list'>"]
        for i, eq in enumerate(EXAMPLE_QUESTIONS):
            sugg_html.append(
                f"<a class='gg-sugg-item' "
                f"href='?ask={i}' target='_self' "
                f"title='{html_module.escape(eq)}'>"
                f"<span class='gg-sugg-icon'>"
                "<svg width='14' height='14' viewBox='0 0 24 24' "
                "fill='none' stroke='currentColor' stroke-width='2' "
                "stroke-linecap='round' stroke-linejoin='round'>"
                "<circle cx='11' cy='11' r='8'/>"
                "<line x1='21' y1='21' x2='16.65' y2='16.65'/>"
                "</svg></span>"
                f"<span class='gg-sugg-text'>{html_module.escape(eq)}</span>"
                f"</a>"
            )
        sugg_html.append("</div>")
        with st.expander(
            "**Suggested Questions**", expanded=not will_show_answer,
        ):
            st.markdown("\n".join(sugg_html), unsafe_allow_html=True)

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

                # Lazily open a server-side conversation on the first turn.
                if st.session_state.conversation_id is None:
                    st.session_state.conversation_id = create_conversation(
                        title=question[:80]
                    )

                history_payload = [
                    {
                        "question": t["question"],
                        "answer": (t.get("result") or {}).get("answer", ""),
                    }
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
                if result is not None:
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


# --- password gate ---------------------------------------------------------


def _check_password() -> bool:
    """Single shared password gate. Returns True once the user has entered
    the correct password this session.

    Password source (in priority order):
      1. ``st.secrets["app_password"]``  (preferred — set in
         ``.streamlit/secrets.toml`` or Streamlit Cloud's Secrets UI)
      2. ``GI_APP_PASSWORD`` env var
    If neither is set, the gate is disabled (open access)."""
    expected = None
    # Only consult st.secrets when a secrets.toml actually exists —
    # otherwise Streamlit prints a noisy "No secrets found" banner at
    # the top of the page even when we catch the exception.
    secrets_paths = [
        Path.home() / ".streamlit" / "secrets.toml",
        REPO_ROOT / ".streamlit" / "secrets.toml",
    ]
    if any(p.exists() for p in secrets_paths):
        try:
            expected = st.secrets.get("app_password")  # type: ignore[attr-defined]
        except Exception:
            expected = None
    if not expected:
        expected = os.environ.get("GI_APP_PASSWORD")
    if not expected:
        return True

    if st.session_state.get("authed"):
        return True

    st.title("GaineyGuidelines")
    with st.form("login"):
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        if hmac.compare_digest(pw, expected):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


# --- multi-page setup -------------------------------------------------------


def main() -> None:
    if not _check_password():
        return
    # default=True triggers an internal ``_default`` attribute access on
    # streamlit 1.40.x that crashes; positional ordering already makes the
    # first page the default, so we just rely on that.
    pg_main = st.Page(render_main_page, title="Ask", icon="🔎")
    pg_about = st.Page(about_page.render, title="About", icon="ℹ")
    nav = st.navigation([pg_main, pg_about])
    nav.run()


main()
