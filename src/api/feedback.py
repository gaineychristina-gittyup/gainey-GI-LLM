"""User-submitted error reports / feedback.

The Streamlit UI lets a clinician flag a problem with the most recent
answer (wrong, refused-but-shouldn't-have, missing source, UI bug, etc).
The handler persists each report to two places:

1. ``logs/feedback_log.jsonl`` — append-only, durable across DB resets,
   safe to grep / pipe through scripts.
2. The Postgres ``feedback`` table — queryable, joinable to conversations.

We also produce a small triage record (``triage_hint``) summarising what
the developer should look at first based on the category. The Streamlit
UI shows this to the user as a "what we'll do with this" message so the
report doesn't feel like it's vanishing into the void.
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import psycopg

REPO_ROOT = Path(__file__).resolve().parents[2]
FEEDBACK_LOG_PATH = REPO_ROOT / "logs" / "feedback_log.jsonl"
FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


CATEGORY_TRIAGE: dict[str, str] = {
    "wrong_answer": (
        "Logged for retrieval/answer review. We'll re-run the question "
        "against the corpus, compare the cited passages to the source PDFs, "
        "and tune retrieval or prompt grounding if the cited evidence is "
        "thin or off-topic."
    ),
    "missing_source": (
        "Logged for corpus review. We'll check whether the document you "
        "expected is in the snapshot and whether the relevant chunk was "
        "filtered out by the rerank or society/year filters."
    ),
    "wrongly_refused": (
        "Logged for refusal-rate tuning. We'll re-run the retrieval with "
        "looser filters and lower thresholds to confirm whether the corpus "
        "actually covers the question."
    ),
    "ui_bug": "Logged for UI triage.",
    "other": "Logged for review.",
}


@contextmanager
def _db() -> Iterable[psycopg.Connection]:
    url = os.environ.get(
        "DATABASE_URL", "postgresql://gi:gi@localhost:5432/gi_guidelines"
    )
    conn = psycopg.connect(url)
    try:
        yield conn
    finally:
        conn.close()


def _ensure_table() -> None:
    """Create the feedback table on first use. Idempotent."""
    try:
        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id BIGSERIAL PRIMARY KEY,
                    report_id TEXT UNIQUE NOT NULL,
                    submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    category TEXT NOT NULL,
                    description TEXT NOT NULL,
                    contact TEXT,
                    conversation_id BIGINT,
                    question TEXT,
                    answer TEXT,
                    context JSONB,
                    status TEXT NOT NULL DEFAULT 'open'
                )
                """,
            )
            conn.commit()
    except Exception:
        # DB may not be reachable — JSONL is the durable backstop.
        pass


def submit_feedback(
    *,
    category: str,
    description: str,
    contact: Optional[str] = None,
    conversation_id: Optional[int] = None,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Persist one feedback record. Returns the report metadata."""
    _ensure_table()

    cat = (category or "other").lower()
    if cat not in CATEGORY_TRIAGE:
        cat = "other"
    triage = CATEGORY_TRIAGE[cat]

    report_id = f"FB-{uuid.uuid4().hex[:10].upper()}"
    submitted_at = datetime.now(timezone.utc).isoformat()

    record = {
        "report_id": report_id,
        "submitted_at": submitted_at,
        "category": cat,
        "description": description,
        "contact": contact,
        "conversation_id": conversation_id,
        "question": question,
        "answer": answer,
        "context": context or {},
        "triage_hint": triage,
        "status": "open",
    }

    # 1) JSONL append (best-effort, never raises)
    try:
        with FEEDBACK_LOG_PATH.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass

    # 2) DB row (best-effort — JSONL is the source of truth if DB is down)
    try:
        with _db() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (
                    report_id, category, description, contact,
                    conversation_id, question, answer, context
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    report_id, cat, description, contact,
                    conversation_id, question, answer,
                    json.dumps(context or {}),
                ),
            )
            conn.commit()
    except Exception:
        pass

    return {
        "report_id": report_id,
        "submitted_at": submitted_at,
        "category": cat,
        "triage_hint": triage,
        "status": "open",
    }


def list_feedback(limit: int = 100, status: Optional[str] = None) -> list[dict[str, Any]]:
    """Return recent feedback records for triage view."""
    _ensure_table()
    sql = (
        "SELECT report_id, submitted_at, category, description, contact, "
        "conversation_id, question, status "
        "FROM feedback "
    )
    args: list[Any] = []
    if status:
        sql += "WHERE status = %s "
        args.append(status)
    sql += "ORDER BY submitted_at DESC LIMIT %s"
    args.append(limit)
    try:
        with _db() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(args))
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception:
        return []
