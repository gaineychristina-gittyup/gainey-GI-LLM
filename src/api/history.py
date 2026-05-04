"""Conversation persistence for the Phase 4 UI.

Each Streamlit session opens a conversation row and appends a turn per Q/A.
The UI uses :func:`list_conversations` for the history sidebar and
:func:`get_conversation` to reload a prior thread.

JSONB columns (``citations`` / ``usage`` / ``verification``) carry whatever
the answer dict produced; we don't normalize on the way in.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Iterable, Optional

import psycopg


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


def create_conversation(title: Optional[str] = None) -> int:
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO conversations (title) VALUES (%s) RETURNING id",
            (title,),
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        return new_id


def save_turn(
    conversation_id: int,
    question: str,
    answer_dict: dict[str, Any],
) -> int:
    """Append a Q/A turn. Returns the turn id."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(MAX(turn_index), -1) + 1
              FROM conversation_turns
             WHERE conversation_id = %s
            """,
            (conversation_id,),
        )
        next_index = cur.fetchone()[0]

        cur.execute(
            """
            INSERT INTO conversation_turns (
                conversation_id, turn_index, question, answer,
                citations, usage, verification, refused
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                conversation_id,
                next_index,
                question,
                answer_dict.get("answer", ""),
                json.dumps(answer_dict.get("citations") or []),
                json.dumps(answer_dict.get("usage") or {}),
                json.dumps(answer_dict.get("verification") or {}),
                bool(answer_dict.get("refused")),
            ),
        )
        turn_id = cur.fetchone()[0]

        # If this is the first turn, derive a title from the question for the
        # sidebar.
        if next_index == 0:
            title = question.strip().splitlines()[0][:80]
            cur.execute(
                "UPDATE conversations SET title = COALESCE(title, %s) WHERE id = %s",
                (title, conversation_id),
            )

        conn.commit()
        return turn_id


def list_conversations(limit: int = 50) -> list[dict[str, Any]]:
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.title, c.started_at, COUNT(t.id) AS n_turns
              FROM conversations c
              LEFT JOIN conversation_turns t ON t.conversation_id = c.id
             GROUP BY c.id
             ORDER BY c.started_at DESC
             LIMIT %s
            """,
            (limit,),
        )
        return [
            {"id": cid, "title": title, "started_at": ts, "n_turns": n}
            for (cid, title, ts, n) in cur.fetchall()
        ]


def get_conversation(conversation_id: int) -> dict[str, Any]:
    """Return {meta: ..., turns: [...]} for a conversation."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, title, started_at FROM conversations WHERE id = %s",
            (conversation_id,),
        )
        row = cur.fetchone()
        if not row:
            return {"meta": None, "turns": []}
        meta = {"id": row[0], "title": row[1], "started_at": row[2]}

        cur.execute(
            """
            SELECT turn_index, question, answer, citations, usage,
                   verification, refused, asked_at
              FROM conversation_turns
             WHERE conversation_id = %s
             ORDER BY turn_index
            """,
            (conversation_id,),
        )
        turns = []
        for r in cur.fetchall():
            turns.append({
                "turn_index": r[0],
                "question": r[1],
                "answer": r[2],
                "citations": r[3] or [],
                "usage": r[4] or {},
                "verification": r[5] or {},
                "refused": r[6],
                "asked_at": r[7],
            })
        return {"meta": meta, "turns": turns}


def delete_conversation(conversation_id: int) -> None:
    """Hard-delete a conversation and its turns. Only safe to call when the
    UI has confirmed the user really wants to."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))
        conn.commit()
