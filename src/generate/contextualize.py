"""Fold conversation history into a standalone retrieval query.

The answer() pipeline threads conversation history into Claude's message
list so the synthesis side can resolve "what about for ACG?" against the
prior turn. But retrieval still runs on only the current bare question —
"what about for ACG?" by itself retrieves nothing relevant because the
embedder has no idea what the topic was.

This module asks Haiku to read the prior turns and the current follow-up,
and emit ONE standalone query that captures the user's actual retrieval
intent. The standalone query is used for retrieve() only; the user's
original phrasing is still what Claude sees, so the answer stays
conversational.

Failure mode: any error returns the current question unchanged. The
caller falls back to retrieving on the bare follow-up.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

CONTEXT_MODEL = "claude-haiku-4-5-20251001"

CONTEXT_SYSTEM = """You convert clinical follow-up questions into standalone retrieval queries.

You'll see the prior conversation turns (each with the user's question and the system's answer summary) and the user's current follow-up. Your job: produce ONE standalone search query that captures what the user is actually asking for, given the prior context.

Rules:
- If the follow-up is a refinement ("what about for ACG?", "and the GRADE?", "in pregnancy?", "what's the dose?"), expand it to a full search query that incorporates the prior topic.
- If the follow-up is a true topic change (the prior turn was about gastroparesis, current is "what's the recommended H. pylori regimen?"), output the current question unchanged.
- Keep the query dense and clinical — use formal guideline language, drug class names, and intervention terms.
- 8-25 words, no question mark, no preamble, no commentary.
- Output the query on a single line, nothing else."""


_client_cache: dict = {}


def _get_client():
    if "client" not in _client_cache:
        from anthropic import Anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set; cannot contextualize.")
        _client_cache["client"] = Anthropic(api_key=api_key)
    return _client_cache["client"]


def standalone_query(
    question: str,
    history: list[dict[str, str]] | None,
    *,
    model: str = CONTEXT_MODEL,
    max_history_turns: int = 4,
) -> str:
    """Return a retrieval query that incorporates history when present.

    Returns the bare ``question`` unchanged when:
      - no history
      - the question is already standalone (clearly self-contained)
      - the Haiku call fails
    """
    if not history:
        return question
    if not question or not question.strip():
        return question

    # Use only the last few turns to keep the prompt bounded.
    turns = [t for t in history if t.get("question") and t.get("answer")][-max_history_turns:]
    if not turns:
        return question

    convo = []
    for t in turns:
        q = t["question"].strip()
        a = t["answer"].strip()
        # Truncate long prior answers to keep the prompt small.
        if len(a) > 600:
            a = a[:600] + "..."
        convo.append(f"USER: {q}\nASSISTANT: {a}")
    convo_text = "\n\n".join(convo)

    user_msg = f"PRIOR TURNS:\n\n{convo_text}\n\nCURRENT FOLLOW-UP: {question.strip()}"
    try:
        resp = _get_client().messages.create(
            model=model,
            max_tokens=200,
            temperature=0.0,
            system=CONTEXT_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text.strip() if resp.content else ""
    except Exception as e:
        logger.warning("contextualize failed: %s", e)
        return question

    if not text:
        return question
    line = text.splitlines()[0].strip().strip('"').strip("'")
    if line.endswith("?"):
        line = line.rstrip("?").rstrip()
    if not line or len(line) < 4:
        return question
    return line
