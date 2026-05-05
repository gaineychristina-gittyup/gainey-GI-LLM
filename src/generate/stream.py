"""Streaming variant of :func:`src.generate.answer.answer`.

Uses Anthropic's ``messages.stream()`` so the UI can render token-by-token.
The generator yields one of three event shapes:

  {"type": "token", "text": str}            -- a delta to append to the answer
  {"type": "done",  "result": dict}         -- final dict (same shape as answer())
  {"type": "error", "message": str}         -- streaming failed; result key absent

Callers should iterate to completion to capture the ``done`` payload — that's
where citations / verification / usage are populated.
"""

from __future__ import annotations

from typing import Any, Iterator

from src.generate.answer import (
    _extract_cited_indices,
    _citation_record,
    _get_anthropic_client,
    _load_config,
    _model_rejects_temperature,
)
from src.generate.prompt import SYSTEM_PROMPT, build_user_message
from src.generate.verify import verify_answer
from src.retrieve import retrieve


def answer_stream(
    question: str,
    *,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    rerank: bool = True,
    history: list[dict[str, str]] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield streaming token events plus a final ``done`` event.

    Same retrieval + grounding as :func:`answer`, only the LLM call is
    streamed. We accumulate the text locally so the final dict still has
    the full answer plus extracted citations and verification results.
    """
    cfg = _load_config()
    gen_cfg = cfg.get("generation", {})
    model = gen_cfg.get("model", "claude-sonnet-4-5")
    max_tokens = int(gen_cfg.get("max_tokens", 1500))
    temperature = float(gen_cfg.get("temperature", 0.0))

    rcfg = cfg.get("retrieval", {})
    top_k = int(top_k or rcfg.get("top_k", 6))

    chunks = retrieve(question, filters=filters, top_k=top_k, rerank=rerank)
    if not chunks:
        yield {
            "type": "done",
            "result": {
                "answer": "The provided guidelines do not directly address this. "
                          "(No chunks were retrieved — check filters or corpus state.)",
                "chunks": [],
                "citations": [],
                "model": model,
                "usage": {},
                "refused": True,
                "verification": {"ok": True, "n_citations": 0,
                                 "out_of_range": [], "unsupported": [],
                                 "min_partial_ratio": 75},
            },
        }
        return

    user_msg = build_user_message(question, chunks)
    client = _get_anthropic_client()

    messages: list[dict[str, Any]] = []
    for turn in history or []:
        prev_q = (turn.get("question") or "").strip()
        prev_a = (turn.get("answer") or "").strip()
        if not prev_q or not prev_a:
            continue
        messages.append({"role": "user", "content": f"QUESTION: {prev_q}"})
        messages.append({"role": "assistant", "content": prev_a})
    messages.append({"role": "user", "content": user_msg})

    create_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        "messages": messages,
    }
    if not _model_rejects_temperature(model):
        create_kwargs["temperature"] = temperature

    accumulated: list[str] = []
    try:
        with client.messages.stream(**create_kwargs) as stream:
            for delta in stream.text_stream:
                accumulated.append(delta)
                yield {"type": "token", "text": delta}
            final_message = stream.get_final_message()
    except Exception as e:  # network drop, etc.
        yield {"type": "error", "message": repr(e)}
        return

    text = "".join(accumulated)
    cited_indices = _extract_cited_indices(text)
    citations = [
        _citation_record(chunks[i - 1], i) for i in cited_indices if 1 <= i <= len(chunks)
    ]
    refused = text.strip().startswith("The provided guidelines do not directly address this.")
    usage = {
        "input_tokens": getattr(final_message.usage, "input_tokens", None),
        "output_tokens": getattr(final_message.usage, "output_tokens", None),
        "cache_creation_input_tokens": getattr(
            final_message.usage, "cache_creation_input_tokens", 0
        ),
        "cache_read_input_tokens": getattr(
            final_message.usage, "cache_read_input_tokens", 0
        ),
    }
    verification = verify_answer(text, chunks)

    yield {
        "type": "done",
        "result": {
            "answer": text,
            "chunks": chunks,
            "citations": citations,
            "model": model,
            "usage": usage,
            "refused": refused,
            "verification": verification,
        },
    }
