"""Grounded answer generator. Calls retrieve() then Claude.

Returns a dict so the CLI / future API layer can show both the prose answer
and the per-citation metadata side-by-side.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.generate.prompt import SYSTEM_PROMPT, build_user_message
from src.generate.verify import verify_answer
from src.retrieve import retrieve

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@lru_cache(maxsize=1)
def _load_config(path: str = str(DEFAULT_CONFIG)) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _get_anthropic_client():
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; cannot generate grounded answers."
        )
    return anthropic.Anthropic(api_key=api_key)


def answer(
    question: str,
    *,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    rerank: bool = True,
    model: str | None = None,
) -> dict[str, Any]:
    """Run retrieve → Claude → return a structured answer.

    Returns a dict:
        {
          "answer": str,                  # the prose answer with [N] citations
          "chunks": list[dict],           # the retrieved chunks (rank order)
          "citations": list[dict],        # only the chunks the answer cited
          "model": str,                   # model id used
          "usage": dict,                  # token counts incl. cache hits
          "refused": bool,                # True if the model said it can't answer
        }
    """
    cfg = _load_config()
    gen_cfg = cfg.get("generation", {})
    model = model or gen_cfg.get("model", "claude-sonnet-4-5")
    max_tokens = int(gen_cfg.get("max_tokens", 1500))
    temperature = float(gen_cfg.get("temperature", 0.0))

    rcfg = cfg.get("retrieval", {})
    top_k = int(top_k or rcfg.get("top_k", 6))

    chunks = retrieve(question, filters=filters, top_k=top_k, rerank=rerank)
    if not chunks:
        return {
            "answer": "The provided guidelines do not directly address this. "
                      "(No chunks were retrieved — check filters or corpus state.)",
            "chunks": [],
            "citations": [],
            "model": model,
            "usage": {},
            "refused": True,
        }

    user_msg = build_user_message(question, chunks)
    client = _get_anthropic_client()

    # Claude Opus 4.7 (and other 4.x extended-thinking-class models) reject
    # the temperature parameter. We pass it for older models only.
    create_kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        # Cache the (large, static) system prompt so re-asking different
        # questions in the same session doesn't re-bill the preamble.
        "system": [{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        "messages": [{"role": "user", "content": user_msg}],
    }
    if not _model_rejects_temperature(model):
        create_kwargs["temperature"] = temperature
    resp = client.messages.create(**create_kwargs)

    text = "".join(block.text for block in resp.content if block.type == "text")
    cited_indices = _extract_cited_indices(text)
    citations = [
        _citation_record(chunks[i - 1], i) for i in cited_indices if 1 <= i <= len(chunks)
    ]
    refused = text.strip().startswith("The provided guidelines do not directly address this.")

    usage = {
        "input_tokens": getattr(resp.usage, "input_tokens", None),
        "output_tokens": getattr(resp.usage, "output_tokens", None),
        "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0),
        "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0),
    }

    verification = verify_answer(text, chunks)

    return {
        "answer": text,
        "chunks": chunks,
        "citations": citations,
        "model": model,
        "usage": usage,
        "refused": refused,
        "verification": verification,
    }


def _model_rejects_temperature(model: str) -> bool:
    """Whether ``temperature`` is a deprecated/rejected kwarg for this model.

    Claude 4.x models in the extended-thinking class (Opus 4.7, Sonnet 4.6,
    Haiku 4.5) reject ``temperature``. Older models still accept it.
    """
    return any(model.startswith(p) for p in (
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ))


def _extract_cited_indices(text: str) -> list[int]:
    """Pull the unique [N] citation numbers in the order they first appear."""
    seen: list[int] = []
    for match in _CITATION_RE.finditer(text):
        for piece in match.group(1).split(","):
            try:
                n = int(piece.strip())
            except ValueError:
                continue
            if n not in seen:
                seen.append(n)
    return seen


def _citation_record(chunk: dict[str, Any], idx: int) -> dict[str, Any]:
    """Trim a chunk to the fields a citation footnote needs.

    Includes everything the UI's source-viewer pane needs to render the
    cited content directly (full text, structured table HTML, figure
    image path) plus the IDs needed to deep-link back to the source PDF.
    """
    return {
        "n": idx,
        "chunk_id": chunk.get("chunk_id"),
        "document_id": chunk.get("document_id"),
        "society": chunk.get("society"),
        "year": chunk.get("year"),
        "title": chunk.get("title"),
        "section_title": chunk.get("section_title"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "recommendation_id": chunk.get("recommendation_id"),
        "grade_strength": chunk.get("grade_strength"),
        "grade_evidence": chunk.get("grade_evidence"),
        "element_type": chunk.get("element_type"),
        "doi": chunk.get("doi"),
        "source_url": chunk.get("source_url"),
        # Full source content for in-UI rendering.
        "text": chunk.get("text"),
        "table_html": chunk.get("table_html"),
        "figure_image_path": chunk.get("figure_image_path"),
    }
