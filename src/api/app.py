"""FastAPI backend for the Phase 4 clinician UI.

The handlers are intentionally thin — they call straight into
:mod:`src.retrieve`, :mod:`src.generate`, and :mod:`src.api.history`. There
is no business logic here that isn't also exercised by the CLI scripts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
# Must run BEFORE importing the answer/retrieve modules — those grab
# ANTHROPIC_API_KEY / VOYAGE_API_KEY / COHERE_API_KEY at first call.
# override=True so the .env file wins over an empty/stale value the
# parent shell may have exported (e.g. ANTHROPIC_API_KEY="" inherited
# from a wrapper); without it, load_dotenv silently refuses to
# overwrite an already-set var and the request fails with a confusing
# 500 from deep inside the embedder/Anthropic client.
load_dotenv(REPO_ROOT / ".env", override=True)

from src.api import history  # noqa: E402
from src.generate import answer, answer_stream  # noqa: E402
from src.retrieve import retrieve  # noqa: E402

app = FastAPI(title="GI Guidelines RAG API", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev — Streamlit fetches from localhost
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- request/response shapes ------------------------------------------------


class Filters(BaseModel):
    society: Optional[list[str]] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    topic: Optional[list[str]] = None
    element_types: Optional[list[str]] = None
    doc_type: Optional[list[str]] = None  # 'Guideline' | 'Guidance' | 'Standards' | 'Other'


class RetrieveRequest(BaseModel):
    query: str
    filters: Optional[Filters] = None
    top_k: int = Field(default=6, ge=1, le=30)
    rerank: bool = True


class AnswerRequest(BaseModel):
    query: str
    filters: Optional[Filters] = None
    top_k: int = Field(default=6, ge=1, le=20)
    rerank: bool = True
    stream: bool = False
    conversation_id: Optional[int] = None
    save: bool = True


class NewConversationRequest(BaseModel):
    title: Optional[str] = None


# --- helpers ---------------------------------------------------------------


def _filters_dict(f: Optional[Filters]) -> Optional[dict[str, Any]]:
    if f is None:
        return None
    out = f.model_dump(exclude_none=True)
    return out or None


# --- routes ----------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    """DB liveness + presence of API keys (boolean only — no values)."""
    db_ok = False
    chunk_count = 0
    try:
        url = os.environ.get(
            "DATABASE_URL", "postgresql://gi:gi@localhost:5432/gi_guidelines"
        )
        with psycopg.connect(url, connect_timeout=2) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
            chunk_count = cur.fetchone()[0]
            db_ok = True
    except Exception:
        pass

    return {
        "db_ok": db_ok,
        "chunk_count": chunk_count,
        "voyage_key_set": bool(os.environ.get("VOYAGE_API_KEY")),
        "cohere_key_set": bool(os.environ.get("COHERE_API_KEY")),
        "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


@app.post("/retrieve")
def retrieve_endpoint(req: RetrieveRequest) -> dict[str, Any]:
    rows = retrieve(
        req.query,
        filters=_filters_dict(req.filters),
        top_k=req.top_k,
        rerank=req.rerank,
    )
    # Strip the raw embedding (huge, useless to the UI) before returning.
    return {"chunks": [_strip_embedding(r) for r in rows]}


@app.post("/answer")
def answer_endpoint(req: AnswerRequest):
    filters = _filters_dict(req.filters)

    if req.stream:
        def event_source():
            final: dict[str, Any] | None = None
            for ev in answer_stream(
                req.query, filters=filters, top_k=req.top_k, rerank=req.rerank,
            ):
                if ev["type"] == "done":
                    final = ev["result"]
                yield f"data: {json.dumps(_event_safe(ev))}\n\n"
            if final and req.save and req.conversation_id is not None:
                try:
                    history.save_turn(req.conversation_id, req.query, final)
                except Exception as e:  # don't fail the stream over a save
                    yield f"data: {json.dumps({'type': 'warn', 'message': repr(e)})}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    out = answer(
        req.query, filters=filters, top_k=req.top_k, rerank=req.rerank,
    )
    if req.save and req.conversation_id is not None:
        history.save_turn(req.conversation_id, req.query, out)
    out = dict(out)
    out["chunks"] = [_strip_embedding(c) for c in out.get("chunks", [])]
    return out


@app.post("/conversations")
def new_conversation(req: NewConversationRequest) -> dict[str, Any]:
    cid = history.create_conversation(req.title)
    return {"id": cid}


@app.get("/conversations")
def list_convs(limit: int = 50) -> dict[str, Any]:
    return {"conversations": history.list_conversations(limit=limit)}


@app.get("/conversations/{conversation_id}")
def get_conv(conversation_id: int) -> dict[str, Any]:
    out = history.get_conversation(conversation_id)
    if out["meta"] is None:
        raise HTTPException(404, f"conversation {conversation_id} not found")
    return out


@app.delete("/conversations/{conversation_id}")
def delete_conv(conversation_id: int) -> dict[str, Any]:
    history.delete_conversation(conversation_id)
    return {"ok": True}


@app.get("/sources/chunk/{chunk_id}")
def chunk_metadata(chunk_id: int) -> dict[str, Any]:
    url = os.environ.get(
        "DATABASE_URL", "postgresql://gi:gi@localhost:5432/gi_guidelines"
    )
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.document_id, d.society, d.year, d.title, d.doi,
                   d.source_url, d.pdf_path, c.section_title, c.recommendation_id,
                   c.grade_evidence, c.grade_strength, c.element_type,
                   c.page_start, c.page_end, c.text, c.table_html,
                   c.figure_image_path
              FROM chunks c JOIN documents d ON d.id = c.document_id
             WHERE c.id = %s
            """,
            (chunk_id,),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(404, f"chunk {chunk_id} not found")
    cols = [
        "chunk_id", "document_id", "society", "year", "title", "doi",
        "source_url", "pdf_path", "section_title", "recommendation_id",
        "grade_evidence", "grade_strength", "element_type",
        "page_start", "page_end", "text", "table_html", "figure_image_path",
    ]
    return dict(zip(cols, row))


@app.get("/sources/document/{document_id}/pdf")
def serve_pdf(document_id: int):
    """Serve the source PDF locally if available, else redirect to source_url."""
    url = os.environ.get(
        "DATABASE_URL", "postgresql://gi:gi@localhost:5432/gi_guidelines"
    )
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT pdf_path, source_url FROM documents WHERE id = %s",
            (document_id,),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(404, f"document {document_id} not found")
    pdf_path, source_url = row

    if pdf_path:
        local = (REPO_ROOT / pdf_path).resolve()
        try:
            local.relative_to(REPO_ROOT)  # path-traversal guard
        except ValueError:
            raise HTTPException(400, "invalid pdf_path")
        if local.exists():
            return FileResponse(local, media_type="application/pdf",
                                filename=local.name)

    if source_url:
        return RedirectResponse(source_url)

    raise HTTPException(404, "PDF not available locally and no source_url set")


@app.get("/sources/figure/{path:path}")
def serve_figure(path: str):
    """Serve a figure PNG from data/parsed/figures/. The path is relative to
    that directory; we strictly clamp inside it to prevent traversal."""
    base = (REPO_ROOT / "data" / "parsed" / "figures").resolve()
    target = (base / path).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(400, "invalid figure path")
    if not target.exists():
        raise HTTPException(404, "figure not found")
    return FileResponse(target, media_type="image/png")


# --- internals -------------------------------------------------------------


def _strip_embedding(chunk: dict[str, Any]) -> dict[str, Any]:
    """Strip pgvector embedding bytes from chunks before serializing."""
    out = {k: v for k, v in chunk.items() if k != "embedding"}
    return out


def _event_safe(ev: dict[str, Any]) -> dict[str, Any]:
    """Some streamed events carry chunk dicts with raw vectors. Strip them."""
    if "result" in ev and isinstance(ev["result"], dict):
        ev = dict(ev)
        ev["result"] = dict(ev["result"])
        if "chunks" in ev["result"]:
            ev["result"]["chunks"] = [_strip_embedding(c) for c in ev["result"]["chunks"]]
    return ev
