"""PDF parsing for GI clinical guidelines.

We use ``unstructured`` with ``strategy="hi_res"`` because guidelines have:
  - dense multi-column layouts that ``fast`` mangles
  - inline tables of recommendations / GRADE evidence that we want to keep intact
  - headers and titles whose styling is the only signal of section boundaries

If hi_res takes longer than ``parsing.fallback_to_fast_after_seconds`` we fall
back to ``fast`` and warn loudly. hi_res requires Tesseract + Poppler + a
detectron2 model — see the README for system-dep install instructions.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedElement:
    """One semantic element from the PDF (title, text block, list, table cell).

    The chunker downstream consumes these. We keep the unstructured category
    (``Title``, ``NarrativeText``, ``ListItem``, ``Table``) because section-aware
    chunking keys off ``Title`` boundaries.

    For ``Table`` elements, ``table_html`` carries the structured HTML
    representation that ``unstructured`` produces when
    ``infer_table_structure=True``. We embed the plain-text rendering but
    keep the HTML so the UI can re-render structured tables later.

    For ``FigureCaption`` / ``Image`` elements, ``bbox`` carries the
    page-coordinate rectangle (x0, y0, x1, y1) used by
    :mod:`src.ingest.extract_figures` to crop the figure region into a PNG.
    """

    category: str          # unstructured element category, e.g. "Title", "NarrativeText"
    text: str
    page_number: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    table_html: Optional[str] = None        # populated when category == "Table"
    bbox: Optional[tuple[float, float, float, float]] = None  # x0,y0,x1,y1 (PDF coords)


def _parse_with_strategy(pdf_path: str, strategy: str, infer_tables: bool) -> list:
    """Worker that runs unstructured. Imported lazily so the module is cheap to import."""
    from unstructured.partition.pdf import partition_pdf

    return partition_pdf(
        filename=pdf_path,
        strategy=strategy,
        infer_table_structure=infer_tables,
        # extract_images_in_pdf=False keeps memory in check on long guidelines
        extract_images_in_pdf=False,
    )


def parse_pdf(
    pdf_path: Path | str,
    strategy: str = "hi_res",
    infer_table_structure: bool = True,
    fallback_after_seconds: int = 600,
) -> list[ParsedElement]:
    """Parse a PDF into a list of ``ParsedElement``.

    Args:
        pdf_path: filesystem path to the PDF.
        strategy: ``hi_res`` (preferred, slow) or ``fast`` (fallback).
        infer_table_structure: keep tables as structured elements vs. flattened text.
        fallback_after_seconds: if hi_res does not finish in this many seconds,
            kill it and retry with ``fast``. The default of 600s (10 min) is
            generous because a 50-page guideline can take 2-5 min on a laptop.

    Returns:
        A list of ParsedElement, in document order.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # The subprocess-based timeout (``_try_parse_with_timeout``) is opt-in via
    # the env var ``GI_PARSE_USE_TIMEOUT_SUBPROCESS=1``. By default we call
    # unstructured directly because on macOS the spawn child reliably hangs in
    # table-transformer inference on PDFs that the SAME code parses fine in the
    # main process (~30s). When build_index runs with --workers > 1 the pool
    # worker IS already a subprocess, so a second-level subprocess would just
    # nest hangs without adding safety.
    import os
    if os.environ.get("GI_PARSE_USE_TIMEOUT_SUBPROCESS") == "1":
        elements = _try_parse_with_timeout(
            str(pdf_path), strategy, infer_table_structure, fallback_after_seconds
        )
    else:
        try:
            elements = _parse_with_strategy(str(pdf_path), strategy, infer_table_structure)
        except Exception as e:
            logger.warning("hi_res failed (%r) — falling back to fast", e)
            elements = _parse_with_strategy(str(pdf_path), "fast", infer_table_structure)

    parsed: list[ParsedElement] = []
    for el in elements:
        # element.category is set by unstructured; element.metadata has page numbers, etc.
        meta = getattr(el, "metadata", None)
        meta_dict = meta.to_dict() if meta is not None else {}
        category = getattr(el, "category", el.__class__.__name__)
        parsed.append(
            ParsedElement(
                category=category,
                text=str(el).strip(),
                page_number=meta_dict.get("page_number"),
                metadata=meta_dict,
                table_html=meta_dict.get("text_as_html") if category == "Table" else None,
                bbox=_extract_bbox(meta_dict),
            )
        )

    # Drop empties (unstructured occasionally emits whitespace-only elements).
    # Tables can be empty-text-but-HTML-only — keep those.
    parsed = [p for p in parsed if p.text or p.table_html]
    logger.info("Parsed %s into %d elements", pdf_path.name, len(parsed))
    return parsed


def _extract_bbox(meta_dict: dict) -> Optional[tuple[float, float, float, float]]:
    """Pull (x0, y0, x1, y1) in PDF points from unstructured's coordinates dict.

    unstructured stores coordinates as a list of (x, y) corner tuples in
    page-local pixel space, with a ``layout_width`` / ``layout_height`` that
    we ignore here — :mod:`src.ingest.extract_figures` re-projects from the
    layout coords onto the actual PDF page when cropping.
    """
    coords = meta_dict.get("coordinates")
    if not coords:
        return None
    pts = coords.get("points")
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _run_in_subprocess(conn, pdf_path: str, strategy: str, infer_tables: bool):  # pragma: no cover — child
    # Module-level so the spawn context can pickle and re-import it.
    #
    # Single-thread torch / OpenMP / MKL inside the child. The spawn child
    # ends up deadlocking on macOS during table-transformer inference when
    # these default to multi-threaded — empirically we hit a >10 min hang
    # on PDFs that the SAME code parses in ~30s in the main process.
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch  # noqa: F401  — set thread count if torch is available
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        result = _parse_with_strategy(pdf_path, strategy, infer_tables)
        conn.send(("ok", result))
    except Exception as e:
        conn.send(("err", repr(e)))
    finally:
        conn.close()


def _try_parse_with_timeout(
    pdf_path: str, strategy: str, infer_tables: bool, timeout_s: int
) -> list:
    """Run unstructured in a subprocess so we can time-bomb hi_res."""
    if strategy == "fast":
        # No need to subprocess for the fallback path.
        return _parse_with_strategy(pdf_path, "fast", infer_tables)

    start = time.time()
    logger.info("Parsing %s with strategy=%s (timeout=%ds)", pdf_path, strategy, timeout_s)

    # multiprocessing with 'spawn' avoids fork-related issues with detectron2/torch.
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    proc = ctx.Process(
        target=_run_in_subprocess,
        args=(child_conn, pdf_path, strategy, infer_tables),
    )
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
        logger.warning(
            "hi_res exceeded %ds on %s — falling back to fast strategy",
            timeout_s, pdf_path,
        )
        return _parse_with_strategy(pdf_path, "fast", infer_tables)

    if parent_conn.poll():
        status, payload = parent_conn.recv()
        if status == "ok":
            logger.info("hi_res completed in %.1fs", time.time() - start)
            return payload
        logger.warning("hi_res failed (%s) — falling back to fast", payload)
        return _parse_with_strategy(pdf_path, "fast", infer_tables)

    logger.warning("hi_res produced no output — falling back to fast")
    return _parse_with_strategy(pdf_path, "fast", infer_tables)
