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
    """

    category: str          # unstructured element category, e.g. "Title", "NarrativeText"
    text: str
    page_number: Optional[int] = None
    metadata: dict = field(default_factory=dict)


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

    elements = _try_parse_with_timeout(
        str(pdf_path), strategy, infer_table_structure, fallback_after_seconds
    )

    parsed: list[ParsedElement] = []
    for el in elements:
        # element.category is set by unstructured; element.metadata has page numbers, etc.
        meta = getattr(el, "metadata", None)
        meta_dict = meta.to_dict() if meta is not None else {}
        parsed.append(
            ParsedElement(
                category=getattr(el, "category", el.__class__.__name__),
                text=str(el).strip(),
                page_number=meta_dict.get("page_number"),
                metadata=meta_dict,
            )
        )

    # Drop empties (unstructured occasionally emits whitespace-only elements).
    parsed = [p for p in parsed if p.text]
    logger.info("Parsed %s into %d elements", pdf_path.name, len(parsed))
    return parsed


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

    def _run(conn):  # pragma: no cover — runs in child
        try:
            result = _parse_with_strategy(pdf_path, strategy, infer_tables)
            conn.send(("ok", result))
        except Exception as e:
            conn.send(("err", repr(e)))
        finally:
            conn.close()

    proc = ctx.Process(target=_run, args=(child_conn,))
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
