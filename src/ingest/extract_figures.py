"""Crop figure regions out of source PDFs into PNG files.

We crop with PyMuPDF (``fitz``) rather than rasterizing whole pages because
guideline figures are usually accompanied by surrounding prose that we don't
want to muddy the image with. The bbox we crop comes from
``unstructured``'s ``Image`` / ``FigureCaption`` element metadata; if the bbox
is missing or unusable we fall back to rendering the full page so the
caption-chunk still has *some* picture to show.

Coordinate spaces
-----------------
unstructured returns coordinates in a per-element layout pixel space whose
size is given by ``coordinates.layout_width`` / ``layout_height``. PyMuPDF
operates on PDF points (1/72 inch). We linearly remap layout pixels onto
PDF points using each page's ``rect`` so cropping works regardless of which
DPI unstructured rasterized at.

Output naming
-------------
``data/parsed/figures/{document_id}_p{page_num}_fig{n}.png``

``n`` is a 1-based per-document figure counter so the same page can host
multiple figures without collisions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Pad the cropped region so we don't cut off the top of a flowchart, axis
# labels, etc. Tuned to be generous; PyMuPDF clips at the page rect anyway.
_BBOX_PAD_POINTS = 12.0

# Resolution for the rasterized PNG. 200 DPI keeps text in the figure
# readable without producing absurdly large files.
_RENDER_DPI = 200


def crop_figure(
    pdf_path: Path | str,
    page_number: int,
    bbox_layout: Optional[tuple[float, float, float, float]],
    layout_width: Optional[float],
    layout_height: Optional[float],
    output_path: Path | str,
) -> bool:
    """Crop a figure region from one PDF page to a PNG. Returns True on success.

    If ``bbox_layout`` is missing or invalid, renders the full page instead so
    the caller still has a picture to associate with the caption chunk.
    """
    import fitz  # PyMuPDF; lazy-imported to keep module-import cheap

    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        logger.warning("Could not open %s for figure extraction: %s", pdf_path, e)
        return False

    try:
        # PyMuPDF pages are 0-indexed; unstructured page_number is 1-indexed.
        page_idx = max(0, page_number - 1)
        if page_idx >= doc.page_count:
            logger.warning(
                "Page %d out of range for %s (has %d pages)",
                page_number, pdf_path.name, doc.page_count,
            )
            return False
        page = doc[page_idx]

        clip = _to_pdf_rect(page.rect, bbox_layout, layout_width, layout_height)
        if clip is None:
            # Fall back to the full page render — the user explicitly approved
            # this fallback in the Phase 2 spec.
            logger.info(
                "No usable bbox for figure on page %d of %s; rendering full page",
                page_number, pdf_path.name,
            )
            pix = page.get_pixmap(dpi=_RENDER_DPI, clip=None)
        else:
            pix = page.get_pixmap(dpi=_RENDER_DPI, clip=clip)

        pix.save(str(output_path))
        return True
    except Exception as e:
        logger.warning(
            "Figure crop failed for page %d of %s: %s", page_number, pdf_path.name, e,
        )
        return False
    finally:
        doc.close()


def _to_pdf_rect(
    page_rect,
    bbox_layout: Optional[tuple[float, float, float, float]],
    layout_width: Optional[float],
    layout_height: Optional[float],
):
    """Map an unstructured layout-space bbox onto PyMuPDF page coords.

    Returns a ``fitz.Rect`` or None if the bbox is missing/degenerate.
    """
    import fitz

    if not bbox_layout:
        return None
    x0, y0, x1, y1 = bbox_layout
    if x1 <= x0 or y1 <= y0:
        return None

    if layout_width and layout_height and layout_width > 0 and layout_height > 0:
        # Linear remap layout px -> PDF points.
        sx = page_rect.width / layout_width
        sy = page_rect.height / layout_height
        rx0, rx1 = x0 * sx, x1 * sx
        ry0, ry1 = y0 * sy, y1 * sy
    else:
        # Already PDF coords (or close enough).
        rx0, rx1, ry0, ry1 = x0, x1, y0, y1

    rect = fitz.Rect(
        max(page_rect.x0, rx0 - _BBOX_PAD_POINTS),
        max(page_rect.y0, ry0 - _BBOX_PAD_POINTS),
        min(page_rect.x1, rx1 + _BBOX_PAD_POINTS),
        min(page_rect.y1, ry1 + _BBOX_PAD_POINTS),
    )
    if rect.is_empty or rect.is_infinite:
        return None
    return rect
