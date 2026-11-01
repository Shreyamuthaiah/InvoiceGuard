"""Text extraction from digital PDFs with PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import pymupdf


def pdf_to_text(path: str | Path) -> str:
    """Reading-order text of every page. `sort=True` orders blocks top-to-bottom,
    left-to-right, which keeps table rows and label/value pairs together."""
    with pymupdf.open(path) as doc:
        pages = [page.get_text("text", sort=True).strip() for page in doc]
    if len(pages) == 1:
        return pages[0]
    return "\n\n".join(f"--- page {i} ---\n{t}" for i, t in enumerate(pages, 1))


def has_text_layer(path: str | Path, min_chars: int = 50) -> bool:
    """False for image-only PDFs (scans), which need the vision path."""
    with pymupdf.open(path) as doc:
        return sum(len(p.get_text("text").strip()) for p in doc) >= min_chars
