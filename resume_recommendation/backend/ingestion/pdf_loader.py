"""
pdf_loader.py
─────────────
Responsible for extracting raw text from PDF (and plain-text) resume files.
Only pdfplumber is used — no LLM or NLP logic here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)


def extract_text(filepath: str | Path) -> str:
    """
    Extract and return all text from a PDF file.

    Falls back to reading as UTF-8 plain text if the file is not a PDF.

    Parameters
    ----------
    filepath : str | Path
        Absolute or relative path to the file.

    Returns
    -------
    str
        Concatenated text from every page, stripped of leading/trailing
        whitespace.  Returns an empty string if nothing could be extracted.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix == ".pdf":
        return _extract_from_pdf(filepath)
    elif suffix in {".txt", ".text"}:
        return _extract_from_text(filepath)
    else:
        # Best-effort: try PDF first, then plain text
        try:
            return _extract_from_pdf(filepath)
        except Exception:
            return _extract_from_text(filepath)


# ── private helpers ──────────────────────────────────────────────────────────

def _extract_from_pdf(filepath: Path) -> str:
    """Use pdfplumber to pull text from every page of a PDF."""
    pages: list[str] = []
    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                pages.append(text)
            else:
                logger.debug("Page %d of '%s' yielded no text.", page_num, filepath.name)

    return "\n".join(pages).strip()


def _extract_from_text(filepath: Path) -> str:
    """Read a plain-text file and return its contents."""
    return filepath.read_text(encoding="utf-8", errors="replace").strip()

