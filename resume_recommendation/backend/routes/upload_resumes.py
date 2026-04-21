"""
routes/upload_resumes.py
────────────────────────
POST /upload_resumes

Integrated resume ingestion pipeline using Ajay's hybrid matching system.

Flow:
1. Save uploaded files → data/resumes/
2. Trigger Ajay's embedding pipeline → generate artifacts
3. Return summary

This replaces the old FAISS-only approach with Ajay's full pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload_resumes", tags=["Resumes"])

RESUME_DIR = Path(__file__).resolve().parents[3] / "data" / "resumes"
JD_DIR = Path(__file__).resolve().parents[3] / "data" / "jd"
ALLOWED_SUFFIXES = {".pdf", ".txt", ".text"}



@router.post(
    "",
    summary="Upload and ingest resume files",
    status_code=status.HTTP_200_OK,
)
async def upload_resumes(
    files: list[UploadFile] = File(..., description="One or more resume PDFs/TXTs"),
) -> JSONResponse:
    """
    Ingest resumes into Ajay's hybrid matching system.

    This endpoint:
    1. Saves uploaded resume files
    2. Triggers the complete embedding pipeline
    3. Generates all required artifacts for matching
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded.",
        )

    RESUME_DIR.mkdir(parents=True, exist_ok=True)
    JD_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Validate & save files ─────────────────────────────────────────────
    saved_count = 0
    for upload in files:
        _validate_file_type(upload.filename)
        dest = RESUME_DIR / upload.filename  # type: ignore[operator]
        contents = await upload.read()
        dest.write_bytes(contents)
        saved_count += 1
        logger.info("Saved resume: '%s' (%d bytes).", upload.filename, len(contents))

    # ── 2. Run embedding pipeline in background ──────────────────────────────
    try:
        # Import here to avoid circular dependencies
        from backend.ajay_integration.embed_index import generate_artifacts

        # Run in thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        summary = await loop.run_in_executor(
            None,
            generate_artifacts,
            str(RESUME_DIR),
            str(JD_DIR)
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "message": f"Successfully ingested {saved_count} resume(s)",
                "pipeline_summary": summary,
            },
        )

    except Exception as exc:
        logger.error(f"Pipeline execution failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {str(exc)}",
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _validate_file_type(filename: str | None) -> None:
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )



