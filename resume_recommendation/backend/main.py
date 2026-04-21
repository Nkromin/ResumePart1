"""
backend/main.py
───────────────
FastAPI application entry-point for the ingestion pipeline.

Start the server:
    uvicorn backend.main:app --reload --app-dir resume_recommendation

Or from the project root:
    uvicorn resume_recommendation.backend.main:app --reload
"""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Resume Recommendation — Ingestion API",
    description=(
        "Handles Job Description uploads and resume ingestion "
        "(PDF text extraction → SentenceTransformer embeddings → FAISS index)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow the Streamlit frontend (default port 8501) ──────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "*",   # loosen for local dev; tighten in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── routers ───────────────────────────────────────────────────────────────────
from backend.routes.upload_jd import router as jd_router          # noqa: E402
from backend.routes.upload_resumes import router as resumes_router  # noqa: E402
from backend.routes.search import router as search_router          # noqa: E402

app.include_router(jd_router)
app.include_router(resumes_router)
app.include_router(search_router)


# ── health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"], summary="Liveness probe")
async def health() -> dict:
    """Returns 200 OK if the server is running."""
    return {"status": "ok", "service": "ingestion-api"}


# ── startup banner ────────────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup() -> None:
    logger.info("Ingestion API ready.  Docs → http://localhost:8000/docs")

