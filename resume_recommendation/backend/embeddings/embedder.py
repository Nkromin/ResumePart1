"""
embedder.py
───────────
Thin wrapper around SentenceTransformer that exposes a single
`generate_embeddings()` coroutine safe for use inside FastAPI async routes.

The model is loaded once at module import time (singleton pattern) so every
request reuses the same in-memory model — no repeated cold-start cost.
CPU-bound `.encode()` is offloaded to a thread-pool executor so the async
event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── model singleton ───────────────────────────────────────────────────────────

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

logger.info("Loading SentenceTransformer model '%s' …", MODEL_NAME)
_model = SentenceTransformer(MODEL_NAME)
logger.info("Model loaded (dim=%d).", EMBEDDING_DIM)


# ── public API ────────────────────────────────────────────────────────────────

async def generate_embeddings(texts: list[str]) -> np.ndarray:
    """
    Asynchronously generate L2-normalised sentence embeddings.

    Runs the CPU-bound `encode()` call in the default thread-pool executor so
    the FastAPI event loop stays free to handle other requests concurrently.

    Parameters
    ----------
    texts : list[str]
        Non-empty list of document strings to embed.

    Returns
    -------
    np.ndarray
        Shape ``(len(texts), 384)``, dtype ``float32``.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    loop = asyncio.get_running_loop()

    # Wrap the synchronous encode() so it can be passed to run_in_executor
    def sync_encode():
        return _model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # unit-length vectors → cosine ≈ L2
        )

    embeddings: np.ndarray = await loop.run_in_executor(None, sync_encode)
    logger.debug("Generated embeddings: shape=%s", embeddings.shape)
    return embeddings.astype(np.float32)

