"""
faiss_store.py
──────────────
All FAISS index operations for the ingestion pipeline.

Responsibilities
────────────────
* Create a new ``faiss.IndexFlatL2`` index
* Load an existing index from disk (if present)
* Append new embedding vectors to an existing index
* Persist the index + companion metadata (pickle) to disk
* Expose a clean search helper for teammates' use

Files managed
─────────────
faiss_index/resume_index.faiss   — binary FAISS index
faiss_index/meta.pkl             — list[str] of resume filenames, ordered by
                                   their position in the FAISS index
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 384
INDEX_DIR = Path.cwd() / "faiss_index"
INDEX_PATH = INDEX_DIR / "resume_index.faiss"
META_PATH = INDEX_DIR / "meta.pkl"


# ── index lifecycle ───────────────────────────────────────────────────────────

def create_index() -> faiss.IndexFlatL2:
    """Return a fresh, empty ``IndexFlatL2`` with dim=384."""
    return faiss.IndexFlatL2(EMBEDDING_DIM)


def load_index() -> tuple[faiss.IndexFlatL2, list[str]]:
    """
    Load the FAISS index and metadata from disk.

    Returns
    -------
    (index, metadata)
        ``index``    — the FAISS index (may be empty if no file exists)
        ``metadata`` — list of resume filenames already stored
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        logger.info("Loaded existing FAISS index (%d vectors).", index.ntotal)
    else:
        index = create_index()
        logger.info("No existing index found — starting fresh.")

    if META_PATH.exists():
        with META_PATH.open("rb") as fh:
            metadata: list[str] = pickle.load(fh)
    else:
        metadata = []

    return index, metadata  # type: ignore[return-value]


def append_embeddings(
    index: faiss.IndexFlatL2,
    metadata: list[str],
    embeddings: np.ndarray,
    filenames: list[str],
) -> None:
    """
    Add new vectors to *index* and extend *metadata* in-place.

    Parameters
    ----------
    index : faiss.IndexFlatL2
        Live FAISS index (mutated in-place).
    metadata : list[str]
        Existing list of stored filenames (mutated in-place).
    embeddings : np.ndarray
        Shape ``(n, 384)``, dtype ``float32``.
    filenames : list[str]
        Exactly ``n`` filenames, one per embedding row.
    """
    if embeddings.shape[0] == 0:
        return
    index.add(embeddings)
    metadata.extend(filenames)
    logger.info(
        "Appended %d vectors — index total: %d.",
        len(filenames),
        index.ntotal,
    )


def save_index(index: faiss.IndexFlatL2, metadata: list[str]) -> None:
    """Persist the FAISS index and metadata to ``faiss_index/``."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with META_PATH.open("wb") as fh:
        pickle.dump(metadata, fh)
    logger.info("Saved index (%d vectors) and metadata.", index.ntotal)


def purge_index() -> None:
    """Delete the on-disk index and metadata files (used for replace mode)."""
    for path in (INDEX_PATH, META_PATH):
        if path.exists():
            path.unlink()
            logger.info("Deleted '%s'.", path.name)


# ── search helper (for teammates) ────────────────────────────────────────────

def search(
    query_vector: np.ndarray,
    k: int = 10,
) -> tuple[list[str], list[float]]:
    """
    Search the on-disk FAISS index and return the top-k matches.

    Parameters
    ----------
    query_vector : np.ndarray
        Shape ``(384,)`` or ``(1, 384)``, dtype ``float32``.
    k : int
        Number of nearest neighbours to return.

    Returns
    -------
    (filenames, distances)
        Parallel lists of matched resume filenames and their L2 distances.
    """
    index, metadata = load_index()
    if index.ntotal == 0:
        return [], []

    vec = np.array(query_vector, dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(vec, min(k, index.ntotal))

    filenames = [metadata[i] for i in indices[0] if i != -1]
    dists = [float(d) for d, i in zip(distances[0], indices[0]) if i != -1]
    return filenames, dists

