"""
embed_index.py
──────────────
Embedding and FAISS indexing module for Ajay's hybrid matching system.

Modified to work with PDF-based ingestion while preserving Ajay's
embedding format and indexing logic.

Key responsibilities:
- Generate embeddings with proper "passage:" and "query:" prefixes
- Create FAISS IndexFlatIP (cosine similarity via normalized embeddings)
- Generate all artifacts expected by match.py:
  * artifacts/faiss.index
  * artifacts/chunk_embeddings.npy
  * artifacts/chunk_meta.parquet
  * artifacts/job_embeddings.npy
  * artifacts/jobs.parquet
  * artifacts/skill_vocab.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from backend.ajay_integration.preprocessing import (
    parse_resume_pdfs,
    parse_jd_pdfs,
    resume_to_sections,
    chunk_sections,
    extract_skill_vocabulary,
)

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
ARTIFACTS_DIR = Path.cwd() / "artifacts"

# ── Embedding model singleton ─────────────────────────────────────────────────

logger.info(f"Loading embedding model '{MODEL_NAME}'...")
_model = SentenceTransformer(MODEL_NAME)
logger.info("Model loaded successfully")


# ── Embedding functions (Ajay's format) ───────────────────────────────────────

def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Returns L2-normalized embeddings (unit vectors) for cosine similarity.

    Parameters
    ----------
    texts : List[str]
        List of text strings to embed
    batch_size : int
        Batch size for encoding

    Returns
    -------
    np.ndarray
        Shape (len(texts), 384), dtype float32, L2-normalized
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    embeddings = _model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # We'll normalize manually for clarity
    )

    # Convert to float32 and normalize
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)  # In-place L2 normalization

    logger.info(f"Generated {len(embeddings)} embeddings (shape={embeddings.shape})")
    return embeddings


def embed_resume_chunks(chunks_df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Embed resume chunks with proper "passage: {section}: {chunk_text}" format.

    This is Ajay's exact embedding format for resume chunks.

    Parameters
    ----------
    chunks_df : pd.DataFrame
        DataFrame with columns: resume_id, section, chunk_id, chunk_text

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        - embeddings: shape (n_chunks, 384)
        - metadata: DataFrame with resume_id, section, chunk_id, chunk_text
    """
    # Format texts with Ajay's "passage:" prefix
    formatted_texts = [
        f"passage: {row['section']}: {row['chunk_text']}"
        for _, row in chunks_df.iterrows()
    ]

    embeddings = embed_texts(formatted_texts)

    # Create metadata DataFrame
    metadata = chunks_df[["resume_id", "section", "chunk_id", "chunk_text"]].copy()

    return embeddings, metadata


def embed_job_descriptions(jobs_df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Embed job descriptions with proper "query: {job_text}" format.

    This is Ajay's exact embedding format for job queries.

    Parameters
    ----------
    jobs_df : pd.DataFrame
        DataFrame with columns: job_id, job_title, job_text

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        - embeddings: shape (n_jobs, 384)
        - metadata: jobs DataFrame
    """
    # Format texts with Ajay's "query:" prefix
    formatted_texts = [
        f"query: {row['job_text']}"
        for _, row in jobs_df.iterrows()
    ]

    embeddings = embed_texts(formatted_texts)

    return embeddings, jobs_df.copy()


# ── FAISS indexing (Ajay's approach) ──────────────────────────────────────────

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Create a FAISS IndexFlatIP for cosine similarity search.

    Ajay uses IndexFlatIP (inner product) with normalized embeddings,
    which is equivalent to cosine similarity.

    Parameters
    ----------
    embeddings : np.ndarray
        L2-normalized embeddings, shape (n, 384)

    Returns
    -------
    faiss.IndexFlatIP
        FAISS index with embeddings added
    """
    index = faiss.IndexFlatIP(EMBEDDING_DIM)

    if len(embeddings) > 0:
        # Ensure embeddings are normalized (defensive)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

    logger.info(f"Created FAISS index with {index.ntotal} vectors")
    return index


# ── Artifact generation ───────────────────────────────────────────────────────

def generate_artifacts(resume_folder: str | Path, jd_folder: str | Path) -> dict:
    """
    Generate all artifacts required by Ajay's match.py retrieval engine.

    This is the main entry point for the embedding pipeline.

    Parameters
    ----------
    resume_folder : str | Path
        Folder containing resume PDF files
    jd_folder : str | Path
        Folder containing job description PDF files

    Returns
    -------
    dict
        Summary of generated artifacts
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Starting embedding pipeline...")
    logger.info("=" * 80)

    # ── 1. Parse PDFs ─────────────────────────────────────────────────────────
    logger.info("Step 1: Parsing resume PDFs...")
    resume_df = parse_resume_pdfs(resume_folder)
    logger.info(f"Parsed {len(resume_df)} resumes")

    logger.info("Step 2: Parsing job description PDFs...")
    jobs_df = parse_jd_pdfs(jd_folder)
    logger.info(f"Parsed {len(jobs_df)} job descriptions")

    if len(resume_df) == 0:
        raise ValueError("No resumes found. Cannot proceed with embedding pipeline.")

    if len(jobs_df) == 0:
        raise ValueError("No job descriptions found. Cannot proceed with embedding pipeline.")

    # Store original resume data for name mapping
    resume_names = resume_df[['resume_id', 'candidate_name']].copy() if 'candidate_name' in resume_df.columns else None

    # ── 2. Extract skill vocabulary ───────────────────────────────────────────
    logger.info("Step 3: Extracting skill vocabulary...")
    skill_vocab = extract_skill_vocabulary(resume_df, jobs_df)

    skill_vocab_path = ARTIFACTS_DIR / "skill_vocab.json"
    with open(skill_vocab_path, "w") as f:
        json.dump(list(skill_vocab), f, indent=2)
    logger.info(f"Saved skill vocabulary ({len(skill_vocab)} skills) to {skill_vocab_path}")

    # ── 3. Process resumes into sections and chunks ───────────────────────────
    logger.info("Step 4: Converting resumes to sections...")
    sections_df = resume_to_sections(resume_df)
    logger.info(f"Generated {len(sections_df)} sections")

    logger.info("Step 5: Chunking sections...")
    chunks_df = chunk_sections(sections_df, max_chunk_length=512, overlap=50)
    logger.info(f"Generated {len(chunks_df)} chunks")

    # ── 4. Embed resume chunks ────────────────────────────────────────────────
    logger.info("Step 6: Embedding resume chunks...")
    chunk_embeddings, chunk_meta = embed_resume_chunks(chunks_df)

    # Merge candidate names into chunk metadata
    if resume_names is not None:
        chunk_meta = chunk_meta.merge(resume_names, on='resume_id', how='left')
        # Fill missing names with resume_id
        chunk_meta['candidate_name'] = chunk_meta['candidate_name'].fillna(chunk_meta['resume_id'])

    # Save chunk embeddings
    chunk_emb_path = ARTIFACTS_DIR / "chunk_embeddings.npy"
    np.save(chunk_emb_path, chunk_embeddings)
    logger.info(f"Saved chunk embeddings to {chunk_emb_path}")

    # Save chunk metadata
    chunk_meta_path = ARTIFACTS_DIR / "chunk_meta.parquet"
    chunk_meta.to_parquet(chunk_meta_path, index=False)
    logger.info(f"Saved chunk metadata to {chunk_meta_path}")

    # ── 5. Create FAISS index ─────────────────────────────────────────────────
    logger.info("Step 7: Creating FAISS index...")
    faiss_index = create_faiss_index(chunk_embeddings)

    # Save FAISS index
    faiss_path = ARTIFACTS_DIR / "faiss.index"
    faiss.write_index(faiss_index, str(faiss_path))
    logger.info(f"Saved FAISS index to {faiss_path}")

    # ── 6. Embed job descriptions ─────────────────────────────────────────────
    logger.info("Step 8: Embedding job descriptions...")
    job_embeddings, jobs_meta = embed_job_descriptions(jobs_df)

    # Save job embeddings
    job_emb_path = ARTIFACTS_DIR / "job_embeddings.npy"
    np.save(job_emb_path, job_embeddings)
    logger.info(f"Saved job embeddings to {job_emb_path}")

    # Save job metadata
    jobs_path = ARTIFACTS_DIR / "jobs.parquet"
    jobs_meta.to_parquet(jobs_path, index=False)
    logger.info(f"Saved job metadata to {jobs_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("Embedding pipeline completed successfully!")
    logger.info("=" * 80)

    summary = {
        "num_resumes": len(resume_df),
        "num_jobs": len(jobs_df),
        "num_chunks": len(chunks_df),
        "num_skills": len(skill_vocab),
        "artifacts": {
            "faiss_index": str(faiss_path),
            "chunk_embeddings": str(chunk_emb_path),
            "chunk_metadata": str(chunk_meta_path),
            "job_embeddings": str(job_emb_path),
            "job_metadata": str(jobs_path),
            "skill_vocab": str(skill_vocab_path),
        }
    }

    return summary


# ── Standalone script mode ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python embed_index.py <resume_folder> <jd_folder>")
        sys.exit(1)

    resume_folder = sys.argv[1]
    jd_folder = sys.argv[2]

    summary = generate_artifacts(resume_folder, jd_folder)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2))

