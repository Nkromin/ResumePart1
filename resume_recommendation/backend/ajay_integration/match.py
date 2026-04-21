"""
match.py
────────
Ajay's hybrid resume matching engine.

This module implements the core retrieval and ranking logic:
1. FAISS semantic similarity search
2. Skill overlap scoring
3. Experience alignment
4. Domain matching
5. Hybrid score combination

PRESERVED EXACTLY from Ajay's original system.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Artifacts paths ───────────────────────────────────────────────────────────

ARTIFACTS_DIR = Path.cwd() / "artifacts"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss.index"
CHUNK_EMBEDDINGS_PATH = ARTIFACTS_DIR / "chunk_embeddings.npy"
CHUNK_META_PATH = ARTIFACTS_DIR / "chunk_meta.parquet"
JOB_EMBEDDINGS_PATH = ARTIFACTS_DIR / "job_embeddings.npy"
JOBS_PATH = ARTIFACTS_DIR / "jobs.parquet"
SKILL_VOCAB_PATH = ARTIFACTS_DIR / "skill_vocab.json"


# ── Summary generation ────────────────────────────────────────────────────────

def generate_match_summary(
    candidate_name: str,
    resume_skills: str,
    resume_experience: str,
    job_title: str,
    semantic_score: float,
    skill_score: float,
    experience_score: float,
    domain_score: float,
    matched_sections: list
) -> str:
    """
    Generate a human-readable summary explaining why a candidate matches a job.

    Returns HTML-formatted summary with proper bold tags.

    Parameters
    ----------
    candidate_name : str
        Name of the candidate
    resume_skills : str
        Skills from the resume
    resume_experience : str
        Experience from the resume
    job_title : str
        Job title
    semantic_score : float
        Semantic similarity score (0-1)
    skill_score : float
        Skill overlap score (0-1)
    experience_score : float
        Experience alignment score (0-1)
    domain_score : float
        Domain match score (0-1)
    matched_sections : list
        List of matched resume sections

    Returns
    -------
    str
        HTML-formatted match summary
    """
    summary_parts = []

    # Overall assessment
    hybrid = (0.5 * semantic_score + 0.3 * skill_score +
              0.1 * experience_score + 0.1 * domain_score)

    if hybrid >= 0.8:
        summary_parts.append(f"<strong>Excellent match</strong> for {job_title}.")
    elif hybrid >= 0.6:
        summary_parts.append(f"<strong>Strong match</strong> for {job_title}.")
    elif hybrid >= 0.4:
        summary_parts.append(f"<strong>Good match</strong> for {job_title}.")
    else:
        summary_parts.append(f"<strong>Potential match</strong> for {job_title}.")

    # Semantic similarity insights
    if semantic_score >= 0.7:
        summary_parts.append("Resume content closely aligns with job requirements.")
    elif semantic_score >= 0.5:
        summary_parts.append("Resume shows relevant background for this role.")

    # Skill insights
    if skill_score >= 0.7:
        summary_parts.append("Strong technical skill match.")
    elif skill_score >= 0.4:
        summary_parts.append("Has several required skills.")
    elif skill_score > 0:
        summary_parts.append("Has some relevant skills.")

    # Experience insights
    if experience_score >= 0.8:
        summary_parts.append("Experience level exceeds requirements.")
    elif experience_score >= 0.5:
        summary_parts.append("Meets experience requirements.")
    elif experience_score > 0:
        summary_parts.append("Has relevant experience.")

    # Domain insights
    if domain_score >= 0.7:
        summary_parts.append("Strong domain expertise.")

    # Matched sections
    if matched_sections:
        key_sections = [s for s in ['skills', 'experience', 'projects'] if s in matched_sections]
        if key_sections:
            summary_parts.append(f"Key matches in: {', '.join(key_sections)}.")

    return " ".join(summary_parts)


# ── Artifact loading ──────────────────────────────────────────────────────────

class ArtifactLoader:
    """Singleton pattern for loading Ajay's artifacts once."""

    _instance = None
    _loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._loaded:
            self.load_artifacts()
            self.__class__._loaded = True

    def load_artifacts(self):
        """Load all required artifacts."""
        logger.info("Loading artifacts...")

        # FAISS index
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_PATH}")
        self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")

        # Chunk embeddings
        if not CHUNK_EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(f"Chunk embeddings not found: {CHUNK_EMBEDDINGS_PATH}")
        self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
        logger.info(f"Loaded chunk embeddings: {self.chunk_embeddings.shape}")

        # Chunk metadata
        if not CHUNK_META_PATH.exists():
            raise FileNotFoundError(f"Chunk metadata not found: {CHUNK_META_PATH}")
        self.chunk_meta = pd.read_parquet(CHUNK_META_PATH)
        logger.info(f"Loaded chunk metadata: {len(self.chunk_meta)} chunks")

        # Job embeddings
        if not JOB_EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(f"Job embeddings not found: {JOB_EMBEDDINGS_PATH}")
        self.job_embeddings = np.load(JOB_EMBEDDINGS_PATH)
        logger.info(f"Loaded job embeddings: {self.job_embeddings.shape}")

        # Job metadata
        if not JOBS_PATH.exists():
            raise FileNotFoundError(f"Jobs metadata not found: {JOBS_PATH}")
        self.jobs = pd.read_parquet(JOBS_PATH)
        logger.info(f"Loaded jobs metadata: {len(self.jobs)} jobs")

        # Skill vocabulary
        if not SKILL_VOCAB_PATH.exists():
            raise FileNotFoundError(f"Skill vocabulary not found: {SKILL_VOCAB_PATH}")
        with open(SKILL_VOCAB_PATH, "r") as f:
            self.skill_vocab = set(json.load(f))
        logger.info(f"Loaded skill vocabulary: {len(self.skill_vocab)} skills")


# ── Matching engine ───────────────────────────────────────────────────────────

class HybridMatcher:
    """
    Ajay's hybrid matching engine.

    Combines multiple signals:
    - Semantic similarity (FAISS)
    - Skill overlap
    - Experience alignment
    - Domain match
    """

    def __init__(
        self,
        semantic_weight: float = 0.5,
        skill_weight: float = 0.3,
        experience_weight: float = 0.1,
        domain_weight: float = 0.1,
        domain_sections: Optional[List[str]] = None,
        domain_top_k: int = 3,
        domain_max_sim_weight: float = 0.4,
        domain_topk_mean_weight: float = 0.6,
    ):
        """
        Initialize the hybrid matcher.

        Parameters
        ----------
        semantic_weight : float
            Weight for semantic similarity score (default: 0.5)
        skill_weight : float
            Weight for skill overlap score (default: 0.3)
        experience_weight : float
            Weight for experience alignment score (default: 0.1)
        domain_weight : float
            Weight for domain match score (default: 0.1)
        domain_sections : Optional[List[str]]
            Resume sections used for domain scoring; defaults to skills/experience/projects
        domain_top_k : int
            Number of top chunk similarities to average in the top-k mean component
        domain_max_sim_weight : float
            Weight for max similarity component in weighted domain hybrid
        domain_topk_mean_weight : float
            Weight for top-k mean similarity component in weighted domain hybrid
        """
        self.artifacts = ArtifactLoader()

        # Weights for hybrid scoring
        self.semantic_weight = semantic_weight
        self.skill_weight = skill_weight
        self.experience_weight = experience_weight
        self.domain_weight = domain_weight

        # Normalize weights
        total = sum([semantic_weight, skill_weight, experience_weight, domain_weight])
        self.semantic_weight /= total
        self.skill_weight /= total
        self.experience_weight /= total
        self.domain_weight /= total

        default_sections = ["skills", "experience", "projects"]
        sections = domain_sections if domain_sections is not None else default_sections
        self.domain_sections = {str(section).strip().lower() for section in sections if str(section).strip()}
        if not self.domain_sections:
            self.domain_sections = set(default_sections)

        self.domain_top_k = max(1, int(domain_top_k))

        domain_mix_total = float(domain_max_sim_weight + domain_topk_mean_weight)
        if domain_mix_total <= 0:
            self.domain_max_sim_weight = 0.5
            self.domain_topk_mean_weight = 0.5
        else:
            self.domain_max_sim_weight = float(domain_max_sim_weight) / domain_mix_total
            self.domain_topk_mean_weight = float(domain_topk_mean_weight) / domain_mix_total

        logger.info(f"Initialized HybridMatcher with weights: "
                   f"semantic={self.semantic_weight:.2f}, "
                   f"skill={self.skill_weight:.2f}, "
                   f"experience={self.experience_weight:.2f}, "
                   f"domain={self.domain_weight:.2f}")
        logger.info(
            "Domain score config: sections=%s, top_k=%d, max_w=%.2f, topk_mean_w=%.2f",
            sorted(self.domain_sections),
            self.domain_top_k,
            self.domain_max_sim_weight,
            self.domain_topk_mean_weight,
        )

    def search_candidates(
        self,
        job_id: str,
        top_k: int = 100,
        rerank_top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Search for top matching resumes for a given job.

        Parameters
        ----------
        job_id : str
            Job ID to search for
        top_k : int
            Number of top candidates to return after hybrid scoring
        rerank_top_k : Optional[int]
            If provided, return this many candidates for LLM reranking

        Returns
        -------
        pd.DataFrame
            Ranked candidates with scores
        """
        # Get job information
        job_row = self.artifacts.jobs[self.artifacts.jobs["job_id"] == job_id]
        if len(job_row) == 0:
            raise ValueError(f"Job ID '{job_id}' not found")

        job_row = job_row.iloc[0]
        job_text = job_row["job_text"]
        job_title = job_row["job_title"]

        logger.info(f"Searching candidates for job: {job_id} ({job_title})")

        # Get job embedding
        job_idx = self.artifacts.jobs[self.artifacts.jobs["job_id"] == job_id].index[0]
        job_embedding = self.artifacts.job_embeddings[job_idx:job_idx+1]

        # ── 1. FAISS semantic search ──────────────────────────────────────────
        k_semantic = min(200, self.artifacts.faiss_index.ntotal)  # Retrieve more for reranking
        distances, indices = self.artifacts.faiss_index.search(job_embedding, k_semantic)

        # Get matching chunks
        chunk_matches = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS padding
                continue
            chunk_row = self.artifacts.chunk_meta.iloc[idx]
            chunk_matches.append({
                "resume_id": chunk_row["resume_id"],
                "section": chunk_row["section"],
                "chunk_text": chunk_row["chunk_text"],
                "semantic_score": float(dist),  # Cosine similarity (0-1)
            })

        # Aggregate by resume (max semantic score per resume)
        resume_scores = {}
        for match in chunk_matches:
            resume_id = match["resume_id"]
            if resume_id not in resume_scores:
                resume_scores[resume_id] = {
                    "semantic_score": match["semantic_score"],
                    "matched_sections": [match["section"]],
                }
            else:
                # Keep max semantic score
                if match["semantic_score"] > resume_scores[resume_id]["semantic_score"]:
                    resume_scores[resume_id]["semantic_score"] = match["semantic_score"]
                resume_scores[resume_id]["matched_sections"].append(match["section"])

        # ── 2. Compute hybrid scores ──────────────────────────────────────────
        candidates = []

        for resume_id, scores in resume_scores.items():
            # Get resume data (from chunk metadata)
            resume_chunks = self.artifacts.chunk_meta[
                self.artifacts.chunk_meta["resume_id"] == resume_id
            ]

            # Get candidate name
            candidate_name = resume_chunks['candidate_name'].iloc[0] if 'candidate_name' in resume_chunks.columns else resume_id

            # Extract skills (from 'skills' section if available)
            skills_chunks = resume_chunks[resume_chunks["section"] == "skills"]
            resume_skills = " ".join(skills_chunks["chunk_text"].tolist()) if len(skills_chunks) > 0 else ""

            # Extract experience (from 'experience' section if available)
            exp_chunks = resume_chunks[resume_chunks["section"] == "experience"]
            resume_experience = " ".join(exp_chunks["chunk_text"].tolist()) if len(exp_chunks) > 0 else ""

            # Skill overlap score
            skill_score = self._compute_skill_overlap(resume_skills, job_text)

            # Experience alignment score
            experience_score = self._compute_experience_alignment(resume_experience, job_text)

            # Domain match score (embedding-based, non-hardcoded)
            domain_score = self._compute_domain_match(resume_chunks, job_embedding[0])

            # Hybrid score
            hybrid_score = (
                self.semantic_weight * scores["semantic_score"] +
                self.skill_weight * skill_score +
                self.experience_weight * experience_score +
                self.domain_weight * domain_score
            )

            # Generate match summary
            matched_sections_unique = list(set(scores["matched_sections"]))
            match_summary = generate_match_summary(
                candidate_name=candidate_name,
                resume_skills=resume_skills,
                resume_experience=resume_experience,
                job_title=job_title,
                semantic_score=scores["semantic_score"],
                skill_score=skill_score,
                experience_score=experience_score,
                domain_score=domain_score,
                matched_sections=matched_sections_unique
            )

            candidates.append({
                "resume_id": resume_id,
                "candidate_name": candidate_name,
                "semantic_score": scores["semantic_score"],
                "skill_score": skill_score,
                "experience_score": experience_score,
                "domain_score": domain_score,
                "hybrid_score": hybrid_score,
                "matched_sections": matched_sections_unique,
                "match_summary": match_summary,
            })

        # Convert to DataFrame and sort
        df = pd.DataFrame(candidates)
        df = df.sort_values("hybrid_score", ascending=False).reset_index(drop=True)

        # Return top_k or rerank_top_k
        k = rerank_top_k if rerank_top_k else top_k
        df = df.head(k)

        logger.info(f"Found {len(df)} candidates for job {job_id}")
        return df

    def _compute_skill_overlap(self, resume_skills: str, job_text: str) -> float:
        """Compute skill overlap score (0-1)."""
        if not resume_skills or not job_text:
            return 0.0

        resume_skills_set = set(resume_skills.lower().split())
        job_text_lower = job_text.lower()

        # Count matching skills from vocabulary
        matching = sum(
            1 for skill in resume_skills_set
            if skill in self.artifacts.skill_vocab and skill in job_text_lower
        )

        # Normalize by job skills
        job_skills = sum(1 for skill in self.artifacts.skill_vocab if skill in job_text_lower)

        if job_skills == 0:
            return 0.0

        return min(1.0, matching / job_skills)

    def _compute_experience_alignment(self, resume_experience: str, job_text: str) -> float:
        """Compute experience alignment score (0-1)."""
        if not resume_experience or not job_text:
            return 0.0

        # Simple heuristic: look for year mentions
        import re

        # Extract years from job text
        job_years_match = re.search(r'(\d+)\+?\s*years?', job_text.lower())
        if not job_years_match:
            return 0.5  # Neutral if no explicit requirement

        required_years = int(job_years_match.group(1))

        # Extract years from resume
        resume_years_match = re.search(r'(\d+)\+?\s*years?', resume_experience.lower())
        if not resume_years_match:
            # Count date ranges
            date_ranges = re.findall(r'(20\d{2})\s*[-–]\s*(20\d{2}|present)', resume_experience.lower())
            candidate_years = sum(
                (2026 if end == 'present' else int(end)) - int(start)
                for start, end in date_ranges
            )
        else:
            candidate_years = int(resume_years_match.group(1))

        if candidate_years == 0:
            return 0.0

        # Score based on how well experience matches requirement
        if candidate_years >= required_years:
            return 1.0
        else:
            return candidate_years / required_years

    def _compute_domain_match(self, resume_chunks: pd.DataFrame, job_embedding: np.ndarray) -> float:
        """Compute domain match score (0-1) from embedding affinity, without hardcoded taxonomies."""
        if resume_chunks.empty or job_embedding is None:
            return 0.5  # Neutral if signal is unavailable

        # Domain scoring sections are configurable to support different role families.
        section_names = resume_chunks["section"].astype(str).str.lower()
        filtered_chunks = resume_chunks[section_names.isin(self.domain_sections)]
        if filtered_chunks.empty:
            filtered_chunks = resume_chunks

        chunk_indices = filtered_chunks.index.to_numpy(dtype=int)
        if len(chunk_indices) == 0:
            return 0.5

        if chunk_indices.max(initial=-1) >= len(self.artifacts.chunk_embeddings):
            logger.warning("Chunk metadata/index mismatch detected; using neutral domain score")
            return 0.5

        resume_embeddings = self.artifacts.chunk_embeddings[chunk_indices].astype(np.float32, copy=False)
        if resume_embeddings.size == 0:
            return 0.5

        # Defensive normalization so cosine similarity remains valid.
        resume_norms = np.linalg.norm(resume_embeddings, axis=1, keepdims=True)
        valid_rows = resume_norms.squeeze() > 0
        if not np.any(valid_rows):
            return 0.5
        resume_embeddings = resume_embeddings[valid_rows] / resume_norms[valid_rows]

        job_vec = np.asarray(job_embedding, dtype=np.float32).reshape(-1)
        job_norm = np.linalg.norm(job_vec)
        if job_norm == 0:
            return 0.5
        job_vec = job_vec / job_norm

        similarities = resume_embeddings @ job_vec
        if similarities.size == 0:
            return 0.5

        # Weighted hybrid: specialist signal (max) + broader signal (top-k mean).
        top_k = min(self.domain_top_k, similarities.size)
        top_similarities = np.partition(similarities, -top_k)[-top_k:]
        max_similarity = float(np.max(similarities))
        topk_mean_similarity = float(np.mean(top_similarities))
        cosine_score = (
            self.domain_max_sim_weight * max_similarity +
            self.domain_topk_mean_weight * topk_mean_similarity
        )

        # Map cosine range [-1, 1] into [0, 1] to keep score contract unchanged.
        return float(np.clip((cosine_score + 1.0) / 2.0, 0.0, 1.0))


# ── Public API ────────────────────────────────────────────────────────────────

def search_resumes(
    job_id: str,
    top_k: int = 10,
    semantic_weight: float = 0.5,
    skill_weight: float = 0.3,
    experience_weight: float = 0.1,
    domain_weight: float = 0.1,
    domain_sections: Optional[List[str]] = None,
    domain_top_k: int = 3,
    domain_max_sim_weight: float = 0.4,
    domain_topk_mean_weight: float = 0.6,
) -> List[Dict]:
    """
    Search for top matching resumes for a given job ID.

    Parameters
    ----------
    job_id : str
        Job ID to search for
    top_k : int
        Number of top candidates to return
    semantic_weight : float
        Weight for semantic similarity
    skill_weight : float
        Weight for skill overlap
    experience_weight : float
        Weight for experience alignment
    domain_weight : float
        Weight for domain match
    domain_sections : Optional[List[str]]
        Resume sections used for domain scoring; defaults to skills/experience/projects
    domain_top_k : int
        Number of top chunk similarities to average in domain top-k mean component
    domain_max_sim_weight : float
        Relative weight of max-similarity component for domain scoring
    domain_topk_mean_weight : float
        Relative weight of top-k mean component for domain scoring

    Returns
    -------
    List[Dict]
        List of ranked candidates with scores
    """
    matcher = HybridMatcher(
        semantic_weight=semantic_weight,
        skill_weight=skill_weight,
        experience_weight=experience_weight,
        domain_weight=domain_weight,
        domain_sections=domain_sections,
        domain_top_k=domain_top_k,
        domain_max_sim_weight=domain_max_sim_weight,
        domain_topk_mean_weight=domain_topk_mean_weight,
    )

    results_df = matcher.search_candidates(job_id, top_k=top_k)

    # Convert to list of dicts
    return results_df.to_dict(orient="records")

