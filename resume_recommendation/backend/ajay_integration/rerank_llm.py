"""
rerank_llm.py
─────────────
Prompt-based reranking for top candidates using an Ollama-hosted reasoning LLM.

The reranker reads the full job description plus reconstructed resume context and
assigns a 0-100 relevance score, which is then blended with Ajay's hybrid score.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "You are an expert technical recruiter.\n\n"
    "Job Description:\n{job_description}\n\n"
    "Candidate Resume:\n{resume_text}\n\n"
    "This candidate currently has a hybrid score of {hybrid_score:.2f} on a 0-1 scale.\n"
    "Evaluate how well this candidate matches the job. Consider skill match, experience, \n"
    "domain relevance, and project relevance. Return ONLY a number between 0 and 100 representing\n"
    "how strong the match is."
)

MAX_CONTEXT_CHARS = 6000
LLM_WEIGHT = 0.3
HYBRID_WEIGHT = 0.7
DEFAULT_TOP_K = 10

_LLM_SCORE_CACHE: Dict[Tuple[str, str], float] = {}


class OllamaReranker:
    """Reasoning-based reranker backed by an Ollama LLM."""

    def __init__(
        self,
        model: str = "llama3",
        ollama_host: Optional[str] = None,
        request_timeout: int = 60,
    ) -> None:
        self.model = model
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = request_timeout
        self.enabled = False
        self._session = requests.Session()

        try:
            response = self._session.get(f"{self.ollama_host}/api/tags", timeout=2)
            if response.status_code == 200:
                self.enabled = True
                logger.info("Initialized Ollama reranker with model: %s", model)
            else:
                logger.warning("Ollama server not responding at %s", self.ollama_host)
        except Exception as exc:
            logger.warning("Failed to connect to Ollama: %s. Reranking disabled.", exc)

    def rerank(
        self,
        candidates: pd.DataFrame,
        job_description: str,
        chunk_meta: pd.DataFrame,
        top_k: int = DEFAULT_TOP_K,
        job_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Apply LLM-based reranking to the strongest hybrid candidates."""
        top_candidates = candidates.head(top_k).copy()
        if top_candidates.empty:
            return top_candidates

        if not self.enabled:
            logger.warning("Ollama reranking disabled. Returning hybrid scores as-is.")
            top_candidates["llm_score"] = 0.0
            top_candidates["final_score"] = top_candidates["hybrid_score"]
            return top_candidates

        job_key = self._resolve_job_key(job_id, job_description)
        reranked_rows = []

        for _, candidate in top_candidates.iterrows():
            resume_id = candidate["resume_id"]
            resume_chunks = chunk_meta[chunk_meta["resume_id"] == resume_id]
            resume_text = self._construct_resume_text(resume_chunks)

            llm_score = self._score_candidate(
                job_key=job_key,
                resume_id=resume_id,
                job_description=job_description,
                resume_text=resume_text,
                hybrid_score=candidate["hybrid_score"],
            )

            final_score = HYBRID_WEIGHT * candidate["hybrid_score"] + LLM_WEIGHT * llm_score

            candidate_dict = candidate.to_dict()
            candidate_dict["llm_score"] = llm_score
            candidate_dict["final_score"] = final_score
            reranked_rows.append(candidate_dict)

        reranked_df = pd.DataFrame(reranked_rows)
        reranked_df = reranked_df.sort_values("final_score", ascending=False).reset_index(drop=True)
        logger.info("Successfully reranked %d candidates using Ollama LLM", len(reranked_df))
        return reranked_df

    def _score_candidate(
        self,
        job_key: str,
        resume_id: str,
        job_description: str,
        resume_text: str,
        hybrid_score: float,
    ) -> float:
        cache_key = (job_key, resume_id)
        if cache_key in _LLM_SCORE_CACHE:
            return _LLM_SCORE_CACHE[cache_key]

        prompt = PROMPT_TEMPLATE.format(
            job_description=self._truncate(job_description),
            resume_text=self._truncate(resume_text),
            hybrid_score=hybrid_score,
        )

        try:
            raw_response = self._call_ollama(prompt)
            numeric_score = self._extract_score(raw_response)
            normalized = self._normalize_score(numeric_score)
        except Exception as exc:
            logger.error("LLM scoring failed for %s/%s: %s", job_key, resume_id, exc)
            normalized = 0.0

        _LLM_SCORE_CACHE[cache_key] = normalized
        return normalized

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        response = self._session.post(
            f"{self.ollama_host}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Ollama generate failed: {response.status_code} {response.text}")
        data = response.json()
        return data.get("response", "")

    @staticmethod
    def _extract_score(raw_response: str) -> float:
        match = re.search(r"-?\d+(?:\.\d+)?", raw_response)
        if not match:
            raise ValueError(f"Could not parse score from LLM response: {raw_response!r}")
        return float(match.group())

    @staticmethod
    def _normalize_score(score: float) -> float:
        return max(0.0, min(1.0, score / 100.0))

    @staticmethod
    def _truncate(text: str) -> str:
        if not text:
            return ""
        return text[:MAX_CONTEXT_CHARS]

    @staticmethod
    def _construct_resume_text(resume_chunks: pd.DataFrame) -> str:
        sections: Dict[str, List[str]] = {}
        for _, chunk in resume_chunks.iterrows():
            sections.setdefault(chunk["section"], []).append(chunk["chunk_text"])

        priority_sections = ["skills", "experience", "projects", "education"]
        resume_parts: List[str] = []

        for section in priority_sections:
            if section in sections:
                resume_parts.append(f"\n{section.upper()}:\n")
                resume_parts.append("\n".join(sections[section]))

        for section, chunks in sections.items():
            if section not in priority_sections:
                resume_parts.append(f"\n{section.upper()}:\n")
                resume_parts.append("\n".join(chunks))

        return "\n".join(resume_parts)

    def _resolve_job_key(self, job_id: Optional[str], job_description: str) -> str:
        if job_id:
            return job_id

        try:
            from backend.ajay_integration.match import ArtifactLoader

            artifacts = ArtifactLoader()
            match = artifacts.jobs[artifacts.jobs["job_text"] == job_description]
            if not match.empty:
                return str(match.iloc[0]["job_id"])
        except Exception as exc:
            logger.debug("Failed to infer job_id from description: %s", exc)

        digest = hashlib.sha1(job_description.encode("utf-8")).hexdigest()
        return f"job-{digest}"


def rerank_with_llm(
    candidates: pd.DataFrame,
    job_description: str,
    chunk_meta: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
    model: str = "llama3",
    job_id: Optional[str] = None,
) -> List[Dict]:
    """Public helper that returns reranked candidates as dictionaries."""
    reranker = OllamaReranker(model=model)
    reranked_df = reranker.rerank(
        candidates=candidates,
        job_description=job_description,
        chunk_meta=chunk_meta,
        top_k=top_k,
        job_id=job_id,
    )
    return reranked_df.to_dict(orient="records")



