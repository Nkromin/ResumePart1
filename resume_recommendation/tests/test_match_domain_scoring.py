from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Ensure `backend.*` imports resolve when tests are run from workspace root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from backend.ajay_integration import match as match_module
except ModuleNotFoundError:
    REPO_ROOT = PROJECT_ROOT.parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from resume_recommendation.backend.ajay_integration import match as match_module


class DomainScoreUnitTests(unittest.TestCase):
    def _build_matcher(self, chunk_meta: pd.DataFrame, chunk_embeddings: np.ndarray):
        matcher = match_module.HybridMatcher.__new__(match_module.HybridMatcher)
        matcher.artifacts = SimpleNamespace(
            chunk_meta=chunk_meta,
            chunk_embeddings=chunk_embeddings,
        )
        matcher.domain_sections = {"skills", "experience", "projects"}
        matcher.domain_top_k = 3
        matcher.domain_max_sim_weight = 0.4
        matcher.domain_topk_mean_weight = 0.6
        return matcher

    def test_embedding_domain_score_prefers_semantically_closer_resume(self):
        chunk_meta = pd.DataFrame(
            [
                {"resume_id": "r1", "section": "skills", "chunk_text": "python ml"},
                {"resume_id": "r1", "section": "experience", "chunk_text": "built models"},
                {"resume_id": "r2", "section": "skills", "chunk_text": "react js"},
                {"resume_id": "r2", "section": "projects", "chunk_text": "web apps"},
            ]
        )
        chunk_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
            ],
            dtype=np.float32,
        )
        job_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        matcher = self._build_matcher(chunk_meta=chunk_meta, chunk_embeddings=chunk_embeddings)
        r1_chunks = chunk_meta[chunk_meta["resume_id"] == "r1"]
        r2_chunks = chunk_meta[chunk_meta["resume_id"] == "r2"]

        score_r1 = matcher._compute_domain_match(r1_chunks, job_embedding)
        score_r2 = matcher._compute_domain_match(r2_chunks, job_embedding)

        self.assertGreater(score_r1, score_r2)
        self.assertGreaterEqual(score_r1, 0.0)
        self.assertLessEqual(score_r1, 1.0)
        self.assertGreaterEqual(score_r2, 0.0)
        self.assertLessEqual(score_r2, 1.0)

    def test_domain_score_handles_missing_or_invalid_vectors_neutrally(self):
        chunk_meta = pd.DataFrame([
            {"resume_id": "r1", "section": "education", "chunk_text": "btech"},
        ])
        chunk_embeddings = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        matcher = self._build_matcher(chunk_meta=chunk_meta, chunk_embeddings=chunk_embeddings)
        score = matcher._compute_domain_match(chunk_meta, np.array([0.0, 0.0, 0.0], dtype=np.float32))

        self.assertEqual(score, 0.5)

    def test_weighted_hybrid_configuration_changes_domain_score(self):
        chunk_meta = pd.DataFrame(
            [
                {"resume_id": "r1", "section": "skills", "chunk_text": "close match"},
                {"resume_id": "r1", "section": "experience", "chunk_text": "strong but less"},
                {"resume_id": "r1", "section": "projects", "chunk_text": "weak"},
            ]
        )
        chunk_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.7, 0.3, 0.0],
                [0.2, 0.8, 0.0],
            ],
            dtype=np.float32,
        )
        job_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        matcher = self._build_matcher(chunk_meta=chunk_meta, chunk_embeddings=chunk_embeddings)
        matcher.domain_sections = {"skills", "experience", "projects"}
        matcher.domain_top_k = 3

        matcher.domain_max_sim_weight = 1.0
        matcher.domain_topk_mean_weight = 0.0
        score_max_only = matcher._compute_domain_match(chunk_meta, job_embedding)

        matcher.domain_max_sim_weight = 0.0
        matcher.domain_topk_mean_weight = 1.0
        score_mean_only = matcher._compute_domain_match(chunk_meta, job_embedding)

        self.assertGreater(score_max_only, score_mean_only)

    def test_domain_sections_are_configurable(self):
        chunk_meta = pd.DataFrame(
            [
                {"resume_id": "r1", "section": "skills", "chunk_text": "python"},
                {"resume_id": "r1", "section": "education", "chunk_text": "history"},
            ]
        )
        chunk_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        job_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        matcher = self._build_matcher(chunk_meta=chunk_meta, chunk_embeddings=chunk_embeddings)
        matcher.domain_top_k = 2
        matcher.domain_max_sim_weight = 0.5
        matcher.domain_topk_mean_weight = 0.5

        matcher.domain_sections = {"skills"}
        score_skills_only = matcher._compute_domain_match(chunk_meta, job_embedding)

        matcher.domain_sections = {"education"}
        score_education_only = matcher._compute_domain_match(chunk_meta, job_embedding)

        self.assertGreater(score_skills_only, score_education_only)


class DomainScoreArtifactSmokeTests(unittest.TestCase):
    SNAPSHOT_PATH = PROJECT_ROOT / "tests" / "snapshots" / "domain_regression_snapshot.json"

    @staticmethod
    def _to_regression_signature(results: list[dict], top_k: int = 3) -> list[dict]:
        signature = []
        for row in results[:top_k]:
            signature.append(
                {
                    "resume_id": row["resume_id"],
                    "candidate_name": row["candidate_name"],
                    "hybrid_score": round(float(row["hybrid_score"]), 6),
                    "domain_score": round(float(row["domain_score"]), 6),
                }
            )
        return signature

    @classmethod
    def _write_snapshot(cls, payload: dict) -> None:
        cls.SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.SNAPSHOT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def test_hybrid_search_runs_with_artifacts_and_domain_score_is_bounded(self):
        artifacts_dir = PROJECT_ROOT / "artifacts"
        required = [
            artifacts_dir / "faiss.index",
            artifacts_dir / "chunk_embeddings.npy",
            artifacts_dir / "chunk_meta.parquet",
            artifacts_dir / "job_embeddings.npy",
            artifacts_dir / "jobs.parquet",
            artifacts_dir / "skill_vocab.json",
        ]
        if not all(path.exists() for path in required):
            self.skipTest("Required artifacts are not available for smoke test")

        # Patch module-level paths so test is independent of process CWD.
        match_module.ARTIFACTS_DIR = artifacts_dir
        match_module.FAISS_INDEX_PATH = artifacts_dir / "faiss.index"
        match_module.CHUNK_EMBEDDINGS_PATH = artifacts_dir / "chunk_embeddings.npy"
        match_module.CHUNK_META_PATH = artifacts_dir / "chunk_meta.parquet"
        match_module.JOB_EMBEDDINGS_PATH = artifacts_dir / "job_embeddings.npy"
        match_module.JOBS_PATH = artifacts_dir / "jobs.parquet"
        match_module.SKILL_VOCAB_PATH = artifacts_dir / "skill_vocab.json"

        # Reset singleton to force reload from patched paths.
        match_module.ArtifactLoader._instance = None
        match_module.ArtifactLoader._loaded = False

        matcher = match_module.HybridMatcher()
        job_id = str(matcher.artifacts.jobs.iloc[0]["job_id"])
        results = matcher.search_candidates(job_id=job_id, top_k=5)

        self.assertFalse(results.empty)
        self.assertIn("domain_score", results.columns)
        self.assertTrue(results["domain_score"].between(0.0, 1.0).all())
        self.assertFalse(results["domain_score"].isna().any())

    def test_artifact_regression_snapshot(self):
        artifacts_dir = PROJECT_ROOT / "artifacts"
        required = [
            artifacts_dir / "faiss.index",
            artifacts_dir / "chunk_embeddings.npy",
            artifacts_dir / "chunk_meta.parquet",
            artifacts_dir / "job_embeddings.npy",
            artifacts_dir / "jobs.parquet",
            artifacts_dir / "skill_vocab.json",
        ]
        if not all(path.exists() for path in required):
            self.skipTest("Required artifacts are not available for regression snapshot test")

        match_module.ARTIFACTS_DIR = artifacts_dir
        match_module.FAISS_INDEX_PATH = artifacts_dir / "faiss.index"
        match_module.CHUNK_EMBEDDINGS_PATH = artifacts_dir / "chunk_embeddings.npy"
        match_module.CHUNK_META_PATH = artifacts_dir / "chunk_meta.parquet"
        match_module.JOB_EMBEDDINGS_PATH = artifacts_dir / "job_embeddings.npy"
        match_module.JOBS_PATH = artifacts_dir / "jobs.parquet"
        match_module.SKILL_VOCAB_PATH = artifacts_dir / "skill_vocab.json"
        match_module.ArtifactLoader._instance = None
        match_module.ArtifactLoader._loaded = False

        jobs = match_module.ArtifactLoader().jobs["job_id"].tolist()
        regression_payload = {
            "config": {
                "domain_sections": ["skills", "experience", "projects"],
                "domain_top_k": 3,
                "domain_max_sim_weight": 0.4,
                "domain_topk_mean_weight": 0.6,
            },
            "jobs": {},
        }

        for job_id in jobs:
            rows = match_module.search_resumes(
                job_id=job_id,
                top_k=5,
                domain_sections=regression_payload["config"]["domain_sections"],
                domain_top_k=regression_payload["config"]["domain_top_k"],
                domain_max_sim_weight=regression_payload["config"]["domain_max_sim_weight"],
                domain_topk_mean_weight=regression_payload["config"]["domain_topk_mean_weight"],
            )
            regression_payload["jobs"][job_id] = self._to_regression_signature(rows, top_k=3)

        if os.getenv("UPDATE_REGRESSION_SNAPSHOTS") == "1" or not self.SNAPSHOT_PATH.exists():
            self._write_snapshot(regression_payload)

        expected = json.loads(self.SNAPSHOT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(regression_payload, expected)


if __name__ == "__main__":
    unittest.main()


