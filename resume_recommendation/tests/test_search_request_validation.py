from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from backend.routes.search import SearchRequest
except ModuleNotFoundError:
    REPO_ROOT = PROJECT_ROOT.parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from resume_recommendation.backend.routes.search import SearchRequest


class SearchRequestValidationTests(unittest.TestCase):
    def test_valid_request_passes(self):
        req = SearchRequest(job_id="jd_backend_engineer")
        self.assertEqual(req.job_id, "jd_backend_engineer")
        self.assertEqual(req.top_k, 10)

    def test_empty_job_id_rejected(self):
        with self.assertRaises(ValidationError):
            SearchRequest(job_id="   ")

    def test_zero_hybrid_weights_rejected(self):
        with self.assertRaises(ValidationError):
            SearchRequest(
                job_id="jd_backend_engineer",
                semantic_weight=0.0,
                skill_weight=0.0,
                experience_weight=0.0,
                domain_weight=0.0,
            )

    def test_zero_domain_mix_rejected(self):
        with self.assertRaises(ValidationError):
            SearchRequest(
                job_id="jd_backend_engineer",
                domain_max_sim_weight=0.0,
                domain_topk_mean_weight=0.0,
            )

    def test_domain_sections_are_normalized_and_deduplicated(self):
        req = SearchRequest(
            job_id="jd_backend_engineer",
            domain_sections=[" Skills ", "projects", "skills", "EXPERIENCE"],
        )
        self.assertEqual(req.domain_sections, ["skills", "projects", "experience"])

    def test_blank_domain_sections_rejected(self):
        with self.assertRaises(ValidationError):
            SearchRequest(job_id="jd_backend_engineer", domain_sections=["", "   "])


if __name__ == "__main__":
    unittest.main()

