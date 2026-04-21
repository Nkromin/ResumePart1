"""
routes/search.py
────────────────
POST /search

Search endpoint for matching resumes to job descriptions using Ajay's
hybrid matching system.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])


# ── Request/Response models ───────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    job_id: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=200)
    use_llm_reranking: bool = False
    semantic_weight: float = Field(default=0.5, ge=0.0)
    skill_weight: float = Field(default=0.3, ge=0.0)
    experience_weight: float = Field(default=0.1, ge=0.0)
    domain_weight: float = Field(default=0.1, ge=0.0)
    domain_sections: list[str] | None = None
    domain_top_k: int = Field(default=3, ge=1, le=50)
    domain_max_sim_weight: float = Field(default=0.4, ge=0.0)
    domain_topk_mean_weight: float = Field(default=0.6, ge=0.0)

    @field_validator("job_id")
    @classmethod
    def _validate_job_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("job_id cannot be empty")
        return value

    @field_validator("domain_sections")
    @classmethod
    def _validate_domain_sections(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None

        cleaned = [section.strip().lower() for section in value if section and section.strip()]
        if not cleaned:
            raise ValueError("domain_sections must contain at least one non-empty section")

        # Deduplicate while preserving input order.
        return list(dict.fromkeys(cleaned))

    @model_validator(mode="after")
    def _validate_weight_totals(self) -> "SearchRequest":
        hybrid_total = self.semantic_weight + self.skill_weight + self.experience_weight + self.domain_weight
        if hybrid_total <= 0:
            raise ValueError("semantic_weight + skill_weight + experience_weight + domain_weight must be > 0")

        domain_mix_total = self.domain_max_sim_weight + self.domain_topk_mean_weight
        if domain_mix_total <= 0:
            raise ValueError("domain_max_sim_weight + domain_topk_mean_weight must be > 0")

        return self


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "",
    summary="Search for matching resumes",
    status_code=status.HTTP_200_OK,
)
async def search_resumes(request: SearchRequest) -> JSONResponse:
    """
    Search for top matching resumes for a given job ID.

    Uses Ajay's hybrid matching system with configurable weights:
    - Semantic similarity (FAISS vector search)
    - Skill overlap
    - Experience alignment
    - Domain match

    Optionally applies LLM-based reranking if enabled.
    """
    try:
        from backend.ajay_integration.match import search_resumes as ajay_search

        # Perform hybrid search
        results = ajay_search(
            job_id=request.job_id,
            top_k=request.top_k if not request.use_llm_reranking else request.top_k * 2,
            semantic_weight=request.semantic_weight,
            skill_weight=request.skill_weight,
            experience_weight=request.experience_weight,
            domain_weight=request.domain_weight,
            domain_sections=request.domain_sections,
            domain_top_k=request.domain_top_k,
            domain_max_sim_weight=request.domain_max_sim_weight,
            domain_topk_mean_weight=request.domain_topk_mean_weight,
        )

        # Optional LLM reranking
        if request.use_llm_reranking:
            try:
                from backend.ajay_integration.rerank_llm import rerank_with_llm
                from backend.ajay_integration.match import ArtifactLoader
                import pandas as pd

                artifacts = ArtifactLoader()
                job_row = artifacts.jobs[artifacts.jobs["job_id"] == request.job_id].iloc[0]

                results = rerank_with_llm(
                    candidates=pd.DataFrame(results),
                    job_description=job_row["job_text"],
                    chunk_meta=artifacts.chunk_meta,
                    top_k=request.top_k,
                )

                logger.info(f"Applied LLM reranking for job {request.job_id}")

            except Exception as e:
                logger.warning(f"LLM reranking failed, using hybrid scores: {e}")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "job_id": request.job_id,
                "num_results": len(results),
                "results": results,
            },
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifacts not found. Please upload resumes and job descriptions first. Error: {str(e)}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@router.get(
    "/jobs",
    summary="List all available jobs",
    status_code=status.HTTP_200_OK,
)
async def list_jobs() -> JSONResponse:
    """
    List all available job IDs and titles.

    Useful for getting job_id values to use in search requests.
    """
    try:
        from backend.ajay_integration.match import ArtifactLoader

        artifacts = ArtifactLoader()
        jobs = artifacts.jobs[["job_id", "job_title"]].to_dict(orient="records")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "num_jobs": len(jobs),
                "jobs": jobs,
            },
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job data not found. Please upload job descriptions first. Error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list jobs: {str(e)}",
        )


@router.get(
    "/resumes",
    summary="List all available resumes",
    status_code=status.HTTP_200_OK,
)
async def list_resumes() -> JSONResponse:
    """
    List all available resume IDs.
    """
    try:
        from backend.ajay_integration.match import ArtifactLoader

        artifacts = ArtifactLoader()
        resume_ids = artifacts.chunk_meta["resume_id"].unique().tolist()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "num_resumes": len(resume_ids),
                "resume_ids": resume_ids,
            },
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resume data not found. Please upload resumes first. Error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Failed to list resumes: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list resumes: {str(e)}",
        )

