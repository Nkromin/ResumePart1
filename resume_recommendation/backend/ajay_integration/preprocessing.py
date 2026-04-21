"""
preprocessing.py
────────────────
Data preprocessing utilities for Ajay's hybrid matching system.

Modified to support PDF-based ingestion instead of CSV.

Key functions:
- parse_resume_pdfs()  — replaces read_resumes()
- parse_jd_pdfs()      — replaces read_jobs()
- resume_to_sections() — unchanged from Ajay's original
- chunk_sections()     — unchanged from Ajay's original
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from backend.ingestion.pdf_loader import extract_text
from backend.ajay_integration.pdf_parser import parse_resume_pdf, parse_jd_pdf

logger = logging.getLogger(__name__)


# ── PDF-based ingestion (replaces CSV readers) ────────────────────────────────

def parse_resume_pdfs(resume_folder: str | Path) -> pd.DataFrame:
    """
    Parse all resume PDFs in a folder into a DataFrame.

    This replaces Ajay's original `read_resumes(resume_csv)` function.

    Parameters
    ----------
    resume_folder : str | Path
        Path to folder containing resume PDF files

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: resume_id, candidate_name, skills, experience,
        projects, education, certifications
    """
    resume_folder = Path(resume_folder)

    if not resume_folder.exists():
        logger.warning(f"Resume folder does not exist: {resume_folder}")
        return pd.DataFrame(columns=[
            "resume_id", "candidate_name", "skills", "experience", "projects",
            "education", "certifications"
        ])

    resumes = []

    # Process all PDF files in the folder
    pdf_files = list(resume_folder.glob("*.pdf")) + list(resume_folder.glob("*.txt"))

    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing resume: {pdf_file.name}")

            # Extract text from PDF
            text = extract_text(pdf_file)

            if not text.strip():
                logger.warning(f"Empty text extracted from {pdf_file.name}, skipping")
                continue

            # Parse into structured format
            resume_data = parse_resume_pdf(pdf_file, text)
            resumes.append(resume_data)

        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {e}", exc_info=True)
            continue

    if not resumes:
        logger.warning("No resumes were successfully parsed")
        return pd.DataFrame(columns=[
            "resume_id", "candidate_name", "skills", "experience", "projects",
            "education", "certifications"
        ])

    df = pd.DataFrame(resumes)
    logger.info(f"Successfully parsed {len(df)} resumes")

    return df


def parse_jd_pdfs(jd_folder: str | Path) -> pd.DataFrame:
    """
    Parse all job description PDFs in a folder into a DataFrame.

    This replaces Ajay's original `read_jobs(job_csv)` function.

    Parameters
    ----------
    jd_folder : str | Path
        Path to folder containing job description PDF files

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: job_id, job_title, job_text
    """
    jd_folder = Path(jd_folder)

    if not jd_folder.exists():
        logger.warning(f"JD folder does not exist: {jd_folder}")
        return pd.DataFrame(columns=["job_id", "job_title", "job_text"])

    jobs = []

    # Process all PDF files in the folder
    pdf_files = list(jd_folder.glob("*.pdf")) + list(jd_folder.glob("*.txt"))

    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing job description: {pdf_file.name}")

            # Extract text from PDF
            text = extract_text(pdf_file)

            if not text.strip():
                logger.warning(f"Empty text extracted from {pdf_file.name}, skipping")
                continue

            # Parse into structured format
            jd_data = parse_jd_pdf(pdf_file, text)
            jobs.append(jd_data)

        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {e}", exc_info=True)
            continue

    if not jobs:
        logger.warning("No job descriptions were successfully parsed")
        return pd.DataFrame(columns=["job_id", "job_title", "job_text"])

    df = pd.DataFrame(jobs)
    logger.info(f"Successfully parsed {len(df)} job descriptions")

    return df


# ── Ajay's original preprocessing functions (unchanged) ───────────────────────

def resume_to_sections(resume_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert resume DataFrame into section-based format.

    Each row becomes multiple rows (one per section).

    Parameters
    ----------
    resume_df : pd.DataFrame
        DataFrame with resume_id and section columns

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: resume_id, section, text
    """
    sections_list = []

    for _, row in resume_df.iterrows():
        resume_id = row["resume_id"]

        # Map section names to content
        section_mapping = {
            "skills": row.get("skills", ""),
            "experience": row.get("experience", ""),
            "projects": row.get("projects", ""),
            "education": row.get("education", ""),
            "certifications": row.get("certifications", ""),
        }

        for section_name, section_text in section_mapping.items():
            if section_text and str(section_text).strip():
                sections_list.append({
                    "resume_id": resume_id,
                    "section": section_name,
                    "text": str(section_text).strip()
                })

    return pd.DataFrame(sections_list)


def chunk_sections(
    sections_df: pd.DataFrame,
    max_chunk_length: int = 512,
    overlap: int = 50
) -> pd.DataFrame:
    """
    Split long section texts into smaller chunks.

    This is Ajay's original chunking logic preserved exactly.

    Parameters
    ----------
    sections_df : pd.DataFrame
        DataFrame with resume_id, section, text columns
    max_chunk_length : int
        Maximum characters per chunk
    overlap : int
        Number of overlapping characters between chunks

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: resume_id, section, chunk_id, chunk_text
    """
    chunks_list = []

    for _, row in sections_df.iterrows():
        resume_id = row["resume_id"]
        section = row["section"]
        text = row["text"]

        # Split text into chunks
        text_len = len(text)

        if text_len <= max_chunk_length:
            # Single chunk
            chunks_list.append({
                "resume_id": resume_id,
                "section": section,
                "chunk_id": 0,
                "chunk_text": text
            })
        else:
            # Multiple chunks with overlap
            chunk_id = 0
            start = 0

            while start < text_len:
                end = min(start + max_chunk_length, text_len)
                chunk_text = text[start:end]

                chunks_list.append({
                    "resume_id": resume_id,
                    "section": section,
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text
                })

                chunk_id += 1
                start += (max_chunk_length - overlap)

    return pd.DataFrame(chunks_list)


# ── Skill vocabulary extraction (Ajay's logic) ────────────────────────────────

def extract_skill_vocabulary(resume_df: pd.DataFrame, job_df: pd.DataFrame) -> set[str]:
    """
    Extract a unified skill vocabulary from resumes and job descriptions.

    This is Ajay's original skill extraction logic.

    Parameters
    ----------
    resume_df : pd.DataFrame
        Resume data with skills column
    job_df : pd.DataFrame
        Job data with job_text column

    Returns
    -------
    set[str]
        Set of unique skill keywords (normalized to lowercase)
    """
    skills = set()

    # Extract from resume skills
    for skill_text in resume_df.get("skills", pd.Series()):
        if pd.notna(skill_text):
            # Split by common delimiters
            skill_items = str(skill_text).lower().replace(",", " ").replace(";", " ").split()
            skills.update(skill_items)

    # Extract from job texts (simple keyword extraction)
    common_skills = [
        "python", "java", "javascript", "c++", "sql", "nosql",
        "machine learning", "deep learning", "nlp", "cv",
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
        "docker", "kubernetes", "aws", "azure", "gcp", "cloud",
        "react", "angular", "vue", "node", "django", "flask", "fastapi",
        "git", "ci/cd", "agile", "scrum", "devops",
        "api", "rest", "graphql", "microservices",
        "database", "mongodb", "postgresql", "mysql", "redis",
    ]

    for job_text in job_df.get("job_text", pd.Series()):
        if pd.notna(job_text):
            text_lower = str(job_text).lower()
            for skill in common_skills:
                if skill in text_lower:
                    skills.add(skill)

    logger.info(f"Extracted {len(skills)} unique skills")
    return skills


def compute_skill_overlap(resume_skills: str, job_text: str, skill_vocab: set[str]) -> float:
    """
    Compute skill overlap ratio between resume and job.

    Parameters
    ----------
    resume_skills : str
        Comma-separated skills from resume
    job_text : str
        Full job description text
    skill_vocab : set[str]
        Unified skill vocabulary

    Returns
    -------
    float
        Overlap ratio (0.0 to 1.0)
    """
    if not resume_skills or not job_text:
        return 0.0

    resume_skills_lower = set(str(resume_skills).lower().replace(",", " ").split())
    job_text_lower = str(job_text).lower()

    # Count matching skills
    matching_skills = sum(1 for skill in resume_skills_lower if skill in job_text_lower and skill in skill_vocab)

    # Normalize by number of skills in vocabulary present in job
    job_skills = sum(1 for skill in skill_vocab if skill in job_text_lower)

    if job_skills == 0:
        return 0.0

    return matching_skills / job_skills

