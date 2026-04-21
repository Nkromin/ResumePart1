"""
pdf_parser.py
─────────────
PDF parsing utilities for extracting structured information from resumes and job descriptions.

Extracts sections like skills, experience, projects, education, and certifications from
resume PDFs to create the DataFrame schema expected by Ajay's pipeline.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PDFSectionParser:
    """
    Parses PDF text content to extract structured sections.

    Uses regex patterns to identify section headers and extract relevant content.
    """

    # Common section header patterns (case-insensitive)
    SECTION_PATTERNS = {
        "skills": r"(?:^|\n)(?:technical\s+)?skills?\s*:?\s*\n",
        "experience": r"(?:^|\n)(?:work\s+)?experience\s*:?\s*\n",
        "projects": r"(?:^|\n)projects?\s*:?\s*\n",
        "education": r"(?:^|\n)education\s*:?\s*\n",
        "certifications": r"(?:^|\n)certifications?\s*:?\s*\n",
    }

    @staticmethod
    def extract_section(text: str, section_name: str, next_section_pos: Optional[int] = None) -> str:
        """
        Extract text content for a specific section.

        Parameters
        ----------
        text : str
            Full document text
        section_name : str
            Name of the section to extract
        next_section_pos : Optional[int]
            Position of the next section header (to bound the extraction)

        Returns
        -------
        str
            Extracted section content, cleaned and stripped
        """
        pattern = PDFSectionParser.SECTION_PATTERNS.get(section_name.lower())
        if not pattern:
            return ""

        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if not match:
            return ""

        start_pos = match.end()
        end_pos = next_section_pos if next_section_pos else len(text)

        section_text = text[start_pos:end_pos].strip()

        # Clean up the section text
        section_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', section_text)  # Remove excessive newlines
        return section_text

    @staticmethod
    def parse_resume_sections(text: str) -> Dict[str, str]:
        """
        Parse resume text into structured sections.

        Parameters
        ----------
        text : str
            Raw resume text extracted from PDF

        Returns
        -------
        Dict[str, str]
            Dictionary containing extracted sections
        """
        text = text.lower()  # Normalize for pattern matching

        # Find all section positions
        section_positions: List[tuple[str, int]] = []
        for section_name, pattern in PDFSectionParser.SECTION_PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                section_positions.append((section_name, match.start()))

        # Sort by position
        section_positions.sort(key=lambda x: x[1])

        # Extract each section
        sections = {}
        for i, (section_name, pos) in enumerate(section_positions):
            next_pos = section_positions[i + 1][1] if i + 1 < len(section_positions) else None
            content = PDFSectionParser.extract_section(text, section_name, next_pos)
            sections[section_name] = content

        return sections

    @staticmethod
    def extract_skills_heuristic(text: str) -> str:
        """
        Heuristic-based skill extraction when no clear section is found.

        Looks for common skill keywords and patterns.
        """
        skill_keywords = [
            "python", "java", "javascript", "c++", "sql", "nosql",
            "machine learning", "deep learning", "nlp", "computer vision",
            "tensorflow", "pytorch", "keras", "scikit-learn",
            "docker", "kubernetes", "aws", "azure", "gcp",
            "react", "angular", "vue", "node.js", "django", "flask",
            "git", "ci/cd", "agile", "scrum",
        ]

        found_skills = []
        text_lower = text.lower()

        for skill in skill_keywords:
            if skill in text_lower:
                found_skills.append(skill)

        return ", ".join(found_skills) if found_skills else ""

    @staticmethod
    def extract_years_of_experience(text: str) -> int:
        """
        Extract years of experience from resume text.

        Looks for patterns like "5 years", "5+ years", etc.
        """
        # Pattern: X years, X+ years, X-Y years
        patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience\s*:\s*(\d+)\+?\s*years?',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))

        # Count date ranges (e.g., "2019-2023")
        date_ranges = re.findall(r'(20\d{2})\s*[-–]\s*(20\d{2}|present|current)', text, re.IGNORECASE)
        if date_ranges:
            total_years = 0
            for start, end in date_ranges:
                end_year = 2026 if end.lower() in ('present', 'current') else int(end)
                total_years += max(0, end_year - int(start))
            return total_years

        return 0


def extract_candidate_name(text: str) -> str:
    """
    Extract candidate name from resume text.

    Looks for common patterns in resume headers.
    """
    # Get first few lines (usually contains name)
    lines = text.strip().split('\n')[:10]

    # Pattern 1: Look for name in first line (most common)
    first_line = lines[0].strip() if lines else ""

    # Remove common prefixes
    first_line = re.sub(r'^(resume|cv|curriculum vitae)\s*[:-]?\s*', '', first_line, flags=re.IGNORECASE)

    # If first line looks like a name (2-4 words, capitalized, not too long)
    words = first_line.split()
    if 2 <= len(words) <= 4 and len(first_line) < 50:
        # Check if words are mostly capitalized (likely a name)
        if sum(1 for w in words if w and w[0].isupper()) >= len(words) * 0.5:
            return first_line

    # Pattern 2: Look for "Name:" pattern
    for line in lines[:5]:
        match = re.search(r'(?:name|candidate)\s*[:-]\s*(.+)', line, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Pattern 3: Look for email and extract name before @
    email_pattern = r'([a-zA-Z]+(?:\.[a-zA-Z]+)?(?:\.[a-zA-Z]+)?)@'
    for line in lines[:10]:
        match = re.search(email_pattern, line)
        if match:
            email_name = match.group(1).replace('.', ' ').title()
            if len(email_name.split()) >= 2:
                return email_name

    # Fallback: return first line if it's reasonable
    if first_line and len(first_line) < 50:
        return first_line

    return "Unknown Candidate"


def parse_resume_pdf(file_path: Path, text: str) -> Dict[str, str]:
    """
    Parse a single resume PDF and extract structured fields.

    Parameters
    ----------
    file_path : Path
        Path to the PDF file
    text : str
        Extracted text from the PDF

    Returns
    -------
    Dict[str, str]
        Dictionary with resume_id and extracted sections
    """
    resume_id = file_path.stem  # Use filename without extension as ID

    # Extract candidate name
    candidate_name = extract_candidate_name(text)

    # Parse sections
    sections = PDFSectionParser.parse_resume_sections(text)

    # Extract or infer required fields
    skills = sections.get("skills", "") or PDFSectionParser.extract_skills_heuristic(text)
    experience = sections.get("experience", "")
    projects = sections.get("projects", "")
    education = sections.get("education", "")
    certifications = sections.get("certifications", "")

    # If experience section is empty, use years of experience
    if not experience:
        years = PDFSectionParser.extract_years_of_experience(text)
        experience = f"{years} years" if years > 0 else "Not specified"

    return {
        "resume_id": resume_id,
        "candidate_name": candidate_name,
        "skills": skills,
        "experience": experience,
        "projects": projects,
        "education": education,
        "certifications": certifications,
    }


def parse_jd_pdf(file_path: Path, text: str) -> Dict[str, str]:
    """
    Parse a single job description PDF.

    Parameters
    ----------
    file_path : Path
        Path to the PDF file
    text : str
        Extracted text from the PDF

    Returns
    -------
    Dict[str, str]
        Dictionary with job_id, job_title, and job_text
    """
    job_id = file_path.stem  # Use filename without extension as ID

    # Try to extract job title from the first few lines
    lines = text.strip().split('\n')
    job_title = lines[0].strip() if lines else "Unknown Position"

    # Common title patterns
    title_patterns = [
        r'(?:position|role|title)\s*:\s*(.+)',
        r'^(.+?(?:engineer|developer|scientist|analyst|manager|director|lead))',
    ]

    for pattern in title_patterns:
        match = re.search(pattern, text[:500], re.IGNORECASE | re.MULTILINE)
        if match:
            job_title = match.group(1).strip()
            break

    return {
        "job_id": job_id,
        "job_title": job_title,
        "job_text": text.strip(),
    }

