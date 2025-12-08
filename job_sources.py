"""
Utilities for fetching and parsing job listings from public GitHub repositories.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import requests

try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:  # pragma: no cover
    BeautifulSoup = None  # type: ignore


FULL_TIME_README_URL = "https://raw.githubusercontent.com/SimplifyJobs/New-Grad-Positions/dev/README.md"
INTERNSHIP_README_URL = "https://raw.githubusercontent.com/SimplifyJobs/Summer2026-Internships/dev/README.md"

FULL_TIME_SECTION_MARKERS = {
    "software_engineering": "## 💻 Software Engineering New Grad Roles",
    "data_science": "## 🤖 Data Science, AI & Machine Learning New Grad Roles",
    "product_management": "## 📱 Product Management New Grad Roles",
    "quant": "## 📈 Quantitative Finance New Grad Roles",
    "hardware": "## 🔧 Hardware Engineering New Grad Roles",
    "other": "## 💼 Other New Grad Roles",
}

INTERNSHIP_SECTION_MARKERS = {
    "software_engineering": "## 💻 Software Engineering Internship Roles",
    "data_science": "## 🤖 Data Science, AI & Machine Learning Internship Roles",
    "product_management": "## 📱 Product Management Internship Roles",
    "quant": "## 📈 Quantitative Finance Internship Roles",
    "hardware": "## 🔧 Hardware Engineering Internship Roles",
}


@dataclass
class JobPosting:
    job_id: str 
    company: str
    role: str
    location: str
    apply_url: Optional[str]
    age: Optional[str]
    category: str
    source: str


class JobSourceError(Exception):
    """Raised when job data cannot be fetched or parsed."""


def fetch_markdown(url: str, timeout: int = 10) -> str:
    """Download raw Markdown from GitHub."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network error
        raise JobSourceError(f"Failed to fetch job data: {exc}") from exc
    return response.text


def _extract_table_html(markdown_text: str, marker: str) -> Optional[str]:
    """Return the first HTML table string following the marker."""
    if not marker:
        return None

    marker_index = markdown_text.lower().find(marker.lower())
    if marker_index == -1:
        return None

    search_region = markdown_text[marker_index:]
    match = re.search(r"<table.*?</table>", search_region, re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    return match.group(0)


def _parse_table(table_html: str, category: str, source: str) -> List[JobPosting]:
    """Convert an HTML table into structured job postings."""
    if not BeautifulSoup:  # pragma: no cover - dependency missing
        logging.warning("BeautifulSoup is not available; cannot parse job tables.")
        return []

    soup = BeautifulSoup(table_html, "html.parser")
    rows = soup.find_all("tr")
    postings: List[JobPosting] = []

    for row in rows[1:]:  # skip header
        cols = row.find_all("td")
        if len(cols) < 4:
            continue

        company = cols[0].get_text(strip=True)
        role = cols[1].get_text(strip=True)
        location = cols[2].get_text(" ", strip=True)

        apply_link = None
        for link in cols[3].find_all("a", href=True):
            apply_link = link["href"]
            if apply_link:
                break

        age = cols[4].get_text(strip=True) if len(cols) > 4 else None

        base_id = f"{source}:{category}:{company}:{role}:{location}"
        job_id = re.sub(r"\s+", "_", base_id.strip().lower())
        
        postings.append(
            JobPosting(
                job_id=job_id,
                company=company,
                role=role,
                location=location,
                apply_url=apply_link,
                age=age,
                category=category,
                source=source,
            )
        )

    return postings


def parse_job_section(markdown_text: str, marker: str, category_key: str, source: str) -> List[JobPosting]:
    """Extract and parse a single section from the README."""
    table_html = _extract_table_html(markdown_text, marker)
    if not table_html:
        logging.warning("No table found for marker '%s'", marker)
        return []
    return _parse_table(table_html, category_key, source)


def get_full_time_jobs(markdown_text: str, category_key: str) -> List[JobPosting]:
    """Return full-time roles for the requested category."""
    marker = FULL_TIME_SECTION_MARKERS.get(category_key)
    if not marker:
        return []
    return parse_job_section(markdown_text, marker, category_key, "full_time")


def get_internship_jobs(markdown_text: str, category_key: str) -> List[JobPosting]:
    """Return internship roles for the requested category."""
    marker = INTERNSHIP_SECTION_MARKERS.get(category_key)
    if not marker:
        return []
    return parse_job_section(markdown_text, marker, category_key, "internship")

