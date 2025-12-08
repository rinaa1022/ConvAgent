"""
Lightweight caching layer for job listings fetched from GitHub.
"""

from __future__ import annotations

import time
from typing import Dict, List, Literal, Optional, Tuple

from job_sources import (
    JobPosting,
    JobSourceError,
    fetch_markdown,
    get_full_time_jobs,
    get_internship_jobs,
    FULL_TIME_README_URL,
    INTERNSHIP_README_URL,
)

from job_parser import JobParser
from unified_neo4j_manager import UnifiedNeo4jManager
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from unified_schema import normalize_skill_name, expand_skill_name

CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours
MAX_POSTINGS_PER_CATEGORY = 50

USE_JOB_PARSER = True

SourceType = Literal["full_time", "internship"]

ALL_CATEGORIES_BY_SOURCE: Dict[SourceType, list[str]] = {
    "internship": [
        "software_engineering",
        "data_science",
        "product_management",
        "quant",
        "hardware",
    ],
    "full_time": [
        "software_engineering",
        "data_science",
        "product_management",
        "quant",
        "hardware",
        "other",
    ],
}


class JobCacheEntry:
    __slots__ = ("postings", "timestamp")

    def __init__(self, postings: List[JobPosting], timestamp: float):
        self.postings = postings
        self.timestamp = timestamp


class JobCache:
    def __init__(self):
        # (source, category) -> JobCacheEntry
        self._store: Dict[Tuple[SourceType, str], JobCacheEntry] = {}

    def get_entry(self, source: SourceType, category: str) -> Optional[JobCacheEntry]:
        entry = self._store.get((source, category))
        if not entry:
            return None
        if time.time() - entry.timestamp > CACHE_TTL_SECONDS:
            # expired → drop it
            self._store.pop((source, category), None)
            return None
        return entry

    def set_entry(self, source: SourceType, category: str, postings: List[JobPosting]) -> None:
        self._store[(source, category)] = JobCacheEntry(postings, time.time())


cache = JobCache()

# --- Neo4j & JobParser singletons ---
_neo4j_manager: Optional[UnifiedNeo4jManager] = None
_job_parser: Optional[JobParser] = None


def _get_neo4j_manager() -> Optional[UnifiedNeo4jManager]:
    """Initialize UnifiedNeo4jManager, preferring Streamlit config."""
    global _neo4j_manager
    if _neo4j_manager is not None:
        return _neo4j_manager

    # Try to read the same config that app.py uses
    try:
        import streamlit as st
        uri = st.session_state.get("config_neo4j_uri", NEO4J_URI)
        user = st.session_state.get("config_neo4j_user", NEO4J_USER)
        pwd  = st.session_state.get("config_neo4j_password", NEO4J_PASSWORD)
    except Exception:
        uri, user, pwd = NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

    try:
        _neo4j_manager = UnifiedNeo4jManager(uri, user, pwd)
    except Exception as e:
        print(f"[job_cache] Could not initialize Neo4j manager: {e}")
        _neo4j_manager = None
    return _neo4j_manager


def _get_job_parser() -> Optional[JobParser]:
    """
    Initialize JobParser with the same LLM config used for resume parsing.

    If we can't read Streamlit config or provider/key are missing, we still create
    a JobParser, but it will just return empty results (no fake skills).
    """
    global _job_parser
    if not USE_JOB_PARSER:
        return None
    if _job_parser is not None:
        return _job_parser

    llm_provider = None
    api_key = None
    try:
        import streamlit as st
        llm_provider = st.session_state.get("config_llm_provider")
        api_key = st.session_state.get("config_api_key")
    except Exception:
        # Not running inside Streamlit; leave as None → parser returns empty
        pass

    try:
        _job_parser = JobParser(llm_provider=llm_provider, api_key=api_key)
    except Exception as e:
        print(f"[job_cache] JobParser not available: {e}")
        _job_parser = None

    return _job_parser


def sync_jobs_to_neo4j(postings: List[JobPosting]) -> None:
    """
    Push a batch of JobPosting objects into Neo4j as Job nodes + relationships.

    For each JobPosting:
    - JobParser fetches the REAL job description from apply_url
    - LLM extracts "skills_required" + "description"
    - We expand composite names (e.g., "C/C++") into multiple skills
    - Store in KG via UnifiedNeo4jManager.upsert_jobs(...)
    """
    manager = _get_neo4j_manager()
    if manager is None:
        return

    parser = _get_job_parser()

    jobs_payload = []
    for p in postings:
        try:
            skills: List[str] = []
            description: str = ""

            if parser is not None:
                try:
                    parsed = parser.parse_job(p)
                    skills = parsed.get("skills_required") or []
                    description = parsed.get("description") or ""
                except Exception as e:
                    print(f"[job_cache] JobParser failed on {p.company} / {p.role}: {e}")

            # No fake skills: if JobParser returned none, skills stays [].

            # Expand names like "C/C++" into ["C", "C++"] via your ontology helper
            normalized_skills: List[str] = []
            for s in skills:
                if not isinstance(s, str) or not s.strip():
                    continue
                normalized_skills.extend(expand_skill_name(str(s)))

            # Build a stable job_id (in case JobPosting doesn't have one)
            job_id = getattr(p, "job_id", None) or f"{p.source}:{p.company}:{p.role}"

            jobs_payload.append(
                {
                    "job_id": job_id,
                    "company": p.company,
                    "role": p.role,
                    "location": p.location,
                    "apply_url": p.apply_url,
                    "age": p.age,
                    "category": p.category,
                    "source": p.source,
                    "skills_required": normalized_skills,
                    "description": description,
                }
            )
        except Exception as e:
            print(f"[job_cache] Skipping job during payload build: {e}")
            continue

    if not jobs_payload:
        return

    manager.upsert_jobs(jobs_payload)


def _fetch_and_parse(source: SourceType, category: str) -> List[JobPosting]:
    """
    Fetch JUST the markdown once and parse ONLY the requested category.
    We keep up to MAX_POSTINGS_PER_CATEGORY in memory, but only sync 1 page
    at a time into Neo4j.
    """
    if source == "full_time":
        markdown = fetch_markdown(FULL_TIME_README_URL)
        postings = get_full_time_jobs(markdown, category)
    else:
        markdown = fetch_markdown(INTERNSHIP_README_URL)
        postings = get_internship_jobs(markdown, category)

    if MAX_POSTINGS_PER_CATEGORY is not None:
        postings = postings[:MAX_POSTINGS_PER_CATEGORY]

    return postings


def get_jobs(source: SourceType, category: str, force_refresh: bool = False, page: int = 1, page_size: int = 100) -> List[JobPosting]:
    """
    Return *one page* of job postings for (source, category).

    Behaviour:
    - Cache stores the FULL list of postings (up to MAX_POSTINGS_PER_CATEGORY)
      per (source, category), refreshed every 24h.
    - Each call only syncs the requested page slice into Neo4j.
      e.g., page=1 → jobs[0:50], page=2 → jobs[50:100].
    """

    if page < 1:
        page = 1
    if page_size <= 0:
        page_size = 50

    #Load from cache or GitHub
    postings: List[JobPosting]

    if not force_refresh:
        entry = cache.get_entry(source, category)
        if entry is not None:
            postings = entry.postings
        else:
            postings = _fetch_and_parse(source, category)
            cache.set_entry(source, category, postings)
    else:
        postings = _fetch_and_parse(source, category)
        cache.set_entry(source, category, postings)

    if not postings:
        return []

    # Slice the requested page
    start = (page - 1) * page_size
    end = start + page_size
    page_postings = postings[start:end]

    if not page_postings:
        # No more jobs in this category for that page
        return []

    # Sync ONLY this page into Neo4j as Job nodes
    try:
        sync_jobs_to_neo4j(page_postings)
    except Exception as e:
        print(f"[job_cache] Failed to sync jobs to Neo4j (page={page}): {e}")

    return page_postings


def get_cache_age_hours(source: SourceType, category: str) -> Optional[float]:
    entry = cache.get_entry(source, category)
    if not entry:
        return None
    return (time.time() - entry.timestamp) / 3600
