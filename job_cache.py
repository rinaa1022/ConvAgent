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

SourceType = Literal["full_time", "internship"]

CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours


class JobCacheEntry:
    __slots__ = ("postings", "timestamp")

    def __init__(self, postings: List[JobPosting], timestamp: float):
        self.postings = postings
        self.timestamp = timestamp


class JobCache:
    def __init__(self):
        self._store: Dict[Tuple[SourceType, str], JobCacheEntry] = {}

    def get_entry(self, source: SourceType, category: str) -> Optional[JobCacheEntry]:
        entry = self._store.get((source, category))
        if not entry:
            return None
        if time.time() - entry.timestamp > CACHE_TTL_SECONDS:
            self._store.pop((source, category), None)
            return None
        return entry

    def set(self, source: SourceType, category: str, postings: List[JobPosting]) -> None:
        self._store[(source, category)] = JobCacheEntry(postings, time.time())


cache = JobCache()


def _fetch_and_parse(source: SourceType, category: str) -> List[JobPosting]:
    if source == "full_time":
        markdown = fetch_markdown(FULL_TIME_README_URL)
        return get_full_time_jobs(markdown, category)
    markdown = fetch_markdown(INTERNSHIP_README_URL)
    return get_internship_jobs(markdown, category)


def get_jobs(source: SourceType, category: str, force_refresh: bool = False) -> List[JobPosting]:
    """
    Return cached job postings for the source/category combination, refreshing if needed.
    Raises JobSourceError when data cannot be fetched.
    """
    if not force_refresh:
        entry = cache.get_entry(source, category)
        if entry is not None:
            return entry.postings

    postings = _fetch_and_parse(source, category)
    cache.set(source, category, postings)
    return postings


def get_cache_age_hours(source: SourceType, category: str) -> Optional[float]:
    entry = cache.get_entry(source, category)
    if not entry:
        return None
    return (time.time() - entry.timestamp) / 3600

