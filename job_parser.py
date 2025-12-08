"""
JobParser: fetch real job descriptions from apply_url and use the configured
LLM (OpenAI / Anthropic / Google) to extract required skills + a short description.

- It SCRAPES the job URL (requests + BeautifulSoup)
- Sends the cleaned text to the LLM
- Expects JSON: { "skills_required": [...], "description": "..." }

If anything fails (HTTP, parsing, LLM, JSON), it returns empty skills/description.
NO fake heuristic fallback.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import requests
from bs4 import BeautifulSoup

from job_sources import JobPosting


class JobParser:
    """
    LLM-based JobParser that works off the REAL job description text.
    """

    def __init__(self, llm_provider: Optional[str], api_key: Optional[str], timeout: int = 10):
        """
        llm_provider: "OpenAI" | "Anthropic" | "Google" | None
        api_key: API key for the chosen provider

        If provider or api_key is None, parse_job will simply return empty results.
        """
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_job(self, posting: JobPosting) -> Dict[str, Any]:
        """
        Parse a JobPosting and return:
            {
              "skills_required": [str, ...],
              "description": str
            }

        Steps:
          1. Fetch HTML from posting.apply_url
          2. Extract visible text
          3. Call LLM with that text + metadata
          4. Parse JSON from LLM

        If any step fails, we return empty lists/strings (NO fake skills).
        """
        # If we don't have an LLM configured, do nothing.
        if not self.llm_provider or not self.api_key:
            print("[JobParser] No LLM provider/API key configured; skipping job parsing.")
            return {"skills_required": [], "description": ""}

        url = posting.apply_url
        if not url:
            return {"skills_required": [], "description": ""}

        html = self._fetch_html(url)
        if not html:
            # Could not fetch the page
            return {"skills_required": [], "description": ""}

        text = self._extract_visible_text(html)
        if not text:
            # No visible text extracted
            return {"skills_required": [], "description": ""}

        # Truncate text so prompt isn't enormous
        job_text_snippet = text[:20000]

        try:
            parsed = self._parse_with_llm(posting, job_text_snippet)
            skills = parsed.get("skills_required") or []
            desc = parsed.get("description") or ""

            # Clean up skills list
            skills = [str(s).strip() for s in skills if str(s).strip()]

            return {
                "skills_required": skills,
                "description": str(desc).strip(),
            }

        except Exception as e:
            print(f"[JobParser] LLM parse error for {posting.company} / {posting.role}: {e}")
            return {"skills_required": [], "description": ""}

    # ------------------------------------------------------------------
    # LLM logic
    # ------------------------------------------------------------------

    def _parse_with_llm(self, posting: JobPosting, text: str) -> Dict[str, Any]:
        """
        Call the configured LLM with the real job text and metadata,
        expect a JSON object back.
        """
        category = posting.category or ""

        system_prompt = """
        You are a precise job-description parser.

        Your input:
        - Real job description text scraped from the company's careers page
        - Metadata: Company, Role, Category, Location, Source, URL

        Categories (from a curated GitHub list):
        - software_engineering
        - data_science
        - product_management
        - quant
        - hardware
        - other

        Use the Category as a hint, BUT:
        - Your PRIMARY source of truth is the job description TEXT.
        - Only include skills that are clearly supported by the text
        (requirements, responsibilities, preferred qualifications, required qualifications, skills, what we're looking for, what you'll bring, etc.).

        Output:
        A SINGLE JSON object with this exact structure:

        {
        "skills_required": ["Skill1", "Skill2", ...],
        "description": "1–2 sentence plain-English summary of the role based on the text."
        }

        Rules:
        - "skills_required" should include ALL distinct skills that are clearly supported
          by the text (requirements, responsibilities, preferred qualifications, required qualifications, skills, what we're looking for, what you'll bring, etc.).
          Aim to capture every concrete technology, tool, language, framework,
          platform, and any clearly mentioned soft skills.
        - Each skill should be a short name or phrase, e.g. "Python", "C++", "Git",
          "Linux", "Vue.js", "Selenium", "AWS", "Teamwork", "Communication".
        - Include BOTH technical and non-technical skills if they are clearly implied
          (e.g. "communication", "teamwork").
        - Do NOT hallucinate skills not supported by the text.

        """

        user_prompt = f"""
        Company: {posting.company}
        Role: {posting.role}
        Category: {category}
        Location: {posting.location}
        Source: {posting.source}
        Apply URL: {posting.apply_url or "N/A"}

        JOB DESCRIPTION TEXT (scraped):

        {text}

        Now return ONLY a JSON object in this format:

        {{
        "skills_required": ["Skill1", "Skill2"],
        "description": "Short summary here."
        }}
        """

        raw = self._call_llm(system_prompt, user_prompt)

        # Sometimes models emit ```json ... ``` – strip if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("LLM did not return a JSON object.")
        return data

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the configured LLM provider and return raw text content.
        Mirrors your resume parser style.
        """
        provider = self.llm_provider
        api_key = self.api_key

        if provider == "OpenAI":
            import openai

            openai.api_key = api_key
            resp = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1000,
            )
            return resp.choices[0].message.content or ""

        elif provider == "Anthropic":
            import requests as _req

            headers = {
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }
            data = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1000,
                "temperature": 0.2,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            resp = _req.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
            )
            resp.raise_for_status()
            j = resp.json()
            blocks = j.get("content", [])
            if not blocks:
                return ""
            # each block: {"type": "text", "text": "..."}
            return "".join(b.get("text", "") for b in blocks)

        elif provider == "Google":
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-pro")
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            resp = model.generate_content(
                full_prompt,
                generation_config={
                    "max_output_tokens": 1000,
                    "temperature": 0.2,
                },
            )
            return resp.text or ""

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _fetch_html(self, url: str) -> Optional[str]:
        """
        Fetch raw HTML from the job's apply URL.
        """
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            print(f"[JobParser] Failed to fetch URL {url}: {e}")
            return None

    def _extract_visible_text(self, html: str) -> str:
        """
        Strip scripts/styles/nav/etc and return visible text.
        """
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()
