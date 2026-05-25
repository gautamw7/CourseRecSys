"""Gemini LLM service for profile extraction and course ranking."""

import json
import logging
import re
from typing import Any, Dict, List

from google import genai
import tenacity

from app.models.user_profile import UserProfile

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a resume parser. Extract structured information from the resume text below.

Return ONLY a valid JSON object with these exact keys:
{{
  "name": "full name or empty string",
  "email": "email or empty string",
  "phone": "phone number or empty string",
  "education_level": "one of: High School, Bachelor's, Master's, PhD, Other",
  "education_field": "field of study, e.g. Computer Science",
  "years_experience": "integer (0 if student/fresher)",
  "skills": ["list", "of", "technical", "skills"],
  "interests": ["topics", "the person", "mentions interest in"],
  "certificates": ["certification names only, not courses taken"],
  "profile_summary": "2-3 sentence neutral summary of who this person is professionally"
}}

Rules:
- skills must include ALL technical tools, languages, frameworks mentioned
- certificates means formal certifications (AWS Certified, Google Analytics, etc.), NOT coursework
- If a field cannot be determined, use the default (empty string, 0, or [])
- Do NOT wrap the JSON in markdown code blocks
- Do NOT add any explanation text, return only the JSON

Resume text:
{resume_text}"""

RANKING_PROMPT = """You are a course recommender. Given a user profile and a list of courses, rank the courses by fit.

User Profile:
{profile_json}

Courses (with index):
{courses_json}

Return ONLY a valid JSON array where each element is:
{{
  "index": original_course_index,
  "relevance_score": float between 0 and 1,
  "explanation": "one sentence: why this course fits this user specifically"
}}

Order by relevance_score descending. Include all courses. Do NOT add markdown or explanation text."""


class GeminiService:
    """Service for Gemini LLM API calls with retry logic."""

    def __init__(self, api_key: str, model_name: str = "gemini-3.5-flash"):
        """
        Initialize Gemini service.

        Args:
            api_key: Gemini API key
            model_name: Model to use (default: gemini-3.5-flash)
        """
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def extract_profile(self, pdf_text: str) -> UserProfile:
        """
        Extract structured profile from resume text using Gemini.

        Args:
            pdf_text: Raw resume text extracted from PDF

        Returns:
            UserProfile with extracted fields

        Raises:
            ValueError: If Gemini response cannot be parsed as valid JSON
        """
        prompt = EXTRACTION_PROMPT.format(resume_text=pdf_text)
        response = self._client.models.generate_content(
            model=self._model_name, contents=prompt
        )
        json_str = self._extract_json(response.text)

        try:
            data = json.loads(json_str)
            return UserProfile(**data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {response.text[:500]}")
            raise ValueError(f"Invalid JSON from Gemini: {e}") from e

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def rank_courses(
        self, profile: UserProfile, courses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rank courses by relevance to user profile using Gemini.

        Args:
            profile: Extracted user profile
            courses: List of course dicts to rank

        Returns:
            List of courses ranked by relevance_score descending.
            Each item: {"index", "relevance_score", "explanation"}

        Raises:
            ValueError: If Gemini response cannot be parsed as valid JSON
        """
        profile_json = profile.model_dump_json(exclude={"raw_text"}, indent=2)
        courses_json = json.dumps(
            [{"index": i, **course} for i, course in enumerate(courses)], indent=2
        )

        prompt = RANKING_PROMPT.format(
            profile_json=profile_json, courses_json=courses_json
        )
        response = self._client.models.generate_content(
            model=self._model_name, contents=prompt
        )
        json_str = self._extract_json(response.text)

        try:
            ranked = json.loads(json_str)
            ranked.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            return ranked
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini ranking response: {response.text[:500]}")
            raise ValueError(f"Invalid ranking JSON from Gemini: {e}") from e

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Extract JSON from text, handling Gemini's occasional prose wrapping.

        If text starts with raw JSON, return as-is.
        If not, try regex to find JSON object `{...}` or array `[...]`.

        Args:
            text: Response text from Gemini

        Returns:
            JSON string (object or array)

        Raises:
            ValueError: If no valid JSON found
        """
        text = text.strip()

        if text.startswith("{") or text.startswith("["):
            return text

        match = re.search(r"[{\[].*[}\]]", text, re.DOTALL)
        if match:
            return match.group(0)

        raise ValueError(f"No JSON found in Gemini response: {text[:200]}")
