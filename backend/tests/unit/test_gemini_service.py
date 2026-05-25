"""TDD tests for Gemini service (mocked, no real API calls)."""

import json
import pytest
from unittest.mock import MagicMock, patch

from app.services.gemini_service import GeminiService
from app.models.user_profile import UserProfile


MOCK_EXTRACTION_RESPONSE = {
    "name": "Test User",
    "email": "test@example.com",
    "phone": "+1234567890",
    "education_level": "Bachelor's",
    "education_field": "Computer Science",
    "years_experience": 1,
    "skills": ["Python", "SQL", "React"],
    "interests": ["Machine Learning", "Web Development"],
    "certificates": [],
    "profile_summary": "A CS graduate with Python and SQL skills interested in ML.",
}

MOCK_RANKING_RESPONSE = [
    {
        "index": 0,
        "relevance_score": 0.92,
        "explanation": "Perfect match for Python and ML skills.",
    },
    {
        "index": 2,
        "relevance_score": 0.75,
        "explanation": "Good coverage of SQL fundamentals.",
    },
    {
        "index": 1,
        "relevance_score": 0.60,
        "explanation": "Tangentially related to interests.",
    },
]


class TestGeminiExtractProfile:
    """Test profile extraction with mocked Gemini API."""

    def test_valid_response_returns_user_profile(self):
        """Valid Gemini JSON response should return UserProfile instance."""
        with patch("google.generativeai.GenerativeModel") as MockModel:
            mock_response = MagicMock()
            mock_response.text = json.dumps(MOCK_EXTRACTION_RESPONSE)
            MockModel.return_value.generate_content.return_value = mock_response

            service = GeminiService(api_key="fake-key")
            profile = service.extract_profile("any resume text")

            assert isinstance(profile, UserProfile)
            assert profile.education_level == "Bachelor's"
            assert profile.name == "Test User"

    def test_skills_always_populated(self):
        """Skills field should always be a list."""
        with patch("google.generativeai.GenerativeModel") as MockModel:
            mock_response = MagicMock()
            mock_response.text = json.dumps(MOCK_EXTRACTION_RESPONSE)
            MockModel.return_value.generate_content.return_value = mock_response

            service = GeminiService(api_key="fake-key")
            profile = service.extract_profile("resume")

            assert profile.skills is not None
            assert isinstance(profile.skills, list)
            assert len(profile.skills) > 0

    def test_education_level_always_populated(self):
        """Education level should never be empty."""
        with patch("google.generativeai.GenerativeModel") as MockModel:
            response_data = MOCK_EXTRACTION_RESPONSE.copy()
            response_data["education_level"] = "Master's"
            mock_response = MagicMock()
            mock_response.text = json.dumps(response_data)
            MockModel.return_value.generate_content.return_value = mock_response

            service = GeminiService(api_key="fake-key")
            profile = service.extract_profile("resume")

            assert profile.education_level != "Other"

    def test_malformed_json_raises_value_error(self):
        """Malformed Gemini response should raise ValueError."""
        with patch("google.generativeai.GenerativeModel") as MockModel:
            mock_response = MagicMock()
            mock_response.text = "Sorry, I cannot parse this resume format."
            MockModel.return_value.generate_content.return_value = mock_response

            service = GeminiService(api_key="fake-key")

            with pytest.raises(ValueError):
                service.extract_profile("malformed input")

    def test_partial_response_still_valid(self):
        """Missing optional fields should still produce valid UserProfile."""
        minimal_response = {
            "name": "User",
            "skills": ["Python"],
        }
        with patch("google.generativeai.GenerativeModel") as MockModel:
            mock_response = MagicMock()
            mock_response.text = json.dumps(minimal_response)
            MockModel.return_value.generate_content.return_value = mock_response

            service = GeminiService(api_key="fake-key")
            profile = service.extract_profile("resume")

            assert profile.name == "User"
            assert profile.skills == ["Python"]
            assert profile.education_level == "Other"  # default


class TestGeminiRankCourses:
    """Test course ranking with mocked Gemini API."""

    def test_ranking_returns_sorted_list(self):
        """Ranking should return list sorted by relevance_score descending."""
        with patch("google.generativeai.GenerativeModel") as MockModel:
            mock_response = MagicMock()
            mock_response.text = json.dumps(MOCK_RANKING_RESPONSE)
            MockModel.return_value.generate_content.return_value = mock_response

            service = GeminiService(api_key="fake-key")
            profile = UserProfile(**MOCK_EXTRACTION_RESPONSE)
            courses = [
                {"title": "Python ML", "course_id": 0},
                {"title": "React Basics", "course_id": 1},
                {"title": "SQL Advanced", "course_id": 2},
            ]

            result = service.rank_courses(profile, courses)

            assert isinstance(result, list)
            assert len(result) == 3
            assert result[0]["relevance_score"] >= result[1]["relevance_score"]
            assert result[1]["relevance_score"] >= result[2]["relevance_score"]

    def test_ranking_includes_explanations(self):
        """Each ranked course should include an explanation."""
        with patch("google.generativeai.GenerativeModel") as MockModel:
            mock_response = MagicMock()
            mock_response.text = json.dumps(MOCK_RANKING_RESPONSE)
            MockModel.return_value.generate_content.return_value = mock_response

            service = GeminiService(api_key="fake-key")
            profile = UserProfile(**MOCK_EXTRACTION_RESPONSE)
            courses = [{"title": "Course 1"}]

            result = service.rank_courses(profile, courses)

            for item in result:
                assert "explanation" in item
                assert isinstance(item["explanation"], str)
                assert len(item["explanation"]) > 0

    def test_malformed_ranking_response_raises(self):
        """Invalid ranking JSON should raise ValueError."""
        with patch("google.generativeai.GenerativeModel") as MockModel:
            mock_response = MagicMock()
            mock_response.text = "Cannot rank these courses."
            MockModel.return_value.generate_content.return_value = mock_response

            service = GeminiService(api_key="fake-key")
            profile = UserProfile(**MOCK_EXTRACTION_RESPONSE)

            with pytest.raises(ValueError):
                service.rank_courses(profile, [{"title": "Course 1"}])
