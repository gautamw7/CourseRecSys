"""User profile schema extracted from resume."""

from typing import List, Literal
from pydantic import BaseModel


class UserProfile(BaseModel):
    """Structured user profile extracted from resume via Gemini."""

    name: str = ""
    email: str = ""
    phone: str = ""
    education_level: Literal[
        "High School", "Bachelor's", "Master's", "PhD", "Other"
    ] = "Other"
    education_field: str = ""
    years_experience: int = 0
    skills: List[str] = []
    interests: List[str] = []
    certificates: List[str] = []
    profile_summary: str = ""
    raw_text: str = ""
