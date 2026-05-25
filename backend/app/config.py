"""
Application configuration loaded from environment variables.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from .env file."""

    GEMINI_API_KEY: str
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = True


def get_settings() -> Settings:
    """Get singleton settings instance."""
    return Settings()
