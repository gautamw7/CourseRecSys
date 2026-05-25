"""Resume upload and recommendation endpoint."""

import logging
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.services.pdf_extractor import PDFExtractorService
from app.services.gemini_service import GeminiService
from app.services.vector_search import VectorSearchService
from app.models.user_profile import UserProfile
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["recommendations"])


class RecommendationResponse(BaseModel):
    """API response for recommendations."""

    status: str
    user_profile: UserProfile
    recommendations: list[dict]


@router.post("/upload", response_model=RecommendationResponse)
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload resume and get course recommendations.

    Request:
      file: PDF resume file

    Response:
      RecommendationResponse with extracted profile and ranked courses

    Raises:
      HTTPException: 400 if file is invalid, 500 if processing fails
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name missing")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    settings = get_settings()

    try:
        # 1. Extract text from PDF
        pdf_extractor = PDFExtractorService()
        file_content = await file.read()

        temp_dir = Path(tempfile.gettempdir())
        temp_path = temp_dir / file.filename

        with open(temp_path, "wb") as f:
            f.write(file_content)

        pdf_text = pdf_extractor.extract_text(str(temp_path))

        if not pdf_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF. Please try another file.",
            )

        logger.info(f"Extracted {len(pdf_text)} chars from {file.filename}")

        # 2. Parse resume with Gemini
        gemini = GeminiService(api_key=settings.GEMINI_API_KEY)
        profile = gemini.extract_profile(pdf_text)

        logger.info(
            f"Extracted profile: {profile.name or 'Unknown'}, "
            f"skills={len(profile.skills)}, exp={profile.years_experience}yr"
        )

        # 3. Search for relevant courses
        search_service = VectorSearchService(
            qdrant_url=settings.QDRANT_URL,
            qdrant_api_key=settings.QDRANT_API_KEY,
        )

        # Build query from profile
        query_parts = [
            profile.profile_summary or "Courses for learning",
            " ".join(profile.skills[:5]) if profile.skills else "",
            " ".join(profile.interests[:3]) if profile.interests else "",
        ]
        query = " ".join([p for p in query_parts if p]).strip()

        if not query:
            query = "Professional development courses"

        courses = search_service.search(query, limit=10)
        logger.info(f"Found {len(courses)} candidate courses")

        # 4. Rank courses with Gemini, then merge full course data back in
        ranked = gemini.rank_courses(profile, courses)
        recommendations = []
        for item in ranked:
            idx = item.get("index", 0)
            if 0 <= idx < len(courses):
                course = courses[idx].copy()
                course["relevance_score"] = item["relevance_score"]
                course["explanation"] = item["explanation"]
                recommendations.append(course)

        logger.info(f"Ranked {len(recommendations)} courses by relevance")

        return RecommendationResponse(
            status="success",
            user_profile=profile,
            recommendations=recommendations,
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400, detail=f"Invalid resume format: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to process resume: {str(e)}"
        )
