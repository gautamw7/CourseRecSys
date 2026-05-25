"""Health check endpoint."""

from fastapi import APIRouter
from app.services.vector_search import VectorSearchService
from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """
    Check service health.

    Returns cluster status and readiness.
    """
    settings = get_settings()
    search_service = VectorSearchService(
        qdrant_url=settings.QDRANT_URL,
        qdrant_api_key=settings.QDRANT_API_KEY,
    )

    is_ready = search_service.health_check()

    return {
        "status": "ready" if is_ready else "degraded",
        "qdrant": "connected" if is_ready else "disconnected",
    }
