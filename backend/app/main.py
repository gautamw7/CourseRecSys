"""FastAPI application factory and startup."""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import health_router, upload_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Course Recommendation API",
        description="Personalized course recommendations based on resume analysis",
        version="1.0.0",
    )

    # CORS for frontend (localhost:5173 in dev)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(health_router)
    app.include_router(upload_router)

    @app.on_event("startup")
    async def startup():
        logger.info("Course Recommendation API starting up")

    @app.on_event("shutdown")
    async def shutdown():
        logger.info("Course Recommendation API shutting down")

    return app


app = create_app()
