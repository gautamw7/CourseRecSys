"""
Vector search service: Query Qdrant for courses matching user profiles.

This service encapsulates Qdrant interactions:
- Connect to database
- Encode user profiles to vectors
- Search for similar courses
- Return ranked results
"""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


class VectorSearchService:
    """Search courses using semantic similarity in Qdrant."""

    def __init__(self, qdrant_url: str, qdrant_api_key: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector search service.

        Args:
            qdrant_url: Qdrant cluster URL
            qdrant_api_key: Qdrant API key
            embedding_model: SentenceTransformer model name
        """
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.collection_name = "courses"

    def _encode_text(self, text: str) -> List[float]:
        """Encode text to embedding vector."""
        return self.embedding_model.encode(text).tolist()

    def search(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for courses matching query text.

        Args:
            query_text: User profile or query text (title, skills, description, etc.)
            limit: Number of results to return

        Returns:
            List of courses ranked by similarity score, each with:
            - title, platform, provider, skills, rating, url, etc.
            - score: Similarity score (0-1, higher = better match)
        """
        # Encode query
        query_vector = self._encode_text(query_text)

        # Search Qdrant
        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        # Format results
        courses = []
        for point in search_results.points:
            course = point.payload.copy()
            course["score"] = float(point.score)  # Similarity score
            courses.append(course)

        return courses

    def health_check(self) -> bool:
        """Check if Qdrant connection is healthy."""
        try:
            collections = self.qdrant_client.get_collections()
            return True
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
