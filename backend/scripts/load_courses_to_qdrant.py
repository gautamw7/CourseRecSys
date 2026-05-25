"""
Load courses from cleaned CSV into Qdrant vector database.

This script:
1. Reads Data/courses_cleaned.csv
2. Generates embeddings with SentenceTransformer
3. Uploads to Qdrant (cloud or local)
4. Creates searchable collection

Usage:
    python scripts/load_courses_to_qdrant.py
"""

import csv
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# Add parent directory to path so we can import app config
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings

# Configuration
COLLECTION_NAME = "courses"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DATA_FILE = Path(__file__).parent.parent.parent / "Data" / "courses_cleaned.csv"


def load_courses_from_csv(filepath: str) -> List[Dict[str, Any]]:
    """Load courses from cleaned CSV file."""
    courses = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            course = {
                "id": idx,
                "title": row.get("title", "").strip(),
                "platform": row.get("platform", "").strip(),
                "provider": row.get("provider", "").strip(),
                "skills": row.get("skills", "").strip(),
                "rating": row.get("rating", "").strip(),
                "review_count": row.get("review_count", "").strip(),
                "url": row.get("url", "").strip(),
                "description": row.get("description", "").strip(),
                "difficulty": row.get("difficulty", "").strip(),
                "duration": row.get("duration", "").strip(),
                "language": row.get("language", "").strip(),
            }
            courses.append(course)
    return courses


def build_search_text(course: Dict[str, Any]) -> str:
    """Build semantic text for embedding from course fields."""
    parts = [
        course.get("title", ""),
        course.get("description", ""),
        course.get("skills", ""),
        f"Difficulty: {course.get('difficulty', '')}",
    ]
    # Filter out empty parts and join
    text = " ".join([p for p in parts if p])
    return text


def generate_embeddings(courses: List[Dict[str, Any]], model: SentenceTransformer) -> List[List[float]]:
    """Generate embeddings for all courses."""
    print("[*] Generating embeddings...")
    texts = [build_search_text(course) for course in courses]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()


def create_qdrant_collection(client: QdrantClient) -> None:
    """Create Qdrant collection if it doesn't exist."""
    print(f"[*] Checking collection '{COLLECTION_NAME}'...")

    try:
        # Try to get collection info
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"[OK] Collection already exists: {collection_info.points_count} points")
        return
    except Exception as e:
        print(f"[*] Collection doesn't exist, creating...")

    # Create collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"[OK] Created collection: {COLLECTION_NAME}")


def upload_to_qdrant(client: QdrantClient, courses: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
    """Upload courses with embeddings to Qdrant."""
    print(f"[*] Uploading {len(courses)} courses to Qdrant...")

    points = []
    for idx, (course, embedding) in enumerate(zip(courses, embeddings)):
        point = PointStruct(
            id=idx,
            vector=embedding,
            payload={
                "title": course["title"],
                "platform": course["platform"],
                "provider": course["provider"],
                "skills": course["skills"],
                "rating": course["rating"],
                "review_count": course["review_count"],
                "url": course["url"],
                "description": course["description"],
                "difficulty": course["difficulty"],
                "duration": course["duration"],
                "language": course["language"],
            },
        )
        points.append(point)

    # Upload in batches to avoid timeout
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"[OK] Uploaded {min(i+batch_size, len(points))}/{len(points)}")

    print(f"[OK] All courses uploaded")


def main():
    print("=" * 60)
    print("Qdrant Course Loading Pipeline")
    print("=" * 60)

    # Load settings
    settings = get_settings()
    print(f"\n[*] Connecting to Qdrant: {settings.QDRANT_URL}")

    # Connect to Qdrant
    try:
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        client.get_collections()  # Test connection
        print("[OK] Connected to Qdrant")
    except Exception as e:
        print(f"[ERROR] Failed to connect to Qdrant: {e}")
        sys.exit(1)

    # Load courses
    if not DATA_FILE.exists():
        print(f"[ERROR] Data file not found: {DATA_FILE}")
        sys.exit(1)

    print(f"\n[*] Loading courses from {DATA_FILE}")
    courses = load_courses_from_csv(str(DATA_FILE))
    print(f"[OK] Loaded {len(courses)} courses")

    # Load embedding model
    print(f"\n[*] Loading embedding model: {EMBEDDING_MODEL}")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"[OK] Model loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # Generate embeddings
    embeddings = generate_embeddings(courses, model)
    print(f"[OK] Generated {len(embeddings)} embeddings")

    # Create collection
    create_qdrant_collection(client)

    # Upload to Qdrant
    try:
        upload_to_qdrant(client, courses, embeddings)
        print(f"\n[OK] All courses indexed in Qdrant")
    except Exception as e:
        print(f"[ERROR] Failed to upload: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
