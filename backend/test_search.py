"""
Quick integration test: Verify Qdrant search functionality.
Run after load_courses_to_qdrant.py completes.
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import get_settings
from app.services.vector_search import VectorSearchService


def main():
    print("=" * 60)
    print("Integration Test: Vector Search")
    print("=" * 60)

    settings = get_settings()
    print(f"\n1. Loading settings...")
    print(f"   QDRANT_URL: {settings.QDRANT_URL}")
    print(f"   QDRANT_API_KEY: {'*' * 20}...")

    print(f"\n2. Initializing VectorSearchService...")
    try:
        search_service = VectorSearchService(
            qdrant_url=settings.QDRANT_URL,
            qdrant_api_key=settings.QDRANT_API_KEY
        )
        print("   [OK] Service initialized")
    except Exception as e:
        print(f"   [ERROR] Failed to initialize: {e}")
        return False

    print(f"\n3. Running health check...")
    try:
        health = search_service.health_check()
        if health:
            print("   [OK] Qdrant cluster is healthy")
        else:
            print("   [ERROR] Health check failed")
            return False
    except Exception as e:
        print(f"   [ERROR] Health check exception: {e}")
        return False

    print(f"\n4. Testing search with sample queries...")
    test_queries = [
        "machine learning python",
        "data science",
        "web development",
        "database SQL",
    ]

    all_passed = True
    for query in test_queries:
        try:
            results = search_service.search(query, limit=3)
            print(f"\n   Query: '{query}'")
            print(f"   Results: {len(results)} courses found")

            for i, course in enumerate(results, 1):
                title = course.get('title', 'N/A')[:50]
                score = course.get('score', 'N/A')
                print(f"     {i}. {title}... (score: {score:.3f})")

            if len(results) == 0:
                print(f"     [WARNING] No results for query: {query}")
                all_passed = False

        except Exception as e:
            print(f"   [ERROR] Search failed for '{query}': {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("SUCCESS: All tests passed!")
        print("Vector search is operational.")
    else:
        print("FAILURE: Some tests did not pass.")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
