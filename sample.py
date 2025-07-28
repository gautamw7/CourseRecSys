# Using only the Qdrant working great !!! :) 
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import PayloadSchemaType

qdrant_client = QdrantClient(
    url="url here",  # Replace with your actual Qdrant URL
    api_key="api_key_here",  # Replace with your actual API key
)

model = SentenceTransformer('all-mpnet-base-v2')

kw_model = KeyBERT('all-MiniLM-L6-v2')
user_query = ["Python basic course"]

keywords = kw_model.extract_keywords(user_query, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
keys = [kw[0] for kw in keywords]
user_query = " ".join(keys)

query_vector = model.encode(user_query).tolist()


qdrant_client.create_payload_index(
    collection_name="courses",
    field_name="difficulty",
    field_schema=PayloadSchemaType.KEYWORD
)

search_result = qdrant_client.query_points(
    collection_name="courses",
    query=query_vector,
    limit=10,
    with_payload=True,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="difficulty",
                match=MatchValue(value="Advanced")
            )
        ]
    )
)

print(f"Search results for query: {user_query}")
for point in search_result.points:
    print(f"Title: {point.payload.get('title')}, Score: {point.score}, Tags: {point.payload.get('tag')}, Organization: {point.payload.get('organization')}")
    print(f"Score : {point.score}")