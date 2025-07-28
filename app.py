import streamlit as st
from sentence_transformers import SentenceTransformer
import warnings
from keybert import KeyBERT
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import PayloadSchemaType
import time
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

warnings.filterwarnings("ignore")

# Function to get Qdrant client connection
@st.cache_resource # Cache the connection to avoid re-creating it
def getConnection():
    qdrant_client = QdrantClient(
    url="https://ed779a30-c6e7-4c82-9aba-d4f31c6b789e.eu-central-1-0.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.rfTON-MO3MX_U2hr0Oz0fbUEACEZMB3cLwnKPuiz_dc",
    )
    return qdrant_client

# Function to load models
@st.cache_resource # Cache the models to avoid re-loading them
def load_models():
    model = SentenceTransformer('all-mpnet-base-v2')
    kw_model = KeyBERT('all-MiniLM-L6-v2')
    return model, kw_model

# Function to get user query and vector
def get_user_query(model, kw_model, user_input):
    keywords = kw_model.extract_keywords(user_input, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
    keys = [kw[0] for kw in keywords]
    user_input = " ".join(keys)
    query_vector = model.encode(user_input).tolist()
    return user_input, query_vector

# Function to create payload index
def search_courses(qdrant_client, query_vector):
    search_result = qdrant_client.query_points(
        collection_name="courses",
        query=query_vector,
        limit=10,
        with_payload=True
    )
    return search_result

# As search results, has duplicates, we can use a set to store unique results
# Return the top 5 results based on score
def get_top_results(search_result, max_results=5):
    seen_titles = set()
    results = []

    for point in search_result.points:
        title = point.payload.get('title')
        if title and title not in seen_titles:
            seen_titles.add(title)
            result = {
                "title": title,
                "score": f"{float(point.score) * 100:.2f}%",  # Score as percentage string
                "tags": point.payload.get('tag'),
                "organization": point.payload.get('organization'),
                "url": point.payload.get('url'),
                "description": point.payload.get('description'),
                "difficulty": point.payload.get('difficulty'),
                "duration": point.payload.get('duration'),
                "platform": point.payload.get('platform'),
                "rating": point.payload.get('rating'),
                "review_count": point.payload.get('review_count')
            }
            results.append(result)

            if len(results) >= max_results:
                break

    return results

def main():
    # Check the connection
    try:
        qdrant_client = getConnection()
    except Exception as e:
        st.error(f"❌ Failed to connect to Qdrant: {e}")
        return
    msg_qdrant = st.empty()
    msg_qdrant.success("✅ Connected to Qdrant successfully!")

    # Load models
    model, kw_model = load_models()
    # Temporary model load success
    msg_model = st.empty()
    msg_model.success("✅ Models loaded successfully!")

    time.sleep(1)  # Simulate loading time
    msg_qdrant.empty()
    msg_model.empty()

    # Streamlit app title
    st.title("🎓 Course Recommendation System")
    st.markdown("Enter your query to find relevant courses:")

    user_input = st.text_input("🔎 Query", placeholder="e.g. Python basic course")

    if user_input:
        # Get user query and vector
        user_query_text, query_vector = get_user_query(model, kw_model, user_input)

        if not query_vector:
            st.error("❌ Failed to process the query vector.")
            return

        st.subheader("📘 Search Results")
        st.write(f"Search results for query: `{user_input}`")

        with st.spinner("Searching courses..."):
            search_result = search_courses(qdrant_client, query_vector)

        if not search_result.points:
            st.warning("⚠️ No courses found for the given query.")
            return

        top_results = get_top_results(search_result)
        st.info(f"📌 Found {len(top_results)} courses matching your query.")
        titles = [result['title'] for result in top_results]
        print(f"Titles found: {titles}")
        st.markdown("#### 🏆 Top 5 Results")
        for result in top_results:
            st.markdown(f"---\n### 📘 {result['title']} ({result['score']})")
            st.markdown(f"🔗 [Course Link]({result['url']})")
            st.markdown(
                f"🏷 **Tags**: {result['tags']}  \n"
                f"🏫 **Organization**: {result['organization']}  \n"
                f"📝 **Description**: {result['description']}  \n"
                f"📊 **Difficulty**: {result['difficulty']}  \n"
                f"⏱ **Duration**: {result['duration']}  \n"
                f"💻 **Platform**: {result['platform']}  \n"
                f"⭐ **Rating**: {result['rating']} ({result['review_count']} reviews)"
            )

        st.success("✅ Search completed successfully!")
    else:
        st.info("ℹ️ Please enter a query above to search for courses.")



if __name__ == "__main__":
    main()