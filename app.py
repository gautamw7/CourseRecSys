import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import warnings

warnings.filterwarnings("ignore")
# Suppress warnings from faiss
faiss.omp_set_num_threads(1)  # Set FAISS to use only one thread

def getConnection():
    username = "Username" # Replace with your MongoDB username
    password = "password" # Replace with your MongoDB password

    # Replace with your MongoDB connection string
    uri = "mongodb+srv://Username:password@cluster.b0nw9v5.mongodb.net/?retryWrites=true&w=majority"

    client = MongoClient(uri, server_api=ServerApi('1'))

    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        return client
    except Exception as e:
        print(e)

def get_courses():
    try:
        client = getConnection()
        db = client['course_recommender']
        collection = db['courses']
        courses = list(collection.find())
        return courses
    except Exception as e:
        print(e)
    finally:
        client.close()

def get_index():
    try:
        client = getConnection()
        db = client['course_recommender']
        index_collection = db['index']
        index_data_from_mongo = index_collection.find_one({"index_type": "Faiss IndexFlatL2"})
        if index_data_from_mongo and 'index_data' in index_data_from_mongo:
            index_bytes_from_mongo = index_data_from_mongo['index_data']
            load_index = faiss.deserialize_index(np.frombuffer(index_bytes_from_mongo, dtype=np.uint8))
            print("FAISS index loaded successfully from MongoDB!")
            print(f"Loaded index has {load_index.ntotal} vectors.")
            return load_index
        else:
            print("Could not find or load the FAISS index from MongoDB.")
    except Exception as e:
        print(e)
    finally:
        client.close()

def load_courses_and_index():
    courses = get_courses()
    index = get_index()
    if not courses:
        print("No courses found in the database.")
    else:
        print("Available Courses:")
        print("Example Courses: ", [course['title'] for course in courses[:5]])

    index = get_index()
    if index is None:
        print("Index could not be retrieved.")
    else:
        print("Index retrieved successfully.")

    return courses, index

def generate_query_embedding(query: str) -> np.ndarray:
    """
    Generate an embedding for the input query using the specified model.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    if not query:
        raise ValueError("Query cannot be empty.")
    if not isinstance(query, str):
        raise TypeError("Query must be a string.")
    if len(query) > 512:
        raise ValueError("Query exceeds maximum length of 512 characters.")
    query_embedding = model.encode(query, convert_to_tensor=True)
    query_embedding = query_embedding.cpu().numpy()
    return query_embedding

def generate_courses(query: str, courses: list, index: faiss.IndexFlatL2, top_k: int = 5):
    query_embedding = generate_query_embedding(query)
    distances, indices = index.search(np.array([query_embedding]), top_k)
    similar_courses = [courses[i] for i in indices[0]]

    print(f"Query: {query}")
    print("Similar Courses:")
    for course in similar_courses:
        print(f"- {course['title']} (ID: {course['_id']})")
        print(f"  Description: {course['description']}"
              f"\n  URL: {course['url']}\n")
        print(f"  Organization: {course['organization']}")
        print(f"  Rating: {course['rating']}")
    return similar_courses

def main():
    st.title("Course Recommender System")
    st.write("Enter a query to find similar courses.")

    
    courses, index = load_courses_and_index()

    if not courses or index is None:
        st.error("Failed to load courses or index. Please check the database connection.")
        return

    query = st.text_input("Enter your query:")
    # Textbar to input the query
    st.sidebar.header("Query Input")
    if not query:
        st.warning("Please enter a query to search for courses.")
    else:
        st.write(f"Searching for courses similar to: **{query}**")
    if st.button("Search"):
        if query:
            print(f"Searching for courses similar to: {query}")
            similar_courses = generate_courses(query, courses, index)
            if similar_courses:
                st.success(f"Found {len(similar_courses)} similar courses.")
                for course in similar_courses:
                    st.markdown(f"### {course['title']} (ID: {course['_id']})")
                    st.markdown(course['description'].encode('latin1').decode('utf-8', errors='ignore'))
                    st.markdown(f"[View Course]({course['url']})", unsafe_allow_html=False)  
                    st.markdown(f"<div style='color:#BBBBBB; font-size: 14px'>{course['description']}</div>", unsafe_allow_html=True)
                    st.write(f"Organization: {course.get('organization', 'N/A')}")
                    st.write(f"Tags: {', '.join(course.get('tags', [])) if 'tags' in course else 'N/A'}")
                    col1, col2 = st.columns(2)
                    col1.markdown(f"**Rating:** {course.get('rating', 'N/A')}")
                    col2.markdown(f"**Duration:** {course.get('duration', 'N/A')}")
                    st.write("---")
            else:
                st.warning("No similar courses found.")
        else:
            st.error("Please enter a valid query.")
    st.sidebar.header("About")
    st.sidebar.write("This application uses a pre-trained model to recommend courses based on user queries.")
    st.sidebar.write("Developed by Gautamw7.")
    st.sidebar.write("For more information, visit the [GitHub repository](https://github.com/gautamw7/CourseRecSys).")

if __name__ == "__main__":
    main()
