from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["course_recommender"]
courses_col = db["courses"]

sample_course = {
    "course_id": "crs001",
    "title": "Introduction to Python",
    "platform": "Coursera",
    "description": "Learn the basics of Python programming.",
    "tags": ["Python", "Programming"],
    "difficulty": "Beginner",
    "vector": [0.12, 0.53, 0.19]  # Example shortened vector
}

courses_col.insert_one(sample_course)
print("Inserted successfully")
