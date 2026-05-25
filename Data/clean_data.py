"""
Data cleaning pipeline: Merge 4 Coursera/NPTEL datasets into unified schema.
Output: courses_cleaned.csv ready for Qdrant embedding + indexing.
"""

import csv
import os
from pathlib import Path

def clean_skills(skills_str):
    """Parse skills string into list, handle various formats."""
    if not skills_str or skills_str == '[]':
        return []

    # Remove brackets if present
    skills_str = skills_str.strip().strip('[]').strip('"')

    # Split by comma or semicolon
    skills = [s.strip().strip('"') for s in skills_str.replace(';', ',').split(',')]

    # Filter empty strings
    return [s for s in skills if s]

def parse_rating(rating_str):
    """Convert rating string to float, handle 'not-mentioned'."""
    if not rating_str or rating_str.lower() == 'not-mentioned' or rating_str == '':
        return ''

    try:
        # Handle ratings like "4.8stars"
        rating_str = rating_str.replace('stars', '').strip()
        return float(rating_str)
    except:
        return ''

def load_coursera_2024():
    """Load and transform 2024 Coursera dataset."""
    courses = []
    filepath = 'Coursera Courses & Skills dataset 2024.csv'

    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return courses

    with open(filepath, encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get('Title', '').strip()
            if not title:
                continue

            skills_list = clean_skills(row.get('Skills', ''))
            skills_text = f"Skills: {', '.join(skills_list)}" if skills_list else ""

            course = {
                'title': title,
                'platform': 'Coursera',
                'provider': row.get('Organization', '').strip(),
                'skills': skills_text,
                'rating': parse_rating(row.get('Ratings', '')),
                'review_count': row.get('Review counts', '').strip() or '',
                'url': '',
                'description': '',
                'difficulty': '',
                'duration': '',
                'language': 'English'
            }
            courses.append(course)

    print(f"[OK] Loaded {len(courses)} courses from {filepath}")
    return courses

def load_coursera_2021():
    """Load and transform 2021 Coursera dataset."""
    courses = []
    filepath = 'Coursera Courses Dataset 2021.csv'

    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return courses

    with open(filepath, encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get('Course Name', '').strip()
            if not title:
                continue

            skills_list = clean_skills(row.get('Skills', ''))
            skills_text = f"Skills: {', '.join(skills_list)}" if skills_list else ""

            course = {
                'title': title,
                'platform': 'Coursera',
                'provider': row.get('University', '').strip(),
                'skills': skills_text,
                'rating': parse_rating(row.get('Course Rating', '')),
                'review_count': '',
                'url': row.get('Course URL', '').strip(),
                'description': row.get('Course Description', '').strip()[:500],  # Truncate to 500 chars
                'difficulty': row.get('Difficulty Level', '').strip(),
                'duration': '',
                'language': 'English'
            }
            courses.append(course)

    print(f"[OK] Loaded {len(courses)} courses from {filepath}")
    return courses

def load_coursera_random():
    """Load and transform random Coursera dataset."""
    courses = []
    filepath = 'coursera_dataset_on_courses Random.csv'

    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return courses

    with open(filepath, encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get('C_Nm', '').strip()
            if not title:
                continue

            course = {
                'title': title,
                'platform': 'Coursera',
                'provider': row.get('Tutor', '').strip(),
                'skills': '',
                'rating': parse_rating(row.get('Rating', '')),
                'review_count': row.get('Review', '').strip() or '',
                'url': '',
                'description': '',
                'difficulty': row.get('lvl', '').strip(),
                'duration': row.get('Time', '').strip(),
                'language': 'English'
            }
            courses.append(course)

    print(f"[OK] Loaded {len(courses)} courses from {filepath}")
    return courses

def load_nptel():
    """Load and transform NPTEL dataset."""
    courses = []
    filepath = 'NPTEL Course List (July - Dec 2025).csv'

    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return courses

    with open(filepath, encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get('Course Name', '').strip()
            if not title:
                continue

            # For NPTEL, keep all since language not clearly specified in data
            # Assuming most are in English or translate well
            course = {
                'title': title,
                'platform': 'NPTEL',
                'provider': row.get('Institute', '').strip() or row.get('Co-ordinating Institute', '').strip(),
                'skills': row.get('Applicable NPTEL Domain', '').strip(),
                'rating': '',
                'review_count': '',
                'url': row.get('NPTEL URL', '').strip() or row.get('NPTEL URL.1', '').strip(),
                'description': '',
                'difficulty': '',
                'duration': row.get('Duration', '').strip(),
                'language': 'English'
            }
            courses.append(course)

    print(f"[OK] Loaded {len(courses)} courses from {filepath}")
    return courses

def deduplicate_courses(all_courses):
    """Remove duplicate courses by title (case-insensitive)."""
    seen = {}
    duplicates = 0

    for course in all_courses:
        title_key = course['title'].lower().strip()
        if title_key not in seen:
            seen[title_key] = course
        else:
            duplicates += 1

    if duplicates > 0:
        print(f"[OK] Removed {duplicates} duplicate courses")

    return list(seen.values())

def main():
    print("=" * 60)
    print("Data Cleaning Pipeline")
    print("=" * 60)

    print("\n1. Loading datasets...")
    courses = []
    courses.extend(load_coursera_2024())
    courses.extend(load_coursera_2021())
    courses.extend(load_coursera_random())
    courses.extend(load_nptel())

    print(f"\nTotal courses loaded: {len(courses)}")

    print("\n2. Deduplicating...")
    courses = deduplicate_courses(courses)
    print(f"Total after dedup: {len(courses)}")

    # Sort by title for consistent output
    courses.sort(key=lambda x: x['title'])

    # Add course_id to each course
    for idx, course in enumerate(courses, start=1):
        course['course_id'] = idx

    print("\n3. Writing to CSV...")
    output_file = 'courses_cleaned.csv'
    fieldnames = ['course_id', 'title', 'platform', 'provider', 'skills', 'rating', 'review_count',
                  'url', 'description', 'difficulty', 'duration', 'language']

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for course in courses:
            writer.writerow(course)

    print(f"[OK] Cleaned data written to {output_file}")
    print(f"\nFinal count: {len(courses)} courses")
    print("=" * 60)

if __name__ == '__main__':
    main()
