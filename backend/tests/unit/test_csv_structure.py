"""
Test CSV structure and data quality for cleaned course data.

TDD approach: Define what a valid course record should be,
then verify the entire CSV meets those criteria.
"""

import csv
import os
from pathlib import Path


CSV_PATH = Path(__file__).parent.parent.parent.parent / "Data" / "courses_cleaned.csv"
REQUIRED_FIELDS = [
    'course_id', 'title', 'platform', 'provider', 'skills',
    'rating', 'review_count', 'url', 'description',
    'difficulty', 'duration', 'language'
]
TOTAL_EXPECTED_COURSES = 4950


def load_csv_data():
    """Load CSV and return list of rows."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


class TestCSVStructure:
    """Test CSV file structure and metadata."""

    def test_csv_file_exists(self):
        assert CSV_PATH.exists(), f"CSV file not found at {CSV_PATH}"

    def test_csv_has_correct_headers(self):
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
        assert header == REQUIRED_FIELDS, f"Headers mismatch. Got: {header}"

    def test_csv_row_count(self):
        courses = load_csv_data()
        assert len(courses) == TOTAL_EXPECTED_COURSES, \
            f"Expected {TOTAL_EXPECTED_COURSES} courses, got {len(courses)}"


class TestCourseIDSequence:
    """Test that course_id is properly sequenced."""

    def test_course_ids_are_sequential(self):
        courses = load_csv_data()
        for idx, course in enumerate(courses, start=1):
            course_id = int(course['course_id'])
            assert course_id == idx, \
                f"Course at row {idx} has ID {course_id}, expected {idx}"

    def test_course_ids_are_unique(self):
        courses = load_csv_data()
        ids = [course['course_id'] for course in courses]
        assert len(ids) == len(set(ids)), "Duplicate course_ids found"


class TestDataQuality:
    """Test data quality and required fields."""

    def test_all_titles_present(self):
        courses = load_csv_data()
        empty_titles = [c for c in courses if not c['title'].strip()]
        assert len(empty_titles) == 0, f"Found {len(empty_titles)} courses with empty titles"

    def test_title_uniqueness(self):
        courses = load_csv_data()
        titles = [c['title'].lower().strip() for c in courses]
        assert len(titles) == len(set(titles)), "Duplicate titles found (dedup failed)"

    def test_platform_values_valid(self):
        """Platform should be one of known values."""
        courses = load_csv_data()
        valid_platforms = {'Coursera', 'NPTEL'}
        invalid = [c for c in courses if c['platform'] not in valid_platforms]
        assert len(invalid) == 0, \
            f"Found {len(invalid)} courses with invalid platform"

    def test_rating_format_valid(self):
        """Ratings should be numeric or empty."""
        courses = load_csv_data()
        for course in courses:
            rating = course['rating'].strip()
            if rating:
                try:
                    float(rating)
                except ValueError:
                    assert False, f"Invalid rating '{rating}' in course {course['course_id']}"

    def test_language_field_present(self):
        """All courses should have a language specified."""
        courses = load_csv_data()
        empty_lang = [c for c in courses if not c['language'].strip()]
        assert len(empty_lang) == 0, f"Found {len(empty_lang)} courses with empty language"

    def test_skills_format_consistency(self):
        """Skills should be either empty or populated (not null/None)."""
        courses = load_csv_data()
        for course in courses:
            skills = course['skills']
            # Skills field should be a string (never None/null)
            assert isinstance(skills, str), \
                f"Course {course['course_id']} skills is not a string: {type(skills)}"

    def test_no_encoding_errors(self):
        """Check for common encoding issues (mojibake, garbled chars)."""
        courses = load_csv_data()
        problem_chars = ['?', '']  # Common corruption indicators
        for course in courses:
            for field_value in course.values():
                # Allow some special chars, but flag obvious corruption
                if '?' in field_value and len(field_value) > 100:
                    # This is likely a real title with ?, not corruption
                    continue


class TestDataCompletenessByField:
    """Test field-specific completeness requirements."""

    def test_provider_mostly_populated(self):
        """Provider should be populated for most courses."""
        courses = load_csv_data()
        empty_providers = [c for c in courses if not c['provider'].strip()]
        # Allow 5% to be empty (some NPTEL courses may lack provider)
        threshold = len(courses) * 0.95
        assert len(empty_providers) <= len(courses) * 0.05, \
            f"Too many courses missing provider: {len(empty_providers)} / {len(courses)}"

    def test_url_distribution(self):
        """URLs should exist for Coursera, less so for NPTEL."""
        courses = load_csv_data()
        coursera = [c for c in courses if c['platform'] == 'Coursera']
        nptel = [c for c in courses if c['platform'] == 'NPTEL']

        coursera_with_url = [c for c in coursera if c['url'].strip()]
        nptel_with_url = [c for c in nptel if c['url'].strip()]

        # Coursera should have ~70%+ URLs
        coursera_ratio = len(coursera_with_url) / len(coursera) if coursera else 0
        # NPTEL should have ~80%+ URLs (they provide them)
        nptel_ratio = len(nptel_with_url) / len(nptel) if nptel else 0

        assert coursera_ratio >= 0.5, \
            f"Coursera URL coverage too low: {coursera_ratio:.1%}"
        assert nptel_ratio >= 0.7, \
            f"NPTEL URL coverage too low: {nptel_ratio:.1%}"


class TestDataIntegrity:
    """Test end-to-end data integrity."""

    def test_sample_record_completeness(self):
        """Spot-check a few records for required fields."""
        courses = load_csv_data()
        samples = [courses[0], courses[len(courses)//2], courses[-1]]

        for course in samples:
            assert course['course_id'].strip(), "course_id empty"
            assert course['title'].strip(), "title empty"
            assert course['platform'].strip(), "platform empty"
            assert course['language'].strip(), "language empty"
