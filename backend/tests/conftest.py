"""
Pytest configuration and fixtures for course recommendation system tests.
"""

import pytest


@pytest.fixture
def sample_course():
    """Fixture providing a valid course dict for testing."""
    return {
        'course_id': '1',
        'title': 'Machine Learning Basics',
        'platform': 'Coursera',
        'provider': 'Stanford University',
        'skills': 'Skills: machine learning, python, data science',
        'rating': '4.8',
        'review_count': '1500',
        'url': 'https://www.coursera.org/learn/ml-basics',
        'description': 'Learn the fundamentals of machine learning.',
        'difficulty': 'Intermediate',
        'duration': '6 weeks',
        'language': 'English'
    }


@pytest.fixture
def sample_courses():
    """Fixture providing multiple valid courses."""
    return [
        {
            'course_id': '1',
            'title': 'Python for Data Science',
            'platform': 'Coursera',
            'provider': 'University of Michigan',
            'skills': 'Skills: python, data science',
            'rating': '4.7',
            'review_count': '2000',
            'url': 'https://www.coursera.org/learn/python-data-science',
            'description': 'Learn Python for data science applications.',
            'difficulty': 'Beginner',
            'duration': '4 weeks',
            'language': 'English'
        },
        {
            'course_id': '2',
            'title': 'Advanced SQL',
            'platform': 'NPTEL',
            'provider': 'IIT Bombay',
            'skills': 'Skills: SQL, databases',
            'rating': '4.5',
            'review_count': '500',
            'url': 'https://nptel.ac.in/courses/sql-advanced',
            'description': 'Master advanced SQL concepts.',
            'difficulty': 'Advanced',
            'duration': '8 weeks',
            'language': 'English'
        }
    ]
