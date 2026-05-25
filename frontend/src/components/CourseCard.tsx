import type { Recommendation } from "../types/api";
import "./CourseCard.css";

interface CourseCardProps {
  course: Recommendation & { index?: number };
  rank?: number;
}

export function CourseCard({ course, rank }: CourseCardProps) {
  const score = course.relevance_score ? (course.relevance_score * 100).toFixed(0) : null;

  return (
    <div className="course-card">
      {rank && <div className="course-rank">#{rank}</div>}

      <div className="course-header">
        <h3 className="course-title">{course.title}</h3>
        {score && <div className="course-score">{score}% match</div>}
      </div>

      <div className="course-meta">
        <span className="course-platform">{course.platform}</span>
        {course.provider && <span className="course-provider">{course.provider}</span>}
        {course.difficulty && <span className="course-difficulty">{course.difficulty}</span>}
      </div>

      {course.description && (
        <p className="course-description">{course.description.substring(0, 150)}...</p>
      )}

      {course.explanation && (
        <p className="course-explanation">
          <strong>Why it's a fit:</strong> {course.explanation}
        </p>
      )}

      <div className="course-skills">
        {course.explanation && <span className="skill-tag">Recommended</span>}
      </div>

      {course.url && (
        <a
          href={course.url}
          target="_blank"
          rel="noopener noreferrer"
          className="course-link"
        >
          View Course →
        </a>
      )}
    </div>
  );
}
