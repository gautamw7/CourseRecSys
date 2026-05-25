import type { UserProfile, Recommendation } from "../types/api";
import { CourseCard } from "../components/CourseCard";
import "./ResultsPage.css";

interface ResultsPageProps {
  profile: UserProfile;
  recommendations: Recommendation[];
  onTryAgain: () => void;
}

export function ResultsPage({
  profile,
  recommendations,
  onTryAgain,
}: ResultsPageProps) {
  return (
    <div className="results-page">
      <div className="results-container">
        <button className="back-button" onClick={onTryAgain}>
          ← Upload Another Resume
        </button>

        <div className="results-header">
          <h1>Your Personalized Recommendations</h1>

          <div className="profile-summary">
            <div className="profile-card">
              {profile.name && <h3>{profile.name}</h3>}
              <div className="profile-details">
                {profile.education_level && (
                  <p>
                    <strong>Education:</strong> {profile.education_level}
                    {profile.education_field && ` in ${profile.education_field}`}
                  </p>
                )}
                {profile.years_experience > 0 && (
                  <p>
                    <strong>Experience:</strong> {profile.years_experience} years
                  </p>
                )}
                {profile.skills.length > 0 && (
                  <p>
                    <strong>Skills:</strong> {profile.skills.slice(0, 5).join(", ")}
                    {profile.skills.length > 5 && ` +${profile.skills.length - 5} more`}
                  </p>
                )}
                {profile.profile_summary && (
                  <p className="summary">{profile.profile_summary}</p>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="results-content">
          <h2>
            Recommended Courses ({recommendations.length})
          </h2>

          {recommendations.length > 0 ? (
            <div className="courses-grid">
              {recommendations.map((course, index) => (
                <CourseCard
                  key={index}
                  course={{ ...course, index }}
                  rank={index + 1}
                />
              ))}
            </div>
          ) : (
            <div className="no-results">
              <p>No recommendations found. Please try uploading another resume.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
