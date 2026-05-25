import { DropZone } from "../components/DropZone";
import "./UploadPage.css";

interface UploadPageProps {
  onUpload: (file: File) => void;
  isLoading?: boolean;
}

export function UploadPage({ onUpload, isLoading = false }: UploadPageProps) {
  return (
    <div className="upload-page">
      <div className="upload-container">
        <h1 className="upload-title">Course Recommendation System</h1>
        <p className="upload-subtitle">
          Upload your resume to get personalized course recommendations
        </p>

        <DropZone onFileSelect={onUpload} isLoading={isLoading} />

        <div className="upload-info">
          <h3>How it works</h3>
          <ul>
            <li>Upload your resume in PDF format</li>
            <li>We extract your skills, education, and experience</li>
            <li>Get personalized course recommendations based on your profile</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
