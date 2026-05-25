import { useState } from "react";
import { UploadPage } from "./pages/UploadPage";
import { ResultsPage } from "./pages/ResultsPage";
import { uploadResume } from "./api/client";
import type { UserProfile, Recommendation } from "./types/api";
import "./App.css";

type Page = "upload" | "results";

function App() {
  const [page, setPage] = useState<Page>("upload");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const [results, setResults] = useState<{
    profile: UserProfile;
    recommendations: Recommendation[];
  } | null>(null);

  const handleUpload = async (file: File) => {
    setIsLoading(true);
    setError("");

    try {
      const response = await uploadResume(file);

      if (response.status === "success") {
        setResults({
          profile: response.user_profile,
          recommendations: response.recommendations,
        });
        setPage("results");
      } else {
        setError(response.error || "Failed to process resume");
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error occurred";
      setError(message);
      console.error("Upload error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTryAgain = () => {
    setPage("upload");
    setError("");
    setResults(null);
  };

  return (
    <div className="app">
      {error && (
        <div className="error-banner">
          <p>{error}</p>
          <button onClick={() => setError("")}>×</button>
        </div>
      )}

      {page === "upload" ? (
        <UploadPage onUpload={handleUpload} isLoading={isLoading} />
      ) : results ? (
        <ResultsPage
          profile={results.profile}
          recommendations={results.recommendations}
          onTryAgain={handleTryAgain}
        />
      ) : null}
    </div>
  );
}

export default App;
