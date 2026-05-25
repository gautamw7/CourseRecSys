import { useRef, useState } from "react";
import "./DropZone.css";

interface DropZoneProps {
  onFileSelect: (file: File) => void;
  isLoading?: boolean;
}

export function DropZone({ onFileSelect, isLoading = false }: DropZoneProps) {
  const [isDragActive, setIsDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type === "application/pdf") {
        onFileSelect(file);
      } else {
        alert("Please drop a PDF file");
      }
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  const handleClick = () => {
    inputRef.current?.click();
  };

  return (
    <div
      className={`dropzone ${isDragActive ? "active" : ""}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={handleClick}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf"
        onChange={handleChange}
        style={{ display: "none" }}
        disabled={isLoading}
      />
      <div className="dropzone-content">
        {isLoading ? (
          <>
            <p className="dropzone-title">Processing your resume...</p>
            <p className="dropzone-subtitle">This may take a moment</p>
          </>
        ) : (
          <>
            <p className="dropzone-title">Drag and drop your resume here</p>
            <p className="dropzone-subtitle">or click to select a PDF file</p>
          </>
        )}
      </div>
    </div>
  );
}
