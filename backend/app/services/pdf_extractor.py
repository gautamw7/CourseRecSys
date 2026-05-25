"""PDF text extraction service."""

import logging
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)


class PDFExtractorService:
    """Extract text from PDF files."""

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text from all pages joined by newline

        Raises:
            ValueError: If file not found or not a PDF
        """
        path = Path(pdf_path)

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected .pdf file, got: {path.suffix}")

        if not path.exists():
            raise ValueError(f"PDF not found: {pdf_path}")

        with pdfplumber.open(str(path)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]

        text = "\n".join(pages).strip()

        if not text:
            logger.warning(
                f"No extractable text in {pdf_path} — may be image-based PDF"
            )

        return text
