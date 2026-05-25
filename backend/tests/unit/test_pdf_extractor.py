"""TDD tests for PDF text extraction service."""

import pytest
from pathlib import Path

from app.services.pdf_extractor import PDFExtractorService


RESUMES_DIR = Path(__file__).parent.parent.parent.parent / "Data" / "resumes"
RESUME_FILES = sorted(RESUMES_DIR.glob("*.pdf"))


class TestPDFTextExtraction:
    """Test PDF text extraction on real resume files."""

    def test_all_8_resumes_discovered(self):
        """Verify test data exists."""
        assert len(RESUME_FILES) == 8, f"Expected 8 resumes, found {len(RESUME_FILES)}"

    @pytest.mark.parametrize(
        "pdf_path", RESUME_FILES, ids=[p.name for p in RESUME_FILES]
    )
    def test_extract_returns_nonempty_text(self, pdf_path):
        """Each resume should extract at least 100 characters of text."""
        service = PDFExtractorService()
        text = service.extract_text(str(pdf_path))

        assert isinstance(text, str), f"Expected str, got {type(text)}"
        assert len(text) > 100, f"Resume {pdf_path.name} extracted only {len(text)} chars"

    def test_extract_nonexistent_file_raises(self):
        """Nonexistent PDF should raise ValueError."""
        service = PDFExtractorService()

        with pytest.raises(ValueError, match="not found"):
            service.extract_text("nonexistent.pdf")

    def test_extract_non_pdf_file_raises(self):
        """Non-PDF file should raise ValueError."""
        service = PDFExtractorService()

        with pytest.raises(ValueError, match="Expected .pdf"):
            service.extract_text("test.txt")

    def test_extract_text_contains_recognizable_content(self):
        """Spot-check a known resume for recognizable term."""
        sample = RESUMES_DIR / "ML_Suyash_Bhemde_VIT.pdf"
        if not sample.exists():
            pytest.skip(f"Sample resume not found: {sample}")

        service = PDFExtractorService()
        text = service.extract_text(str(sample))

        assert (
            "VIT" in text or "vit" in text.lower()
        ), "VIT resume should contain 'VIT' text"
