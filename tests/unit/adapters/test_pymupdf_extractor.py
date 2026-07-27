"""Unit tests for monkeygrab.adapters.extraction.pymupdf_extractor.PymupdfExtractor.

Stubs pymupdf4llm entirely -- no real PDF, no filesystem access -- so these
run in milliseconds and never touch the real library.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from monkeygrab.adapters.extraction import pymupdf_extractor as module
from monkeygrab.adapters.extraction.pymupdf_extractor import PymupdfExtractor
from monkeygrab.domain.extracted_page import ExtractedPage


def test_extract_converts_page_chunks_to_extracted_pages_and_normalizes_page_numbers(monkeypatch):
    """pymupdf4llm reports 1-based page numbers; extract() must normalize to 0-based."""
    calls = {}

    def fake_to_markdown(pdf_path, page_chunks):
        calls["pdf_path"] = pdf_path
        calls["page_chunks"] = page_chunks
        return [
            {"metadata": {"page": 1}, "text": "first page text"},
            {"metadata": {"page": 2}, "text": "second page text"},
        ]

    monkeypatch.setattr(module.pymupdf4llm, "to_markdown", fake_to_markdown)

    extractor = PymupdfExtractor()
    pages = extractor.extract("some/document.pdf")

    # The exact path given is what reaches the library -- not a hardcoded one.
    assert calls["pdf_path"] == "some/document.pdf"
    assert calls["page_chunks"] is True

    assert pages == [
        ExtractedPage(page=0, text="first page text"),
        ExtractedPage(page=1, text="second page text"),
    ]


def test_extract_defaults_missing_text_to_empty_string(monkeypatch):
    monkeypatch.setattr(
        module.pymupdf4llm, "to_markdown",
        lambda pdf_path, page_chunks: [{"metadata": {"page": 1}}],
    )

    pages = PymupdfExtractor().extract("doc.pdf")

    assert pages == [ExtractedPage(page=0, text="")]


def test_extract_hard_fails_when_pymupdf4llm_raises_instead_of_falling_back_to_pypdf(monkeypatch):
    """The old pipeline silently retries with pypdf on failure; this adapter must not.

    Nothing here should attempt an alternate extraction path -- a raised
    exception from pymupdf4llm must surface as a raised exception from
    extract(), never as an empty/partial page list.
    """
    def raising_to_markdown(pdf_path, page_chunks):
        raise ValueError("corrupt PDF stream")

    monkeypatch.setattr(module.pymupdf4llm, "to_markdown", raising_to_markdown)

    extractor = PymupdfExtractor()
    with pytest.raises(RuntimeError, match="pymupdf4llm failed"):
        extractor.extract("broken.pdf")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
