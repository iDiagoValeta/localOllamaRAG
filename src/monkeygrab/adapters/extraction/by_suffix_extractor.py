"""Extractors that route one file to the adapter owning its suffix."""
from pathlib import Path
from typing import Dict, List

from monkeygrab.adapters.extraction.text_extractor import TEXT_SUFFIXES
from monkeygrab.domain.extracted_image import ExtractedImage
from monkeygrab.domain.extracted_page import ExtractedPage
from monkeygrab.ports.image_extractor import ImageExtractor
from monkeygrab.ports.pdf_extractor import PdfExtractor


class BySuffixExtractor:
    """``PdfExtractor`` that sends ``.pdf`` to one adapter and text files to another.

    Composition, not fallback: the choice is made from the file name before
    anything runs, and an unknown suffix raises rather than trying both.
    """

    def __init__(self, pdf: PdfExtractor, text: PdfExtractor):
        self._pdf = pdf
        self._text = text

    def extract(self, pdf_path: str) -> List[ExtractedPage]:
        suffix = Path(pdf_path).suffix.lower()
        if suffix == ".pdf":
            return self._pdf.extract(pdf_path)
        if suffix in TEXT_SUFFIXES:
            return self._text.extract(pdf_path)
        raise ValueError(f"Unsupported document type {suffix!r}: {pdf_path}")


class PdfOnlyImageExtractor:
    """``ImageExtractor`` that only asks the inner adapter about PDFs.

    A text file has no rasterized figures, so an empty mapping is the true
    answer, not a degraded one.
    """

    def __init__(self, inner: ImageExtractor):
        self._inner = inner

    def extract(self, pdf_path: str) -> Dict[int, List[ExtractedImage]]:
        if Path(pdf_path).suffix.lower() != ".pdf":
            return {}
        return self._inner.extract(pdf_path)
