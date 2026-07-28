"""ExtractedPage -- one page of raw text pulled from a PDF, pre-chunking."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedPage:
    """Raw per-page text, the input to ``dividir_en_chunks``.

    Page numbering is normalized to zero-based before constructing this
    domain entity.

    Attributes:
        page: Zero-based page number within the source PDF.
        text: Raw extracted text for this page (before chunking).
    """

    page: int
    text: str
