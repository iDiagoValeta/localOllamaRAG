"""ExtractedPage -- one page of raw text pulled from a PDF, pre-chunking.

# ─────────────────────────────────────────────
# SECTION 1: ENTITY
# ─────────────────────────────────────────────
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedPage:
    """Raw per-page text, the input to ``dividir_en_chunks``.

    Mirrors what ``rag.engine.indexing.indexar_documentos`` reads from
    both extraction paths before chunking: ``page_info['metadata']['page']
    - 1`` / ``page_info['text']`` from ``pymupdf4llm.to_markdown(...,
    page_chunks=True)``, and ``i`` / ``page.extract_text()`` from the
    ``pypdf.PdfReader`` fallback loop. Page numbering is 0-based in both
    cases (pymupdf4llm reports 1-based page numbers; indexing subtracts 1
    to normalize).

    Attributes:
        page: Zero-based page number within the source PDF.
        text: Raw extracted text for this page (before chunking).
    """

    page: int
    text: str
