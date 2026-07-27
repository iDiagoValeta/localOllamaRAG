"""PdfExtractor -- PDF file to per-page raw text."""

from typing import List, Protocol

from monkeygrab.domain.extracted_page import ExtractedPage


class PdfExtractor(Protocol):
    """Reads one PDF file and returns its text, one entry per page.

    Matches the only entry point the pipeline calls today: the
    ``indexar_documentos`` loop in ``rag/engine/indexing.py`` extracts a
    whole PDF at once (``pymupdf4llm.to_markdown(ruta_pdf,
    page_chunks=True)`` or, on the pypdf fallback path,
    ``PdfReader(ruta_pdf).pages``) and only afterwards splits per-page text
    into chunks via ``dividir_en_chunks``. Splitting into chunks is
    intentionally NOT this port's job -- it stays a pure domain operation
    the caller applies to the returned pages.

    Failure policy: hard-fail. Raise on any extraction failure (corrupt
    PDF, unsupported format, missing file). The three-level fallback this
    replaces (MinerU -> pymupdf4llm -> pypdf, see
    docs/design/2026-06-26-mineru-indexing-design.md) was superseded by the
    spike's hard-fail behavior (docs/design/2026-07-26-monkeygrab-v2.md,
    section 3): a caller that wants a fallback chain composes two
    ``PdfExtractor`` adapters explicitly, the port itself never swallows an
    error and returns partial or empty output.
    """

    def extract(self, pdf_path: str) -> List[ExtractedPage]:
        """Extract every page of ``pdf_path`` as raw text.

        Args:
            pdf_path: Absolute or relative path to the PDF file.

        Returns:
            One ``ExtractedPage`` per page, in page order.

        Raises:
            Exception: On any extraction failure. Never returns partial
                results silently.
        """
        ...
