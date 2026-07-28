"""PdfExtractor -- PDF file to per-page raw text."""

from typing import List, Protocol

from monkeygrab.domain.extracted_page import ExtractedPage


class PdfExtractor(Protocol):
    """Reads one PDF file and returns its text, one entry per page.

    The indexing loop extracts a whole PDF and only afterwards splits
    per-page text into chunks. Splitting into chunks is
    intentionally NOT this port's job -- it stays a pure domain operation
    the caller applies to the returned pages.

    Failure policy: hard-fail. Raise on any extraction failure (corrupt
    PDF, unsupported format, missing file). A caller that wants a fallback
    chain composes two
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
