"""ImageExtractor -- PDF file to raster images grouped by page."""

from typing import Dict, List, Protocol

from monkeygrab.domain.extracted_image import ExtractedImage


class ImageExtractor(Protocol):
    """Reads one PDF file and returns its qualifying raster images, per page.

    Size thresholds and how far below an image to search for its caption are
    adapter configuration, read once at construction: they are properties of
    a whole indexing run, not of one PDF.

    Describing what an image depicts is not this port's job. That is a
    vision-model call, already covered by ``ChatModel.generate(prompt,
    images=[...])`` with a model wired to the "ocr" role.

    Failure policy: this port is the documented carve-out from the project's
    hard-fail default (design doc, section 3). Failures are logged and
    swallowed at two levels -- an unopenable PDF yields ``{}``, and an
    unreadable single image is skipped. Image extraction is optional
    enrichment behind a flag, and one corrupt embedded image must not abort
    indexing of the document's text. A caller wanting hard failure wraps
    this port itself.
    """

    def extract(self, pdf_path: str) -> Dict[int, List[ExtractedImage]]:
        """Extract every qualifying image of ``pdf_path``, grouped by page.

        Args:
            pdf_path: Absolute or relative path to the PDF file.

        Returns:
            Mapping of zero-based page number to the images found on that
            page, in document order. Pages with no qualifying images (below
            the adapter's minimum size, or none present) are omitted from
            the mapping -- never present with an empty list. Also empty
            (``{}``) if the whole PDF could not be opened.
        """
        ...
