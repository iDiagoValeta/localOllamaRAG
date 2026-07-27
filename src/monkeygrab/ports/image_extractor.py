"""ImageExtractor -- PDF file to raster images grouped by page."""

from typing import Dict, List, Protocol

from monkeygrab.domain.extracted_image import ExtractedImage


class ImageExtractor(Protocol):
    """Reads one PDF file and returns its qualifying raster images, per page.

    Matches the single entry point the indexing pipeline calls today:
    ``extraer_imagenes_pdf(ruta_pdf, max_por_pagina, min_size_px)`` in
    ``rag/engine/images.py``. Both size arguments never vary per call (every
    call site in ``indexar_documentos`` passes the same two config-derived
    values for the whole run), so they are adapter configuration -- read once
    at construction, like every other adapter's config -- not per-call
    Protocol parameters. ``caption_margin_px`` (how far below the image the
    adapter searches for a caption) is adapter configuration for the same
    reason.

    Describing an image's visual content is NOT this port's job: that is a
    vision-model call (``describir_imagen_con_llm`` in ``rag/engine/images.py``),
    already covered by ``ChatModel.generate(prompt, images=[...])`` -- no
    separate port is needed for it, only a ``ChatModel`` wired to the "ocr"
    role.

    Failure policy: unlike most ports in this project, extraction failures
    are swallowed and logged rather than raised -- matching
    ``extraer_imagenes_pdf``'s own two failure layers exactly (opening the
    whole PDF with ``fitz`` fails -> return ``{}``; extracting one image's
    bytes fails -> skip that image, keep going). This is a deliberate
    carve-out from the project's hard-fail default (docs/design/2026-07-26-
    monkeygrab-v2.md, section 3), not an oversight: image extraction is
    optional enrichment gated by ``usar_embeddings_imagen``, and one corrupt
    embedded image on one page must not abort indexing of the PDF's text,
    exactly as it does not today. A caller that wants hard-fail-on-image-
    error wraps this port itself.
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
