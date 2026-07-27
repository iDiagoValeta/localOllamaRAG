"""PymupdfImageExtractor -- ImageExtractor adapter over PyMuPDF (fitz)."""

import logging
from typing import Dict, List

import fitz

from monkeygrab.config.chunking import ChunkingConfig
from monkeygrab.domain.extracted_image import ExtractedImage

# Not subclassed from monkeygrab.ports.image_extractor.ImageExtractor: Protocol
# conformance here is structural (duck typing), the same contract every other
# adapter in this package satisfies without inheriting its port.


class PymupdfImageExtractor:
    """Extracts raster images from a PDF via PyMuPDF, grouped by page.

    Literal port of ``extraer_imagenes_pdf`` (``rag/engine/images.py``):
    opens the PDF once with ``fitz``, walks every page's image list, skips
    images below the configured minimum size, and looks for a caption in
    the margin immediately below each image's bounding box.

    Per the ``ImageExtractor`` port's documented carve-out from this
    project's hard-fail default, both of the original's failure layers are
    reproduced exactly: if ``fitz`` cannot open the file at all, this
    returns ``{}`` (logged, not raised) instead of aborting the whole PDF's
    indexing; if one image's bytes cannot be extracted, that image is
    skipped and extraction continues with the rest.
    """

    def __init__(self, chunking: ChunkingConfig):
        """Args:
            chunking: Chunking config; ``max_images_per_page``,
                ``min_image_size_px`` and ``caption_margin_px`` are read.
        """
        self._max_per_page = chunking.max_images_per_page
        self._min_size_px = chunking.min_image_size_px
        self._caption_margin_px = chunking.caption_margin_px

    def extract(self, pdf_path: str) -> Dict[int, List[ExtractedImage]]:
        """Extract every qualifying image of ``pdf_path``, grouped by page.

        Args:
            pdf_path: Absolute or relative path to the PDF file.

        Returns:
            Mapping of zero-based page number to the images found on that
            page, in document order. ``{}`` if the file could not be opened.
        """
        images_by_page: Dict[int, List[ExtractedImage]] = {}

        try:
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                page_images: List[ExtractedImage] = []

                for img_info in image_list:
                    if len(page_images) >= self._max_per_page:
                        break

                    xref = img_info[0]
                    try:
                        img_data = doc.extract_image(xref)
                        width = img_data.get("width", 0)
                        height = img_data.get("height", 0)

                        if width < self._min_size_px or height < self._min_size_px:
                            continue

                        caption_text = ""
                        try:
                            img_rects = page.get_image_rects(xref)
                            if img_rects:
                                img_rect = img_rects[0]
                                below_rect = fitz.Rect(
                                    img_rect.x0,
                                    img_rect.y1,
                                    img_rect.x1,
                                    img_rect.y1 + self._caption_margin_px,
                                )
                                candidate = page.get_text("text", clip=below_rect).strip()
                                candidate = " ".join(candidate.split())
                                if candidate and len(candidate) <= 300:
                                    caption_text = candidate
                        except Exception:
                            pass  # caption is optional; never interrupt image extraction

                        page_images.append(ExtractedImage(
                            image_bytes=img_data["image"],
                            width=width,
                            height=height,
                            ext=img_data.get("ext", "png"),
                            caption=caption_text,
                        ))
                    except Exception as exc:
                        logging.warning(
                            "Error extracting image xref=%s from %s page %s: %s",
                            xref, pdf_path, page_num, exc,
                        )

                if page_images:
                    images_by_page[page_num] = page_images

            doc.close()

        except Exception as exc:
            logging.warning("Error reading %s with fitz for image extraction: %s", pdf_path, exc)

        return images_by_page
