"""ExtractedImage -- one raster image pulled from a PDF page, pre-description."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedImage:
    """Raw image bytes plus the caption found near it, the input to image
    description (a vision-capable ``ChatModel.generate(..., images=[...])``
    call).

    Field-for-field match of the dicts ``rag.engine.images.extraer_imagenes_pdf``
    returns per page (``"image_bytes"``, ``"width"``, ``"height"``, ``"ext"``,
    ``"caption"``).

    Attributes:
        image_bytes: Raw image bytes, in whatever format the source PDF
            embedded (PNG, JPEG, ...).
        width: Pixel width.
        height: Pixel height.
        ext: Image format, e.g. ``"png"``.
        caption: Caption text found immediately below the image's bounding
            box on the same page; empty string when none was found.
    """

    image_bytes: bytes
    width: int
    height: int
    ext: str
    caption: str = ""
