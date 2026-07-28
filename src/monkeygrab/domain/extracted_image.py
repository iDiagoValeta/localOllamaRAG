"""ExtractedImage -- one raster image pulled from a PDF page, pre-embedding."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedImage:
    """Raw image bytes plus the caption found near it.

    The multimodal embedder receives the pixels and optional caption together,
    preserving visual content while adding nearby textual context.

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
