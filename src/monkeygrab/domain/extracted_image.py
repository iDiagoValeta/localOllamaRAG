"""ExtractedImage -- one raster image pulled from a PDF page, pre-description."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedImage:
    """Raw image bytes plus the caption found near it.

    The input to image description: a vision model is asked what the image
    depicts, and the caption goes in with it as context, because a figure
    caption usually names what the pixels alone are ambiguous about.

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
