"""ChunkMetadata -- position and format of a stored chunk."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ChunkMetadata:
    """Identity and shape of a chunk inside its source document.

    These fields are what the vector store persists alongside each chunk, and
    together they locate it precisely enough to derive its id, find its
    neighbors and cite it back to the user.

    Attributes:
        source: PDF filename the chunk was extracted from.
        page: Zero-based page number within the source PDF.
        chunk: Zero-based chunk index within the page. Image-derived chunks
            start at a high offset so they cannot collide with text chunk
            indices on the same page. Defaults to 0, since older stored
            metadata may omit it.
        total_chunks_in_page: Total number of text chunks produced for this
            page, used to detect the first and last chunk of a page during
            neighbor expansion. ``None`` for incomplete metadata.
        format: Extraction format -- ``"markdown"``, ``"plain_text"``, or
            ``"image"`` for the description of an extracted figure or table.
            ``None`` when unset.
        section_header: Nearest Markdown header above this chunk (empty
            string if none), or the OCR caption context for images.
        image_width: Pixel width, only set for ``format == "image"``.
        image_height: Pixel height, only set for ``format == "image"``.
    """

    source: str
    page: int
    chunk: int = 0
    total_chunks_in_page: Optional[int] = None
    format: Optional[str] = None
    section_header: str = ""
    image_width: Optional[int] = None
    image_height: Optional[int] = None
