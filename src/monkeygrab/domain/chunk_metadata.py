"""ChunkMetadata -- position and format of a stored chunk."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ChunkMetadata:
    """Identity and shape of a chunk inside its source document.

    Mirrors, field for field, the metadata dict written by
    ``rag.engine.indexing.indexar_documentos`` (``source``, ``page``,
    ``chunk``, ``total_chunks_in_page``, ``format``, ``section_header``,
    ``image_width``, ``image_height``) and read back throughout
    ``rag/engine/retrieval.py``, ``rag/engine/lexical.py`` and
    ``rag/engine/generation.py`` (e.g. the chunk-id formula
    ``f"{source}_pag{page}_chunk{chunk}"`` in
    ``realizar_busqueda_hibrida``/``expandir_con_chunks_adyacentes``).

    ``chunk`` is the field name used in the original metadata dict (not
    "index"); kept as-is here so the derivation formula stays a literal
    match with the running pipeline.

    Attributes:
        source: PDF filename the chunk was extracted from.
        page: Zero-based page number within the source PDF.
        chunk: Zero-based chunk index within the page. Image-derived
            chunks use an offset range (see ``_IMAGEN_CHUNK_OFFSET`` in
            ``rag/chat_pdfs.py``) to avoid colliding with text chunk indices.
            Defaults to 0, mirroring the defensive ``metadata.get('chunk', 0)``
            reads in ``rag/engine/retrieval.py`` and ``rag/engine/chunking.py``.
        total_chunks_in_page: Total number of text chunks produced for this
            page, used to detect the first/last chunk of a page during
            neighbor expansion. ``None`` for legacy/incomplete metadata.
        format: Extraction format -- ``"markdown"`` (pymupdf4llm),
            ``"plain_text"`` (pypdf fallback), or ``"image"`` (OCR
            description of an extracted figure/table). ``None`` when unset.
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
