"""ChunkingConfig -- indexing-time text splitting parameters and the PDF extractor feeding them.

Defaults are a field-for-field copy of ``rag/chat_pdfs.py`` section 3.7
(the ``RAG_*``/``CONTEXTUAL_DOC_CHARS`` group of ``_leer_env_int`` calls).
"""

from dataclasses import dataclass

PDF_EXTRACTORS = ("mineru", "docling")


@dataclass(frozen=True)
class ChunkingConfig:
    """Parameters ``split_markdown_into_chunks`` takes.

    Attributes:
        chunk_size: Maximum character length per text chunk (``RAG_CHUNK_SIZE``).
        chunk_overlap: Trailing characters from the previous chunk prepended
            to the next one (``RAG_CHUNK_OVERLAP``).
        min_chunk_length: Minimum character length for a chunk to be kept
            (``RAG_MIN_CHUNK_LENGTH``).
        contextual_doc_chars: Size of the document-level text sample used to
            generate contextual-retrieval summaries (``CONTEXTUAL_DOC_CHARS``).
        pdf_extractor: Which extractor turns a PDF into page text and
            figures, one of ``PDF_EXTRACTORS`` (``MONKEYGRAB_PDF_EXTRACTOR``).
            It decides what an index stores, so it is part of the index
            fingerprint.
    """

    chunk_size: int = 2000
    chunk_overlap: int = 400
    min_chunk_length: int = 150
    contextual_doc_chars: int = 24000
    pdf_extractor: str = "mineru"

    def __post_init__(self) -> None:
        """Reject an unknown extractor: a typo must not index with one nobody chose."""
        if self.pdf_extractor not in PDF_EXTRACTORS:
            raise ValueError(
                f"Invalid value for MONKEYGRAB_PDF_EXTRACTOR: {self.pdf_extractor!r}. "
                f"Must be one of {PDF_EXTRACTORS}"
            )
