"""ChunkingConfig -- indexing-time text splitting parameters.

Defaults are a field-for-field copy of ``rag/chat_pdfs.py`` section 3.7
(the ``RAG_*``/``CONTEXTUAL_DOC_CHARS`` group of ``_leer_env_int`` calls).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    """Parameters ``rag.engine.chunking.dividir_en_chunks`` reads.

    Attributes:
        chunk_size: Maximum character length per text chunk (``RAG_CHUNK_SIZE``).
        chunk_overlap: Trailing characters from the previous chunk prepended
            to the next one (``RAG_CHUNK_OVERLAP``).
        min_chunk_length: Minimum character length for a chunk to be kept
            (``RAG_MIN_CHUNK_LENGTH``).
        contextual_doc_chars: Size of the document-level text sample used to
            generate contextual-retrieval summaries (``CONTEXTUAL_DOC_CHARS``).
    """

    chunk_size: int = 2000
    chunk_overlap: int = 400
    min_chunk_length: int = 150
    contextual_doc_chars: int = 24000
