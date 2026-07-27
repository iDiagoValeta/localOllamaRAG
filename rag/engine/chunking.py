"""Auxiliary implementation module for rag.chat_pdfs.

This module keeps business logic split out of the public facade. Runtime
configuration stays owned by rag.chat_pdfs and is read lazily through ``cfg``
(a live reference to that module), so web/API toggles and test monkeypatches
are observed without any per-call synchronization.
"""

from typing import Any, Dict, List

from monkeygrab.application.text_chunking import (
    adjacent_chunk_ids,
    split_markdown_into_chunks,
)
from monkeygrab.domain.chunk_metadata import ChunkMetadata
from rag.engine.runtime import get_runtime

cfg = get_runtime()
# PREPROCESSING AND CHUNKING


def dividir_en_chunks(
    texto: str,
    chunk_size: int = cfg.CHUNK_SIZE,
    overlap: int = cfg.CHUNK_OVERLAP
) -> List[Dict[str, str]]:
    """Split text into chunks by Markdown sections with overlap.

    Uses a recursive separator strategy to break content at natural
    boundaries (paragraphs, sentences, commas, spaces). Each chunk
    preserves its nearest Markdown header for context.

    Args:
        texto: Source text to split.
        chunk_size: Maximum character length per chunk.
        overlap: Number of trailing characters from the previous chunk
            to prepend to the next one.

    Returns:
        List of dicts with ``"text"`` and ``"header"`` keys.
    """
    # Implementation lives in monkeygrab.application.text_chunking
    # (split_markdown_into_chunks) -- a literal port, equivalence-tested
    # against this function in tests/unit/application/test_text_chunking_
    # equivalence.py. cfg.MIN_CHUNK_LENGTH is still read live here (not
    # captured as a stale default) and passed through as an explicit
    # parameter, since the new layer takes config as arguments instead of
    # reading a module global.
    chunks = split_markdown_into_chunks(texto, chunk_size, overlap, cfg.MIN_CHUNK_LENGTH)
    return [{"text": c.text, "header": c.header} for c in chunks]


def expandir_con_chunks_adyacentes(
    chunk_id: str,
    metadata: Dict[str, Any],
    n_vecinos: int = 1
) -> List[str]:
    """Build IDs for neighboring chunks (same-page and cross-page) for context expansion.

    Args:
        chunk_id: Identifier of the anchor chunk.
        metadata: Metadata dict with ``source``, ``page``, ``chunk``,
            and optionally ``total_chunks_in_page``.
        n_vecinos: How many neighbors to include on each side.

    Returns:
        List of neighboring chunk IDs.
    """
    # Implementation lives in monkeygrab.application.text_chunking
    # (adjacent_chunk_ids). chunk_id is accepted for API compatibility but,
    # like the original, never read -- every id built here derives from
    # metadata alone.
    meta = ChunkMetadata(
        source=metadata['source'],
        page=metadata['page'],
        chunk=metadata.get('chunk', 0),
        total_chunks_in_page=metadata.get('total_chunks_in_page', None),
    )
    return adjacent_chunk_ids(meta, n_neighbors=n_vecinos)


