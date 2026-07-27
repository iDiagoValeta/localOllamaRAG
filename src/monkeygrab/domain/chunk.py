"""Chunk -- a text unit ready to be embedded and stored.

# ─────────────────────────────────────────────
# SECTION 1: ENTITY
# ─────────────────────────────────────────────
"""

from dataclasses import dataclass

from monkeygrab.domain.chunk_metadata import ChunkMetadata


@dataclass(frozen=True)
class Chunk:
    """One indexing-time unit: the text handed to the embedder and stored.

    Combines two things that are split across two source modules today:
    the chunking output of ``rag.engine.chunking.dividir_en_chunks``
    (``{"text": ..., "header": ...}``) and the positional metadata
    ``rag.engine.indexing.indexar_documentos`` attaches before calling
    ``collection.add`` (``source``, ``page``, ``chunk`` index,
    ``total_chunks_in_page``, ``format``, ``section_header``). Indexing
    already renames chunking's ``header`` key to ``section_header`` when it
    builds that metadata dict (see ``metadata["section_header"] =
    chunk_header`` in ``indexar_documentos``); ``Chunk.metadata.section_header``
    is that same value, carried in ``ChunkMetadata`` instead of a second field.

    ``text`` holds the final text sent to the embedder -- for chunks with
    contextual retrieval enabled that is ``contexto_situacional + chunk_text``,
    matching what ``indexar_documentos`` stores as both ``documents`` and the
    embedding input.

    Attributes:
        text: Text to embed and store.
        metadata: Position and format of this chunk within its document.
    """

    text: str
    metadata: ChunkMetadata

    @property
    def id(self) -> str:
        """Chunk id, e.g. ``"paper.pdf_pag2_chunk3"``.

        Same formula used throughout the pipeline
        (``f"{source}_pag{page}_chunk{chunk}"``) to build/look up chunk ids
        -- kept as a derived property, not a stored field, so a ``Chunk``
        can never carry an id that disagrees with its own metadata.
        """
        return f"{self.metadata.source}_pag{self.metadata.page}_chunk{self.metadata.chunk}"
