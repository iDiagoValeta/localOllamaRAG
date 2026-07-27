"""VectorStore -- add/query/get/count over embedded chunks.

# ─────────────────────────────────────────────
# SECTION 1: PORT
# ─────────────────────────────────────────────
"""

from typing import List, Optional, Protocol, Sequence

from monkeygrab.domain.chunk import Chunk
from monkeygrab.domain.fragment import Fragment


class VectorStore(Protocol):
    """Exactly the five Chroma operations the pipeline exercises today.

    No more, no less -- each maps to one real call site, and no ``where``
    filter is used anywhere in the codebase, so it is deliberately absent
    here (a metadata-sidecar index such as FAISS can implement this whole
    port without a query planner):

    - ``add``: ``collection.add(ids=[id_doc], embeddings=[embedding],
      documents=[chunk_doc_text], metadatas=[metadata])`` in
      ``_indexar_chunk`` (``rag/engine/indexing.py``), always one chunk
      at a time.
    - ``query``: ``collection.query(query_embeddings=[...], n_results=...,
      include=[...])`` in ``realizar_busqueda_hibrida``
      (``rag/engine/retrieval.py``), once per query variant.
    - ``get_by_ids``: ``collection.get(ids=ids_vecinos, include=[...])``
      in ``_expandir_fragmentos_contexto`` (``rag/engine/generation.py``),
      to fetch specific neighbor chunks.
    - ``get_page``: ``collection.get(limit=..., offset=..., include=[...])``
      in ``_construir_indice_bm25`` (``rag/engine/lexical.py``, batched
      full-corpus scan to build the BM25 index) and the unpaginated
      ``collection.get(include=['metadatas'])`` in
      ``obtener_documentos_indexados`` (``rag/engine/indexing.py``,
      ``limit=None`` here means "everything").
    - ``count``: ``collection.count()``, used both by
      ``_obtener_indice_bm25``'s cache key and inside ``get_page``'s
      batch loop.

    Every method returns ``Fragment`` -- the shape callers immediately
    rebuild raw Chroma rows into anyway (``all_semantic_results`` in
    ``realizar_busqueda_hibrida``, the neighbor dicts in
    ``_expandir_fragmentos_contexto``, the BM25 ``corpus_entries`` in
    ``_construir_indice_bm25`` all use ``doc``/``metadata``/``id`` with
    scores left at their defaults). Ranking (RRF, reranking) is computed
    by the retrieval use case, never by this port.

    Failure policy: hard-fail. Raise on any storage or query failure --
    ``obtener_documentos_indexados`` today swallows errors and returns an
    empty list; that silent-empty behavior does not carry over.
    """

    def add(self, chunk: Chunk, embedding: Sequence[float]) -> None:
        """Store one embedded chunk.

        Args:
            chunk: The chunk to store; ``chunk.id`` is the storage key.
            embedding: The chunk's embedding vector.

        Raises:
            Exception: On any storage failure.
        """
        ...

    def query(self, embedding: Sequence[float], n_results: int) -> List[Fragment]:
        """Semantic search: the ``n_results`` nearest chunks to ``embedding``.

        Args:
            embedding: Query embedding vector.
            n_results: Maximum number of results to return.

        Returns:
            Fragments ordered nearest-first, with ``distancia`` set to the
            L2 distance and every score field left at its default (RRF
            fusion and reranking happen outside this port).

        Raises:
            Exception: On any query failure.
        """
        ...

    def get_by_ids(self, ids: Sequence[str]) -> List[Fragment]:
        """Fetch specific chunks by id (used for neighbor-context expansion).

        Args:
            ids: Chunk ids to fetch. Ids with no match are simply absent
                from the result -- this is not a failure.

        Returns:
            Fragments for the ids that exist, in no particular order.

        Raises:
            Exception: On any storage failure.
        """
        ...

    def get_page(self, limit: Optional[int], offset: int) -> List[Fragment]:
        """Fetch a page of the whole collection, for full-corpus scans.

        Args:
            limit: Maximum number of chunks to return, or ``None`` for
                every remaining chunk from ``offset`` onward.
            offset: Number of chunks to skip from the start.

        Returns:
            Fragments in storage order.

        Raises:
            Exception: On any storage failure.
        """
        ...

    def count(self) -> int:
        """Total number of chunks currently stored.

        Raises:
            Exception: On any storage failure.
        """
        ...
