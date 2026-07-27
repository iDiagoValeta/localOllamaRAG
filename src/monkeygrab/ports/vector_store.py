"""VectorStore -- add/query/get/count over embedded chunks."""

from typing import List, Optional, Protocol, Sequence

from monkeygrab.domain.chunk import Chunk
from monkeygrab.domain.fragment import Fragment


class VectorStore(Protocol):
    """Storage and retrieval of embedded chunks.

    Five operations, each backing one thing the pipeline actually does:
    ``add`` stores a chunk during indexing, ``query`` is semantic search,
    ``get_by_ids`` fetches neighbor chunks for context expansion,
    ``get_page`` supports full-corpus scans (building the lexical index,
    listing indexed documents) and ``count`` reports corpus size.

    There is no ``where`` filter because nothing in the pipeline filters by
    metadata. Keeping it out means a plain vector index with a metadata
    sidecar, such as FAISS, can satisfy this port without a query planner.

    Every method returns ``Fragment`` with its score fields at their
    defaults. This port stores and fetches; ranking belongs to the retrieval
    use case.

    Failure policy: hard-fail. Raise on any storage or query failure. An
    empty result must mean the corpus held nothing matching, never that a
    read failed.
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
