"""LexicalIndex adapter over rank_bm25's Okapi BM25 implementation."""

from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi

from monkeygrab.application.keywords import tokenize_bm25
from monkeygrab.config.retrieval import RetrievalConfig
from monkeygrab.domain.fragment import Fragment
from monkeygrab.ports.vector_store import VectorStore

# Not subclassed from monkeygrab.ports.lexical_index.LexicalIndex: Protocol
# conformance here is structural (duck typing), the same contract every other
# adapter in this package satisfies without inheriting its port.


class Bm25LexicalIndex:
    """Ranks chunks from an injected ``VectorStore`` against a query via Okapi BM25.

    The ``LexicalIndex`` port has no "index these chunks" step and no
    ``collection`` parameter on ``search`` -- building and keeping the index in
    sync with the corpus is explicitly an adapter concern (see the port
    docstring). This adapter gets its corpus by scanning a ``VectorStore``
    port instance (``get_page``/``count``), so it depends on the storage
    *port*, not on FAISS specifically.

    Scanning and tokenizing a whole corpus is far too expensive to repeat per
    query, so the built index is cached under a ``(count, k1, b)`` key held on
    the instance -- not in a module global. Two indexes over two corpora, or
    two tests, therefore never invalidate each other. The ceiling is that an
    in-place edit keeping the same chunk count will not refresh the index;
    reindexing changes the count and does.

    Failure policy: hard-fail for anything reaching the underlying
    ``VectorStore`` -- its own hard-fail policy applies transitively, since
    nothing here catches its exceptions. A query with no positive BM25 match
    returns an empty list, which is a result and not a failure.
    """

    def __init__(self, vector_store: VectorStore, retrieval: RetrievalConfig):
        """Args:
            vector_store: ``VectorStore``-conforming instance whose stored
                chunks make up the BM25 corpus.
            retrieval: Retrieval config; only ``bm25_k1``/``bm25_b`` are read.
        """
        self._vector_store = vector_store
        self._k1 = retrieval.bm25_k1
        self._b = retrieval.bm25_b
        self._cache_key: Optional[Tuple[int, float, float]] = None
        self._bm25: Optional[BM25Okapi] = None
        self._entries: List[Fragment] = []

    def _ensure_index(self) -> None:
        """Rebuild the BM25 index iff the corpus size or BM25 parameters changed."""
        key = (self._vector_store.count(), self._k1, self._b)
        if key == self._cache_key:
            return

        entries = self._vector_store.get_page(limit=None, offset=0)
        corpus_tokens = [tokenize_bm25(entry.doc) for entry in entries]

        self._entries = entries
        self._bm25 = BM25Okapi(corpus_tokens, k1=self._k1, b=self._b) if any(corpus_tokens) else None
        self._cache_key = key

    def search(self, query: str, top_n: int) -> List[Fragment]:
        """Rank stored chunks against ``query`` by Okapi BM25 relevance.

        Args:
            query: User query text (tokenized internally).
            top_n: Maximum number of ranked results to return.

        Returns:
            Fragments ranked best-first (``score_keyword``/``score_final``
            left at their defaults -- fusion happens outside this port), or
            an empty list when the query has no tokens or no chunk scores
            above zero.
        """
        query_tokens = tokenize_bm25(query)
        if not query_tokens:
            return []

        self._ensure_index()
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: List[Fragment] = []
        for i in ranked_idx:
            if scores[i] <= 0:
                break
            results.append(self._entries[i])
            if len(results) >= top_n:
                break
        return results
