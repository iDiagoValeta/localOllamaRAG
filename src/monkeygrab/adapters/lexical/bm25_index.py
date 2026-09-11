"""LexicalIndex adapter over rank_bm25's Okapi BM25 implementation."""

import threading
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi

from monkeygrab.application.keywords import tokenize_bm25
from monkeygrab.config.retrieval import RetrievalConfig
from monkeygrab.domain.fragment import Fragment
from monkeygrab.ports.vector_store import VectorStore

# Not subclassed from monkeygrab.ports.lexical_index.LexicalIndex: Protocol
# conformance here is structural (duck typing), the same contract every other
# adapter in this package satisfies without inheriting its port.

_IndexSnapshot = Tuple[Optional[Tuple[int, float, float]], List[Fragment], Optional[BM25Okapi]]


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

    ``rag/engine/wiring.py`` caches one instance per ``(store, k1, b)`` and
    hands it to every concurrent Flask request thread (``threaded=True``).
    ``_ensure_index`` is written for that: see its docstring for the race a
    shared instance is exposed to and how the lock closes it (#238).

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
        # (cache_key, entries, bm25) published as one immutable tuple behind
        # one attribute rather than three separate ones, so a reader gets a
        # matched triple from a single dereference -- never entries from one
        # rebuild paired with a bm25 from another, which three independently
        # read/written attributes cannot promise (this is the same technique
        # rag/engine/wiring.py's vector_store()/lexical_index() caches use,
        # for the identical reason -- see that module's #57 docstrings).
        self._index: _IndexSnapshot = (None, [], None)
        self._rebuild_lock = threading.Lock()

    def _ensure_index(self) -> Tuple[List[Fragment], Optional[BM25Okapi]]:
        """Return the current ``(entries, bm25)`` pair, rebuilding iff the
        corpus size or the BM25 parameters changed since the last rebuild.

        ``rag/engine/wiring.py`` shares one instance across concurrent Flask
        request threads. Rebuilding was previously three unsynchronized
        writes (``self._entries``, then ``self._bm25``, then
        ``self._cache_key``) with no lock at all: a thread scheduled out
        mid-rebuild -- a slow ``BM25Okapi()`` call, or plain GIL contention
        under load -- could resume and publish its own snapshot, read
        *before* a concurrent ``delete_source()``, *after* a second thread's
        rebuild (triggered by that same delete) had already published a
        fresher one. The slower write landed last and won for all three
        fields at once, silently reverting the index to cite content the
        caller had just deleted (#238).

        The whole rebuild -- read corpus, tokenize, construct ``BM25Okapi``,
        publish -- now runs inside ``_rebuild_lock``, not just the publish
        step. That serializes rebuilds end to end: whichever one starts
        later can only do so after the earlier one has already published, so
        it always builds from a corpus snapshot at least as fresh, and it is
        the one left standing. A request whose own read began before a
        concurrent mutation can still see the pre-mutation corpus for that
        one call -- there is no way to retroact on data already read without
        locking the vector store itself during a rebuild, which is out of
        this adapter's scope -- but that state is never a torn mix, and a
        slower rebuild can never overwrite a fresher one that a concurrent
        caller already observed.

        The cheap, unlocked key check outside the lock keeps the common case
        (cache already valid) lock-free, mirroring the double-checked
        locking ``vector_store()``/``lexical_index()`` use in
        ``rag/engine/wiring.py``.

        Returns:
            The fragments backing the index and the built ``BM25Okapi``
            (``None`` when every chunk tokenized to nothing).
        """
        cache_key, entries, bm25 = self._index
        key = (self._vector_store.count(), self._k1, self._b)
        if key == cache_key:
            return entries, bm25

        with self._rebuild_lock:
            cache_key, entries, bm25 = self._index
            key = (self._vector_store.count(), self._k1, self._b)
            if key == cache_key:
                return entries, bm25

            entries = self._vector_store.get_page(limit=None, offset=0)
            corpus_tokens = [tokenize_bm25(entry.doc) for entry in entries]
            bm25 = BM25Okapi(corpus_tokens, k1=self._k1, b=self._b) if any(corpus_tokens) else None

            self._index = (key, entries, bm25)
            return entries, bm25

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

        entries, bm25 = self._ensure_index()
        if bm25 is None:
            return []

        scores = bm25.get_scores(query_tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: List[Fragment] = []
        for i in ranked_idx:
            if scores[i] <= 0:
                break
            results.append(entries[i])
            if len(results) >= top_n:
                break
        return results
