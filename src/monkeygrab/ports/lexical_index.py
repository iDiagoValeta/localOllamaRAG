"""LexicalIndex -- BM25-style lexical (keyword) search.

# ─────────────────────────────────────────────
# SECTION 1: PORT
# ─────────────────────────────────────────────
"""

from typing import List, Protocol

from monkeygrab.domain.fragment import Fragment


class LexicalIndex(Protocol):
    """Ranks stored chunks against a query by lexical (term) overlap.

    Matches the single public entry point the pipeline calls today:
    ``busqueda_lexica_bm25(pregunta, collection, top_n)`` in
    ``rag/engine/lexical.py``. How the index is built and kept in sync with
    the corpus (today: an Okapi BM25 index scanned and cached per
    collection, invalidated by a ``(name, count, k1, b)`` cache key -- see
    ``_obtener_indice_bm25``) is an adapter concern, not part of this
    port's contract: nothing in the pipeline calls a separate "index
    these chunks" step, so this Protocol does not invent one.

    Failure policy: hard-fail. Raise if the search cannot be performed
    (e.g. the underlying index cannot be built or read). Returning an
    empty list is reserved for the genuine "no positive match" case
    (all BM25 scores <= 0, exactly as ``busqueda_lexica_bm25`` does today),
    not for swallowed errors.
    """

    def search(self, query: str, top_n: int) -> List[Fragment]:
        """Rank stored chunks against ``query`` by lexical relevance.

        Args:
            query: User query text (tokenized internally by the adapter).
            top_n: Maximum number of ranked results to return.

        Returns:
            Fragments ranked best-first, or an empty list when no chunk
            has a positive lexical match. ``score_keyword``/``score_final``
            are left at their defaults: this port ranks, it does not fuse
            with semantic search.

        Raises:
            Exception: On any search failure.
        """
        ...
