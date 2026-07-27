"""LexicalIndex -- BM25-style lexical (keyword) search."""

from typing import List, Protocol

from monkeygrab.domain.fragment import Fragment


class LexicalIndex(Protocol):
    """Ranks stored chunks against a query by lexical (term) overlap.

    The lexical branch of hybrid retrieval: it finds chunks that share
    literal terms with the question, which is what semantic search is worst
    at -- identifiers, acronyms, figures and numbers.

    There is deliberately no "index these chunks" method. Building the index
    and keeping it in sync with the corpus is an adapter concern, and the
    retrieval use case has no business knowing whether one exists.

    Failure policy: hard-fail. Raise if the search cannot be performed at
    all. An empty list means "no chunk matched", which is a result; it must
    never stand in for a swallowed error.
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
