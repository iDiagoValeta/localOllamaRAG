"""Reranker -- Cross-Encoder-style re-scoring of retrieval candidates."""

from typing import List, Protocol, Sequence

from monkeygrab.domain.fragment import Fragment


class Reranker(Protocol):
    """Re-scores and truncates retrieval candidates against the query.

    Fusion ranks by how often and how highly each branch retrieved a chunk;
    this port instead reads the query and the chunk together and scores their
    actual relevance. It is expensive, which is why it runs last and only on
    the candidates fusion already shortlisted.

    Failure policy: hard-fail. Raise if the model cannot load or cannot
    score, rather than returning the input unranked -- an adapter that could
    not rerank must not report results as if it had. A GPU-to-CPU retry is
    allowed, but as the adapter's own explicit two-step logic, never as a
    caller-invisible substitution of "not reranked" for "reranked".
    """

    def rerank(self, query: str, fragments: Sequence[Fragment], top_k: int) -> List[Fragment]:
        """Re-score ``fragments`` against ``query`` and keep the best ``top_k``.

        Args:
            query: User query text.
            fragments: Candidate fragments from prior retrieval stages.
            top_k: Maximum number of fragments to return.

        Returns:
            Up to ``top_k`` fragments, best-first, each with
            ``score_reranker`` set to the model's relevance score.

        Raises:
            Exception: On any reranking failure.
        """
        ...
