"""Reranker -- Cross-Encoder-style re-scoring of retrieval candidates.

# ─────────────────────────────────────────────
# SECTION 1: PORT
# ─────────────────────────────────────────────
"""

from typing import List, Protocol, Sequence

from monkeygrab.domain.fragment import Fragment


class Reranker(Protocol):
    """Re-scores and truncates retrieval candidates against the query.

    Matches ``rerank_resultados(pregunta, documentos_recuperados, top_k)``
    in ``rag/engine/reranking.py``: query plus candidate fragments in,
    the top ``top_k`` fragments out with a fresh relevance score.

    Failure policy: hard-fail. Raise if the model cannot score the
    candidates (fails to load, inference error). Today's three-level
    silent degradation -- GPU load failure falls back to CPU, and if the
    reranker cannot be loaded at all ``rerank_resultados`` returns the
    input unranked and untruncated -- does not carry over: an adapter that
    cannot rerank must not pretend it did. If GPU-with-CPU-fallback is
    still wanted operationally, it is the adapter's own explicit two-step
    logic (attempt, catch, retry once), not a caller-invisible substitution
    of "reranked" for "not reranked".
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
