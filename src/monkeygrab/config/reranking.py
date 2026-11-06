"""RerankingConfig -- BGE Cross-Encoder relevance threshold.

Default is a copy of ``rag/chat_pdfs.py`` section 3.7
(``RAG_UMBRAL_SCORE_RERANKER``). The model itself is fixed to
``BAAI/bge-reranker-v2-m3`` by the adapter.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RerankingConfig:
    """Parameters ``rag.engine.generation._filtrar_por_umbral_reranker`` reads.

    Attributes:
        score_threshold: Minimum reranker (or, absent a rerank score, RRF
            ``score_final``) a fragment must reach to survive filtering
            before generation; inclusive (``>=``) (``RAG_UMBRAL_SCORE_RERANKER``).
    """

    score_threshold: float = 0.65
    #: Opt-in alternative to the per-fragment threshold (``RAG_SUELO_SCORE_RERANKER``;
    #: 0 disables it). When set, the question is judged by its BEST fragment:
    #: at or above the floor, every reranked fragment goes to the generator;
    #: below it, nothing does. Tables, notes and multi-part questions score
    #: their relevant chunks well under 0.65 (0.03-0.3 measured) while
    #: off-topic questions stay near 0.0-0.04, so a per-fragment cut either
    #: drops real evidence or lets noise through, and the best score separates
    #: the two cases cleanly.
    relevance_floor: float = 0.0
