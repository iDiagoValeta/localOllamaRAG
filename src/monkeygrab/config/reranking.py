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
