"""RerankingConfig -- Cross-Encoder relevance threshold.

Default is a copy of ``rag/chat_pdfs.py`` section 3.7
(``RAG_UMBRAL_SCORE_RERANKER``). The reranker *model variant*
(``RERANKER_QUALITY``, "quality"/"fast") lives in ``config.models.ModelsConfig
.reranker_quality`` instead: it selects which Cross-Encoder checkpoint
loads, the same kind of decision as any other ``MODELO_*`` role, not a
reranking-behavior parameter.
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
