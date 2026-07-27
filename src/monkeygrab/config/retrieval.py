"""RetrievalConfig -- semantic/BM25 fan-out and Reciprocal-Rank-Fusion.

Defaults mirror the ones ``rag/chat_pdfs.py`` reads from the environment;
``tests/unit/test_app_config_defaults.py`` fails if the two drift apart.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalConfig:
    """How wide each retrieval branch casts its net, and how the two combine.

    Attributes:
        n_semantic_results: Candidates fetched per semantic query variant
            (``RAG_N_RESULTADOS_SEMANTICOS``).
        n_keyword_results: Maximum BM25 candidates returned per query
            (``RAG_N_RESULTADOS_KEYWORD``).
        top_k_rerank_candidates: Fused candidates handed to the reranker
            (``RAG_TOP_K_RERANK_CANDIDATES``).
        top_k_final: Final number of fragments kept after reranking
            (``RAG_TOP_K_FINAL``).
        n_top_for_expansion: How many top fragments get neighbor-chunk
            expansion (``RAG_N_TOP_PARA_EXPANSION``).
        rrf_k: Reciprocal-Rank-Fusion rank-damping constant (``RAG_RRF_K``).
        bm25_k1: Okapi BM25 term-frequency saturation parameter (``RAG_BM25_K1``).
        bm25_b: Okapi BM25 length-normalization parameter (``RAG_BM25_B``).
        weight_semantic_rrf: Fusion weight applied to the semantic RRF score
            (``RAG_PESO_SEMANTICO_RRF``).
        weight_bm25_rrf: Fusion weight applied to the BM25 RRF score
            (``RAG_PESO_BM25_RRF``).
        min_question_length: Minimum question length in characters below
            which the pipeline does not run at all
            (``RAG_MIN_LONGITUD_PREGUNTA``).
    """

    n_semantic_results: int = 80
    n_keyword_results: int = 40
    top_k_rerank_candidates: int = 200
    top_k_final: int = 8
    n_top_for_expansion: int = 3
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    weight_semantic_rrf: float = 0.55
    weight_bm25_rrf: float = 0.45
    min_question_length: int = 10
