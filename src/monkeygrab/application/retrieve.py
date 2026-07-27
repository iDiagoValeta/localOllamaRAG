"""Retrieve -- query decomposition (opt.) -> semantic + lexical search ->
RRF fusion -> reranking (opt.) -> reranker-threshold filter.

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  +-- RetrieveResult              -- fragments + observability metrics
#  +-- _generate_subqueries        -- query decomposition via ChatModel (optional)
#  +-- _relevance_score            -- moved from generation.py's _score_relevancia_fragmento
#  +-- _filter_by_reranker_threshold -- moved from generation.py's _filtrar_por_umbral_reranker
#  +-- _embedding_keep_alive       -- moved from retrieval.py's _embedding_keep_alive
#  +-- Retrieve                    -- the use case
#
# ─────────────────────────────────────────────

Orchestrates ``rag.engine.retrieval.realizar_busqueda_hibrida`` (multi-query
semantic search + BM25 + RRF fusion + optional reranking) followed by the
reranker-threshold filter that used to live at the top of
``rag.engine.generation.preparar_fragmentos_para_generacion``. Everything
that talks to infrastructure goes through an injected port (``Embedder``,
``VectorStore``, ``LexicalIndex``, ``Reranker``, ``ChatModel``); the RRF math
itself lives in ``monkeygrab.application.rrf_fusion`` (equivalence-tested
separately).

Known, deliberate scope reduction vs. the original
``realizar_busqueda_hibrida``: when LLM query decomposition yields no
sub-queries (disabled, question too short, or the LLM call fails), the
original falls back to appending ONE extra query variant built from
``extraer_keywords``/``_validar_coherencia_query`` (``rag/engine/lexical.py``
and ``rag/engine/reranking.py``). Those two helpers are not part of the four
modules this migration is scoped to move and are not exposed by any port, so
that fallback query variant is not reproduced here -- see the class
docstring and the final report for why, and why the *coherence check* on the
LLM sub-queries themselves needed no equivalent (it is provably a no-op in
the original: see below).
"""

import dataclasses
from typing import Any, Dict, List, Optional

from monkeygrab.application.rrf_fusion import fuse_semantic_and_keyword
from monkeygrab.config.app_config import AppConfig
from monkeygrab.domain.fragment import Fragment
from monkeygrab.ports.chat_model import ChatModel
from monkeygrab.ports.embedder import Embedder
from monkeygrab.ports.lexical_index import LexicalIndex
from monkeygrab.ports.reranker import Reranker
from monkeygrab.ports.vector_store import VectorStore


@dataclasses.dataclass
class RetrieveResult:
    """Output of ``Retrieve.run``.

    Attributes:
        fragments: Final ranked, filtered fragments (at most
            ``retrieval.top_k_final``).
        metrics: Observability data (query variants used, candidate counts
            per stage, whether reranking ran). Not part of any port's
            contract -- see ``monkeygrab.application``'s module docstring
            for the project-wide decision to carry metrics on a result
            object instead of module-level logging or tuple returns.
    """

    fragments: List[Fragment]
    metrics: Dict[str, Any]


def _parse_subqueries(raw_response: str) -> List[str]:
    """Parse an LLM's raw sub-query response into up to 3 query strings.

    Moved from the parsing tail of
    ``rag.engine.reranking.generar_queries_con_llm``.
    """
    queries = [
        q.strip().lstrip("0123456789.-) ")
        for q in raw_response.strip().split("\n")
        if q.strip() and len(q.strip()) > 20
    ]
    return queries[:3]


def _generate_subqueries(chat_model: ChatModel, pregunta: str) -> List[str]:
    """Generate up to 3 search sub-queries via an auxiliary LLM.

    Prompt text is a verbatim copy of
    ``rag.engine.reranking.generar_queries_con_llm``'s prompt. Sampling
    parameters (temperature 0.5, num_predict 400, stop sequence) are not
    passed here -- the ``ChatModel`` port has no per-call options parameter
    by design (see the port docstring); they belong to how the injected
    ``ChatModel`` adapter for the "chat" role was constructed.

    Args:
        chat_model: ``ChatModel`` adapter wired to the query-decomposition
            role (``MODELO_CHAT``).
        pregunta: The original user question.

    Returns:
        Up to 3 generated queries, or an empty list on failure.
    """
    prompt = (
        "Generate exactly 3 search queries to retrieve relevant content "
        "from academic documents about the question below.\n\n"
        "Requirements:\n"
        "- Each query must target a DIFFERENT semantic aspect of the question\n"
        "- Write every query in the EXACT SAME LANGUAGE as the question\n"
        "- Output ONLY the 3 queries, one per line\n"
        "- No numbering, no bullets, no labels, no explanations\n\n"
        f"Question: {pregunta}\n\n"
        "Queries:\n"
    )
    try:
        raw = chat_model.generate(prompt)
    except Exception:
        # Explicit use-case-level fallback: the ChatModel port's failure
        # policy sanctions exactly this pattern ("a caller that wants a
        # fallback makes it an explicit decision using two ports"). Query
        # decomposition is optional enrichment (Retrieve's docstring: "opt.")
        # -- its failure degrades to "search with the original question
        # only", matching generar_queries_con_llm's own
        # `except Exception: return []`.
        return []
    return _parse_subqueries(raw)


def _relevance_score(fragment: Fragment) -> float:
    """Return the active relevance score for post-reranker filtering.

    Moved from ``rag.engine.generation._score_relevancia_fragmento``.
    """
    try:
        value = fragment.score_reranker if fragment.score_reranker is not None else fragment.score_final
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _filter_by_reranker_threshold(
    fragments: List[Fragment], usar_reranker: bool, threshold: float
) -> List[Fragment]:
    """Apply the single reranker relevance threshold.

    Moved from ``rag.engine.generation._filtrar_por_umbral_reranker``.
    """
    if not usar_reranker:
        return list(fragments)
    return [f for f in fragments if _relevance_score(f) >= threshold]


def _embedding_keep_alive(q_idx: int, n_queries: int) -> Optional[int]:
    """VRAM residency for one query-variant embedding call.

    Literal port of ``_embedding_keep_alive`` in ``rag/engine/retrieval.py``:
    the embedding model is unloaded (``keep_alive=0``) after the LAST query
    variant, freeing VRAM for the RAG generator that runs immediately
    afterward; every earlier variant keeps the server's own default residency
    (``None``) since another embedding call is about to follow right away.

    Args:
        q_idx: Zero-based index of this query variant in ``queries``.
        n_queries: Total number of query variants.

    Returns:
        ``0`` on the last variant, ``None`` otherwise.
    """
    return 0 if q_idx >= n_queries - 1 else None


class Retrieve:
    """Query decomposition (opt.) -> semantic + lexical search -> RRF fusion
    -> reranking (opt.) -> reranker-threshold filter.

    Ports are all read/used at call time from the constructor-injected
    instances -- never captured as a default argument -- so a config change
    between two ``run()`` calls is observed immediately (the structural fix
    for the bug in ``tests/characterization/test_stale_default_config_bug.py``).

    ``lexical_index``, ``reranker`` and ``query_decomposer`` are ``Optional``:
    when a caller does not wire one in, the corresponding pipeline flag is
    treated as if it were off for that stage, regardless of what
    ``config.flags`` says -- there is nothing to invoke otherwise. This
    mirrors the original's `if cfg.USAR_X and <thing available>` guards,
    just made explicit instead of relying on a lazily-loaded singleton that
    might return ``None`` (``obtener_modelo_reranker``).
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        config: AppConfig,
        lexical_index: Optional[LexicalIndex] = None,
        reranker: Optional[Reranker] = None,
        query_decomposer: Optional[ChatModel] = None,
    ):
        """Args:
            embedder: Embeds each query variant.
            vector_store: Semantic search backend.
            config: Root config; ``flags``, ``retrieval``, ``reranking`` and
                ``models.embed_prefix_query`` are read fresh on every ``run()``.
            lexical_index: BM25-style lexical search, or ``None`` to disable
                hybrid search regardless of ``flags.usar_busqueda_hibrida``.
            reranker: Cross-Encoder reranking, or ``None`` to disable
                reranking regardless of ``flags.usar_reranker``.
            query_decomposer: ``ChatModel`` wired to the "chat" role, or
                ``None`` to disable query decomposition regardless of
                ``flags.usar_llm_query_decomposition``.
        """
        self._embedder = embedder
        self._vector_store = vector_store
        self._lexical_index = lexical_index
        self._reranker = reranker
        self._query_decomposer = query_decomposer
        self._config = config

    def run(self, question: str) -> RetrieveResult:
        """Run the full retrieval pipeline for ``question``.

        Args:
            question: User query.

        Returns:
            ``RetrieveResult`` with the final ranked fragments and metrics.
        """
        config = self._config
        flags = config.flags
        retrieval = config.retrieval

        llm_queries: List[str] = []
        if (
            flags.usar_llm_query_decomposition
            and len(question) > 60
            and self._query_decomposer is not None
        ):
            llm_queries = _generate_subqueries(self._query_decomposer, question)

        # Net result of the original's "if llm_queries: append llm_queries[0]
        # if coherent" branch followed unconditionally by "for lq in
        # llm_queries: append if not already present" is IDENTICAL to just
        # appending every llm_queries entry in order once each -- the
        # coherence check on llm_queries[0] never changes which queries end
        # up in the final list (an incoherent llm_queries[0] gets added
        # anyway by the unconditional loop). Provably equivalent, not a
        # simplification.
        queries = [question]
        for lq in llm_queries:
            if lq not in queries:
                queries.append(lq)

        semantic_hits_per_query: List[List[Fragment]] = []
        for q_idx, q in enumerate(queries):
            embedding = self._embedder.embed(
                f"{config.models.embed_prefix_query}{q}",
                keep_alive=_embedding_keep_alive(q_idx, len(queries)),
            )
            semantic_hits_per_query.append(
                self._vector_store.query(embedding, retrieval.n_semantic_results)
            )

        keyword_hits: List[Fragment] = []
        if flags.usar_busqueda_hibrida and self._lexical_index is not None:
            keyword_hits = self._lexical_index.search(question, retrieval.n_keyword_results)

        fused = fuse_semantic_and_keyword(
            semantic_hits_per_query,
            keyword_hits,
            retrieval.rrf_k,
            retrieval.weight_semantic_rrf,
            retrieval.weight_bm25_rrf,
        )

        reranked = False
        if flags.usar_reranker and self._reranker is not None and fused:
            candidatos_rerank = fused[: retrieval.top_k_rerank_candidates]
            fused = self._reranker.rerank(question, candidatos_rerank, top_k=retrieval.top_k_final)
            reranked = True

        filtered = _filter_by_reranker_threshold(fused, flags.usar_reranker, config.reranking.score_threshold)
        # Matches preparar_fragmentos_para_generacion's `fragmentos_base =
        # list(candidatos[:cfg.TOP_K_FINAL])`: applied after the threshold
        # filter regardless of whether reranking already truncated to
        # top_k_final (a no-op re-slice in that case).
        final_fragments = filtered[: retrieval.top_k_final]

        metrics = {
            "query_variants": queries,
            "sub_queries": llm_queries,
            "semantic_candidates": sum(len(hits) for hits in semantic_hits_per_query),
            "keyword_candidates": len(keyword_hits),
            "fused_candidates": len(fused),
            "reranked": reranked,
            "candidates_above_threshold": len(filtered),
            "final_count": len(final_fragments),
        }
        return RetrieveResult(fragments=final_fragments, metrics=metrics)
