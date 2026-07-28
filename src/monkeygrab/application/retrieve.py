"""The retrieval use case: from a user question to ranked evidence.

Runs optional query decomposition, semantic search over every query variant,
optional BM25 lexical search, Reciprocal-Rank-Fusion of the two branches,
optional Cross-Encoder reranking, and the relevance-threshold filter that
decides what is worth generating from.

Everything touching infrastructure arrives as an injected port (``Embedder``,
``VectorStore``, ``LexicalIndex``, ``Reranker``, ``ChatModel``), so this
module has no notion of Ollama, FAISS or sentence-transformers. The fusion
arithmetic lives in ``monkeygrab.application.rrf_fusion`` and the text
analysis in ``monkeygrab.application.keywords``; both are pure and tested on
their own.

This is the single retrieval path: the CLI and web interfaces reach it
through ``rag.engine.retrieval``, and the evaluation gate constructs it
directly. Neither can drift from the other.
"""

import dataclasses
import time
from typing import Any, Dict, List, Optional

from monkeygrab.application.keywords import extract_keywords, is_coherent_query
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

    Models emit numbering and bullets despite being told not to, and pad
    with stray fragments; both are normalized away here.
    """
    queries = [
        q.strip().lstrip("0123456789.-) ")
        for q in raw_response.strip().split("\n")
        if q.strip() and len(q.strip()) > 20
    ]
    return queries[:3]


def _generate_subqueries(chat_model: ChatModel, pregunta: str) -> List[str]:
    """Generate up to 3 search sub-queries via an auxiliary LLM.

    Sampling parameters are deliberately absent: the ``ChatModel`` port has
    no per-call options, so how warm this generation runs is decided when
    the adapter for the "chat" role is constructed.

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
        # only".
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


def _distance_summary(semantic_hits_per_query: List[List[Fragment]]) -> Dict[str, Any]:
    """Summarize semantic distances over the deduplicated hit set.

    A chunk retrieved by several query variants is counted once, at its best
    (smallest) distance, so the summary describes distinct evidence rather
    than rewarding fan-out.

    Args:
        semantic_hits_per_query: Hits per query variant, in embedding order.

    Returns:
        Unique-fragment count plus min/max/mean L2 distance. Zeroed when
        nothing was retrieved.
    """
    best_per_id: Dict[str, float] = {}
    for hits in semantic_hits_per_query:
        for hit in hits:
            if hit.id not in best_per_id or hit.distancia < best_per_id[hit.id]:
                best_per_id[hit.id] = hit.distancia

    distances = list(best_per_id.values())
    return {
        "unique_fragments": len(best_per_id),
        "l2_min": min(distances) if distances else 0.0,
        "l2_max": max(distances) if distances else 0.0,
        "l2_mean": sum(distances) / len(distances) if distances else 0.0,
    }


def _branch_counts(fused: List[Fragment]) -> Dict[str, int]:
    """Count how many fused fragments each retrieval branch accounts for.

    Cross-branch agreement is the signal hybrid retrieval exists to produce,
    so these three counts are what tell you whether fusion did anything.

    Args:
        fused: Fragments after RRF fusion.

    Returns:
        Counts of semantic-only, lexical-only and both-branch fragments.
    """
    return {
        "semantic_only": sum(1 for f in fused if f.score_semantic > 0 and f.score_keyword == 0),
        "lexical_only": sum(1 for f in fused if f.score_keyword > 0 and f.score_semantic == 0),
        "both_branches": sum(1 for f in fused if f.score_semantic > 0 and f.score_keyword > 0),
    }


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
    made explicit in the constructor instead of hidden behind a lazily
    loaded singleton that might quietly turn out to be unavailable.
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
                the configured embedder is read fresh on every ``run()``.
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

        keywords = extract_keywords(question)
        queries = self._build_query_variants(question, llm_queries, keywords)

        semantic_hits_per_query: List[List[Fragment]] = []
        for q_idx, q in enumerate(queries):
            embedding = self._embedder.embed(
                q,
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
        fused_candidates = len(fused)
        branch_counts = _branch_counts(fused)

        reranked = False
        rerank_seconds = 0.0
        if flags.usar_reranker and self._reranker is not None and fused:
            started = time.perf_counter()
            candidatos_rerank = fused[: retrieval.top_k_rerank_candidates]
            fused = self._reranker.rerank(question, candidatos_rerank, top_k=retrieval.top_k_final)
            rerank_seconds = time.perf_counter() - started
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
            "keywords": keywords,
            "semantic_candidates": sum(len(hits) for hits in semantic_hits_per_query),
            "semantic_hits_per_query": [len(hits) for hits in semantic_hits_per_query],
            "semantic_distances": _distance_summary(semantic_hits_per_query),
            "keyword_candidates": len(keyword_hits),
            "fused_candidates": fused_candidates,
            "fusion_branches": branch_counts,
            "reranked": reranked,
            "rerank_seconds": rerank_seconds,
            "candidates_above_threshold": len(filtered),
            "final_count": len(final_fragments),
        }
        return RetrieveResult(fragments=final_fragments, metrics=metrics)

    @staticmethod
    def _build_query_variants(
        question: str, llm_queries: List[str], keywords: List[str]
    ) -> List[str]:
        """Assemble the semantic query variants to embed, original first.

        Sub-queries from the LLM are used when available. When there are none
        -- decomposition disabled, question too short, or the call failed --
        the leading extracted keywords are joined into one extra variant
        instead, so a short question still gets a second angle on the corpus.
        That fallback is dropped when the joined keywords do not read as a
        coherent query, since embedding a bag of words retrieves noise.

        Args:
            question: The original user question; always the first variant.
            llm_queries: Sub-queries from decomposition, possibly empty.
            keywords: Keywords extracted from the question, most specific first.

        Returns:
            Query variants in embedding order, without duplicates.
        """
        queries = [question]

        if not llm_queries and keywords:
            keyword_query = " ".join(keywords[:6]).strip()
            if keyword_query and is_coherent_query(keyword_query) and keyword_query not in queries:
                queries.append(keyword_query)

        # Appending every sub-query once, in order, is equivalent to the
        # original's "append the first if coherent, then append any missing
        # one": an incoherent first sub-query was added by the second loop
        # anyway, so the coherence check never changed the resulting list.
        for lq in llm_queries:
            if lq not in queries:
                queries.append(lq)

        return queries
