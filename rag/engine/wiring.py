"""Bridge from the mutable runtime configuration to the hexagonal core.

``rag.chat_pdfs`` owns configuration as module-level globals that the web
control panel and the CLI mutate in place. The core under ``src/monkeygrab``
takes an immutable ``AppConfig`` instead. This module is the single place
where the former is turned into the latter, and the single place that builds
the port adapters the use cases need.

Building an ``AppConfig`` is cheap and happens per call, so a model or flag
changed at runtime takes effect on the next query with no restart. What is
not cheap is cached here instead: the Cross-Encoder weights and the tokenized
BM25 corpus would otherwise be rebuilt on every question.
"""

import threading
from typing import Any, Dict

from monkeygrab.adapters.chat.ollama_chat import OllamaChatModel
from monkeygrab.adapters.chat.ollama_model_unloader import OllamaModelUnloader
from monkeygrab.adapters.lexical.bm25_index import Bm25LexicalIndex
from monkeygrab.adapters.reranking.cross_encoder_reranker import CrossEncoderReranker
from monkeygrab.application.answer import Answer
from monkeygrab.composition import build_embedder, build_vector_store
from monkeygrab.config.app_config import AppConfig
from monkeygrab.domain.chunk_metadata import ChunkMetadata
from monkeygrab.domain.fragment import Fragment
from monkeygrab.ports.vector_store import VectorStore
from rag.engine.runtime import get_runtime

cfg = get_runtime()

# Runtime state that AppConfig mirrors. Kept as an explicit list rather than
# introspection so adding a global without deciding whether the core needs it
# is a visible omission rather than a silent one.
_RUNTIME_OVERRIDES = {
    "paths.docs_folder": "CARPETA_DOCS",
    "models.rag": "MODELO_RAG",
    "models.chat": "MODELO_CHAT",
    "models.contextual": "MODELO_CONTEXTUAL",
    "models.recomp": "MODELO_RECOMP",
    "flags.usar_contextual_retrieval": "USAR_CONTEXTUAL_RETRIEVAL",
    "flags.usar_embeddings_imagen": "USAR_EMBEDDINGS_IMAGEN",
    "flags.usar_llm_query_decomposition": "USAR_LLM_QUERY_DECOMPOSITION",
    "flags.usar_busqueda_hibrida": "USAR_BUSQUEDA_HIBRIDA",
    "flags.usar_reranker": "USAR_RERANKER",
    "flags.expandir_contexto": "EXPANDIR_CONTEXTO",
    "flags.usar_optimizacion_contexto": "USAR_OPTIMIZACION_CONTEXTO",
    "flags.usar_recomp_synthesis": "USAR_RECOMP_SYNTHESIS",
    "flags.logging_metricas": "LOGGING_METRICAS",
    "flags.guardar_debug_rag": "GUARDAR_DEBUG_RAG",
}


def app_config_from_runtime() -> AppConfig:
    """Build an ``AppConfig`` reflecting the current runtime state.

    The fixed MinerU, Jina CLIP and FAISS composition sits outside this
    configuration object. Paths, model roles and pipeline flags are overridden
    from the live ``rag.chat_pdfs`` globals, so
    ``set_docs_folder_runtime``, ``set_model_roles_runtime`` and
    ``set_pipeline_flags`` keep working exactly as before.

    Returns:
        A fresh, immutable config for this call.
    """
    return AppConfig.from_env().with_overrides(
        **{key: getattr(cfg, name) for key, name in _RUNTIME_OVERRIDES.items()}
    )


_store_cache: Dict[str, Any] = {"key": None, "store": None}
_store_cache_lock = threading.Lock()
_embedder_cache: Dict[str, Any] = {"embedder": None}
_embedder_cache_lock = threading.Lock()


def vector_store(config: AppConfig) -> VectorStore:
    """Return the FAISS store for the active corpus.

    Double-checked locking: two concurrent callers racing a corpus switch
    must not both open the same FAISS index, so the key check is repeated
    once the lock is held. The cheap, unlocked check outside the lock keeps
    the common case (cache already valid) lock-free.
    """
    key = (config.paths.path_db, config.paths.collection_name)
    if _store_cache["key"] != key:
        with _store_cache_lock:
            if _store_cache["key"] != key:
                _store_cache.update(key=key, store=build_vector_store(config))
    return _store_cache["store"]


def reset_vector_store_cache() -> None:
    with _store_cache_lock:
        _store_cache.update(key=None, store=None)


def embedder(config: AppConfig):
    """Return the reusable jina-clip-v2 worker.

    Double-checked locking: two concurrent first callers must not both
    construct one, since each duplicate later starts its own worker
    subprocess and loads jina-clip on first use (~29s), and the losing
    instance is unreachable yet pinned alive by its own stdout/stderr pump
    threads once its worker has started, so nothing ever closes it (#46).
    """
    if _embedder_cache["embedder"] is None:
        with _embedder_cache_lock:
            if _embedder_cache["embedder"] is None:
                _embedder_cache["embedder"] = build_embedder(config)
    return _embedder_cache["embedder"]


def rag_chat_model(config: AppConfig) -> OllamaChatModel:
    """Build the model that writes the final answer.

    Sampling is cold and repetition-penalised: the answer must stay inside the
    retrieved evidence, and a wandering generator is worse than a terse one.
    ``num_predict`` is unbounded because a truncated answer to a document
    question is indistinguishable from a wrong one.

    The unloader is wired in here because this model runs last and needs the
    VRAM the embedder and reranker were holding.
    """
    ollama = config.models.ollama
    return OllamaChatModel(
        config.models.rag,
        num_ctx=ollama.rag_num_ctx,
        keep_alive=ollama.keep_alive,
        request_timeout=ollama.request_timeout,
        generate_retries=ollama.generate_retries,
        generate_retry_delay=ollama.generate_retry_delay,
        options={
            "temperature": 0.15,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
            "repeat_last_n": 64,
            "num_predict": -1,
        },
        model_unloader=OllamaModelUnloader(config.models),
    )


def recomp_chat_model(config: AppConfig) -> OllamaChatModel:
    """Build the model that compresses retrieved evidence before generation."""
    ollama = config.models.ollama
    return OllamaChatModel(
        config.models.recomp,
        num_ctx=ollama.recomp_num_ctx,
        keep_alive=ollama.keep_alive,
        request_timeout=ollama.request_timeout,
        generate_retries=ollama.generate_retries,
        generate_retry_delay=ollama.generate_retry_delay,
        options={
            "temperature": 0.1,
            "num_predict": 1500,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
        },
    )


def query_decomposer(config: AppConfig) -> OllamaChatModel:
    """Build the auxiliary chat model that generates search sub-queries.

    Sampling is deliberately warmer than the answer generator's: sub-queries
    should explore different angles on the question, and identical phrasings
    would add nothing to retrieve.
    """
    ollama = config.models.ollama
    return OllamaChatModel(
        config.models.chat,
        num_ctx=ollama.query_num_ctx,
        keep_alive=ollama.keep_alive,
        request_timeout=ollama.request_timeout,
        generate_retries=ollama.generate_retries,
        generate_retry_delay=ollama.generate_retry_delay,
        options={"temperature": 0.5, "num_predict": 400, "stop": ["\n\n\n"]},
    )


# Expensive, reusable components keyed by what would invalidate them. Held at
# module level because the interfaces call the pipeline as free functions and
# have nowhere else to keep them for the life of the process.
_lexical_cache: Dict[str, Any] = {"key": None, "index": None}
_lexical_cache_lock = threading.Lock()
_reranker_cache: Dict[str, Any] = {"reranker": None}
_reranker_cache_lock = threading.Lock()


def lexical_index(store: VectorStore, config: AppConfig) -> Bm25LexicalIndex:
    """Return the BM25 index for this collection, building it only when needed.

    The index itself re-scans the corpus when the chunk count changes; this
    cache exists one level up, so that swapping corpora or retuning the BM25
    parameters gets a fresh index while ordinary queries reuse the tokenized
    one.

    Double-checked locking: the key is re-read once the lock is held so two
    concurrent callers racing the same new key build only one index (#46).

    Args:
        collection: Collection whose chunks form the BM25 corpus.
        config: Current config; the BM25 parameters are read from it.

    Returns:
        A cached or freshly built lexical index.
    """
    # Real collections have a stable name; fall back to object identity so two
    # unnamed collections (test doubles, typically) never share a cache entry.
    key = (id(store), config.retrieval.bm25_k1, config.retrieval.bm25_b)

    if _lexical_cache["key"] != key:
        with _lexical_cache_lock:
            if _lexical_cache["key"] != key:
                _lexical_cache.update(
                    key=key, index=Bm25LexicalIndex(store, config.retrieval)
                )
    return _lexical_cache["index"]


def release_gpu_models() -> None:
    """Free the cached embedder worker and reranker weights, if either is loaded.

    Called from the generation entry points (rag/engine/generation.py) right
    before Ollama is asked to load the generator, never from the retrieval
    path itself: on an 8 GB card, an idle jina-clip worker or a resident
    Cross-Encoder is enough to keep Ollama from loading at all, and Ollama
    does not fail fast when memory is short -- it blocks until the request
    times out (#25). This mirrors tests/eval/run_eval.py's
    _release_gpu_models, which does the same thing for the embedder/reranker
    instances its own Retrieve use case holds; this function releases this
    module's caches instead, since the CLI and the web app get their
    embedder/reranker through embedder()/reranker() above rather than
    holding their own references.

    Safe to call with nothing cached (both hooks are then no-ops) and safe
    to call more than once (a second call finds the slot already cleared).
    Each cache slot is swapped for ``None`` while holding its own lock --
    the same lock embedder()/reranker() take to build -- so a build racing
    this call either wins the swap and gets the live instance back (release
    then closes the now-orphaned one), or loses it and rebuilds against an
    empty slot; either way nothing ever calls close()/release() on an
    instance another thread is mid-build on, or hands out a reference to one
    that is mid-close.
    """
    with _embedder_cache_lock:
        emb, _embedder_cache["embedder"] = _embedder_cache["embedder"], None
    if emb is not None:
        emb.close()

    with _reranker_cache_lock:
        rer, _reranker_cache["reranker"] = _reranker_cache["reranker"], None
    if rer is not None:
        rer.release()


def reranker(config: AppConfig) -> CrossEncoderReranker:
    """Return the Cross-Encoder reranker, loading its weights at most once.

    Rebuilding this per query would reload hundreds of megabytes of model
    weights each time. The fixed BGE adapter loads lazily, so holding one costs
    nothing until something is actually reranked.

    Double-checked locking: two concurrent first callers must not both
    construct one, since each duplicate would separately load hundreds of
    megabytes of weights on first ``rerank()`` call, competing for the same
    GPU (#46).

    Args:
        config: Current pipeline configuration.

    Returns:
        A cached or freshly built reranker.
    """
    del config
    if _reranker_cache["reranker"] is None:
        with _reranker_cache_lock:
            if _reranker_cache["reranker"] is None:
                _reranker_cache["reranker"] = CrossEncoderReranker()
    return _reranker_cache["reranker"]


def answer(store: VectorStore, config: AppConfig) -> Answer:
    """Build the generation use case for this collection.

    RECOMP synthesis is wired only when its flag is on: an absent model means
    the stage is skipped, which is how the use case expresses "not configured"
    without consulting a flag it was not given.

    Args:
        collection: Collection used for neighbour-chunk expansion.
        config: Current config.

    Returns:
        A ready-to-run ``Answer``.
    """
    return Answer(
        store,
        rag_chat_model(config),
        config,
        recomp_chat_model=recomp_chat_model(config) if config.flags.usar_recomp_synthesis else None,
        system_prompt=cfg.SYSTEM_PROMPT_RAG,
    )


def metadata_to_dict(metadata) -> Dict[str, Any]:
    """Serialize ``ChunkMetadata`` back into the dict shape the interfaces read.

    The CLI, the web app and the debug dump all index into fragment metadata
    as a plain dict for the interfaces. Optional fields
    that are unset are omitted rather than written as ``None``, matching what
    is on disk.

    Args:
        metadata: The ``ChunkMetadata`` to serialize.

    Returns:
        Metadata as a plain dict.
    """
    data: Dict[str, Any] = {
        "source": metadata.source,
        "page": metadata.page,
        "chunk": metadata.chunk,
        "section_header": metadata.section_header,
    }
    if metadata.total_chunks_in_page is not None:
        data["total_chunks_in_page"] = metadata.total_chunks_in_page
    if metadata.format is not None:
        data["format"] = metadata.format
    if metadata.image_width is not None:
        data["image_width"] = metadata.image_width
    if metadata.image_height is not None:
        data["image_height"] = metadata.image_height
    return data


def fragment_to_dict(fragment) -> Dict[str, Any]:
    """Convert a ``Fragment`` into the dict the CLI, web and debug dump expect.

    Args:
        fragment: The domain fragment to convert.

    Returns:
        The fragment as a plain dict, including its derived ``id``.
    """
    data = {
        "doc": fragment.doc,
        "metadata": metadata_to_dict(fragment.metadata),
        "distancia": fragment.distancia,
        "id": fragment.id,
        "score_semantic": fragment.score_semantic,
        "score_keyword": fragment.score_keyword,
        "matches": list(fragment.matches),
        "query_matches": list(fragment.query_matches),
        "score_final": fragment.score_final,
    }
    # Absent rather than None when the fragment was never reranked. Readers do
    # `frag.get("score_reranker", frag["score_final"])`, and a present None
    # defeats that fallback -- the web app's source panel then compares None
    # against None and raises. Only reachable with reranking off, which the
    # control panel can do at runtime.
    if fragment.score_reranker is not None:
        data["score_reranker"] = fragment.score_reranker
    return data


def fragment_from_dict(data: Dict[str, Any]) -> Fragment:
    """Rebuild a ``Fragment`` from the dict shape the interfaces pass around.

    The inverse of ``fragment_to_dict``. Needed because the facade's public API
    is dict-based while the core speaks domain entities: fragments cross that
    boundary out of retrieval and back in at generation.

    Args:
        data: A fragment dict, as produced by retrieval or neighbour expansion.

    Returns:
        The equivalent domain fragment.
    """
    meta = data.get("metadata") or {}
    return Fragment(
        doc=data.get("doc", ""),
        metadata=ChunkMetadata(
            source=meta.get("source", ""),
            page=meta.get("page", 0),
            chunk=meta.get("chunk", 0),
            total_chunks_in_page=meta.get("total_chunks_in_page"),
            format=meta.get("format"),
            section_header=meta.get("section_header", ""),
            image_width=meta.get("image_width"),
            image_height=meta.get("image_height"),
        ),
        distancia=data.get("distancia", float("inf")),
        score_semantic=data.get("score_semantic", 0.0),
        score_keyword=data.get("score_keyword", 0.0),
        score_final=data.get("score_final", 0.0),
        score_reranker=data.get("score_reranker"),
        matches=tuple(data.get("matches") or ()),
        query_matches=tuple(data.get("query_matches") or ()),
    )


def retrieval_metrics_to_legacy(metrics: Dict[str, Any], config: AppConfig) -> Dict[str, Any]:
    """Reshape ``RetrieveResult.metrics`` into the keys the debug dump reads.

    The debug dump predates the use case and speaks a Spanish, phase-oriented
    vocabulary. Rather than rewriting every historical dump reader, the
    translation is done here, in the layer that already exists to adapt
    between the two vocabularies.

    Args:
        metrics: Metrics as produced by ``Retrieve.run``.
        config: Config used for the run, for the parameters worth recording.

    Returns:
        Metrics under the legacy phase-oriented keys.
    """
    distances = metrics["semantic_distances"]
    branches = metrics["fusion_branches"]
    retrieval = config.retrieval

    return {
        "fase_semantica": {
            "queries_generadas": len(metrics["query_variants"]),
            "fragmentos_unicos": distances["unique_fragments"],
            "modelo_embedding": "jinaai/jina-clip-v2",
            "n_resultados_por_query": retrieval.n_semantic_results,
            "resultados_por_query": metrics["semantic_hits_per_query"],
            "distancia_l2_min": distances["l2_min"],
            "distancia_l2_max": distances["l2_max"],
            "distancia_l2_media": distances["l2_mean"],
        },
        "fase_keywords": {
            "resultados_totales": metrics["keyword_candidates"],
        },
        "fase_fusion": {
            "peso_semantico": retrieval.weight_semantic_rrf,
            "peso_lexico": retrieval.weight_bm25_rrf,
            "rrf_k": retrieval.rrf_k,
            "candidatos_totales": metrics["fused_candidates"],
            "solo_semantica": branches["semantic_only"],
            "solo_lexica": branches["lexical_only"],
            "ambas_ramas": branches["both_branches"],
        },
        "fase_reranking": {
            "candidatos_entrada": metrics["fused_candidates"],
            "resultados_salida": metrics["final_count"],
            "tiempo_reranking": metrics["rerank_seconds"],
            "modelo_usado": "BAAI/bge-reranker-v2-m3",
        } if metrics["reranked"] else {},
        "candidatos_fusion": metrics["fused_candidates"],
        "resultados_finales": metrics["final_count"],
        "sub_queries": metrics["sub_queries"],
        "queries_semanticas": metrics["query_variants"],
        "keywords": metrics["keywords"],
    }


__all__ = [
    "answer",
    "app_config_from_runtime",
    "embedder",
    "fragment_from_dict",
    "fragment_to_dict",
    "lexical_index",
    "metadata_to_dict",
    "query_decomposer",
    "rag_chat_model",
    "recomp_chat_model",
    "release_gpu_models",
    "reranker",
    "reset_vector_store_cache",
    "retrieval_metrics_to_legacy",
    "vector_store",
]
