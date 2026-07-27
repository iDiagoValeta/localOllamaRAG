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

from typing import Any, Dict, Optional

import chromadb

from monkeygrab.adapters.chat.ollama_chat import OllamaChatModel
from monkeygrab.adapters.embedding.ollama_embedder import OllamaEmbedder
from monkeygrab.adapters.lexical.bm25_index import Bm25LexicalIndex
from monkeygrab.adapters.reranking.cross_encoder_reranker import CrossEncoderReranker
from monkeygrab.adapters.vectorstore.chroma_store import ChromaVectorStore
from monkeygrab.config.app_config import AppConfig
from rag.engine.runtime import get_runtime

cfg = get_runtime()

# Runtime state that AppConfig mirrors. Kept as an explicit list rather than
# introspection so adding a global without deciding whether the core needs it
# is a visible omission rather than a silent one.
_RUNTIME_OVERRIDES = {
    "paths.docs_folder": "CARPETA_DOCS",
    "models.rag": "MODELO_RAG",
    "models.chat": "MODELO_CHAT",
    "models.embedding": "MODELO_EMBEDDING",
    "models.contextual": "MODELO_CONTEXTUAL",
    "models.recomp": "MODELO_RECOMP",
    "models.ocr": "MODELO_OCR",
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

    Stack selection (extractor, vector store, embedder) comes from the process
    environment via ``AppConfig.from_env``. Paths, model roles and pipeline
    flags are then overridden from the live ``rag.chat_pdfs`` globals, so
    ``set_docs_folder_runtime``, ``set_model_roles_runtime`` and
    ``set_pipeline_flags`` keep working exactly as before.

    Returns:
        A fresh, immutable config for this call.
    """
    return AppConfig.from_env().with_overrides(
        **{key: getattr(cfg, name) for key, name in _RUNTIME_OVERRIDES.items()}
    )


def vector_store(collection: chromadb.Collection) -> ChromaVectorStore:
    """Adapt a caller-owned Chroma collection to the ``VectorStore`` port.

    The CLI, the web app and the evaluation runner all open the collection
    themselves and pass it in; wrapping that same object keeps their own
    ``count``/``get`` views consistent with what the pipeline reads and writes.

    Args:
        collection: An open ChromaDB collection.

    Returns:
        The port-conforming wrapper.
    """
    return ChromaVectorStore.wrap_collection(collection)


def embedder(config: AppConfig) -> OllamaEmbedder:
    """Build the query/document embedder for the configured embedding model."""
    return OllamaEmbedder(config.models)


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
_reranker_cache: Dict[str, Any] = {"key": None, "reranker": None}


def lexical_index(
    collection: chromadb.Collection, config: AppConfig
) -> Bm25LexicalIndex:
    """Return the BM25 index for this collection, building it only when needed.

    The index itself re-scans the corpus when the chunk count changes; this
    cache exists one level up, so that swapping corpora or retuning the BM25
    parameters gets a fresh index while ordinary queries reuse the tokenized
    one.

    Args:
        collection: Collection whose chunks form the BM25 corpus.
        config: Current config; the BM25 parameters are read from it.

    Returns:
        A cached or freshly built lexical index.
    """
    # Real collections have a stable name; fall back to object identity so two
    # unnamed collections (test doubles, typically) never share a cache entry.
    collection_key = getattr(collection, "name", None) or id(collection)
    key = (collection_key, config.retrieval.bm25_k1, config.retrieval.bm25_b)

    if _lexical_cache["key"] != key:
        _lexical_cache.update(
            key=key, index=Bm25LexicalIndex(vector_store(collection), config.retrieval)
        )
    return _lexical_cache["index"]


def reranker(config: AppConfig) -> CrossEncoderReranker:
    """Return the Cross-Encoder reranker, loading its weights at most once.

    Rebuilding this per query would reload hundreds of megabytes of model
    weights each time, so the instance is cached until the configured quality
    tier changes. The adapter loads lazily, so holding one costs nothing until
    something is actually reranked.

    Args:
        config: Current config; the reranker quality tier is read from it.

    Returns:
        A cached or freshly built reranker.
    """
    key = config.models.reranker_quality
    if _reranker_cache["key"] != key:
        _reranker_cache.update(key=key, reranker=CrossEncoderReranker(config.models))
    return _reranker_cache["reranker"]


def reset_component_cache() -> None:
    """Drop the cached lexical index and reranker.

    Only tests need this: within a process the caches invalidate themselves on
    the state that matters. Releasing the reranker also returns its GPU memory.
    """
    cached: Optional[CrossEncoderReranker] = _reranker_cache["reranker"]
    if cached is not None:
        cached.release()
    _lexical_cache.update(key=None, index=None)
    _reranker_cache.update(key=None, reranker=None)


def metadata_to_dict(metadata) -> Dict[str, Any]:
    """Serialize ``ChunkMetadata`` back into the dict shape the interfaces read.

    The CLI, the web app and the debug dump all index into fragment metadata
    as a plain dict, which is also how ChromaDB stores it. Optional fields
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
    return {
        "doc": fragment.doc,
        "metadata": metadata_to_dict(fragment.metadata),
        "distancia": fragment.distancia,
        "id": fragment.id,
        "score_semantic": fragment.score_semantic,
        "score_keyword": fragment.score_keyword,
        "score_reranker": fragment.score_reranker,
        "matches": list(fragment.matches),
        "query_matches": list(fragment.query_matches),
        "score_final": fragment.score_final,
    }


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
            "modelo_embedding": config.models.embedding,
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
            "modelo_usado": config.models.reranker_quality,
        } if metrics["reranked"] else {},
        "candidatos_fusion": metrics["fused_candidates"],
        "resultados_finales": metrics["final_count"],
        "sub_queries": metrics["sub_queries"],
        "queries_semanticas": metrics["query_variants"],
        "keywords": metrics["keywords"],
    }


__all__ = [
    "app_config_from_runtime",
    "embedder",
    "fragment_to_dict",
    "lexical_index",
    "metadata_to_dict",
    "query_decomposer",
    "reranker",
    "reset_component_cache",
    "retrieval_metrics_to_legacy",
    "vector_store",
]
