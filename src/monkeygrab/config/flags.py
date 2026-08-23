"""PipelineFlagsConfig -- togglable pipeline stages.

Defaults copy ``rag/chat_pdfs.py`` section 3.4 literally: every flag there
is a hardcoded ``True``, not read from the environment at all -- so
``AppConfig.from_env`` does not read any env var for this section either,
it just sets the same literals. (``rag/chat_pdfs.py``'s ``set_pipeline_flags``
only lets a subset -- the ones marked "runtime" below -- change after
startup; index-time flags require a fresh vector store to compare fairly.
``AppConfig.with_overrides`` does not enforce that split: unlike the mutable
module-global it replaces, overriding an index-time flag here cannot corrupt
already-running state -- it just produces a new, distinct ``AppConfig``.)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineFlagsConfig:
    """Attributes:
        usar_contextual_retrieval: Index-time. Generate a situational summary
            per chunk before storing it (``USAR_CONTEXTUAL_RETRIEVAL``).
        usar_llm_query_decomposition: Runtime. Generate LLM sub-queries for
            questions over 60 chars (``USAR_LLM_QUERY_DECOMPOSITION``).
        usar_busqueda_hibrida: Runtime. Add BM25 lexical search, fused via
            RRF with semantic search (``USAR_BUSQUEDA_HIBRIDA``).
        usar_reranker: Runtime. Cross-Encoder rerank of fused candidates
            (``USAR_RERANKER``).
        expandir_contexto: Runtime. Append neighbor chunks around the top
            fragments (``EXPANDIR_CONTEXTO``).
        usar_optimizacion_contexto: Runtime. Strip PDF extraction noise from
            fragment text before generation (``USAR_OPTIMIZACION_CONTEXTO``).
        usar_recomp_synthesis: Runtime. Synthesize a facts briefing instead of
            feeding raw chunks to the generator (``USAR_RECOMP_SYNTHESIS``).
        usar_embeddings_imagen: Index-time. Extract and directly embed PDF
            images as extra chunks (``USAR_EMBEDDINGS_IMAGEN``).
        usar_descripcion_imagen: Index-time. Describe each extracted figure
            with the vision-capable chat model and store that description as
            the image chunk's text (``USAR_DESCRIPCION_IMAGEN``). Requires
            ``usar_embeddings_imagen`` -- there is nothing to describe
            otherwise.
        logging_metricas: Log per-stage pipeline metrics (``LOGGING_METRICAS``).
        guardar_debug_rag: Write a debug dump per RAG interaction
            (``GUARDAR_DEBUG_RAG``).
    """

    usar_contextual_retrieval: bool = True
    usar_llm_query_decomposition: bool = True
    usar_busqueda_hibrida: bool = True
    usar_reranker: bool = True
    expandir_contexto: bool = True
    usar_optimizacion_contexto: bool = True
    usar_recomp_synthesis: bool = True
    usar_embeddings_imagen: bool = True
    usar_descripcion_imagen: bool = False
    logging_metricas: bool = True
    guardar_debug_rag: bool = True
