"""Runtime synchronization helpers for split RAG engine modules."""

import sys
from types import ModuleType
from typing import MutableMapping, Any

_RUNTIME_MODULE = "rag.chat_pdfs"

# Names intentionally owned by rag.chat_pdfs. Auxiliary modules refresh them
# before every call so runtime toggles, evaluation overrides and monkeypatches
# are observed after the split.
_RUNTIME_NAMES = {
    "Any", "Dict", "List", "Optional", "Tuple",
    "base64", "chromadb", "fitz", "io", "json", "logging", "ollama", "os",
    "pymupdf4llm", "re", "requests", "sys", "ui", "PdfReader", "CrossEncoder",
    "PYMUPDF_AVAILABLE", "FITZ_DISPONIBLE", "RERANKER_AVAILABLE",
    "MODELO_RAG", "MODELO_CHAT", "MODELO_EMBEDDING", "MODELO_CONTEXTUAL",
    "MODELO_RECOMP", "MODELO_OCR", "MODELO_DESC",
    "USAR_CONTEXTUAL_RETRIEVAL", "USAR_LLM_QUERY_DECOMPOSITION",
    "USAR_BUSQUEDA_HIBRIDA", "USAR_BUSQUEDA_EXHAUSTIVA", "USAR_RERANKER",
    "EXPANDIR_CONTEXTO", "USAR_OPTIMIZACION_CONTEXTO", "USAR_RECOMP_SYNTHESIS",
    "USAR_EMBEDDINGS_IMAGEN", "EVAL_RAGBENCH_RERANKER_LOW_SCORE_FALLBACK",
    "LOGGING_METRICAS", "GUARDAR_DEBUG_RAG", "PIPELINE_RUNTIME_FLAGS",
    "BASE_DIR", "CARPETA_DOCS", "PATH_DB", "COLLECTION_NAME", "HISTORIAL_PATH",
    "SYSTEM_PROMPT_RAG", "_modelo_necesita_system_prompt",
    "MAX_HISTORIAL_MENSAJES", "CARPETA_DEBUG_RAG", "EMBED_PREFIX_QUERY",
    "EMBED_PREFIX_DOC", "MAX_CHARS_EMBED", "CHUNK_SIZE", "CHUNK_OVERLAP",
    "MIN_CHUNK_LENGTH", "MAX_IMAGENES_POR_PAGINA", "MIN_IMAGEN_SIZE_PX",
    "CAPTION_MARGIN_PX", "_IMAGEN_CHUNK_OFFSET", "N_RESULTADOS_SEMANTICOS",
    "N_RESULTADOS_KEYWORD", "TOP_K_RERANK_CANDIDATES", "TOP_K_AFTER_RERANK",
    "TOP_K_FINAL", "N_TOP_PARA_EXPANSION", "RERANKER_MODEL_QUALITY",
    "UMBRAL_RELEVANCIA", "UMBRAL_SCORE_RERANKER", "RRF_K",
    "MIN_LONGITUD_PREGUNTA_RAG", "MAX_CONTEXTO_CHARS", "SYSTEM_PROMPT_CHAT",
    "STOPWORDS", "TERMINOS_EXPANSION", "GENERIC_TERMS_BLACKLIST",
    "OLLAMA_BASE_URL", "_RECOMP_FACTS_HEADER",
    "_inferir_descripcion_modelo", "cargar_historial", "guardar_historial",
    "limpiar_historial", "extraer_header_markdown", "dividir_en_chunks",
    "expandir_con_chunks_adyacentes", "extraer_keywords", "busqueda_por_keywords",
    "busqueda_exhaustiva_texto", "_detectar_dispositivo_reranker",
    "obtener_modelo_reranker", "rerank_resultados", "generar_queries_con_llm",
    "_validar_coherencia_query", "_filtrar_terminos_criticos",
    "realizar_busqueda_hibrida", "_es_continuacion_parrafo", "_reunir_parrafos",
    "optimizar_texto_contexto", "_marcar_fragmento_incompleto",
    "_texto_fuente_fragmento", "_strip_ollama_think_blocks",
    "_normalizar_salida_recomp", "construir_contexto_para_modelo",
    "sintetizar_contexto_recomp", "guardar_debug_rag", "_ollama_generate_stream",
    "_preparar_mensaje_usuario_rag", "_generar_respuesta_stream", "generar_respuesta",
    "generar_respuesta_silenciosa", "evaluar_pregunta_rag", "_detectar_idioma",
    "generar_contexto_situacional", "_es_descripcion_spam", "_es_prompt_echo",
    "_es_solo_caption", "extraer_imagenes_pdf", "describir_imagen_con_llm",
    "indexar_documentos", "obtener_documentos_indexados",
}


def get_runtime() -> ModuleType:
    module = sys.modules.get(_RUNTIME_MODULE)
    if module is None:
        # Direct-run scenario: `python chat_pdfs.py` registers as __main__, not rag.chat_pdfs.
        # Identify it by a distinctive constant that is guaranteed to exist before the engine
        # imports fire (all constants are defined before line 441 in chat_pdfs.py).
        main = sys.modules.get("__main__")
        if main is not None and hasattr(main, "MODELO_RAG"):
            return main
        import rag.chat_pdfs as module  # type: ignore[no-redef]
    return module


def sync_runtime_globals(namespace: MutableMapping[str, Any]) -> None:
    runtime = get_runtime()
    for name in _RUNTIME_NAMES:
        if hasattr(runtime, name):
            namespace[name] = getattr(runtime, name)
