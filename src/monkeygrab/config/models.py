"""ModelsConfig -- model roles and the Ollama runtime they execute under.

Defaults below are a field-for-field copy of ``rag/chat_pdfs.py`` section
3.3 (model roles + ``OLLAMA_*`` runtime) -- see that file for the
authoritative values if this ever needs re-syncing.
"""

from dataclasses import dataclass
from typing import Tuple


# ─────────────────────────────────────────────
# OLLAMA RUNTIME


@dataclass(frozen=True)
class OllamaRuntimeConfig:
    """Per-call-site context windows and retry/keep-alive behavior.

    Mirrors ``rag/chat_pdfs.py`` section 3.3: one ``num_ctx`` per model
    role (they differ because roles have very different prompt sizes --
    the RAG generator's context includes the full retrieved evidence,
    the query-decomposition role sees just the question) plus the shared
    HTTP timeout, keep-alive and retry knobs used by
    ``rag.engine.generation._ollama_generate_stream``.
    """

    num_ctx: int = 8192  # OLLAMA_NUM_CTX
    rag_num_ctx: int = 16384  # OLLAMA_RAG_NUM_CTX
    aux_num_ctx: int = 8192  # OLLAMA_AUX_NUM_CTX
    query_num_ctx: int = 2048  # OLLAMA_QUERY_NUM_CTX
    recomp_num_ctx: int = 8192  # OLLAMA_RECOMP_NUM_CTX
    contextual_num_ctx: int = 32768  # OLLAMA_CONTEXTUAL_NUM_CTX
    ocr_num_ctx: int = 8192  # OLLAMA_OCR_NUM_CTX
    request_timeout: int = 900  # OLLAMA_REQUEST_TIMEOUT
    keep_alive: int = 0  # OLLAMA_KEEP_ALIVE -- 0 unloads weights after each call
    generate_retries: int = 2  # OLLAMA_GENERATE_RETRIES
    generate_retry_delay: int = 3  # OLLAMA_GENERATE_RETRY_DELAY


# MODEL ROLES


@dataclass(frozen=True)
class ModelsConfig:
    """Ollama model assigned to each pipeline role, plus derived fields.

    ``desc`` and the ``embed_prefix_*`` pair are not independent
    configuration -- they are always a pure function of ``rag`` and
    ``embedding`` respectively (see ``infer_model_description`` /
    ``derive_embedding_prefixes`` below). They are still stored as fields,
    not recomputed on every access, because ``AppConfig.with_overrides``
    needs to refresh them explicitly the same way
    ``set_model_roles_runtime`` does in ``rag/chat_pdfs.py`` today when
    ``rag``/``embedding`` change.

    Attributes:
        rag: RAG generator model (``OLLAMA_RAG_MODEL``).
        chat: Chat + query-decomposition model (``OLLAMA_CHAT_MODEL``).
        embedding: Embedding model (``OLLAMA_EMBED_MODEL``); namespaces
            the vector store path (see ``config.paths``).
        contextual: Contextual-retrieval summary model (``OLLAMA_CONTEXTUAL_MODEL``).
        recomp: RECOMP context-synthesis model (``OLLAMA_RECOMP_MODEL``).
        ocr: Vision/OCR model for image indexing (``OLLAMA_OCR_MODEL``).
        desc: Short description derived from ``rag`` (tag stripped),
            e.g. ``"gemma4:e4b"`` -> ``"gemma4"``.
        embed_prefix_query: Task prefix prepended to queries before
            embedding (``search_query: `` for Nomic-family models, ``""``
            otherwise).
        embed_prefix_doc: Task prefix prepended to documents before
            embedding (``search_document: `` for Nomic-family models,
            ``""`` otherwise).
        reranker_quality: Cross-Encoder variant (``RERANKER_QUALITY``):
            ``"quality"`` (BAAI/bge-reranker-v2-m3) or ``"fast"``
            (cross-encoder/ms-marco-MiniLM-L-6-v2).
        ollama: Per-role context windows and retry/keep-alive settings.
    """

    rag: str = "gemma4:e4b"
    chat: str = "gemma4:e4b"
    embedding: str = "embeddinggemma:latest"
    contextual: str = "gemma4:e4b"
    recomp: str = "gemma4:e4b"
    ocr: str = "gemma4:e4b"
    desc: str = "gemma4"
    embed_prefix_query: str = ""
    embed_prefix_doc: str = ""
    reranker_quality: str = "quality"
    ollama: OllamaRuntimeConfig = OllamaRuntimeConfig()


# PURE DERIVATIONS


def infer_model_description(model_name: str) -> str:
    """Strip the ``:tag`` suffix from an Ollama model name.

    Same rule as ``_inferir_descripcion_modelo`` in ``rag/chat_pdfs.py``.

    Args:
        model_name: Full model identifier (e.g. ``"gemma4:e4b"``).

    Returns:
        Model name without the colon-separated tag.
    """
    return model_name.split(":")[0]


def derive_embedding_prefixes(embedding_model: str) -> Tuple[str, str]:
    """Return ``(query_prefix, doc_prefix)`` task prefixes for an embedding model.

    Same rule as ``_derivar_prefijos_embedding`` in ``rag/chat_pdfs.py``:
    only Nomic-family models (base name contains ``"nomic"``,
    case-insensitive) get the ``search_query:`` / ``search_document:``
    task prefixes; every other model embeds text unmodified.

    Args:
        embedding_model: Ollama embedding model name.

    Returns:
        Tuple ``(query_prefix, doc_prefix)``.
    """
    name = embedding_model.lower().split(":")[0]
    if "nomic" in name:
        return "search_query: ", "search_document: "
    return "", ""
