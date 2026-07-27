"""Composition root -- builds the adapter graph a StackConfig asks for.

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  +-- 1. build_extractor      text extraction backend
#  +-- 2. build_vector_store   vector index backend
#  +-- 3. build_embedder       embedding backend
#  +-- 4. build_stack          all three at once, as a named tuple
#
# ─────────────────────────────────────────────

This is the only module that knows which concrete adapter exists. Use cases
receive ports, so nothing above this layer changes when a technology is swapped:
comparing two stacks is an environment variable plus a second run of the gate.

Every import here is deferred inside its factory, and that is not a style
preference. Importing an adapter imports its library, so a module-level import
would drag chromadb, faiss and torch into every process regardless of what was
selected -- paying the startup cost of three backends to use one, and making a
missing optional dependency fatal for users who never asked for it.

Dependencies: none at import time, by construction.
"""

from typing import Any, NamedTuple

from monkeygrab.config.app_config import AppConfig
from monkeygrab.config.stack import (
    EMBEDDER_JINA_CLIP,
    EMBEDDER_OLLAMA,
    EXTRACTOR_MINERU,
    EXTRACTOR_PYMUPDF,
    VECTOR_STORE_CHROMA,
    VECTOR_STORE_FAISS,
)


class Stack(NamedTuple):
    """The three swappable adapters, built and ready to inject.

    Attributes:
        extractor: Implements ``ports.PdfExtractor``.
        vector_store: Implements ``ports.VectorStore``.
        embedder: Implements ``ports.Embedder``.
    """

    extractor: Any
    vector_store: Any
    embedder: Any


# ─────────────────────────────────────────────
# SECTION 1: EXTRACTOR
# ─────────────────────────────────────────────


def build_extractor(config: AppConfig) -> Any:
    """Build the text extraction adapter named by ``config.stack.extractor``.

    Args:
        config: Application configuration, injected into the adapter.

    Returns:
        An object satisfying ``ports.PdfExtractor``.

    Raises:
        ValueError: If the selector holds an unimplemented value. StackConfig
            validates on construction, so reaching this means the two lists
            disagree -- a bug here, not bad input.
    """
    choice = config.stack.extractor
    if choice == EXTRACTOR_PYMUPDF:
        from monkeygrab.adapters.extraction.pymupdf_extractor import PymupdfExtractor

        return PymupdfExtractor(config.chunking)
    if choice == EXTRACTOR_MINERU:
        from monkeygrab.adapters.extraction.mineru_extractor import MineruExtractor

        return MineruExtractor(config.chunking)
    raise ValueError(f"No extractor adapter for {choice!r}")


# ─────────────────────────────────────────────
# SECTION 2: VECTOR STORE
# ─────────────────────────────────────────────


def build_vector_store(config: AppConfig) -> Any:
    """Build the vector store adapter named by ``config.stack.vector_store``.

    The index path carries the stack slug, because two stacks produce vectors of
    different dimension and meaning: sharing a directory would silently mix them.

    Args:
        config: Application configuration, injected into the adapter.

    Returns:
        An object satisfying ``ports.VectorStore``.

    Raises:
        ValueError: If the selector holds an unimplemented value.
    """
    choice = config.stack.vector_store
    if choice == VECTOR_STORE_CHROMA:
        from monkeygrab.adapters.vectorstore.chroma_store import ChromaVectorStore

        return ChromaVectorStore(config.paths.path_db, config.paths.collection_name)
    if choice == VECTOR_STORE_FAISS:
        from monkeygrab.adapters.vectorstore.faiss_store import FaissVectorStore

        return FaissVectorStore(f"{config.paths.path_db}_faiss")
    raise ValueError(f"No vector store adapter for {choice!r}")


# ─────────────────────────────────────────────
# SECTION 3: EMBEDDER
# ─────────────────────────────────────────────


def build_embedder(config: AppConfig) -> Any:
    """Build the embedding adapter named by ``config.stack.embedder``.

    Args:
        config: Application configuration, injected into the adapter.

    Returns:
        An object satisfying ``ports.Embedder``.

    Raises:
        ValueError: If the selector holds an unimplemented value.
    """
    choice = config.stack.embedder
    if choice == EMBEDDER_OLLAMA:
        from monkeygrab.adapters.embedding.ollama_embedder import OllamaEmbedder

        return OllamaEmbedder(config.models)
    if choice == EMBEDDER_JINA_CLIP:
        from monkeygrab.adapters.embedding.jina_clip_embedder import JinaClipEmbedder

        return JinaClipEmbedder()
    raise ValueError(f"No embedder adapter for {choice!r}")


# ─────────────────────────────────────────────
# SECTION 4: WHOLE STACK
# ─────────────────────────────────────────────


def build_stack(config: AppConfig) -> Stack:
    """Build every swappable adapter for ``config``.

    Args:
        config: Application configuration.

    Returns:
        A ``Stack`` with the three adapters.
    """
    return Stack(
        extractor=build_extractor(config),
        vector_store=build_vector_store(config),
        embedder=build_embedder(config),
    )
