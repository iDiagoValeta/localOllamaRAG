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

import dataclasses
from pathlib import Path
from typing import Any, NamedTuple

from monkeygrab.config.app_config import AppConfig
from monkeygrab.config.paths import PathsConfig
from monkeygrab.config.stack import (
    EMBEDDER_JINA_CLIP,
    EMBEDDER_OLLAMA,
    EXTRACTOR_MINERU,
    EXTRACTOR_PYMUPDF,
    VECTOR_STORE_CHROMA,
    VECTOR_STORE_FAISS,
)

# Historic production path has no stack slug; every other combination must not
# share that directory (incompatible vector spaces / dimensions).
_DEFAULT_STACK_SLUG = f"{EXTRACTOR_PYMUPDF}-{EMBEDDER_OLLAMA}-{VECTOR_STORE_CHROMA}"


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


def paths_for_stack(config: AppConfig) -> PathsConfig:
    """Return store paths, namespaced by stack slug when not the default stack.

    The default ``pymupdf-ollama-chroma`` path stays exactly as
    ``derive_db_paths`` produced it, so existing indexes and PATH_DB
    characterization keep working. Any other combination appends
    ``__{slug}`` so A/B indexes cannot collide.
    """
    paths = config.paths
    slug = config.stack.slug
    if slug == _DEFAULT_STACK_SLUG:
        return paths
    return dataclasses.replace(paths, path_db=f"{paths.path_db}__{slug}")


# ─────────────────────────────────────────────
# SECTION 1: EXTRACTOR
# ─────────────────────────────────────────────


def build_extractor(config: AppConfig) -> Any:
    """Build the text extraction adapter named by ``config.stack.extractor``.

    Args:
        config: Application configuration (selector only; extractors take their
            own optional constructor args from the environment).

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

        return PymupdfExtractor()
    if choice == EXTRACTOR_MINERU:
        from monkeygrab.adapters.extraction.mineru_extractor import MineruExtractor

        return MineruExtractor()
    raise ValueError(f"No extractor adapter for {choice!r}")


# ─────────────────────────────────────────────
# SECTION 2: VECTOR STORE
# ─────────────────────────────────────────────


def build_vector_store(config: AppConfig) -> Any:
    """Build the vector store adapter named by ``config.stack.vector_store``.

    The index path carries the stack slug (except for the historic default),
    because two stacks produce vectors of different dimension and meaning:
    sharing a directory would silently mix them.

    Args:
        config: Application configuration, injected into the adapter.

    Returns:
        An object satisfying ``ports.VectorStore``.

    Raises:
        ValueError: If the selector holds an unimplemented value.
    """
    choice = config.stack.vector_store
    paths = paths_for_stack(config)
    if choice == VECTOR_STORE_CHROMA:
        from monkeygrab.adapters.vectorstore.chroma_store import ChromaVectorStore

        return ChromaVectorStore(paths)
    if choice == VECTOR_STORE_FAISS:
        from monkeygrab.adapters.vectorstore.faiss_store import FaissVectorStore

        return FaissVectorStore(paths)
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

        # Same isolated interpreter MinerU uses; required (no silent default
        # inside the adapter) so a missing venv fails here with a clear path.
        root = Path(__file__).resolve().parents[2]
        python_exe = root / ".venv-mineru" / "Scripts" / "python.exe"
        if not python_exe.is_file():
            python_exe = root / ".venv-mineru" / "bin" / "python"
        if not python_exe.is_file():
            raise FileNotFoundError(
                f"jina_clip embedder needs the isolated MinerU venv python at "
                f"{python_exe}; create .venv-mineru and install jina-clip there"
            )
        return JinaClipEmbedder(str(python_exe))

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
