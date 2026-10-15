"""Filesystem locations for the single multimodal vector store."""

import os
from dataclasses import dataclass
from typing import Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
RAG_BASE_DIR = os.path.join(_PROJECT_ROOT, "rag")

# One index directory per embedder: the two produce vectors in different spaces,
# so sharing a path would let one query the other's store. The jina slug is the
# historical name and must not change -- existing indexes live under it.
_EMBEDDER_SLUGS = {"jina": "jina_clip", "gemma2": "eg2"}


def derive_db_paths(docs_folder: str, data_dir: str, embedder: str = "jina") -> Tuple[str, str]:
    """Return the FAISS path and collection name for a PDF corpus.

    Args:
        docs_folder: PDF directory; its basename names the store.
        data_dir: Writable root that holds ``vector_db/``.
        embedder: Active embedder key; selects the per-embedder path suffix.

    Returns:
        Tuple ``(path_db, collection_name)``.

    Raises:
        ValueError: If ``embedder`` is not a known embedder key.
    """
    if embedder not in _EMBEDDER_SLUGS:
        valid = ", ".join(_EMBEDDER_SLUGS)
        raise ValueError(f"Unknown embedder {embedder!r}. Valid: {valid}")
    name = os.path.basename(os.path.abspath(docs_folder))
    return (
        os.path.join(data_dir, "vector_db", f"{name}_{_EMBEDDER_SLUGS[embedder]}"),
        f"docs_{name}",
    )


_DEFAULT_PATH, _DEFAULT_COLLECTION = derive_db_paths(
    os.path.join(RAG_BASE_DIR, "docs", "en"), RAG_BASE_DIR
)


@dataclass(frozen=True)
class PathsConfig:
    base_dir: str = RAG_BASE_DIR
    data_dir: str = RAG_BASE_DIR
    docs_folder: str = os.path.join(RAG_BASE_DIR, "docs", "en")
    path_db: str = _DEFAULT_PATH
    collection_name: str = _DEFAULT_COLLECTION
    historial_path: str = os.path.join(RAG_BASE_DIR, "historial_chat.json")
    max_historial_mensajes: int = 40
    carpeta_debug_rag: str = os.path.join(RAG_BASE_DIR, "debug_rag")
