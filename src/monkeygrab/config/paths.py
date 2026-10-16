"""Filesystem locations for the single multimodal vector store."""

import os
from dataclasses import dataclass
from typing import Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
RAG_BASE_DIR = os.path.join(_PROJECT_ROOT, "rag")

# The suffix names the embedder that built the store: a different embedder
# writes a different vector space and must never share its directory.
_EMBEDDER_SLUG = "eg2"


def derive_db_paths(docs_folder: str, data_dir: str) -> Tuple[str, str]:
    """Return the FAISS path and collection name for a PDF corpus.

    Args:
        docs_folder: PDF directory; its basename names the store.
        data_dir: Writable root that holds ``vector_db/``.

    Returns:
        Tuple ``(path_db, collection_name)``.
    """
    name = os.path.basename(os.path.abspath(docs_folder))
    return (
        os.path.join(data_dir, "vector_db", f"{name}_{_EMBEDDER_SLUG}"),
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
