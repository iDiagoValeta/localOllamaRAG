"""Filesystem locations for the single multimodal vector store."""

import os
from dataclasses import dataclass
from typing import Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
RAG_BASE_DIR = os.path.join(_PROJECT_ROOT, "rag")


def derive_db_paths(docs_folder: str, data_dir: str) -> Tuple[str, str]:
    """Return the FAISS path and collection name for a PDF corpus."""
    name = os.path.basename(os.path.abspath(docs_folder))
    return (
        os.path.join(data_dir, "vector_db", f"{name}_jina_clip"),
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
