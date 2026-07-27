"""PathsConfig -- filesystem locations, derived vector-store paths."""

import os
from dataclasses import dataclass
from typing import Tuple

# Mirrors rag/chat_pdfs.py's BASE_DIR (= the ``rag/`` package directory),
# computed independently -- this module must not import rag.chat_pdfs (that
# would pull chromadb/ollama/fitz/... into config/, which stays infra-free
# like domain/ and ports/). This file lives at
# <root>/src/monkeygrab/config/paths.py; three ``dirname`` calls from its own
# directory land on <root>, then "rag" reproduces the same absolute path
# rag/chat_pdfs.py computes as ``os.path.dirname(os.path.abspath(__file__))``
# from inside rag/chat_pdfs.py itself.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
RAG_BASE_DIR = os.path.join(_PROJECT_ROOT, "rag")

# Matches models.ModelsConfig.embedding's default ("embeddinggemma:latest").
# Duplicated as a literal (rather than imported) to avoid a paths<->models
# import edge for a single string; kept in sync by
# test_app_config_defaults.py comparing the built AppConfig against
# rag.chat_pdfs's live PATH_DB/COLLECTION_NAME.
_DEFAULT_EMBEDDING_MODEL = "embeddinggemma:latest"


# PURE DERIVATION


def derive_db_paths(docs_folder: str, embedding_model: str, data_dir: str) -> Tuple[str, str]:
    """Derive ``(path_db, collection_name)`` from a docs folder and embedding model.

    Same rule as ``_derivar_paths_db`` in ``rag/chat_pdfs.py``: the
    embedding slug is part of the vector-store path so stores built with
    different embedding models never collide on disk.

    Args:
        docs_folder: PDF directory (absolute or relative).
        embedding_model: Ollama embedding model name; the tag is stripped
            for the slug.
        data_dir: Writable data root the vector store lives under.

    Returns:
        Tuple ``(path_db, collection_name)``.
    """
    nombre = os.path.basename(os.path.abspath(docs_folder))
    slug = embedding_model.split(":")[0].replace("/", "_")
    return (
        os.path.join(data_dir, "vector_db", f"{nombre}_{slug}"),
        f"docs_{nombre}",
    )


# CONFIG


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations, mirroring ``rag/chat_pdfs.py`` section 3.5.

    Attributes:
        base_dir: The ``rag/`` package directory.
        data_dir: Writable data root for the vector store, history and
            debug dumps -- the package dir in source/dev runs, a per-user
            location in the packaged desktop app (``MONKEYGRAB_DATA_DIR``).
        docs_folder: PDF source directory (``DOCS_FOLDER``).
        path_db: Vector-store directory, derived from ``docs_folder`` and
            ``models.embedding`` via ``derive_db_paths``.
        collection_name: Vector-store collection name, derived alongside
            ``path_db``.
        historial_path: CHAT-mode history JSON file.
        max_historial_mensajes: Maximum history messages kept (``MAX_HISTORIAL_MENSAJES``).
        carpeta_debug_rag: Directory for per-interaction RAG debug dumps.
    """

    base_dir: str = RAG_BASE_DIR
    data_dir: str = RAG_BASE_DIR
    docs_folder: str = os.path.join(RAG_BASE_DIR, "docs", "en")
    path_db: str = derive_db_paths(
        os.path.join(RAG_BASE_DIR, "docs", "en"), _DEFAULT_EMBEDDING_MODEL, RAG_BASE_DIR
    )[0]
    collection_name: str = derive_db_paths(
        os.path.join(RAG_BASE_DIR, "docs", "en"), _DEFAULT_EMBEDDING_MODEL, RAG_BASE_DIR
    )[1]
    historial_path: str = os.path.join(RAG_BASE_DIR, "historial_chat.json")
    max_historial_mensajes: int = 40
    carpeta_debug_rag: str = os.path.join(RAG_BASE_DIR, "debug_rag")
