"""Characterization tests for configuration derivation in rag/chat_pdfs.py --
pre-migration snapshot.

Covers ``_derivar_paths_db`` / ``_derivar_prefijos_embedding`` (pure
derivation functions) and the runtime mutators ``set_pipeline_flags`` /
``set_model_roles_runtime`` / ``set_docs_folder_runtime`` (what they mutate,
what they return, and their validation errors). Design doc
(docs/design/2026-07-26-monkeygrab-v2.md, section 4) replaces this mutable
global config with an immutable ``AppConfig`` injected per call -- these
tests pin today's mutation semantics so that replacement is a deliberate,
visible decision rather than an accidental behavior change.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

import rag.chat_pdfs as rag


# ─────────────────────────────────────────────
# _derivar_paths_db / _derivar_prefijos_embedding -- pure derivation
# ─────────────────────────────────────────────


def test_derivar_paths_db_namespaces_vector_db_by_folder_basename_and_embedding_slug():
    """PATH_DB is keyed by (docs-folder basename, embedding model slug before
    the ':' tag) so different embedding models never share a vector store."""
    carpeta = os.path.join("rag", "docs", "en")

    path_db, collection_name = rag._derivar_paths_db(carpeta, "embeddinggemma:latest")

    assert path_db == os.path.join(rag.DATA_DIR, "vector_db", "en_embeddinggemma")
    assert collection_name == "docs_en"


def test_derivar_paths_db_slug_strips_tag_and_replaces_slashes():
    carpeta = os.path.join("rag", "docs", "es")

    path_db, collection_name = rag._derivar_paths_db(carpeta, "org/nomic-embed-text:v1.5")

    assert path_db == os.path.join(rag.DATA_DIR, "vector_db", "es_org_nomic-embed-text")
    assert collection_name == "docs_es"


@pytest.mark.parametrize("modelo,expected", [
    ("nomic-embed-text:latest", ("search_query: ", "search_document: ")),
    ("NOMIC-Something:v1", ("search_query: ", "search_document: ")),  # case-insensitive match
    ("embeddinggemma:latest", ("", "")),
    ("bge-m3", ("", "")),
])
def test_derivar_prefijos_embedding_only_nomic_models_get_task_prefixes(modelo, expected):
    """Only embedding models with "nomic" in their base name (case-insensitive)
    get the search_query:/search_document: task prefixes Nomic expects;
    every other model embeds text unmodified."""
    assert rag._derivar_prefijos_embedding(modelo) == expected


# ─────────────────────────────────────────────
# set_pipeline_flags
# ─────────────────────────────────────────────


def test_set_pipeline_flags_mutates_module_globals_and_returns_previous_values():
    previous = rag.set_pipeline_flags({"USAR_RERANKER": False, "USAR_BUSQUEDA_HIBRIDA": False})
    try:
        assert previous["USAR_RERANKER"] is True
        assert previous["USAR_BUSQUEDA_HIBRIDA"] is True
        assert rag.USAR_RERANKER is False
        assert rag.USAR_BUSQUEDA_HIBRIDA is False
        # Flags not mentioned in the override are left untouched.
        assert rag.USAR_LLM_QUERY_DECOMPOSITION is True
    finally:
        rag.set_pipeline_flags(previous)


def test_set_pipeline_flags_rejects_index_time_or_unknown_flags():
    """Index-time flags (e.g. USAR_CONTEXTUAL_RETRIEVAL, USAR_EMBEDDINGS_IMAGEN)
    and arbitrary config constants (e.g. CHUNK_SIZE) are intentionally outside
    PIPELINE_RUNTIME_FLAGS and raise instead of silently mutating."""
    with pytest.raises(ValueError, match="CHUNK_SIZE"):
        rag.set_pipeline_flags({"CHUNK_SIZE": 10})

    with pytest.raises(ValueError, match="USAR_CONTEXTUAL_RETRIEVAL"):
        rag.set_pipeline_flags({"USAR_CONTEXTUAL_RETRIEVAL": False})


# ─────────────────────────────────────────────
# set_model_roles_runtime
# ─────────────────────────────────────────────


def test_set_model_roles_runtime_changing_embedding_recomputes_db_paths_and_prefixes():
    """Reassigning the "embedding" role must also recompute PATH_DB,
    COLLECTION_NAME and the embed prefixes -- they're namespaced by embedding
    model. Other roles must NOT trigger this recomputation."""
    original_roles = rag.get_model_roles()
    original_path_db = rag.PATH_DB
    try:
        rag.set_model_roles_runtime({"embedding": "nomic-embed-text:latest"})

        assert rag.MODELO_EMBEDDING == "nomic-embed-text:latest"
        assert rag.PATH_DB != original_path_db
        assert "nomic-embed-text" in rag.PATH_DB
        assert rag.EMBED_PREFIX_QUERY == "search_query: "
    finally:
        # NOTE: set_model_roles_runtime returns the mapping AFTER applying
        # overrides (unlike set_pipeline_flags / set_docs_folder_runtime,
        # which both return the PREVIOUS state) -- restoring requires
        # capturing get_model_roles() before mutating, not the call's own
        # return value. Documented here because it's an easy trap for a test
        # (or a caller) to fall into.
        rag.set_model_roles_runtime(original_roles)
        assert rag.PATH_DB == original_path_db


def test_set_model_roles_runtime_changing_rag_role_does_not_touch_db_paths():
    original_roles = rag.get_model_roles()
    original_path_db = rag.PATH_DB
    try:
        result = rag.set_model_roles_runtime({"rag": "qwen3:14b"})

        assert result["rag"] == "qwen3:14b"
        assert rag.MODELO_DESC == "qwen3"  # tag stripped
        assert rag.PATH_DB == original_path_db
    finally:
        rag.set_model_roles_runtime(original_roles)


def test_set_model_roles_runtime_ignores_blank_overrides_and_rejects_unknown_roles():
    original_roles = rag.get_model_roles()
    try:
        # Blank/whitespace values are treated as "no change", not cleared.
        result = rag.set_model_roles_runtime({"chat": "  "})
        assert result["chat"] == original_roles["chat"]

        with pytest.raises(ValueError, match="unsupported_role"):
            rag.set_model_roles_runtime({"unsupported_role": "x"})
    finally:
        rag.set_model_roles_runtime(original_roles)


# ─────────────────────────────────────────────
# set_docs_folder_runtime
# ─────────────────────────────────────────────


def test_set_docs_folder_runtime_returns_previous_state_and_none_restores_import_time_defaults():
    original = (rag.CARPETA_DOCS, rag.PATH_DB, rag.COLLECTION_NAME)
    try:
        previous = rag.set_docs_folder_runtime(os.path.join("rag", "docs", "es"))
        assert previous == original

        es_state = (rag.CARPETA_DOCS, rag.PATH_DB, rag.COLLECTION_NAME)
        assert rag.CARPETA_DOCS.endswith(os.path.join("rag", "docs", "es"))
        assert rag.COLLECTION_NAME == "docs_es"

        # carpeta=None restores the values captured at import time, not just
        # "the previous call's carpeta" -- it returns the state right before
        # this call (the "es" state) as its own "previous" tuple.
        restored_from = rag.set_docs_folder_runtime(None)
        assert restored_from == es_state
        assert (rag.CARPETA_DOCS, rag.PATH_DB, rag.COLLECTION_NAME) == original
    finally:
        rag.set_docs_folder_runtime(None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
