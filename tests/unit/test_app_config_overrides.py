"""Unit tests for monkeygrab.config.AppConfig.with_overrides() and validation.

Covers: with_overrides never mutates the original instance, the derived-field
side effects that replace set_model_roles_runtime/set_docs_folder_runtime
(rag -> desc, embedding -> prefixes + db paths, docs_folder -> db paths),
invalid override keys raise, and AppConfig.from_env() hard-fails on
unparseable/unknown environment values instead of silently falling back to a
default.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from monkeygrab.config import AppConfig


# ─────────────────────────────────────────────
# with_overrides -- immutability
# ─────────────────────────────────────────────


def test_with_overrides_returns_new_instance_and_does_not_mutate_original():
    original = AppConfig()
    updated = original.with_overrides(**{"flags.usar_reranker": False})

    assert updated is not original
    assert updated.flags is not original.flags
    assert original.flags.usar_reranker is True  # untouched
    assert updated.flags.usar_reranker is False
    # Every other section is untouched (same nested instance, not just equal).
    assert updated.models is original.models
    assert updated.chunking is original.chunking
    assert updated.paths is original.paths


def test_with_overrides_leaves_unmentioned_fields_in_the_same_section_untouched():
    original = AppConfig()
    updated = original.with_overrides(**{"flags.usar_reranker": False})

    assert updated.flags.usar_busqueda_hibrida is original.flags.usar_busqueda_hibrida is True


# ─────────────────────────────────────────────
# with_overrides -- derived-field side effects
# ─────────────────────────────────────────────


def test_overriding_models_rag_recomputes_desc():
    """Mirrors set_model_roles_runtime's ``if overrides.get("rag"):
    MODELO_DESC = _inferir_descripcion_modelo(MODELO_RAG)`` branch."""
    cfg = AppConfig().with_overrides(**{"models.rag": "qwen3:14b"})

    assert cfg.models.rag == "qwen3:14b"
    assert cfg.models.desc == "qwen3"


def test_overriding_models_embedding_recomputes_prefixes_and_db_paths():
    """Mirrors set_model_roles_runtime's ``embedding_changed`` branch:
    changing the embedding role must recompute both the task prefixes and
    the vector-store path/collection name, since both are namespaced by
    the embedding model."""
    original = AppConfig()
    cfg = original.with_overrides(**{"models.embedding": "nomic-embed-text:latest"})

    assert cfg.models.embedding == "nomic-embed-text:latest"
    assert cfg.models.embed_prefix_query == "search_query: "
    assert cfg.models.embed_prefix_doc == "search_document: "
    assert cfg.paths.path_db != original.paths.path_db
    assert "nomic-embed-text" in cfg.paths.path_db
    # docs folder / collection name basis is unchanged, only the embedding slug moved.
    assert cfg.paths.collection_name == original.paths.collection_name


def test_overriding_models_chat_does_not_touch_desc_or_db_paths():
    """Only the "rag" role feeds desc; only "embedding" feeds db paths."""
    original = AppConfig()
    cfg = original.with_overrides(**{"models.chat": "qwen3:14b"})

    assert cfg.models.chat == "qwen3:14b"
    assert cfg.models.desc == original.models.desc
    assert cfg.paths.path_db == original.paths.path_db


def test_overriding_paths_docs_folder_recomputes_db_paths():
    """Mirrors set_docs_folder_runtime: switching the PDF source folder
    must recompute PATH_DB/COLLECTION_NAME using the (unchanged) embedding
    model."""
    original = AppConfig()
    new_folder = str(Path(original.paths.base_dir) / "docs" / "es")

    cfg = original.with_overrides(**{"paths.docs_folder": new_folder})

    assert cfg.paths.docs_folder == new_folder
    assert cfg.paths.collection_name == "docs_es"
    assert cfg.paths.path_db != original.paths.path_db


def test_overriding_paths_docs_folder_normalizes_a_relative_path():
    """set_docs_folder_runtime (rag/chat_pdfs.py) always applied os.path.abspath
    before assigning CARPETA_DOCS, so the value was guaranteed absolute
    regardless of what the caller passed in. with_overrides must match: a
    relative override left as-is would resolve against whatever the
    process's cwd happens to be when something later reads
    paths.docs_folder (e.g. the web PDF viewer), instead of staying stable."""
    original = AppConfig()

    cfg = original.with_overrides(**{"paths.docs_folder": "relative/subdir"})

    assert cfg.paths.docs_folder == os.path.abspath("relative/subdir")
    assert os.path.isabs(cfg.paths.docs_folder)


def test_overriding_paths_docs_folder_with_an_already_absolute_path_is_unchanged():
    original = AppConfig()
    absolute_folder = str(Path(original.paths.base_dir) / "docs" / "ca")

    cfg = original.with_overrides(**{"paths.docs_folder": absolute_folder})

    assert cfg.paths.docs_folder == absolute_folder


# ─────────────────────────────────────────────
# with_overrides -- validation
# ─────────────────────────────────────────────


def test_with_overrides_rejects_key_without_a_dot():
    with pytest.raises(ValueError, match="section.*field"):
        AppConfig().with_overrides(usar_reranker=False)


def test_with_overrides_rejects_unknown_section():
    with pytest.raises(ValueError, match="bogus_section"):
        AppConfig().with_overrides(**{"bogus_section.x": 1})


def test_with_overrides_rejects_unknown_field_in_known_section():
    with pytest.raises(ValueError, match="bogus_field"):
        AppConfig().with_overrides(**{"flags.bogus_field": True})


# ─────────────────────────────────────────────
# from_env -- hard-fail validation
# ─────────────────────────────────────────────


def test_from_env_raises_on_unparseable_int_env_var(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "not-an-int")
    with pytest.raises(ValueError, match="RAG_CHUNK_SIZE"):
        AppConfig.from_env()


def test_from_env_raises_on_unparseable_float_env_var(monkeypatch):
    monkeypatch.setenv("RAG_BM25_K1", "not-a-float")
    with pytest.raises(ValueError, match="RAG_BM25_K1"):
        AppConfig.from_env()


def test_from_env_raises_on_unknown_reranker_quality(monkeypatch):
    monkeypatch.setenv("RERANKER_QUALITY", "ultra-quality")
    with pytest.raises(ValueError, match="RERANKER_QUALITY"):
        AppConfig.from_env()


def test_from_env_accepts_valid_env_overrides(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "1234")
    monkeypatch.setenv("RERANKER_QUALITY", "fast")
    cfg = AppConfig.from_env()

    assert cfg.chunking.chunk_size == 1234
    assert cfg.models.reranker_quality == "fast"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
