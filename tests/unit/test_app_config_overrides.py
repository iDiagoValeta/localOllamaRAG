"""Unit tests for monkeygrab.config.AppConfig.with_overrides() and validation.

Covers: with_overrides never mutates the original instance, the derived-field
side effects that replace set_model_roles_runtime/set_docs_folder_runtime
(rag -> desc and docs_folder -> db paths),
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

from monkeygrab.config import AppConfig, env
from monkeygrab.config.chunking import ChunkingConfig
from monkeygrab.config.retrieval import RetrievalConfig


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


def test_overriding_models_rag_recomputes_desc():
    """Mirrors set_model_roles_runtime's ``if overrides.get("rag"):
    MODELO_DESC = _inferir_descripcion_modelo(MODELO_RAG)`` branch."""
    cfg = AppConfig().with_overrides(**{"models.rag": "qwen3:14b"})

    assert cfg.models.rag == "qwen3:14b"
    assert cfg.models.desc == "qwen3"


def test_overriding_models_chat_does_not_touch_desc_or_db_paths():
    """Only the RAG role feeds the description."""
    original = AppConfig()
    cfg = original.with_overrides(**{"models.chat": "qwen3:14b"})

    assert cfg.models.chat == "qwen3:14b"
    assert cfg.models.desc == original.models.desc
    assert cfg.paths.path_db == original.paths.path_db


def test_overriding_paths_docs_folder_recomputes_db_paths():
    """Mirrors set_docs_folder_runtime: switching the PDF source folder
    must recompute PATH_DB/COLLECTION_NAME."""
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


def test_with_overrides_rejects_key_without_a_dot():
    with pytest.raises(ValueError, match="section.*field"):
        AppConfig().with_overrides(usar_reranker=False)


def test_with_overrides_rejects_unknown_section():
    with pytest.raises(ValueError, match="bogus_section"):
        AppConfig().with_overrides(**{"bogus_section.x": 1})


def test_with_overrides_rejects_unknown_field_in_known_section():
    with pytest.raises(ValueError, match="bogus_field"):
        AppConfig().with_overrides(**{"flags.bogus_field": True})


def test_from_env_raises_on_unparseable_int_env_var(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "not-an-int")
    with pytest.raises(ValueError, match="RAG_CHUNK_SIZE"):
        AppConfig.from_env()


def test_from_env_raises_on_unparseable_float_env_var(monkeypatch):
    monkeypatch.setenv("RAG_BM25_K1", "not-a-float")
    with pytest.raises(ValueError, match="RAG_BM25_K1"):
        AppConfig.from_env()


def test_from_env_accepts_valid_env_overrides(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "1234")
    cfg = AppConfig.from_env()

    assert cfg.chunking.chunk_size == 1234


# Range validation. A parsed-but-impossible value must abort at construction:
# chunk_size <= 0 makes the recursive splitter stop advancing and exhaust memory,
# and rrf_k <= 0 divides by zero or yields negative fusion weights.


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_app_config_rejects_non_positive_chunk_size(chunk_size):
    with pytest.raises(ValueError, match=f"RAG_CHUNK_SIZE.*{chunk_size}"):
        AppConfig(chunking=ChunkingConfig(chunk_size=chunk_size))


@pytest.mark.parametrize("chunk_overlap", [-1, 2000, 2001])
def test_app_config_rejects_chunk_overlap_outside_its_range(chunk_overlap):
    with pytest.raises(ValueError, match="RAG_CHUNK_OVERLAP"):
        AppConfig(chunking=ChunkingConfig(chunk_size=2000, chunk_overlap=chunk_overlap))


def test_app_config_accepts_the_chunk_overlap_boundaries():
    zero = AppConfig(chunking=ChunkingConfig(chunk_size=2000, chunk_overlap=0))
    largest = AppConfig(chunking=ChunkingConfig(chunk_size=2000, chunk_overlap=1999))

    assert zero.chunking.chunk_overlap == 0
    assert largest.chunking.chunk_overlap == 1999


@pytest.mark.parametrize("rrf_k", [0, -1])
def test_app_config_rejects_non_positive_rrf_k(rrf_k):
    with pytest.raises(ValueError, match=f"RAG_RRF_K.*{rrf_k}"):
        AppConfig(retrieval=RetrievalConfig(rrf_k=rrf_k))


@pytest.mark.parametrize(
    "field,env_name",
    [
        ("top_k_rerank_candidates", "RAG_TOP_K_RERANK_CANDIDATES"),
        ("top_k_final", "RAG_TOP_K_FINAL"),
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_app_config_rejects_non_positive_top_k(field, env_name, value):
    with pytest.raises(ValueError, match=f"{env_name}.*{value}"):
        AppConfig(retrieval=RetrievalConfig(**{field: value}))


def test_from_env_rejects_an_out_of_range_value(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "0")

    with pytest.raises(ValueError, match="RAG_CHUNK_SIZE"):
        AppConfig.from_env()


# Empty strings are how shell exports and .env files spell "not configured";
# abspath("") is the process cwd, which would relocate the vector store, chat
# history and debug dumps of issue #356 to wherever the process was launched.


@pytest.mark.parametrize("blank", ["", "   "])
def test_read_env_str_treats_blank_as_unset(monkeypatch, blank):
    monkeypatch.setenv("MONKEYGRAB_DATA_DIR", blank)

    assert env.read_env_str("MONKEYGRAB_DATA_DIR", "/fallback") == "/fallback"


def test_from_env_treats_empty_data_dir_as_unset(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_DATA_DIR", "")
    cfg = AppConfig.from_env()

    assert cfg.paths.data_dir == cfg.paths.base_dir
    assert cfg.paths.data_dir != os.getcwd()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
