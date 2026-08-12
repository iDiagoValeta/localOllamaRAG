"""Tests for rag/engine/settings_store.py -- the settings.json load/apply
logic shared by the CLI and the web app (issue #36's follow-up from PR #59's
review: before this module existed, the CLI never read the file the web
control panel persisted model roles and pipeline flags to, so a change made
in one interface silently never reached the other).

Uses the real rag.chat_pdfs module rather than a hand-rolled fake: precedence
against an actual environment variable and the sanctioned
set_model_roles_runtime switch are exactly what needs end-to-end coverage
here. monkeypatch restores every attribute this touches after each test.
"""

import json
import os

import rag.chat_pdfs as rag_engine
import rag.web.app as web_app
from rag.engine import settings_store


def _write_settings(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# --- path resolution ---------------------------------------------------
# MONKEYGRAB_DATA_DIR (the packaged desktop app's writable root, vs. the
# package dir in a source/dev run) only ever acts through rag_engine.DATA_DIR
# -- rag.chat_pdfs resolves it once at import time and nothing here
# re-derives it -- so exercising settings_path/settings_file against a
# monkeypatched DATA_DIR covers both cases without needing a second process.

def test_settings_path_is_settings_json_under_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_engine, "DATA_DIR", str(tmp_path))
    assert settings_store.settings_path(rag_engine) == os.path.join(str(tmp_path), "settings.json")


# --- corrupt / partial / missing files never raise ----------------------

def test_missing_settings_file_returns_empty_dict(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_engine, "DATA_DIR", str(tmp_path))
    assert settings_store.load_settings(rag_engine) == {}


def test_corrupt_json_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_engine, "DATA_DIR", str(tmp_path))
    path = settings_store.settings_path(rag_engine)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not valid json")

    assert settings_store.load_settings(rag_engine) == {}


def test_settings_file_that_is_a_json_array_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_engine, "DATA_DIR", str(tmp_path))
    _write_settings(settings_store.settings_path(rag_engine), ["not", "an", "object"])

    assert settings_store.load_settings(rag_engine) == {}


def test_wrong_shaped_sections_do_not_raise_when_applied(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_engine, "DATA_DIR", str(tmp_path))
    _write_settings(settings_store.settings_path(rag_engine), {"roles": "oops", "flags": 42})

    data = settings_store.load_settings(rag_engine)
    settings_store.apply_model_roles(rag_engine, data)
    settings_store.apply_pipeline_flags(rag_engine, data)


# --- CLI model-role precedence (prefer_env_var=True, the default): ------
# env var > settings.json > hardcoded default

def test_cli_takes_the_persisted_role_when_no_env_var_is_set(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_engine, "DATA_DIR", str(tmp_path))
    monkeypatch.delenv("OLLAMA_CONTEXTUAL_MODEL", raising=False)
    monkeypatch.setattr(rag_engine, "MODELO_CONTEXTUAL", "hardcoded-default")
    _write_settings(
        settings_store.settings_path(rag_engine),
        {"roles": {"contextual": "from-settings-json"}},
    )

    data = settings_store.load_settings(rag_engine)
    settings_store.apply_model_roles(rag_engine, data)  # prefer_env_var defaults True

    assert rag_engine.MODELO_CONTEXTUAL == "from-settings-json"


def test_cli_lets_an_explicit_env_var_win_over_a_persisted_role(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_engine, "DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OLLAMA_CONTEXTUAL_MODEL", "from-env")
    monkeypatch.setattr(rag_engine, "MODELO_CONTEXTUAL", "from-env")
    _write_settings(
        settings_store.settings_path(rag_engine),
        {"roles": {"contextual": "from-settings-json"}},
    )

    data = settings_store.load_settings(rag_engine)
    settings_store.apply_model_roles(rag_engine, data)

    assert rag_engine.MODELO_CONTEXTUAL == "from-env"


def test_cli_preserves_an_explicit_modelo_desc_override(tmp_path, monkeypatch):
    """set_model_roles_runtime recomputes MODELO_DESC from MODELO_RAG whenever
    "rag" is applied. The CLI's env-preferring path must not let that
    clobber a MODELO_DESC the environment set on purpose (.env.example
    documents it as independently overridable)."""
    monkeypatch.setattr(rag_engine, "DATA_DIR", str(tmp_path))
    monkeypatch.delenv("OLLAMA_RAG_MODEL", raising=False)
    monkeypatch.setenv("MODELO_DESC", "custom-desc")
    # set_model_roles_runtime mutates these two globals directly (not through
    # setattr on a fresh binding), so a no-op setattr here is what makes
    # monkeypatch restore rag.chat_pdfs's real values afterwards -- without
    # it this test would permanently leak "other-model:latest"/"custom-desc"
    # into the shared module for the rest of the pytest process.
    monkeypatch.setattr(rag_engine, "MODELO_RAG", rag_engine.MODELO_RAG)
    monkeypatch.setattr(rag_engine, "MODELO_DESC", rag_engine.MODELO_DESC)
    _write_settings(
        settings_store.settings_path(rag_engine),
        {"roles": {"rag": "other-model:latest"}},
    )

    data = settings_store.load_settings(rag_engine)
    settings_store.apply_model_roles(rag_engine, data)

    assert rag_engine.MODELO_RAG == "other-model:latest"
    assert rag_engine.MODELO_DESC == "custom-desc"


def test_unknown_role_key_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_engine, "DATA_DIR", str(tmp_path))
    _write_settings(
        settings_store.settings_path(rag_engine),
        {"roles": {"not_a_real_role": "whatever"}},
    )

    data = settings_store.load_settings(rag_engine)
    settings_store.apply_model_roles(rag_engine, data)  # must not raise


# --- web app model-role precedence (prefer_env_var=False): --------------
# settings.json always wins -- it is the app's own prior state, not a
# remembered default. Regression coverage for a real bug caught in review: an
# earlier version of this module applied env-var precedence unconditionally,
# which meant a role picked in the web UI would silently revert on the next
# restart whenever its OLLAMA_*_MODEL var was set (commonly true for anyone
# following the project's own .env.example), and the next _save_persisted_
# settings call would then permanently overwrite the file with the reverted
# value.

def test_web_app_takes_the_persisted_role_even_with_an_env_var_set(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_engine, "DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OLLAMA_CONTEXTUAL_MODEL", "from-env")
    monkeypatch.setattr(rag_engine, "MODELO_CONTEXTUAL", "from-env")
    _write_settings(
        settings_store.settings_path(rag_engine),
        {"roles": {"contextual": "picked-in-the-web-ui"}},
    )

    data = settings_store.load_settings(rag_engine)
    settings_store.apply_model_roles(rag_engine, data, prefer_env_var=False)

    assert rag_engine.MODELO_CONTEXTUAL == "picked-in-the-web-ui"


def test_web_app_load_then_save_does_not_erase_a_persisted_role(tmp_path, monkeypatch):
    """End-to-end regression coverage for the HIGH bug caught in review: an
    earlier version of settings_store applied the CLI's env-first precedence
    unconditionally, which reverted a role picked in the web UI whenever its
    OLLAMA_*_MODEL var happened to be set, and the very next
    _save_persisted_settings call then permanently overwrote settings.json
    with that reverted value.

    Drives the real rag.web.app._load_persisted_settings /
    _save_persisted_settings pair -- not settings_store.apply_model_roles
    directly -- so removing web_app's prefer_env_var=False call site would
    fail this test, unlike the two unit-level tests above.
    """
    monkeypatch.setattr(rag_engine, "DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OLLAMA_CONTEXTUAL_MODEL", "from-env")
    monkeypatch.setattr(rag_engine, "MODELO_CONTEXTUAL", "from-env")
    _write_settings(
        settings_store.settings_path(rag_engine),
        {"roles": {"contextual": "picked-in-the-web-ui"}},
    )

    web_app._load_persisted_settings()
    assert rag_engine.MODELO_CONTEXTUAL == "picked-in-the-web-ui"

    web_app._save_persisted_settings()
    with open(settings_store.settings_path(rag_engine), encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["roles"]["contextual"] == "picked-in-the-web-ui"


# --- pipeline flags: settings.json > hardcoded default, no env layer, ---
# identical for both callers

def test_persisted_flags_override_the_hardcoded_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_engine, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(rag_engine, "USAR_CONTEXTUAL_RETRIEVAL", True)
    monkeypatch.setattr(rag_engine, "USAR_EMBEDDINGS_IMAGEN", True)
    _write_settings(
        settings_store.settings_path(rag_engine),
        {"flags": {"USAR_CONTEXTUAL_RETRIEVAL": False, "USAR_EMBEDDINGS_IMAGEN": False}},
    )

    data = settings_store.load_settings(rag_engine)
    settings_store.apply_pipeline_flags(rag_engine, data)

    assert rag_engine.USAR_CONTEXTUAL_RETRIEVAL is False
    assert rag_engine.USAR_EMBEDDINGS_IMAGEN is False


def test_unknown_flag_key_is_ignored_and_creates_no_attribute(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_engine, "DATA_DIR", str(tmp_path))
    _write_settings(
        settings_store.settings_path(rag_engine),
        {"flags": {"NOT_A_REAL_FLAG": True}},
    )

    data = settings_store.load_settings(rag_engine)
    settings_store.apply_pipeline_flags(rag_engine, data)

    assert not hasattr(rag_engine, "NOT_A_REAL_FLAG")


def test_web_settings_map_matches_the_persistable_flag_vars():
    """rag/web/app.py's _SETTINGS_MAP (frontend key -> engine var, used by
    /api/settings) and rag.chat_pdfs.PERSISTABLE_FLAG_VARS (what settings.json
    may persist and restore) must name the same set of engine variables --
    otherwise a flag could be saved but never restored, or restored under a
    name the web API no longer recognizes.
    """
    assert set(web_app._SETTINGS_MAP.values()) == set(rag_engine.PERSISTABLE_FLAG_VARS)


# --- the active store is web-only: neither apply function touches it ----

def test_apply_functions_never_touch_the_active_store(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_engine, "DATA_DIR", str(tmp_path))
    original_carpeta = rag_engine.CARPETA_DOCS
    _write_settings(
        settings_store.settings_path(rag_engine),
        {"active_store": "es", "roles": {}, "flags": {}},
    )

    data = settings_store.load_settings(rag_engine)
    settings_store.apply_model_roles(rag_engine, data)
    settings_store.apply_pipeline_flags(rag_engine, data)

    assert rag_engine.CARPETA_DOCS == original_carpeta
