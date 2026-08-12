"""Tests for MonkeyGrabCLI adopting settings.json at startup (issue #36's
follow-up from PR #59's review).

Most tests exercise ``MonkeyGrabCLI._load_persisted_settings`` directly, the
same style as ``test_cli_fingerprint_warning.py`` for
``_check_index_fingerprint``: ``run()`` also does an Ollama health check and
drives the Rich console, neither of which this decision depends on. One test
(``test_run_adopts_settings_before_opening_the_collection``) drives the real
``run()`` instead, specifically to pin the call order against ``run()``
itself rather than against a helper that could silently stop being called
from there.

Imports rag.cli.app (pulls in rich through rag.cli.display), so this file is
skipped by the dependency-free fast CI gate -- see tests/conftest.py.
"""

import json
import os

from rag.cli.app import MonkeyGrabCLI


class _FakeRagEngine:
    """Just enough of the rag_engine surface for _load_persisted_settings."""

    MODEL_ROLE_VARS = {
        "rag": "MODELO_RAG",
        "chat": "MODELO_CHAT",
        "contextual": "MODELO_CONTEXTUAL",
        "recomp": "MODELO_RECOMP",
    }
    MODEL_ROLE_ENV_VARS = {
        "rag": "OLLAMA_RAG_MODEL",
        "chat": "OLLAMA_CHAT_MODEL",
        "contextual": "OLLAMA_CONTEXTUAL_MODEL",
        "recomp": "OLLAMA_RECOMP_MODEL",
    }
    PERSISTABLE_FLAG_VARS = (
        "USAR_CONTEXTUAL_RETRIEVAL",
        "USAR_EMBEDDINGS_IMAGEN",
        "USAR_RERANKER",
    )

    def __init__(self, data_dir):
        self.DATA_DIR = data_dir
        self.MODELO_RAG = "default-rag"
        self.MODELO_CHAT = "default-chat"
        self.MODELO_CONTEXTUAL = "default-contextual"
        self.MODELO_RECOMP = "default-recomp"
        self.USAR_CONTEXTUAL_RETRIEVAL = True
        self.USAR_EMBEDDINGS_IMAGEN = True
        self.USAR_RERANKER = True
        self.docs_folder_switch_calls = []

    def set_model_roles_runtime(self, overrides):
        for role, model in overrides.items():
            setattr(self, self.MODEL_ROLE_VARS[role], model)
        return {role: getattr(self, var) for role, var in self.MODEL_ROLE_VARS.items()}

    def set_docs_folder_runtime(self, folder):
        # Present so a bug that tries to adopt active_store would be caught
        # by a test asserting this was never called, rather than silently
        # swallowed by _load_persisted_settings's top-level try/except.
        self.docs_folder_switch_calls.append(folder)


def _write_settings(data_dir: str, data: dict) -> None:
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "settings.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_adopts_persisted_roles_and_flags_but_not_the_active_store(tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_CONTEXTUAL_MODEL", raising=False)
    engine = _FakeRagEngine(str(tmp_path))
    _write_settings(str(tmp_path), {
        "roles": {"contextual": "other-model"},
        "flags": {"USAR_CONTEXTUAL_RETRIEVAL": False},
        "active_store": "es",
    })
    cli = MonkeyGrabCLI(engine)

    cli._load_persisted_settings()

    assert engine.MODELO_CONTEXTUAL == "other-model"
    assert engine.USAR_CONTEXTUAL_RETRIEVAL is False
    # Point 2 of the task: adopting the store is a much bigger behavioural
    # change than adopting a role or a flag, and the CLI has no picker for
    # it -- it must keep following DOCS_FOLDER, not settings.json.
    assert engine.docs_folder_switch_calls == []


def test_an_explicit_env_var_beats_a_persisted_role(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_CONTEXTUAL_MODEL", "env-model")
    engine = _FakeRagEngine(str(tmp_path))
    _write_settings(str(tmp_path), {"roles": {"contextual": "from-settings-json"}})
    cli = MonkeyGrabCLI(engine)

    cli._load_persisted_settings()

    # The fake never re-derives MODELO_CONTEXTUAL from the env var itself
    # (rag.chat_pdfs does that at import time); what matters here is that the
    # persisted value was *not* applied over whatever was already there.
    assert engine.MODELO_CONTEXTUAL == "default-contextual"


def test_survives_a_corrupt_settings_file(tmp_path):
    os.makedirs(tmp_path, exist_ok=True)
    (tmp_path / "settings.json").write_text("{not valid json", encoding="utf-8")
    engine = _FakeRagEngine(str(tmp_path))
    cli = MonkeyGrabCLI(engine)

    cli._load_persisted_settings()  # must not raise

    assert engine.MODELO_CONTEXTUAL == "default-contextual"


def test_is_a_noop_with_no_settings_file(tmp_path):
    engine = _FakeRagEngine(str(tmp_path))
    cli = MonkeyGrabCLI(engine)

    cli._load_persisted_settings()  # must not raise

    assert engine.MODELO_CONTEXTUAL == "default-contextual"
    assert engine.USAR_CONTEXTUAL_RETRIEVAL is True


def test_run_adopts_settings_before_opening_the_collection(tmp_path, monkeypatch):
    """The order matters, not just that the adoption happens somewhere:
    obtener_vector_store() and _check_index_fingerprint() must see the
    adopted config, or the fix only applies one query too late. Drives the
    real run() rather than calling _load_persisted_settings() directly, so a
    change to run()'s call order would actually fail this test.
    """
    engine = _FakeRagEngine(str(tmp_path))
    engine.CARPETA_DOCS = str(tmp_path)
    engine.OLLAMA_BASE_URL = "http://localhost:11434"
    _write_settings(str(tmp_path), {"flags": {"USAR_CONTEXTUAL_RETRIEVAL": False}})

    seen = {}

    class _FakeCollection:
        def count(self):
            return 0

    def _fake_obtener_vector_store():
        seen["USAR_CONTEXTUAL_RETRIEVAL"] = engine.USAR_CONTEXTUAL_RETRIEVAL
        return _FakeCollection()

    engine.obtener_vector_store = _fake_obtener_vector_store
    engine.indexar_documentos = lambda *a, **k: 0  # 0 chunks -> run() returns before the input loop

    cli = MonkeyGrabCLI(engine)
    monkeypatch.setattr(cli, "_ollama_health", lambda: (True, "ok"))
    monkeypatch.setattr("rag.cli.app.ui.logo", lambda: None)
    monkeypatch.setattr("rag.cli.app.ui.ollama_status", lambda *a, **k: None)
    monkeypatch.setattr("rag.cli.app.ui.no_pdfs", lambda *a, **k: None)
    monkeypatch.setattr("rag.cli.app.ui.warning", lambda *a, **k: None)

    cli.run()

    assert seen["USAR_CONTEXTUAL_RETRIEVAL"] is False
