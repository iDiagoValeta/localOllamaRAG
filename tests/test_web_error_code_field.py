"""Every failure the UI can show carries a stable ``code`` (issue #555).

``error`` stays what it was, Spanish prose for most routes, because API
clients and older tests read it. The UI cannot translate prose, so it maps
``code`` to an i18n key and only falls back to ``error`` for a code it does
not know. What these tests pin is that each route names its failure: a route
that drops ``code`` silently brings back Spanish text in an English session.

Lives with the loose tests because it imports the Flask app, which pulls the
engine stack the no-infra CI job does not have.
"""

import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.web import app as web


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web.rag_engine, "CARPETA_DOCS", str(tmp_path / "docs"))
    monkeypatch.setitem(web._state, "indexing", False)
    web.app.config["TESTING"] = True
    with web.app.test_client() as c:
        yield c


def _boom(*_args, **_kwargs):
    raise RuntimeError("boom")


class TestChatAndMode:
    def test_empty_chat_message(self, client):
        response = client.post("/api/chat", json={"message": "  "})

        assert response.status_code == 400
        assert response.json["code"] == "empty_message"
        assert response.json["error"] == "Mensaje vacío"

    def test_chat_without_streaming(self, client):
        response = client.post("/api/chat", json={"message": "hola", "stream": False})

        assert response.status_code == 400
        assert response.json["code"] == "stream_required"
        assert response.json["error"] == "Usar stream=true"

    def test_invalid_mode(self, client):
        response = client.post("/api/mode", json={"mode": "nope"})

        assert response.status_code == 400
        assert response.json["code"] == "invalid_mode"
        assert response.json["error"] == "Modo inválido"


class TestInit:
    def test_a_failed_index_run_is_indexing_failed(self, client, monkeypatch):
        monkeypatch.setattr(web, "_invalidate_collection_if_deleted", lambda: None)
        monkeypatch.setitem(web._state, "indexing_failed", True)
        monkeypatch.setitem(web._state, "indexing_error", "all files failed")

        response = client.get("/api/init")

        assert response.status_code == 500
        assert response.json["code"] == "indexing_failed"
        assert response.json["error"] == "all files failed"
        assert response.json["detail"] == "all files failed"

    def test_a_failed_run_with_no_message_has_no_detail(self, client, monkeypatch):
        # The UI shows its own translated line when there is no detail, so the
        # Spanish fallback in `error` must not be mistaken for one.
        monkeypatch.setattr(web, "_invalidate_collection_if_deleted", lambda: None)
        monkeypatch.setitem(web._state, "indexing_failed", True)
        monkeypatch.setitem(web._state, "indexing_error", "")

        response = client.get("/api/init")

        assert response.json["code"] == "indexing_failed"
        assert response.json["error"] == "La indexación falló"
        assert response.json["detail"] is None

    def test_a_run_in_progress_is_indexing_in_progress(self, client, monkeypatch):
        monkeypatch.setattr(web, "_invalidate_collection_if_deleted", lambda: None)
        monkeypatch.setattr(web, "_ensure_indexed", lambda: None)
        monkeypatch.setitem(web._state, "indexing_failed", False)
        monkeypatch.setitem(web._state, "indexing", True)
        monkeypatch.setitem(web._state, "indexing_progress", None)

        response = client.get("/api/init")

        assert response.status_code == 202
        assert response.json["code"] == "indexing_in_progress"

    def test_an_unrecoverable_init_is_init_failed(self, client, monkeypatch):
        monkeypatch.setattr(web, "_api_init_logic", _boom)

        response = client.get("/api/init")

        assert response.status_code == 500
        assert response.json["code"] == "init_failed"
        assert response.json["error"] == "boom"


class TestFilenames:
    def test_delete_with_an_invalid_name(self, client):
        response = client.delete("/api/docs/notes.txt")

        assert response.status_code == 400
        assert response.json["code"] == "invalid_filename"
        assert response.json["error"] == "Nombre de archivo inválido"

    def test_serving_an_invalid_name(self, client):
        response = client.get("/api/pdf/notes.txt")

        assert response.status_code == 400
        assert response.json["code"] == "invalid_filename"
        assert response.json["error"] == "Archivo inválido"

    def test_upload_with_an_invalid_name(self, client):
        response = client.post(
            "/api/docs", data={"file": (io.BytesIO(b"x"), "notes.txt")}
        )

        assert response.status_code == 400
        assert response.json["code"] == "invalid_filename"
        assert response.json["error"] == "Nombre de archivo inválido"


class TestOperationFailures:
    def test_delete_that_raises_is_delete_failed(self, client, monkeypatch):
        monkeypatch.setattr(web, "_get_collection", _boom)

        response = client.delete("/api/docs/paper.pdf")

        assert response.status_code == 500
        assert response.json["code"] == "delete_failed"
        assert response.json["error"] == "boom"

    def test_reindex_that_raises_is_reindex_failed(self, client, monkeypatch):
        monkeypatch.setattr(web, "_reset_db", _boom)

        response = client.post("/api/reindex")

        assert response.status_code == 500
        assert response.json["code"] == "reindex_failed"
        assert response.json["error"] == "boom"

    def test_an_upload_without_files(self, client):
        response = client.post("/api/docs", data={}, content_type="multipart/form-data")

        assert response.status_code == 400
        assert response.json["code"] == "no_file"

    def test_an_upload_while_indexing_runs(self, client, monkeypatch):
        monkeypatch.setitem(web._state, "indexing", True)

        response = client.post(
            "/api/docs", data={"file": (io.BytesIO(b"x"), "a.pdf")}, content_type="multipart/form-data"
        )

        assert response.status_code == 409
        assert response.json["code"] == "indexing_in_progress"

    def test_an_upload_that_raises_is_upload_failed(self, client, monkeypatch):
        monkeypatch.setattr(web, "_get_collection", _boom)

        response = client.post(
            "/api/docs", data={"file": (io.BytesIO(b"x"), "a.pdf")}, content_type="multipart/form-data"
        )

        assert response.status_code == 500
        assert response.json["code"] == "upload_failed"
        assert response.json["error"] == "boom"

    def test_ollama_start_that_raises(self, client, monkeypatch):
        monkeypatch.setattr(web, "_ollama_version", lambda: None)
        monkeypatch.setattr(web.shutil, "which", lambda _name: "/usr/bin/ollama")
        monkeypatch.setattr(web, "_start_ollama_process", _boom)

        response = client.post("/api/ollama/start")

        assert response.status_code == 500
        assert response.json["code"] == "ollama_start_failed"
        assert response.json["error"] == "boom"

    def test_ollama_models_that_raise(self, client, monkeypatch):
        monkeypatch.setattr(web, "_ollama_version", lambda: "0.1")
        monkeypatch.setattr(web, "_ollama_models", _boom)

        response = client.get("/api/ollama/models")

        assert response.status_code == 500
        assert response.json["code"] == "ollama_models_failed"
        assert response.json["error"] == "boom"

    def test_a_rejected_model_role_value(self, client, monkeypatch):
        def _reject(_overrides):
            raise ValueError("unknown model")

        monkeypatch.setattr(web.rag_engine, "set_model_roles_runtime", _reject)

        response = client.post("/api/models", json={"rag": "x"})

        assert response.status_code == 400
        assert response.json["code"] == "invalid_model_roles"
        assert response.json["error"] == "unknown model"


class TestStudyFailures:
    @pytest.fixture(autouse=True)
    def _engine(self, monkeypatch):
        monkeypatch.setattr(web, "_get_collection", lambda: object())
        monkeypatch.setattr(
            web.rag_engine, "obtener_documentos_indexados", lambda _coll: ["paper.pdf"]
        )
        monkeypatch.setattr(web.rag_engine, "fragmentos_de_documento", lambda _name: [{"t": 1}])

    def _study(self, client, **body):
        return client.post("/api/study", json={"document": "paper.pdf", **body})

    def test_a_malformed_artifact(self, client, monkeypatch):
        def _malformed(_fragments, **_kwargs):
            raise web.rag_engine.MalformedSummaryError("bad shape")

        monkeypatch.setattr(web.rag_engine, "resumir_fragmentos", _malformed)

        response = self._study(client, kind="summary")

        assert response.status_code == 422
        assert response.json["code"] == "study_malformed"
        assert response.json["kind"] == "malformed"

    def test_an_unusable_question_count(self, client, monkeypatch):
        monkeypatch.setattr(web.rag_engine, "cuestionario_de_fragmentos", _boom)

        response = self._study(client, kind="quiz", question_count="many")

        assert response.status_code == 400
        assert response.json["code"] == "study_invalid_request"

    def test_a_generator_failure(self, client, monkeypatch):
        monkeypatch.setattr(web.rag_engine, "resumir_fragmentos", _boom)

        response = self._study(client, kind="summary")

        assert response.status_code == 500
        assert response.json["code"] == "study_failed"
