"""Stable error codes from /api/rag and /api/chat (issue #379).

The UI translates on ``error`` and keeps ``message`` as the human fallback for
clients without i18n. What these tests pin is the distinction the UI cannot
guess from prose: embeddings run on llama-server, a different process from
Ollama, so an embedder outage must not be reported as "Ollama is down" and
send the user to the wrong control panel.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.web import app as web  # noqa: E402


@pytest.fixture
def client():
    web.app.config["TESTING"] = True
    with web.app.test_client() as c:
        yield c


class TestErrorClassification:
    def test_a_llama_server_failure_is_an_embedder_failure(self):
        payload = web._pipeline_error_payload(
            RuntimeError(
                "http://127.0.0.1:8080: could not reach llama-server: "
                "[Errno 111] Connection refused"
            )
        )

        assert payload["ok"] is False
        assert payload["error"] == "embedder_unavailable"
        assert payload["message"]

    def test_an_ollama_failure_keeps_its_own_code(self):
        payload = web._pipeline_error_payload(
            RuntimeError("Ollama generate failed for model 'x': connection refused")
        )

        assert payload["error"] == "model_unavailable"
        assert payload["message"]

    def test_an_unclassified_failure_keeps_the_generic_code(self):
        payload = web._pipeline_error_payload(ValueError("bad shape"))

        assert payload["error"] == "generation_error"
        assert payload["message"]


class TestTheRoutesCarryTheCode:
    def test_rag_retrieval_failure_names_the_embedder(self, client, monkeypatch):
        monkeypatch.setattr(web, "_get_collection", lambda: object())

        def _boom(*_args, **_kwargs):
            raise RuntimeError(
                "http://127.0.0.1:8080: could not reach llama-server: refused"
            )

        monkeypatch.setattr(web.rag_engine, "realizar_busqueda_hibrida", _boom)

        response = client.post(
            "/api/rag", json={"message": "una pregunta suficientemente larga"}
        )

        assert response.status_code == 200
        body = response.get_json()
        assert body["ok"] is False
        assert body["error"] == "embedder_unavailable"
        assert body["message"]

    def test_chat_stream_failure_carries_a_code_and_a_message(self, client, monkeypatch):
        def _boom(_pregunta):
            raise RuntimeError("Ollama generate failed for model 'x': connection refused")

        monkeypatch.setattr(web, "_chat_stream", _boom)

        response = client.post("/api/chat", json={"message": "hola"})
        text = response.get_data(as_text=True)

        assert response.status_code == 200
        assert "event: error" in text
        assert '"error": "model_unavailable"' in text
        assert '"message":' in text
