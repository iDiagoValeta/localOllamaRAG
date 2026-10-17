"""CHAT mode caps its generation like the RAG role (issue #371).

Loose test: importing rag.web.app needs the engine stack, so it cannot join the
dependency-free CI job.
"""

from monkeygrab.adapters.chat import ollama_chat as ollama_chat_module
from rag.web import app as web_app


def test_chat_stream_sends_a_num_predict_cap(monkeypatch):
    calls = []

    class _Client:
        def chat(self, **kwargs):
            calls.append(kwargs)
            return iter([{"message": {"content": "hola"}}])

    monkeypatch.setattr(ollama_chat_module, "ollama_client_for", lambda *_a, **_kw: _Client())
    monkeypatch.setitem(web_app._state, "historial_chat", [])

    tokens = list(web_app._chat_stream("¿Qué tal?"))

    assert tokens == ["hola"]
    assert calls[0]["options"]["num_predict"] == 4096
