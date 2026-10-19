"""CHAT mode caps its generation like the RAG role (issue #371).

The request is doubled at the HTTP call of the chat role's Ollama adapter,
which is where _chat_stream now goes through.

Loose test: importing rag.web.app needs the engine stack, so it cannot join the
dependency-free CI job.
"""

import json

from monkeygrab.adapters.chat import ollama_chat as ollama_chat_module
from rag.web import app as web_app


class _Response:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self._lines)


def test_chat_stream_sends_a_num_predict_cap(monkeypatch):
    calls = []

    def fake_post(url, json=None, **kwargs):
        calls.append({"url": url, "payload": json})
        return _Response([_line({"response": "hola", "done": False}), _line({"response": "", "done": True})])

    monkeypatch.setenv("MONKEYGRAB_CHAT_BACKEND", "ollama")
    monkeypatch.setattr(ollama_chat_module.requests, "post", fake_post)
    monkeypatch.setitem(web_app._state, "historial_chat", [])

    tokens = list(web_app._chat_stream("¿Qué tal?"))

    assert tokens == ["hola"]
    assert calls[0]["payload"]["options"]["num_predict"] == 4096


def _line(data):
    return json.dumps(data).encode("utf-8")
