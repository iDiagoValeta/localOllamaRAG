"""Tests that ``/api/init`` hands the persisted chat history to the frontend.

The backend reloads ``historial_chat.json`` on every init and builds the next
prompt from it; if the response omits the history the user sees an empty
conversation while the model answers with context that is no longer visible
(issue #403). These tests pin the response shape the frontend paints, with the
engine doubled so they stay independent of indexing and retrieval.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.web import app as web_app


class _FakeStore:
    def __init__(self, count):
        self._count = count

    def count(self):
        return self._count

    def get_page(self, *_a, **_kw):
        return []


def _reset_state(monkeypatch):
    for key, value in (
        ("collection", None),
        ("indexing", False),
        ("indexing_failed", False),
        ("indexing_error", None),
        ("indexing_done_empty", False),
        ("indexing_progress", None),
    ):
        monkeypatch.setitem(web_app._state, key, value)


def _wire_init(monkeypatch, historial):
    _reset_state(monkeypatch)
    monkeypatch.setattr(web_app, "_get_collection", lambda: _FakeStore(count=5))
    monkeypatch.setattr(web_app.rag_engine, "obtener_documentos_indexados", lambda coll: ["a.pdf"])
    monkeypatch.setattr(web_app.rag_engine, "index_fingerprint_mismatch", lambda coll: False)
    monkeypatch.setattr(web_app.rag_engine, "cargar_historial", lambda: historial)


def test_init_returns_the_persisted_history(monkeypatch):
    historial = [
        {"role": "user", "content": "¿De qué habla el paper?"},
        {"role": "assistant", "content": "Habla del método propuesto."},
    ]
    _wire_init(monkeypatch, historial)

    resp = web_app.app.test_client().get("/api/init")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["history"] == historial
    assert body["history_count"] == 2


def test_init_returns_history_when_the_index_is_empty(monkeypatch):
    historial = [
        {"role": "user", "content": "pregunta anterior"},
        {"role": "assistant", "content": "respuesta anterior"},
    ]
    _reset_state(monkeypatch)
    monkeypatch.setitem(web_app._state, "indexing_done_empty", True)
    monkeypatch.setattr(web_app, "_get_collection", lambda: _FakeStore(count=0))
    monkeypatch.setattr(web_app.rag_engine, "cargar_historial", lambda: historial)

    resp = web_app.app.test_client().get("/api/init")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["total_fragments"] == 0
    assert body["history"] == historial


def test_init_skips_entries_that_are_not_a_user_or_assistant_turn(monkeypatch):
    historial = [
        {"role": "user", "content": "pregunta"},
        {"role": "system", "content": "no es un turno del chat"},
        {"role": "assistant", "content": "respuesta"},
    ]
    _wire_init(monkeypatch, historial)

    resp = web_app.app.test_client().get("/api/init")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["history"] == [
        {"role": "user", "content": "pregunta"},
        {"role": "assistant", "content": "respuesta"},
    ]
