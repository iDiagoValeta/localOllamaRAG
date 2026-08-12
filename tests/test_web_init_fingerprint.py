"""Tests for the web app surfacing an index-fingerprint mismatch (issue #36).

Exercises ``_api_init_logic`` with ``_get_collection`` and the fingerprint
check doubled, rather than spinning up the real indexing stack: what matters
here is the shape of the response the frontend banner reads, and that a
mismatch never triggers automatic reindexing -- both isolated from retrieval
and indexing infrastructure already covered elsewhere.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.web import app as web_app  # noqa: E402


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


def _wire_common(monkeypatch, store, mismatch):
    monkeypatch.setattr(web_app, "_get_collection", lambda: store)
    monkeypatch.setattr(web_app.rag_engine, "obtener_documentos_indexados", lambda coll: ["a.pdf"])
    monkeypatch.setattr(web_app.rag_engine, "cargar_historial", lambda: [])
    monkeypatch.setattr(web_app.rag_engine, "index_fingerprint_mismatch", lambda coll: mismatch)


def test_init_reports_a_stale_index_without_reindexing(monkeypatch):
    _reset_state(monkeypatch)
    _wire_common(monkeypatch, _FakeStore(count=5), mismatch=True)

    def _must_not_reindex(*_a, **_kw):
        raise AssertionError("a fingerprint mismatch must not trigger automatic reindexing")

    monkeypatch.setattr(web_app.rag_engine, "indexar_documentos", _must_not_reindex)

    resp, status = web_app._api_init_logic()

    assert status == 200
    assert resp["fingerprint_stale"] is True


def test_init_does_not_report_a_matching_index(monkeypatch):
    _reset_state(monkeypatch)
    _wire_common(monkeypatch, _FakeStore(count=5), mismatch=False)

    resp, status = web_app._api_init_logic()

    assert status == 200
    assert resp["fingerprint_stale"] is False


# Toggling a pipeline flag or a model role is the most likely real-world path
# to a stale index: unlike a settings-file edit, it happens *inside* a running
# session, with no restart and no store switch to trigger a fresh /api/init
# check. Both handlers must refresh the flag they hand back, or the banner
# stays hidden for the rest of the session even though the index and the
# active config have just, provably, diverged.


def test_settings_change_reports_the_resulting_mismatch(monkeypatch):
    _reset_state(monkeypatch)
    # api_settings_post setattr's this straight onto rag.chat_pdfs -- pin the
    # starting value so monkeypatch restores it afterwards regardless of what
    # the request sets it to (real mutation, not itself monkeypatched).
    monkeypatch.setattr(web_app.rag_engine, "USAR_CONTEXTUAL_RETRIEVAL", True)
    monkeypatch.setattr(web_app, "_get_collection", lambda: _FakeStore(count=5))
    monkeypatch.setattr(web_app, "_save_persisted_settings", lambda: None)
    monkeypatch.setattr(web_app.rag_engine, "index_fingerprint_mismatch", lambda coll: True)

    client = web_app.app.test_client()
    resp = client.post("/api/settings", json={"contextualRetrieval": False})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["fingerprint_stale"] is True


def test_settings_change_does_not_report_a_mismatch_when_matching(monkeypatch):
    _reset_state(monkeypatch)
    monkeypatch.setattr(web_app.rag_engine, "USAR_CONTEXTUAL_RETRIEVAL", True)
    monkeypatch.setattr(web_app, "_get_collection", lambda: _FakeStore(count=5))
    monkeypatch.setattr(web_app, "_save_persisted_settings", lambda: None)
    monkeypatch.setattr(web_app.rag_engine, "index_fingerprint_mismatch", lambda coll: False)

    client = web_app.app.test_client()
    resp = client.post("/api/settings", json={"contextualRetrieval": True})

    assert resp.status_code == 200
    assert resp.get_json()["fingerprint_stale"] is False


def test_model_role_change_reports_the_resulting_mismatch(monkeypatch):
    _reset_state(monkeypatch)
    monkeypatch.setattr(web_app, "_get_collection", lambda: _FakeStore(count=5))
    monkeypatch.setattr(web_app, "_save_persisted_settings", lambda: None)
    monkeypatch.setattr(web_app.rag_engine, "index_fingerprint_mismatch", lambda coll: True)
    monkeypatch.setattr(web_app.rag_engine, "set_model_roles_runtime", lambda overrides: {"contextual": "other-model"})

    client = web_app.app.test_client()
    resp = client.post("/api/models", json={"contextual": "other-model"})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["fingerprint_stale"] is True
