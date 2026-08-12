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
