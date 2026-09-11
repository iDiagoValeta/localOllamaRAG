"""Turning the reranker off via /api/settings must release its VRAM (#239).

CrossEncoderReranker.release() existed to give back its CUDA weights once
reranking is done, but nothing called it: rag/engine/wiring.py cached the
reranker as a singleton with no reset path, so once loaded its weights sat
resident for the rest of the process's life -- including after a user
explicitly turns reranking off, at which point nothing will use them again
until it is turned back on. This is the one call site wired for the fix (see
rag/engine/wiring.py's release_reranker and rag_chat_model docstrings for why
the per-query retrieval path itself is deliberately left alone).

Doubles the engine entirely: what is under test is which lifecycle call the
web layer makes on this one settings transition, not reranking itself.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.web import app as web_app  # noqa: E402


class _FakeStore:
    def get_page(self, *_a, **_kw):
        return []


def _wire_common(monkeypatch):
    released = []
    monkeypatch.setattr(web_app.rag_engine, "release_reranker", lambda: released.append(1))
    monkeypatch.setattr(web_app.rag_engine, "guardar_ajustes_persistidos", lambda: None)
    monkeypatch.setattr(web_app.rag_engine, "index_fingerprint_mismatch", lambda coll: False)
    monkeypatch.setattr(web_app, "_get_collection", lambda: _FakeStore())
    monkeypatch.setattr(web_app.rag_engine, "RERANKER_AVAILABLE", True)
    monkeypatch.setattr(web_app.rag_engine, "USAR_RERANKER", True)
    return released


def test_turning_the_reranker_off_releases_its_vram(monkeypatch):
    released = _wire_common(monkeypatch)

    with web_app.app.test_client() as client:
        resp = client.post("/api/settings", json={"reranker": False})

    assert resp.status_code == 200
    assert resp.get_json()["settings"]["reranker"] is False
    assert released == [1]


def test_turning_the_reranker_on_does_not_release_anything(monkeypatch):
    released = _wire_common(monkeypatch)
    web_app.rag_engine.USAR_RERANKER = False

    with web_app.app.test_client() as client:
        resp = client.post("/api/settings", json={"reranker": True})

    assert resp.status_code == 200
    assert resp.get_json()["settings"]["reranker"] is True
    assert released == []


def test_leaving_other_flags_alone_does_not_touch_the_reranker(monkeypatch):
    released = _wire_common(monkeypatch)

    with web_app.app.test_client() as client:
        resp = client.post("/api/settings", json={"expandContext": False})

    assert resp.status_code == 200
    assert released == []
