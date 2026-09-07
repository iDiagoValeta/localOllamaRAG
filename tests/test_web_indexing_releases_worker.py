"""Tests that a failed indexing run releases the jina-clip worker (issue #191).

The observed shape: a re-index failed on every file, and seven minutes later
the worker was still alive holding 1784 MiB. The next attempt then failed with
a *different* error than the first -- CUDA OOM naming the orphan's PID -- so
the second failure blamed memory the first failure was holding.

Doubles the engine entirely: what is under test is which lifecycle call the
web layer makes on each path, not indexing or CUDA.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.web import app as web_app  # noqa: E402


class _FakeStore:
    def count(self):
        return 0


def _reset_state(monkeypatch):
    for key, value in (
        ("indexing", True),
        ("indexing_failed", False),
        ("indexing_error", None),
        ("indexing_done_empty", False),
        ("indexing_progress", None),
    ):
        monkeypatch.setitem(web_app._state, key, value)


def _track_release(monkeypatch):
    calls = []
    monkeypatch.setattr(web_app.rag_engine, "release_embedder", lambda: calls.append(1))
    monkeypatch.setattr(web_app, "_get_collection", lambda: _FakeStore())
    return calls


def test_a_failed_indexing_run_releases_the_worker(monkeypatch):
    _reset_state(monkeypatch)
    released = _track_release(monkeypatch)

    def _boom(*_a, **_kw):
        raise RuntimeError("Indexing failed on all 6 file(s)")

    monkeypatch.setattr(web_app.rag_engine, "indexar_documentos", _boom)

    web_app._run_indexing_bg()

    assert released == [1], "the worker holding VRAM must not outlive the run that failed"
    assert web_app._state["indexing_failed"] is True


def test_a_successful_indexing_run_keeps_the_worker(monkeypatch):
    # Deliberate asymmetry: that worker is the one the next query will use,
    # and reloading jina-clip costs ~29s. Releasing here would trade an
    # orphaned-VRAM bug for a latency one.
    _reset_state(monkeypatch)
    released = _track_release(monkeypatch)
    monkeypatch.setattr(web_app.rag_engine, "indexar_documentos", lambda *_a, **_kw: 885)

    web_app._run_indexing_bg()

    assert released == []
    assert web_app._state["indexing_failed"] is False
