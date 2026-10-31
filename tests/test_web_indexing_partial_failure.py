"""Tests that a partial indexing failure is visible in web state and responses.

Issue #323: ``indexar_documentos`` used to swallow per-file failures, write the
fingerprint and return normally. The background re-index then reported a clean
run, so a corpus that lost a document looked exactly like a complete one.

The engine half of the fix lives in ``tests/unit/test_indexing_fingerprint.py``;
what is under test here is the web layer: it must keep the names of the failed
files and hand them back instead of claiming everything was indexed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.web import app as web_app


class _FakeStore:
    def count(self):
        return 1


def _reset_state(monkeypatch):
    for key, value in (
        ("indexing", True),
        ("indexing_failed", False),
        ("indexing_error", None),
        ("indexing_done_empty", False),
        ("indexing_progress", None),
        ("indexing_failed_files", []),
    ):
        monkeypatch.setitem(web_app._state, key, value)


def _partial_indexer(failed):
    """An indexer that indexes everything except the names in ``failed``."""

    def _run(folder, coll, solo_archivos=None, silent=False, progress_callback=None, **kwargs):
        on_file_failed = kwargs.get("on_file_failed")
        for name in failed:
            if on_file_failed is not None:
                on_file_failed(name, RuntimeError("extraction failed"))
        return 4

    return _run


def test_partial_failure_is_kept_in_web_state(monkeypatch):
    _reset_state(monkeypatch)
    monkeypatch.setattr(web_app, "_get_collection", lambda: _FakeStore())
    monkeypatch.setattr(
        web_app.rag_engine, "indexar_documentos", _partial_indexer(["bad.pdf"])
    )

    web_app._run_indexing_bg()

    # A partial run stored chunks and did not abort the app; it is a warning,
    # not the permanent failure that /api/init turns into an error screen.
    assert web_app._state["indexing_failed"] is False
    assert web_app._state["indexing_failed_files"] == ["bad.pdf"]
    assert web_app._state["indexing"] is False


def test_init_hands_the_failed_files_to_the_client(monkeypatch):
    _reset_state(monkeypatch)
    monkeypatch.setattr(web_app, "_get_collection", lambda: _FakeStore())
    monkeypatch.setattr(web_app.rag_engine, "obtener_documentos_indexados", lambda coll: ["ok.pdf"])
    monkeypatch.setattr(web_app.rag_engine, "cargar_historial", lambda: [])
    monkeypatch.setattr(web_app.rag_engine, "index_fingerprint_mismatch", lambda coll: False)
    monkeypatch.setattr(
        web_app.rag_engine, "indexar_documentos", _partial_indexer(["bad.pdf"])
    )

    web_app._run_indexing_bg()
    resp, status = web_app._api_init_logic()

    assert status == 200
    assert resp["failed_files"] == ["bad.pdf"]
