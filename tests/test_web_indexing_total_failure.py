"""Tests that a re-index failing on every file reaches the interface (issue #192).

The bug this pins: those per-file failures were caught and logged, the run
returned 0 chunks, and the web layer read that 0 as ``indexing_done_empty``.
The user got "no documents indexed" -- the same screen as a genuinely empty
corpus folder -- for a corpus that had six PDFs and lost all of them.

Doubles ``indexar_documentos`` rather than driving the real stack: what is
under test is which state the web layer lands in, not extraction itself, which
tests/unit/test_indexing_fingerprint.py covers.
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


def test_a_run_that_fails_on_every_file_reports_an_error_not_an_empty_corpus(monkeypatch):
    _reset_state(monkeypatch)
    monkeypatch.setattr(web_app, "_get_collection", lambda: _FakeStore())

    def _all_files_fail(*_a, **_kw):
        raise RuntimeError("Indexing failed on all 6 file(s) in rag/docs/en")

    monkeypatch.setattr(web_app.rag_engine, "indexar_documentos", _all_files_fail)

    web_app._run_indexing_bg()

    assert web_app._state["indexing_failed"] is True
    assert "all 6 file(s)" in web_app._state["indexing_error"]
    # The distinction the bug erased: an empty corpus and a corpus that failed
    # to index are different screens, and only one of them tells the user to
    # do something about it.
    assert web_app._state["indexing_done_empty"] is False
    assert web_app._state["indexing"] is False


def test_a_run_with_no_pdfs_still_reports_an_empty_corpus(monkeypatch):
    # The other side of the same branch: nothing to index is not a failure,
    # and must keep reaching the "no documents" screen rather than an error.
    _reset_state(monkeypatch)
    monkeypatch.setattr(web_app, "_get_collection", lambda: _FakeStore())
    monkeypatch.setattr(web_app.rag_engine, "indexar_documentos", lambda *_a, **_kw: 0)

    web_app._run_indexing_bg()

    assert web_app._state["indexing_failed"] is False
    assert web_app._state["indexing_error"] is None
    assert web_app._state["indexing_done_empty"] is True
