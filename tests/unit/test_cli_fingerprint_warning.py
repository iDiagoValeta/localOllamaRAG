"""Tests for the CLI surfacing an index-fingerprint mismatch (issue #36).

Exercises ``MonkeyGrabCLI._check_index_fingerprint`` directly rather than
``run()``: the full startup path also does an Ollama health check and drives
the Rich console, none of which this decision depends on.

Imports rag.cli.app (pulls in rich through rag.cli.display), so this file is
skipped by the dependency-free fast CI gate -- see tests/conftest.py.
"""

from rag.cli.app import MonkeyGrabCLI


class _FakeCollection:
    pass


class _FakeRagEngine:
    """Just enough of the rag_engine surface for __init__ and the check."""

    def __init__(self, mismatch):
        self._mismatch = mismatch

    def index_fingerprint_mismatch(self, collection):
        assert isinstance(collection, _FakeCollection)
        return self._mismatch


def _cli(mismatch: bool) -> MonkeyGrabCLI:
    cli = MonkeyGrabCLI(_FakeRagEngine(mismatch))
    cli.collection = _FakeCollection()
    return cli


def test_warns_when_the_index_no_longer_matches_the_config(monkeypatch):
    warnings = []
    monkeypatch.setattr("rag.cli.app.ui.warning", lambda msg: warnings.append(msg))

    _cli(mismatch=True)._check_index_fingerprint()

    assert len(warnings) == 1


def test_stays_silent_when_the_index_matches(monkeypatch):
    warnings = []
    monkeypatch.setattr("rag.cli.app.ui.warning", lambda msg: warnings.append(msg))

    _cli(mismatch=False)._check_index_fingerprint()

    assert warnings == []
