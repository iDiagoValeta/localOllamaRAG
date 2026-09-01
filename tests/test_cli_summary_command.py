"""/resumen reaches its own paths without raising (issue #148).

`rag/cli/app.py` called `self._s(...)` in nine places for a `_s` that lives on
the `Display` class, not on the CLI. Every branch of `/resumen` hit at least
one, so the command raised `AttributeError` before doing anything -- and no
test constructed `MonkeyGrabCLI` at all, so nothing went red.

These tests are that missing coverage: the wiring between a use case and the
command that calls it. They assert about paths completing and the right engine
call happening, not about wording, because pinning the strings would just move
the untested gap somewhere else.

Imports rag.cli.app (pulls in rich through rag.cli.display), so this file is
skipped by the dependency-free fast CI gate -- see tests/conftest.py.
"""

import pytest

from rag.cli.app import MonkeyGrabCLI


class _FakeRagEngine:
    from monkeygrab.application.study import MalformedSummaryError

    def __init__(self, fragments=None, raises=None):
        self._fragments = [{"doc": "text"}] if fragments is None else fragments
        self._raises = raises
        self.summarised = 0

    def index_fingerprint_mismatch(self, collection):
        return False

    def fragmentos_de_documento(self, nombre):
        return self._fragments

    def resumir_fragmentos(self, fragmentos, idioma=None):
        self.summarised += 1
        if self._raises:
            raise self._raises
        return {"source_document": "paper.pdf", "sections": []}


def _cli(engine, docs=(("paper.pdf",),)):
    cli = MonkeyGrabCLI(engine)
    cli.collection = object()
    cli._get_document_summaries = lambda: [{"source": d} for (d,) in docs]
    return cli


@pytest.fixture(autouse=True)
def _silent_ui(monkeypatch):
    """Swallow the output; these tests are about not raising, not about text."""
    for name in ("info", "error", "warning", "summary_panel"):
        monkeypatch.setattr(f"rag.cli.app.ui.{name}", lambda *a, **k: None)


def test_no_indexed_documents_reports_instead_of_raising():
    engine = _FakeRagEngine()
    assert _cli(engine, docs=())._cmd_summary("") is False
    assert engine.summarised == 0


def test_a_named_document_reaches_the_summariser():
    engine = _FakeRagEngine()
    assert _cli(engine)._cmd_summary("paper.pdf") is False
    assert engine.summarised == 1


def test_a_positional_argument_picks_from_the_list():
    engine = _FakeRagEngine()
    assert _cli(engine)._cmd_summary("1") is False
    assert engine.summarised == 1


def test_a_number_out_of_range_reports_instead_of_raising():
    engine = _FakeRagEngine()
    assert _cli(engine)._cmd_summary("99") is False
    assert engine.summarised == 0


def test_a_name_matching_nothing_reports_instead_of_raising():
    engine = _FakeRagEngine()
    assert _cli(engine)._cmd_summary("nada-coincide") is False
    assert engine.summarised == 0


def test_an_ambiguous_name_reports_instead_of_raising():
    engine = _FakeRagEngine()
    cli = _cli(engine, docs=(("planck-a.pdf",), ("planck-b.pdf",)))
    assert cli._cmd_summary("planck") is False
    assert engine.summarised == 0


def test_a_document_with_no_fragments_reports_instead_of_raising():
    engine = _FakeRagEngine(fragments=[])
    assert _cli(engine)._cmd_summary("paper.pdf") is False
    assert engine.summarised == 0


def test_a_malformed_summary_reports_instead_of_raising():
    engine = _FakeRagEngine(raises=_FakeRagEngine.MalformedSummaryError("not json"))
    assert _cli(engine)._cmd_summary("paper.pdf") is False


def test_an_unexpected_generator_failure_does_not_end_the_session():
    # False keeps the REPL running. A traceback here would close the session
    # over one failed command.
    engine = _FakeRagEngine(raises=RuntimeError("ollama is down"))
    assert _cli(engine)._cmd_summary("paper.pdf") is False


def test_a_bare_command_prompts_and_honours_a_cancel(monkeypatch):
    monkeypatch.setattr("rag.cli.app.ui.ask", lambda *a, **k: "")
    engine = _FakeRagEngine()
    assert _cli(engine)._cmd_summary("") is False
    assert engine.summarised == 0


def test_a_bare_command_prompts_and_uses_the_answer(monkeypatch):
    monkeypatch.setattr("rag.cli.app.ui.ask", lambda *a, **k: "paper.pdf")
    engine = _FakeRagEngine()
    assert _cli(engine)._cmd_summary("") is False
    assert engine.summarised == 1
