"""The three Study commands: /resumen, /esquema and /cuestionario.

Issue #140. The core builds the artifacts and is tested against a doubled
``ChatModel`` under ``tests/unit/application/``. What is left, and what these
cover, is the wiring: that each command reaches its own engine entry point,
renders its own panel, and reports its own failure -- because a quiz that
silently renders as a summary, or an outline failure reported with the
summary's wording, is a bug no core test can see.

Imports rag.cli.app (pulls in rich through rag.cli.display), so this file is
skipped by the dependency-free fast CI gate -- see tests/conftest.py.
"""

import pytest

from rag.cli.app import MonkeyGrabCLI


class _FakeCollection:
    pass


class _FakeRagEngine:
    """Just enough of the rag_engine surface for the three Study commands."""

    # The CLI catches these by identity, so the doubles must be the real
    # classes: catching a look-alike would pass here and miss in production.
    from monkeygrab.application.study import (
        MalformedOutlineError,
        MalformedQuizError,
        MalformedSummaryError,
    )

    def __init__(self, fragments=None, results=None, raises=None):
        self._fragments = [{"doc": "text"}] if fragments is None else fragments
        self._results = results or {}
        self._raises = raises or {}
        self.calls = []

    def index_fingerprint_mismatch(self, collection):
        return False

    def fragmentos_de_documento(self, nombre):
        self.calls.append(("fragmentos", nombre))
        return self._fragments

    def _artifact(self, kind, fragmentos, **kwargs):
        self.calls.append((kind, kwargs))
        if kind in self._raises:
            raise self._raises[kind]
        return self._results.get(kind, {})

    def resumir_fragmentos(self, fragmentos, idioma=None):
        return self._artifact("resumen", fragmentos, idioma=idioma)

    def esquema_de_fragmentos(self, fragmentos, idioma=None):
        return self._artifact("esquema", fragmentos, idioma=idioma)

    def cuestionario_de_fragmentos(self, fragmentos, idioma=None):
        return self._artifact("cuestionario", fragmentos, idioma=idioma)


def _cli(engine) -> MonkeyGrabCLI:
    cli = MonkeyGrabCLI(engine)
    cli.collection = _FakeCollection()
    # The picker is not what these tests are about; it has its own path.
    cli._get_document_summaries = lambda: [{"name": "paper.pdf"}]
    return cli


@pytest.fixture
def panels(monkeypatch):
    """Record which panel each command rendered, without drawing anything."""
    drawn = {}
    for name in ("summary_panel", "outline_panel", "quiz_panel"):
        monkeypatch.setattr(
            f"rag.cli.app.ui.{name}",
            lambda payload, _n=name: drawn.__setitem__(_n, payload),
        )
    monkeypatch.setattr("rag.cli.app.ui.info", lambda *a, **k: None)
    return drawn


@pytest.fixture
def errors(monkeypatch):
    shown = []
    monkeypatch.setattr("rag.cli.app.ui.error", lambda msg: shown.append(msg))
    monkeypatch.setattr("rag.cli.app.ui.info", lambda *a, **k: None)
    return shown


class TestEachCommandReachesItsOwnArtifact:
    @pytest.mark.parametrize(
        "command, kind, panel",
        [
            ("_cmd_summary", "resumen", "summary_panel"),
            ("_cmd_outline", "esquema", "outline_panel"),
            ("_cmd_quiz", "cuestionario", "quiz_panel"),
        ],
    )
    def test_the_right_engine_call_feeds_the_right_panel(self, panels, command, kind, panel):
        engine = _FakeRagEngine(results={kind: {"marker": kind}})
        getattr(_cli(engine), command)("paper.pdf")

        assert [c[0] for c in engine.calls] == ["fragmentos", kind]
        assert panels == {panel: {"marker": kind}}

    def test_all_three_are_registered_as_argument_commands(self):
        cli = _cli(_FakeRagEngine())
        for name in ("/resumen", "/esquema", "/cuestionario"):
            # Both registries: without the no-argument one a bare "/esquema"
            # falls through to the did-you-mean branch and suggests the
            # command the user just typed.
            assert name in cli._commands_with_argument
            assert name in cli._commands


class TestFailuresAreReportedAsTheirOwnArtifact:
    @pytest.mark.parametrize(
        "command, kind, error",
        [
            ("_cmd_summary", "resumen", _FakeRagEngine.MalformedSummaryError("bad")),
            ("_cmd_outline", "esquema", _FakeRagEngine.MalformedOutlineError("bad")),
            ("_cmd_quiz", "cuestionario", _FakeRagEngine.MalformedQuizError("bad")),
        ],
    )
    def test_a_malformed_reply_is_shown_not_swallowed(self, errors, panels, command, kind, error):
        engine = _FakeRagEngine(raises={kind: error})
        getattr(_cli(engine), command)("paper.pdf")

        assert len(errors) == 1
        # No panel: an empty artifact rendered as if it were real is exactly
        # what raising in the core exists to prevent.
        assert panels == {}

    def test_a_quiz_failure_does_not_borrow_the_summarys_wording(self, errors):
        quiz_engine = _FakeRagEngine(
            raises={"cuestionario": _FakeRagEngine.MalformedQuizError("bad")}
        )
        _cli(quiz_engine)._cmd_quiz("paper.pdf")
        summary_engine = _FakeRagEngine(
            raises={"resumen": _FakeRagEngine.MalformedSummaryError("bad")}
        )
        _cli(summary_engine)._cmd_summary("paper.pdf")

        assert errors[0] != errors[1]

    @pytest.mark.parametrize(
        "command, kind",
        [
            ("_cmd_summary", "resumen"),
            ("_cmd_outline", "esquema"),
            ("_cmd_quiz", "cuestionario"),
        ],
    )
    def test_an_unexpected_failure_still_reports_rather_than_crashing_the_repl(
        self, errors, panels, command, kind
    ):
        engine = _FakeRagEngine(raises={kind: RuntimeError("ollama is down")})
        # False = keep the loop running. A traceback here would end the
        # session over one failed command.
        assert getattr(_cli(engine), command)("paper.pdf") is False
        assert len(errors) == 1
        assert panels == {}


class TestNothingToWorkFrom:
    @pytest.mark.parametrize(
        "command", ["_cmd_summary", "_cmd_outline", "_cmd_quiz"]
    )
    def test_a_document_with_no_fragments_never_reaches_the_generator(
        self, errors, panels, command
    ):
        engine = _FakeRagEngine(fragments=[])
        getattr(_cli(engine), command)("paper.pdf")

        assert [c[0] for c in engine.calls] == ["fragmentos"]
        assert len(errors) == 1
        assert panels == {}


class TestThePickerAgreesWithTheProducer:
    """Issue #150: the picker read a key the summaries never carried.

    Every other test here stubs ``_get_document_summaries``, which is fine for
    what they check and useless for this: a double built from the caller's
    assumption reproduces the assumption. These two run the real method over a
    doubled collection, so the contract is anchored at the producer and a
    rename on either side goes red.
    """

    @staticmethod
    def _cli_over_real_summaries():
        from monkeygrab.domain.chunk_metadata import ChunkMetadata
        from monkeygrab.domain.fragment import Fragment

        class _Collection:
            def get_page(self, limit, offset):
                return [
                    Fragment(doc="t", metadata=ChunkMetadata(source="paper.pdf", page=0)),
                    Fragment(doc="t", metadata=ChunkMetadata(source="otro.pdf", page=1)),
                ]

        engine = _FakeRagEngine(results={"resumen": {"ok": True}})
        cli = MonkeyGrabCLI(engine)
        cli.collection = _Collection()
        return cli, engine

    def test_a_filename_from_the_real_summaries_reaches_the_generator(self, panels):
        cli, engine = self._cli_over_real_summaries()
        cli._cmd_summary("paper.pdf")
        assert ("resumen", {"idioma": cli._summary_language()}) in engine.calls

    def test_the_summaries_expose_the_key_the_picker_reads(self):
        cli, _ = self._cli_over_real_summaries()
        docs = cli._get_document_summaries()
        assert docs, "the doubled collection should produce summaries"
        # Stated as the contract rather than as a string comparison in the
        # picker: this is the line issue #150 crossed.
        assert all("name" in d for d in docs)
