"""The silent generation path says which stage it is in, through ``stats``.

Issue #234: a budget-exhausted evaluation record could not say whether the
context stage (RECOMP synthesis on the auxiliary model, when the flag is on)
or the generator itself was running when the cap hit. ``run_eval.py`` bounds
``generar_respuesta_silenciosa`` as one opaque call and reads the marker off
the same ``stats`` dict Ollama's counters land in.

Lives here rather than under tests/unit because it imports the engine
(``rag.engine.generation`` pulls in wiring and the runtime globals), which
the dependency-free CI job cannot load.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag.chat_pdfs  # noqa: F401  (the facade must load before its engine modules)
from rag.engine import generation


def test_the_silent_path_marks_context_then_generator(monkeypatch):
    seen = []
    stats = {}

    def _prepare(pregunta, fragmentos, metricas_out=None):
        # Accepts the RECOMP out-parameter (issue #497); this test is about
        # stage ordering, not what the context stage reports.
        seen.append(("context", stats.get("stage")))
        return "user message"

    def _stream(mensaje_usuario, on_token=None, stats=None):
        seen.append(("generator", stats.get("stage")))
        stats.update({"eval_count": 1, "eval_duration": 1})
        return "answer"

    monkeypatch.setattr(generation.cfg, "_preparar_mensaje_usuario_rag", _prepare)
    monkeypatch.setattr(generation.cfg, "_generar_respuesta_stream", _stream)

    assert generation.generar_respuesta_silenciosa("q", [], stats=stats) == "answer"
    # Each stage sees its own marker already set when it starts, so a cap
    # that hits mid-stage reads the stage that was running.
    assert seen == [("context", "context"), ("generator", "generator")]
    # Ollama's counters land on top of the marker, not instead of it.
    assert stats == {"stage": "generator", "eval_count": 1, "eval_duration": 1}


def test_the_silent_path_still_works_with_no_stats_dict(monkeypatch):
    monkeypatch.setattr(generation.cfg, "_preparar_mensaje_usuario_rag", lambda q, f: "m")
    monkeypatch.setattr(
        generation.cfg, "_generar_respuesta_stream", lambda m, on_token=None, stats=None: "a"
    )
    assert generation.generar_respuesta_silenciosa("q", []) == "a"


class _FakeAnswer:
    def __init__(self, metrics):
        self._metrics = metrics

    def build_user_message(self, pregunta, fragments):
        return "message", dict(self._metrics)


def _wire_answer(monkeypatch, metrics):
    monkeypatch.setattr(generation.wiring, "answer", lambda collection, config: _FakeAnswer(metrics))
    monkeypatch.setattr(generation.wiring, "fragment_from_dict", lambda f: f)


def test_a_recomp_fallback_reaches_the_out_dict(monkeypatch):
    # Issue #497: the use case reports the fallback; the gate only sees it
    # if message preparation hands it on.
    _wire_answer(monkeypatch, {"recomp_used": False, "recomp_error": True})
    out = {}

    generation._preparar_mensaje_usuario_rag("q", [], metricas_out=out)

    assert out == {"recomp_used": False, "recomp_error": True}


def test_a_used_recomp_is_reported_with_no_error(monkeypatch):
    _wire_answer(monkeypatch, {"recomp_used": True})
    out = {}

    generation._preparar_mensaje_usuario_rag("q", [], metricas_out=out)

    assert out == {"recomp_used": True, "recomp_error": False}


def test_the_silent_path_hands_its_stats_to_message_preparation(monkeypatch):
    _wire_answer(monkeypatch, {"recomp_used": False, "recomp_error": True})
    monkeypatch.setattr(generation.cfg, "_preparar_mensaje_usuario_rag", generation._preparar_mensaje_usuario_rag)
    monkeypatch.setattr(generation.cfg, "_generar_respuesta_stream", lambda m, stats=None: "answer")
    stats = {}

    generation.generar_respuesta_silenciosa("q", [], stats=stats)

    assert stats["recomp_error"] is True and stats["recomp_used"] is False
