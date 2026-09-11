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

import rag.chat_pdfs  # noqa: E402,F401  (the facade must load before its engine modules)
from rag.engine import generation  # noqa: E402


def test_the_silent_path_marks_context_then_generator(monkeypatch):
    seen = []
    stats = {}

    def _prepare(pregunta, fragmentos):
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
