"""RECOMP synthesis metrics survive the trip to the caller's ``stats`` dict.

Issue #497: ``Answer._synthesize_recomp`` already marks ``recomp_used`` and
``recomp_error`` on the metrics it returns to ``build_user_message``, but
``_preparar_mensaje_usuario_rag`` dropped them. The evaluation gate bounds
``generar_respuesta_silenciosa`` as one call and reads its ``stats`` dict, so
a RECOMP fallback was invisible in the artifact even though the use case knew
about it.

Lives here rather than under tests/unit because it imports the engine
(``rag.engine.generation`` pulls in wiring and the runtime globals), which the
dependency-free CI job cannot load.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag.chat_pdfs  # noqa: E402,F401  (the facade must load before its engine modules)
from rag.engine import generation  # noqa: E402

from monkeygrab.application.answer import Answer  # noqa: E402
from monkeygrab.config.app_config import AppConfig  # noqa: E402
from monkeygrab.domain.generation_chunk import GenerationChunk  # noqa: E402

_FRAGMENTS = [
    {"doc": "evidence text " * 10, "metadata": {"source": "p.pdf", "page": 1, "chunk": 0}}
]


class _RagModel:
    """Only ``stream`` is reached here; the RECOMP decision happens before it."""

    def stream(self, prompt, *, system=None):
        yield GenerationChunk(text="answer", done=False)
        yield GenerationChunk(done=True)


class _RecompThatFails:
    """What a request over ``recomp_num_ctx`` does: the adapter raises."""

    def generate(self, prompt, *, system=None, images=(), response_format=None):
        raise RuntimeError("HTTP 400: context too long")


class _RecompThatAnswers:
    def generate(self, prompt, *, system=None, images=(), response_format=None):
        return "## Facts relevant to the question\n- a grounded fact"


def _wire(monkeypatch, recomp_model):
    config = AppConfig().with_overrides(
        **{"flags.usar_recomp_synthesis": True, "flags.expandir_contexto": False}
    )
    use_case = Answer(None, _RagModel(), config, recomp_chat_model=recomp_model)
    monkeypatch.setattr(generation.wiring, "app_config_from_runtime", lambda: config)
    monkeypatch.setattr(generation.wiring, "answer", lambda store, cfg: use_case)
    return config


def test_a_failed_recomp_call_reaches_stats_as_the_raw_context_fallback(monkeypatch):
    _wire(monkeypatch, _RecompThatFails())
    stats = {}

    generation.generar_respuesta_silenciosa("q", _FRAGMENTS, stats=stats)

    assert stats["recomp_used"] is False
    assert stats["recomp_error"] is True


def test_a_successful_recomp_call_is_marked_used_without_an_error(monkeypatch):
    _wire(monkeypatch, _RecompThatAnswers())
    stats = {}

    generation.generar_respuesta_silenciosa("q", _FRAGMENTS, stats=stats)

    assert stats["recomp_used"] is True
    assert stats["recomp_error"] is False


def test_recomp_is_reported_unused_when_the_flag_is_off(monkeypatch):
    config = AppConfig().with_overrides(
        **{"flags.usar_recomp_synthesis": False, "flags.expandir_contexto": False}
    )
    use_case = Answer(None, _RagModel(), config)
    monkeypatch.setattr(generation.wiring, "app_config_from_runtime", lambda: config)
    monkeypatch.setattr(generation.wiring, "answer", lambda store, cfg: use_case)
    stats = {}

    generation.generar_respuesta_silenciosa("q", _FRAGMENTS, stats=stats)

    assert stats["recomp_used"] is False
    assert stats["recomp_error"] is False


def test_the_out_parameter_is_optional_and_does_not_change_the_message(monkeypatch):
    _wire(monkeypatch, _RecompThatAnswers())

    without = generation._preparar_mensaje_usuario_rag("q", _FRAGMENTS)
    stats = {}
    with_out = generation._preparar_mensaje_usuario_rag("q", _FRAGMENTS, metricas_out=stats)

    assert without == with_out
    assert stats == {"recomp_used": True, "recomp_error": False}
