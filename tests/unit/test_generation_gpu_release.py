"""Regression tests for #25: the RAG generation entry points must release the
GPU-resident embedder/reranker before Ollama is asked to load the generator.

rag/engine/generation.py's generar_tokens_respuesta is the one function every
RAG generation call funnels through:

- the web app's streaming reply iterates it directly (rag/web/app.py's
  api_rag);
- the web app's non-streaming reply and the CLI's generar_respuesta both
  reach it through _generar_respuesta_stream.

/chat mode (both interfaces) never imports this module at all -- it calls
Ollama directly -- so it needs no test here; there is nothing for it to call.

Everything below is a double: no real Ollama, embedder or reranker is
constructed, so this runs without a GPU (it still needs the project's real
dependencies importable, same as the rest of tests/unit that `import rag`;
see tests/conftest.py's collect_ignore for why the dependency-free CI job
skips this file entirely rather than running it against stubs).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag.chat_pdfs  # noqa: F401 -- import before rag.engine.generation, see wiring's module docstring
from rag.engine import generation, wiring


class _FakeChunk:
    """Minimal stand-in for monkeygrab.application.answer.GenerationChunk."""

    _STAT_FIELDS = (
        "model", "done_reason", "total_duration", "load_duration",
        "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration",
    )

    def __init__(self, text, done=False):
        self.text = text
        self.done = done
        for field in self._STAT_FIELDS:
            setattr(self, field, None)


class _FakeUseCase:
    """Stand-in for monkeygrab.application.answer.Answer.

    ``stream`` records that it was entered, so a test can assert
    release_gpu_models ran strictly before it -- the whole point of #25 is
    ordering, not just that the hook exists somewhere in the call chain.
    """

    def __init__(self, events):
        self._events = events

    def build_user_message(self, pregunta, fragments):
        del pregunta, fragments
        return "mensaje", {}

    def stream(self, mensaje_usuario):
        del mensaje_usuario
        self._events.append("stream_started")
        yield _FakeChunk("hola")
        yield _FakeChunk("", done=True)


def _patch_generation(monkeypatch, events):
    monkeypatch.setattr(wiring, "app_config_from_runtime", lambda: object())
    monkeypatch.setattr(wiring, "answer", lambda *args, **kwargs: _FakeUseCase(events))
    monkeypatch.setattr(wiring, "release_gpu_models", lambda: events.append("released"))


def test_generar_tokens_respuesta_releases_before_streaming_web_streaming_path(monkeypatch):
    """api_rag's SSE path: `for token in rag_engine.generar_tokens_respuesta(...)`."""
    events = []
    _patch_generation(monkeypatch, events)

    tokens = list(generation.generar_tokens_respuesta("mensaje"))

    assert events == ["released", "stream_started"]
    assert tokens == ["hola"]


def test_generar_respuesta_stream_releases_before_streaming_web_nonstreaming_path(monkeypatch):
    """api_rag's non-streaming path calls _generar_respuesta_stream directly."""
    events = []
    _patch_generation(monkeypatch, events)

    respuesta = generation._generar_respuesta_stream("mensaje")

    assert events == ["released", "stream_started"]
    assert respuesta == "hola"


def test_generar_respuesta_releases_before_streaming_cli_path(monkeypatch):
    """rag/cli/app.py's _process_rag calls self.rag.generar_respuesta(...)."""
    events = []
    _patch_generation(monkeypatch, events)
    monkeypatch.setattr(generation.cfg, "guardar_debug_rag", lambda *a, **k: None)

    fragmentos = [{"doc": "texto", "metadata": {"source": "a.pdf", "page": 0, "chunk": 0}}]
    respuesta = generation.generar_respuesta("pregunta", fragmentos)

    assert events == ["released", "stream_started"]
    assert respuesta == "hola"


def test_generar_respuesta_silenciosa_releases_before_streaming_eval_path(monkeypatch):
    """generar_respuesta_silenciosa is what tests/eval/run_eval.py's
    run_factual_case calls per case in phase 2 -- confirming it also releases
    (a no-op there in practice, since the eval runner never populates these
    caches in the first place, but the ordering contract is the same one)."""
    events = []
    _patch_generation(monkeypatch, events)

    fragmentos = [{"doc": "texto", "metadata": {"source": "a.pdf", "page": 0, "chunk": 0}}]
    respuesta = generation.generar_respuesta_silenciosa("pregunta", fragmentos)

    assert events == ["released", "stream_started"]
    assert respuesta == "hola"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
