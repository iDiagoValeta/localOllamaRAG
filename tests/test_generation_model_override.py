"""A per-answer model replaces the rag role for that answer only."""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rag.chat_pdfs  # noqa: F401  (the facade must load before its engine modules)
from rag.engine import generation


def _capture(monkeypatch):
    seen = []

    def fake_answer(store, config):
        seen.append((config.models.rag, config.models.rag_backend))
        return types.SimpleNamespace(stream=lambda message: iter([types.SimpleNamespace(text="ok", done=True)]))

    monkeypatch.setattr(generation.wiring, "answer", fake_answer)
    monkeypatch.setattr(generation, "_generation_stats", lambda chunk: {})
    return seen


def test_the_requested_model_generates_that_answer(monkeypatch):
    seen = _capture(monkeypatch)
    assert list(generation.generar_tokens_respuesta("MSG", model="deepseek-flash", backend="openai")) == ["ok"]
    assert seen == [("deepseek-flash", "openai")]


def test_without_a_model_the_configured_role_answers(monkeypatch):
    seen = _capture(monkeypatch)
    models = generation.wiring.app_config_from_runtime().models
    list(generation.generar_tokens_respuesta("MSG"))
    assert seen == [(models.rag, models.rag_backend)]
