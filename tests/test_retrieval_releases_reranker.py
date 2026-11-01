"""Retrieval hands the GPU to the reranker and back (issue #584).

The observed shape, on an 8 GB card with the embedder resident: with the
reranker kept loaded, Ollama placed 303 MB of the 6 GB ``gemma4:e4b`` on the
GPU and generated on CPU, 10 to 54 s per answer. Releasing only the reranker
moved generation onto the GPU, but the generator then stayed resident for its
keep_alive and the reranker's reload on the next question ran out of memory.
So Ollama's models are unloaded before retrieval and the reranker after it.

Lives with the loose tests because it imports the engine package.
"""

from types import SimpleNamespace

import pytest

import rag.chat_pdfs  # noqa: F401  (loads the engine package in its supported order)
from monkeygrab.config.app_config import AppConfig
from rag.engine import retrieval, wiring


@pytest.mark.parametrize("usar_reranker, expected", [
    (True, ["unload", "run", "release"]),
    (False, ["run"]),
])
def test_reranker_is_released_after_retrieval(monkeypatch, usar_reranker, expected):
    events = []

    class _RecordingRetrieve:
        def __init__(self, *_a, **_kw):
            pass

        def run(self, _question):
            events.append("run")
            return SimpleNamespace(fragments=[], metrics={})

    config = AppConfig().with_overrides(**{
        "flags.usar_reranker": usar_reranker,
        "flags.usar_busqueda_hibrida": False,
        "flags.usar_llm_query_decomposition": False,
        "flags.logging_metricas": False,
    })
    monkeypatch.setattr(wiring, "app_config_from_runtime", lambda: config)
    monkeypatch.setattr(wiring, "embedder", lambda _config: None)
    monkeypatch.setattr(wiring, "reranker", lambda _config: None)
    monkeypatch.setattr(wiring, "release_ollama_models", lambda _config: events.append("unload"))
    monkeypatch.setattr(wiring, "release_reranker", lambda: events.append("release"))
    monkeypatch.setattr(wiring, "retrieval_metrics_to_legacy", lambda _metrics, _config: {})
    monkeypatch.setattr(retrieval, "Retrieve", _RecordingRetrieve)

    retrieval.realizar_busqueda_hibrida("question", collection=None)

    assert events == expected
