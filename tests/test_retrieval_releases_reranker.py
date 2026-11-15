"""Retrieval shares the GPU between the reranker and the generator (issues #584, #630).

#584 observed, on an 8 GB card with the embedder resident: with the reranker
kept loaded before the generator, Ollama placed 303 MB of the 6 GB
``gemma4:e4b`` on the GPU and generated on CPU, 10 to 54 s per answer. #630
measured that ``qwen3:4b-instruct-2507`` (3.6 GB), the reranker (1.4 GB at
batch 8) and the embedder (1.2 GB) fit together on the same card. So the two
stay resident when they fit and take the card in turn when they do not, decided
from measured free memory and Ollama's own placement, never assumed.

Lives with the loose tests because it imports the engine package.
"""

from types import SimpleNamespace

import pytest

import rag.chat_pdfs  # noqa: F401  (loads the engine package in its supported order)
from monkeygrab.config.app_config import AppConfig
from rag.engine import retrieval, wiring


def _config(usar_reranker=True):
    return AppConfig().with_overrides(**{
        "flags.usar_reranker": usar_reranker,
        "flags.usar_busqueda_hibrida": False,
        "flags.usar_llm_query_decomposition": False,
        "flags.logging_metricas": False,
    })


def _run(monkeypatch, *, usar_reranker=True, resident=True, fails=False):
    events = []

    class _Retrieve:
        def __init__(self, *_a, **_kw):
            pass

        def run(self, _question):
            events.append("run")
            if fails:
                raise RuntimeError("CUDA out of memory")
            return SimpleNamespace(fragments=[], metrics={})

    config = _config(usar_reranker)
    monkeypatch.setattr(wiring, "app_config_from_runtime", lambda: config)
    monkeypatch.setattr(wiring, "embedder", lambda _config: None)
    monkeypatch.setattr(wiring, "room_making_reranker", lambda _config: None)
    monkeypatch.setattr(wiring, "generator_resident_on_gpu", lambda _config: resident)
    monkeypatch.setattr(wiring, "release_ollama_models", lambda _config: events.append("unload"))
    monkeypatch.setattr(wiring, "release_reranker", lambda: events.append("release"))
    monkeypatch.setattr(wiring, "retrieval_metrics_to_legacy", lambda _metrics, _config: {})
    monkeypatch.setattr(retrieval, "Retrieve", _Retrieve)
    if fails:
        with pytest.raises(RuntimeError):
            retrieval.realizar_busqueda_hibrida("question", collection=None)
    else:
        retrieval.realizar_busqueda_hibrida("question", collection=None)
    return events


def test_reranker_stays_beside_a_generator_resident_on_the_gpu(monkeypatch):
    assert _run(monkeypatch, resident=True) == ["run"]


def test_reranker_and_a_split_or_absent_generator_take_the_card_in_turn(monkeypatch):
    """Released, and Ollama unloaded too: a split generator stays split until reloaded."""
    assert _run(monkeypatch, resident=False) == ["run", "release", "unload"]


def test_no_reranker_means_nothing_to_hand_over(monkeypatch):
    assert _run(monkeypatch, usar_reranker=False, resident=False) == ["run"]


def test_reranker_is_released_when_retrieval_fails(monkeypatch):
    """Left resident after its own OOM, it kept ~2 GB and every later query failed."""
    assert _run(monkeypatch, resident=True, fails=True) == ["run", "release"]


@pytest.mark.parametrize("needs_room, expected", [
    (True, ["unload", "rerank"]),
    (False, ["rerank"]),
])
def test_reranker_unloads_ollama_only_when_it_would_not_fit(monkeypatch, needs_room, expected):
    events = []

    class _Inner:
        def needs_room(self):
            return needs_room

        def rerank(self, _query, fragments, _top_k):
            events.append("rerank")
            return list(fragments)

    monkeypatch.setattr(wiring, "release_ollama_models", lambda _config: events.append("unload"))
    wrapped = wiring._RoomMakingReranker(_Inner(), _config())

    assert wrapped.rerank("q", [], 8) == []
    assert events == expected


def _roles(**models):
    return AppConfig().with_overrides(**{f"models.{k}": v for k, v in models.items()})


def test_a_role_on_the_generators_model_asks_for_the_generators_context():
    """A different num_ctx makes Ollama reload the runner (2.3 s measured in #630)."""
    config = _roles(rag="gen:4b", chat="gen:4b", recomp="other:1b")
    rag_ctx = config.models.ollama.rag_num_ctx

    assert wiring.role_num_ctx(config, "chat", 2048) == rag_ctx
    assert wiring.role_num_ctx(config, "recomp", 8192) == 8192


def test_a_role_on_another_backend_does_not_share_the_ollama_runner():
    config = _roles(rag="gen:4b", chat="gen:4b").with_overrides(**{"models.chat_backend": "openai"})

    assert not wiring.shares_generator(config, "chat")


@pytest.mark.parametrize("chat, keep_alive_is_zero", [("gen:4b", False), ("other:1b", True)])
def test_query_decomposer_keeps_only_the_shared_runner_loaded(monkeypatch, chat, keep_alive_is_zero):
    seen = {}
    monkeypatch.setattr(
        wiring, "chat_model_for_role",
        lambda _config, role, **kw: seen.update(role=role, **kw),
    )
    config = _roles(rag="gen:4b", chat=chat)

    wiring.query_decomposer(config)

    assert seen["role"] == "chat"
    assert (seen["keep_alive"] == 0) is keep_alive_is_zero
