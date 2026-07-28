"""Tests for the eval runner's phase-2-only OLLAMA_KEEP_ALIVE override.

No GPU or Ollama server needed: these check the environment-scoping
mechanism itself, that it reaches the exact config path (AppConfig.from_env())
that generar_respuesta_silenciosa's fresh app_config_from_runtime() call reads
at generation time (see issue #27), and the explicit unload that runs on
block exit.

``_release_ollama_models`` does its ``import requests`` lazily, same as the
rest of run_eval.py's Ollama-HTTP calls -- this file participates in the fast
CI gate (tests/conftest.py), which deliberately installs no project
dependencies, so it cannot assume the real ``requests`` package exists. The
``_stub_requests`` fixture below plants a fake module in ``sys.modules``
instead of importing the real thing, which satisfies that lazy import
whether or not `requests` is actually installed.
"""

import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

import run_eval  # noqa: E402
from run_eval import (  # noqa: E402
    _EVAL_GENERATION_KEEP_ALIVE_SECONDS,
    _generation_keep_alive,
    _release_ollama_models,
)

from monkeygrab.config.app_config import AppConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _stub_requests(monkeypatch):
    """Give every test's ``_release_ollama_models`` call a captured, no-op
    ``requests.post`` -- ``_generation_keep_alive`` calls it unconditionally
    on exit, so this must be active even for tests that don't care about the
    release calls themselves.
    """
    calls = []
    fake_module = types.ModuleType("requests")
    fake_module.post = lambda url, json, timeout: calls.append(json) or None
    monkeypatch.setitem(sys.modules, "requests", fake_module)
    return calls


def test_keep_alive_is_set_during_the_block(monkeypatch):
    monkeypatch.delenv("OLLAMA_KEEP_ALIVE", raising=False)
    with _generation_keep_alive(["some-model"]):
        assert os.environ["OLLAMA_KEEP_ALIVE"] == _EVAL_GENERATION_KEEP_ALIVE_SECONDS
        # Same call generar_respuesta_silenciosa makes via
        # wiring.app_config_from_runtime() -- this is the proof the override
        # actually reaches generation, not just the retrieval-phase config.
        assert AppConfig.from_env().models.ollama.keep_alive == int(_EVAL_GENERATION_KEEP_ALIVE_SECONDS)


def test_keep_alive_is_removed_after_the_block_when_previously_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_KEEP_ALIVE", raising=False)
    with _generation_keep_alive(["some-model"]):
        pass
    assert "OLLAMA_KEEP_ALIVE" not in os.environ


def test_keep_alive_restores_a_prior_value(monkeypatch):
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "42")
    with _generation_keep_alive(["some-model"]):
        assert os.environ["OLLAMA_KEEP_ALIVE"] == _EVAL_GENERATION_KEEP_ALIVE_SECONDS
    assert os.environ["OLLAMA_KEEP_ALIVE"] == "42"


def test_keep_alive_restores_even_if_generation_raises(monkeypatch):
    monkeypatch.delenv("OLLAMA_KEEP_ALIVE", raising=False)
    try:
        with _generation_keep_alive(["some-model"]):
            raise RuntimeError("simulated generation failure")
    except RuntimeError:
        pass
    assert "OLLAMA_KEEP_ALIVE" not in os.environ


def test_generation_keep_alive_releases_the_models_used_on_exit(_stub_requests):
    """Restoring the env var only changes future requests -- Ollama's own
    120s hold on the last-used models needs an explicit keep_alive=0 POST,
    which must fire on block exit regardless (see the
    _generation_keep_alive docstring)."""
    with _generation_keep_alive(["gen-model:latest", "aux-model:latest"]):
        pass

    released = {c["model"] for c in _stub_requests}
    assert released == {"gen-model:latest", "aux-model:latest"}
    assert all(c["keep_alive"] == 0 for c in _stub_requests)


def test_generation_keep_alive_releases_even_if_generation_raises(_stub_requests):
    try:
        with _generation_keep_alive(["some-model"]):
            raise RuntimeError("simulated generation failure")
    except RuntimeError:
        pass

    assert {c["model"] for c in _stub_requests} == {"some-model"}


def test_release_deduplicates_model_names(_stub_requests):
    _release_ollama_models(["m1", "m1", "m2"])

    assert sorted(c["model"] for c in _stub_requests) == ["m1", "m2"]


def test_release_posts_to_the_module_base_url(monkeypatch):
    calls = []
    fake_module = types.ModuleType("requests")
    fake_module.post = lambda url, json, timeout: calls.append(url) or None
    monkeypatch.setitem(sys.modules, "requests", fake_module)

    _release_ollama_models(["m1"])

    assert calls == [f"{run_eval.OLLAMA_BASE_URL}/api/generate"]


def test_release_swallows_a_post_failure_for_one_model_and_continues(monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append(json["model"])
        if json["model"] == "flaky":
            raise ConnectionError("simulated Ollama unreachable")

    fake_module = types.ModuleType("requests")
    fake_module.post = fake_post
    monkeypatch.setitem(sys.modules, "requests", fake_module)

    _release_ollama_models(["flaky", "other"])  # must not raise

    assert set(calls) == {"flaky", "other"}
