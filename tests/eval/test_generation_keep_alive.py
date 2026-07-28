"""Tests for the eval runner's phase-2-only OLLAMA_KEEP_ALIVE override.

No GPU or Ollama server needed: these check the environment-scoping
mechanism itself and that it reaches the exact config path
(AppConfig.from_env()) that generar_respuesta_silenciosa's fresh
app_config_from_runtime() call reads at generation time (see issue #27).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import _EVAL_GENERATION_KEEP_ALIVE_SECONDS, _generation_keep_alive  # noqa: E402

from monkeygrab.config.app_config import AppConfig  # noqa: E402


def test_keep_alive_is_set_during_the_block(monkeypatch):
    monkeypatch.delenv("OLLAMA_KEEP_ALIVE", raising=False)
    with _generation_keep_alive():
        assert os.environ["OLLAMA_KEEP_ALIVE"] == _EVAL_GENERATION_KEEP_ALIVE_SECONDS
        # Same call generar_respuesta_silenciosa makes via
        # wiring.app_config_from_runtime() -- this is the proof the override
        # actually reaches generation, not just the retrieval-phase config.
        assert AppConfig.from_env().models.ollama.keep_alive == int(_EVAL_GENERATION_KEEP_ALIVE_SECONDS)


def test_keep_alive_is_removed_after_the_block_when_previously_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_KEEP_ALIVE", raising=False)
    with _generation_keep_alive():
        pass
    assert "OLLAMA_KEEP_ALIVE" not in os.environ


def test_keep_alive_restores_a_prior_value(monkeypatch):
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "42")
    with _generation_keep_alive():
        assert os.environ["OLLAMA_KEEP_ALIVE"] == _EVAL_GENERATION_KEEP_ALIVE_SECONDS
    assert os.environ["OLLAMA_KEEP_ALIVE"] == "42"


def test_keep_alive_restores_even_if_generation_raises(monkeypatch):
    monkeypatch.delenv("OLLAMA_KEEP_ALIVE", raising=False)
    try:
        with _generation_keep_alive():
            raise RuntimeError("simulated generation failure")
    except RuntimeError:
        pass
    assert "OLLAMA_KEEP_ALIVE" not in os.environ
