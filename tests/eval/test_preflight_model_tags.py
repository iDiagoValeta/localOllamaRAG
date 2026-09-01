"""Ollama's implicit `:latest`, and the instruction that used to be wrong.

Issue #129. `--models qwen3-coder-30b` aborted with

    SETUP FAILED: 1 required Ollama model(s) not installed:
      ollama pull qwen3-coder-30b

on a machine where the model was installed. `ollama list` shows it as
`qwen3-coder-30b:latest` and `ollama run qwen3-coder-30b` works, because
Ollama resolves a bare name to `:latest`; `/api/tags` always returns the
tagged form, and the check compared the two strings directly.

The failure was fast and its message looked actionable, which is what made it
worth fixing rather than shrugging at: it told the operator to pull something
they already had, and following it is a no-op that teaches them nothing.

No network, no Ollama: `_installed_ollama_models` is replaced with a list.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

import run_eval  # noqa: E402


@pytest.mark.parametrize(
    "name,expected",
    [
        ("qwen3-coder-30b", "qwen3-coder-30b:latest"),
        ("qwen3:8b", "qwen3:8b"),
        ("gemma4:e4b", "gemma4:e4b"),
        # A digest already names an exact blob; a tag on top of it is nonsense.
        ("model@sha256:abc123", "model@sha256:abc123"),
    ],
)
def test_the_implicit_tag_is_applied_only_where_ollama_applies_it(name, expected):
    assert run_eval._with_implicit_tag(name) == expected


def test_a_bare_name_matches_a_model_installed_as_latest(monkeypatch):
    """The #129 reproduction, as a test."""
    monkeypatch.setattr(
        run_eval, "_installed_ollama_models", lambda: ["qwen3-coder-30b:latest"]
    )
    run_eval.preflight_ollama(["qwen3-coder-30b"])


def test_a_tagged_name_still_matches_itself(monkeypatch):
    monkeypatch.setattr(run_eval, "_installed_ollama_models", lambda: ["gemma4:e4b"])
    run_eval.preflight_ollama(["gemma4:e4b"])


def test_a_genuinely_missing_model_still_fails(monkeypatch):
    """The check must not be weakened into never failing."""
    monkeypatch.setattr(run_eval, "_installed_ollama_models", lambda: ["gemma4:e4b"])
    with pytest.raises(run_eval.EvalSetupError) as exc:
        run_eval.preflight_ollama(["nothing-like-this"])
    assert "nothing-like-this" in str(exc.value)


def test_the_message_names_the_model_the_caller_asked_for(monkeypatch):
    """Reported by the caller's spelling, not the normalised one: the message
    has to be something they can paste back."""
    monkeypatch.setattr(run_eval, "_installed_ollama_models", lambda: [])
    with pytest.raises(run_eval.EvalSetupError) as exc:
        run_eval.preflight_ollama(["some-model"])
    message = str(exc.value)
    assert "ollama pull some-model" in message
    assert "some-model:latest" not in message


def test_every_missing_model_is_listed_not_just_the_first(monkeypatch):
    monkeypatch.setattr(run_eval, "_installed_ollama_models", lambda: [])
    with pytest.raises(run_eval.EvalSetupError) as exc:
        run_eval.preflight_ollama(["a", "b"])
    assert "ollama pull a" in str(exc.value) and "ollama pull b" in str(exc.value)
