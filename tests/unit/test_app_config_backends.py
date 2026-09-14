"""Per-role chat backend selection in ``ModelsConfig``.

Pure config: no adapters, no engine, so it runs in the no-infrastructure CI job.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monkeygrab.config import AppConfig  # noqa: E402
from monkeygrab.config.models import CHAT_BACKENDS, OpenAICompatRuntimeConfig  # noqa: E402

_ROLE_VARS = (
    "MONKEYGRAB_RAG_BACKEND",
    "MONKEYGRAB_CHAT_BACKEND",
    "MONKEYGRAB_CONTEXTUAL_BACKEND",
    "MONKEYGRAB_RECOMP_BACKEND",
    "MONKEYGRAB_OPENAI_BASE_URL",
    "MONKEYGRAB_OPENAI_API_KEY",
    "MONKEYGRAB_OPENAI_TIMEOUT",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ROLE_VARS:
        monkeypatch.delenv(name, raising=False)


def test_every_role_defaults_to_ollama():
    models = AppConfig.from_env().models
    assert (models.rag_backend, models.chat_backend, models.contextual_backend, models.recomp_backend) == (
        "ollama", "ollama", "ollama", "ollama"
    )
    assert models.openai == OpenAICompatRuntimeConfig()
    assert models.openai.base_url == "http://127.0.0.1:8000/v1"
    assert models.openai.api_key == ""
    assert models.openai.timeout == 900


def test_roles_can_point_at_openai_independently(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_RAG_BACKEND", "openai")
    monkeypatch.setenv("MONKEYGRAB_OPENAI_BASE_URL", "http://127.0.0.1:8082/v1/models")
    monkeypatch.setenv("MONKEYGRAB_OPENAI_API_KEY", "secret")
    monkeypatch.setenv("MONKEYGRAB_OPENAI_TIMEOUT", "30")
    models = AppConfig.from_env().models
    assert models.rag_backend == "openai"
    assert models.chat_backend == "ollama"
    assert models.openai.base_url == "http://127.0.0.1:8082/v1/models"
    assert models.openai.api_key == "secret"
    assert models.openai.timeout == 30


def test_unknown_backend_name_is_a_hard_error(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_RECOMP_BACKEND", "anthropic")
    with pytest.raises(ValueError, match="MONKEYGRAB_RECOMP_BACKEND"):
        AppConfig.from_env()


def test_backend_choices_are_exactly_the_two_supported():
    assert CHAT_BACKENDS == ("ollama", "openai")
