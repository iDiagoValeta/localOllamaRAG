"""Unit tests for the Ollama endpoint resolution shared by every component.

``OLLAMA_BASE_URL`` used to be documented in ``.env.example`` while only the
CLI's startup health check read it: generation, chat, RECOMP synthesis and
contextual enrichment all went to a hardcoded ``http://localhost:11434``, so
pointing the variable at another server moved the diagnostic without moving
the traffic. These tests pin the single reader that now feeds all of them.

Standard library and ``monkeygrab.config`` only, so the fast CI gate's
``architecture`` job (which installs no infrastructure) runs them.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pytest  # noqa: E402

from monkeygrab.config import AppConfig  # noqa: E402
from monkeygrab.config.env import (  # noqa: E402
    DEFAULT_OLLAMA_BASE_URL,
    read_env_ollama_base_url,
)


@pytest.fixture(autouse=True)
def _clean_ollama_env(monkeypatch):
    """Neither variable set, whatever the developer's shell exports."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)


def test_defaults_to_the_local_server_when_nothing_is_set():
    assert read_env_ollama_base_url() == "http://localhost:11434"
    assert DEFAULT_OLLAMA_BASE_URL == "http://localhost:11434"


def test_ollama_base_url_is_honoured(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434")

    assert read_env_ollama_base_url() == "http://gpu-box:11434"


def test_ollama_host_is_the_fallback(monkeypatch):
    """Ollama's own variable is what the ``ollama`` client and the web control
    panel honoured before this was wired -- it must not stop working."""
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434")

    assert read_env_ollama_base_url() == "http://gpu-box:11434"


def test_ollama_base_url_wins_over_ollama_host(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://explicit:1111")
    monkeypatch.setenv("OLLAMA_HOST", "http://ambient:2222")

    assert read_env_ollama_base_url() == "http://explicit:1111"


def test_a_schemeless_value_gets_http_prepended(monkeypatch):
    """Ollama documents OLLAMA_HOST as ``127.0.0.1:11434``; the adapters build
    request URLs by concatenation, which that spelling would break."""
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")

    assert read_env_ollama_base_url() == "http://127.0.0.1:11434"


def test_a_trailing_slash_is_dropped(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/")

    assert read_env_ollama_base_url() == "http://gpu-box:11434"


def test_an_empty_value_falls_through_to_the_default(monkeypatch):
    """``.env`` files routinely carry ``OLLAMA_BASE_URL=`` with nothing after
    it; that is "unset", not "connect to the empty string"."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "   ")

    assert read_env_ollama_base_url() == "http://localhost:11434"


def test_app_config_carries_the_resolved_endpoint(monkeypatch):
    """The field the adapters are actually built from."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434")

    assert AppConfig.from_env().models.ollama.base_url == "http://gpu-box:11434"


def test_app_config_defaults_to_the_local_server():
    assert AppConfig.from_env().models.ollama.base_url == "http://localhost:11434"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
