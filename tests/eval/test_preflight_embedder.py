"""The gate's preflight must check the embedder, not just Ollama (issue #297).

run_eval.py embeds every gold question and every indexed chunk through
embeddinggemma-2 served by ``llama-server`` at
``config.models.embedding.base_url``. A run with Ollama up and the embedder
down used to pass preflight and fail deep inside retrieval, far from the one
check built to say what is missing.

No network, no Ollama and no embedder: a fake ``requests`` module is planted
in ``sys.modules``, the same way test_generation_keep_alive.py does, so this
file also runs in the dependency-free fast gate.
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

import run_eval  # noqa: E402


class _RequestException(Exception):
    """Stand-in for requests.RequestException, which run_eval catches."""


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _RequestException(f"HTTP {self.status_code}")


def _stub_requests(monkeypatch, response=None, error=None):
    """Plant a fake ``requests`` module and return the URLs it was asked for."""
    calls = []
    fake_module = types.ModuleType("requests")
    fake_module.RequestException = _RequestException

    def _get(url, timeout):
        calls.append(url)
        if error is not None:
            raise error
        return response

    fake_module.get = _get
    monkeypatch.setitem(sys.modules, "requests", fake_module)
    return calls


def test_the_health_check_targets_the_appconfig_embedder_endpoint(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_EMBEDDER_URL", "http://embedder.example:9000/")
    calls = _stub_requests(monkeypatch, _Response(200))

    run_eval.preflight_embedder()

    assert calls == ["http://embedder.example:9000/health"]


def test_the_url_comes_from_appconfig_not_a_fresh_env_read(monkeypatch):
    """The preflight must see the same endpoint the pipeline will call."""
    monkeypatch.setenv("MONKEYGRAB_EMBEDDER_URL", "http://from-env.example:1111")
    config = SimpleNamespace(
        models=SimpleNamespace(
            embedding=SimpleNamespace(base_url="http://from-config.example:2222")
        )
    )
    monkeypatch.setattr(run_eval, "AppConfig", SimpleNamespace(from_env=lambda: config))
    calls = _stub_requests(monkeypatch, _Response(200))

    run_eval.preflight_embedder()

    assert calls == ["http://from-config.example:2222/health"]


def test_a_non_200_health_response_is_a_setup_error(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_EMBEDDER_URL", "http://embedder.example:9000")
    _stub_requests(monkeypatch, _Response(503))

    with pytest.raises(run_eval.EvalSetupError) as exc:
        run_eval.preflight_embedder()

    assert "http://embedder.example:9000" in str(exc.value)


def test_an_unreachable_embedder_is_a_setup_error_that_says_how_to_start_it(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_EMBEDDER_URL", "http://embedder.example:9000")
    _stub_requests(monkeypatch, error=_RequestException("Connection refused"))

    with pytest.raises(run_eval.EvalSetupError) as exc:
        run_eval.preflight_embedder()

    message = str(exc.value)
    assert "http://embedder.example:9000" in message
    assert "llama-server" in message
    assert "--embedding" in message
