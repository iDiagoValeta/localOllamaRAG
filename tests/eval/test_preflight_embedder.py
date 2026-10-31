"""The gate's preflight must check the embedder, not just Ollama (issue #297).

run_eval.py embeds every gold question and every indexed chunk through
embeddinggemma-2 served by ``llama-server`` at
``config.models.embedding.base_url``. A run with Ollama up and the embedder
down used to pass preflight and fail deep inside retrieval, far from the one
check built to say what is missing.

Since issue #384 the preflight also reads ``GET /props``: a llama-server that
answers ``/health`` may still be serving a different model or have been
started without ``--embedding``, and the artifact would label that run
``mineru-eg2-faiss`` anyway. The model the server says it serves is returned
so the run artifact can record it.

No network, no Ollama and no embedder: a fake ``requests`` module is planted
in ``sys.modules``, the same way test_generation_keep_alive.py does, so this
file also runs in the dependency-free fast gate.
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

import run_eval

# What a correctly started llama-server reports: the GGUF it loaded.
_EMBEDDINGGEMMA_PATH = "/models/embeddinggemma-2-300m-Q8_0.gguf"


class _RequestException(Exception):
    """Stand-in for requests.RequestException, which run_eval catches."""


class _Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _RequestException(f"HTTP {self.status_code}")

    def json(self):
        if self._payload is None:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


def _stub_requests(monkeypatch, response=None, error=None, props=None):
    """Plant a fake ``requests`` module and return the URLs it was asked for.

    ``props`` is the ``/props`` body (defaults to a server correctly serving
    embeddinggemma-2); pass a ``_Response`` instead to fake an HTTP or JSON
    failure on that endpoint.
    """
    calls = []
    props = _Response(200, {"model_path": _EMBEDDINGGEMMA_PATH}) if props is None else props
    fake_module = types.ModuleType("requests")
    fake_module.RequestException = _RequestException

    def _get(url, timeout):
        calls.append(url)
        if error is not None:
            raise error
        if url.endswith("/props"):
            if isinstance(props, _Response):
                return props
            return _Response(200, props)
        return response

    fake_module.get = _get
    monkeypatch.setitem(sys.modules, "requests", fake_module)
    return calls


def test_the_health_check_targets_the_appconfig_embedder_endpoint(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_EMBEDDER_URL", "http://embedder.example:9000/")
    calls = _stub_requests(monkeypatch, _Response(200))

    run_eval.preflight_embedder()

    assert calls == [
        "http://embedder.example:9000/health",
        "http://embedder.example:9000/props",
    ]


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

    assert calls == [
        "http://from-config.example:2222/health",
        "http://from-config.example:2222/props",
    ]


def test_the_served_model_path_is_returned_for_the_run_artifact(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_EMBEDDER_URL", "http://embedder.example:9000")
    _stub_requests(
        monkeypatch, _Response(200), props={"model_path": _EMBEDDINGGEMMA_PATH}
    )

    assert run_eval.preflight_embedder() == _EMBEDDINGGEMMA_PATH


def test_the_alias_identifies_the_model_when_the_path_does_not(monkeypatch):
    """A llama-server started with ``--alias`` need not carry the model name in
    the path it was given; the alias it reports is the identifier."""
    monkeypatch.setenv("MONKEYGRAB_EMBEDDER_URL", "http://embedder.example:9000")
    _stub_requests(
        monkeypatch,
        _Response(200),
        props={
            "model_path": "/models/model.gguf",
            "model_alias": "EmbeddingGemma-2-300M",
        },
    )

    assert run_eval.preflight_embedder() == "EmbeddingGemma-2-300M"


def test_a_server_serving_another_model_is_a_setup_error_that_names_it(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_EMBEDDER_URL", "http://embedder.example:9000")
    _stub_requests(
        monkeypatch,
        _Response(200),
        props={
            "model_path": "/models/qwen2.5-7b-instruct-q4_k_m.gguf",
            "model_alias": "qwen2.5-7b-instruct-q4_k_m",
        },
    )

    with pytest.raises(run_eval.EvalSetupError) as exc:
        run_eval.preflight_embedder()

    message = str(exc.value)
    assert "qwen2.5-7b-instruct-q4_k_m" in message
    assert "embeddinggemma" in message
    assert "http://embedder.example:9000" in message


def test_a_props_response_that_identifies_no_model_is_a_setup_error(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_EMBEDDER_URL", "http://embedder.example:9000")
    _stub_requests(monkeypatch, _Response(200), props={})

    with pytest.raises(run_eval.EvalSetupError) as exc:
        run_eval.preflight_embedder()

    message = str(exc.value)
    assert "http://embedder.example:9000" in message
    assert "embeddinggemma" in message


def test_a_non_200_props_response_is_a_setup_error(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_EMBEDDER_URL", "http://embedder.example:9000")
    _stub_requests(monkeypatch, _Response(200), props=_Response(503))

    with pytest.raises(run_eval.EvalSetupError) as exc:
        run_eval.preflight_embedder()

    assert "http://embedder.example:9000/props" in str(exc.value)


def test_a_props_response_that_is_not_json_is_a_setup_error(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_EMBEDDER_URL", "http://embedder.example:9000")
    _stub_requests(monkeypatch, _Response(200), props=_Response(200))

    with pytest.raises(run_eval.EvalSetupError) as exc:
        run_eval.preflight_embedder()

    assert "http://embedder.example:9000/props" in str(exc.value)


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
