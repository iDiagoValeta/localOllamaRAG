"""End-to-end check that OLLAMA_BASE_URL redirects real requests.

The acceptance criterion of #62 is deliberately not "the constant changed":
before this was wired, ``rag.chat_pdfs.OLLAMA_BASE_URL`` and the CLI health
check did read the variable, and generation still went to localhost anyway.
So these tests set the environment variable, build the models through
``rag.engine.wiring`` exactly as the pipeline does, and assert the URL the
adapter would have sent to -- intercepted at the HTTP boundary.

Lives outside ``tests/unit`` because it imports ``rag.engine.wiring``, which
pulls in the engine; the fast CI gate's ``architecture`` job installs no
infrastructure and only runs ``tests/unit``.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class _FakeStreamResponse:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        return iter(self._lines)


@pytest.fixture
def posted_urls(monkeypatch):
    """Capture every URL the chat adapter POSTs to, answering with one done line."""
    from monkeygrab.adapters.chat import ollama_chat

    urls = []

    def fake_post(url, json_=None, **kwargs):
        urls.append(url)
        return _FakeStreamResponse([json.dumps({"response": "hi", "done": True}).encode()])

    monkeypatch.setattr(
        ollama_chat.requests, "post",
        lambda url, **kwargs: fake_post(url, **kwargs),
    )
    return urls


def _wiring_with_env(monkeypatch, base_url=None, host=None):
    """Return ``rag.engine.wiring`` with the endpoint variables set as given.

    No module reload: ``app_config_from_runtime`` calls ``AppConfig.from_env``
    on every invocation, which is what lets a runtime config change take
    effect on the next query. Setting the variable and asking for a model is
    therefore the same sequence a running process goes through.
    """
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    if base_url is not None:
        monkeypatch.setenv("OLLAMA_BASE_URL", base_url)
    if host is not None:
        monkeypatch.setenv("OLLAMA_HOST", host)

    # rag.chat_pdfs first: it is the facade every entry point imports, and
    # rag.engine.wiring imports back into it, so importing wiring on its own
    # lands in a half-initialized module.
    import rag.chat_pdfs  # noqa: F401
    import rag.engine.wiring as wiring

    return wiring


def test_setting_ollama_base_url_redirects_the_rag_generator(monkeypatch, posted_urls):
    wiring = _wiring_with_env(monkeypatch, base_url="http://gpu-box:11434")

    model = wiring.rag_chat_model(wiring.app_config_from_runtime())
    list(model.stream("prompt"))

    assert posted_urls == ["http://gpu-box:11434/api/generate"]


def test_ollama_host_alone_also_redirects_the_rag_generator(monkeypatch, posted_urls):
    wiring = _wiring_with_env(monkeypatch, host="http://gpu-box:11434")

    model = wiring.rag_chat_model(wiring.app_config_from_runtime())
    list(model.stream("prompt"))

    assert posted_urls == ["http://gpu-box:11434/api/generate"]


def test_unset_variables_keep_every_role_on_the_local_server(monkeypatch):
    """The default path must be byte-identical to before this was wired."""
    wiring = _wiring_with_env(monkeypatch)
    config = wiring.app_config_from_runtime()

    endpoints = {
        wiring.rag_chat_model(config)._base_url,
        wiring.recomp_chat_model(config)._base_url,
        wiring.query_decomposer(config)._base_url,
        wiring.rag_chat_model(config)._model_unloader._base_url,
    }

    assert endpoints == {"http://localhost:11434"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
