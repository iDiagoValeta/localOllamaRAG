"""Unit tests for monkeygrab.adapters.chat.ollama_model_unloader.OllamaModelUnloader.

Stubs requests.post entirely -- no Ollama server, no network -- so these run
in milliseconds.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from monkeygrab.adapters.chat import ollama_model_unloader as module  # noqa: E402
from monkeygrab.adapters.chat.ollama_model_unloader import OllamaModelUnloader  # noqa: E402
from monkeygrab.config.models import ModelsConfig  # noqa: E402


def _models(**overrides):
    return ModelsConfig(**overrides) if overrides else ModelsConfig()


def test_unloads_every_configured_model_except_the_one_kept(monkeypatch):
    calls = []
    monkeypatch.setattr(
        module.requests, "post",
        lambda url, json, timeout: calls.append(json) or None,
    )

    models = _models(
        rag="rag-model:latest", chat="chat-model:latest",
        contextual="ctx-model:latest", recomp="recomp-model:latest",
    )
    OllamaModelUnloader(models).unload_all_except("rag-model:latest")

    unloaded = {c["model"] for c in calls}
    assert unloaded == {"chat-model:latest", "ctx-model:latest", "recomp-model:latest"}
    assert "rag-model:latest" not in unloaded
    assert all(c["keep_alive"] == 0 for c in calls)


def test_unloads_every_model_when_keep_is_none(monkeypatch):
    calls = []
    monkeypatch.setattr(
        module.requests, "post",
        lambda url, json, timeout: calls.append(json) or None,
    )

    models = _models(rag="m1:latest", chat="m1:latest",
                      contextual="m1:latest", recomp="m2:latest")
    OllamaModelUnloader(models).unload_all_except(None)

    unloaded = {c["model"] for c in calls}
    assert unloaded == {"m1:latest", "m2:latest"}  # de-duplicated role models


def test_duplicate_role_models_are_only_unloaded_once(monkeypatch):
    """Several roles commonly share the same model (e.g. rag/chat/contextual
    all defaulting to gemma4:e4b) -- must not POST redundantly per role."""
    calls = []
    monkeypatch.setattr(
        module.requests, "post",
        lambda url, json, timeout: calls.append(json) or None,
    )

    models = _models(rag="shared:latest", chat="shared:latest",
                      contextual="shared:latest", recomp="other:latest")
    OllamaModelUnloader(models).unload_all_except("other:latest")

    assert len(calls) == 1
    assert calls[0]["model"] == "shared:latest"


def test_an_unload_failure_for_one_model_does_not_stop_the_others(monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append(json["model"])
        if json["model"] == "flaky:latest":
            raise ConnectionError("model already unloaded")

    monkeypatch.setattr(module.requests, "post", fake_post)

    models = _models(rag="keep:latest", chat="flaky:latest",
                      contextual="other:latest", recomp="keep:latest")
    OllamaModelUnloader(models).unload_all_except("keep:latest")  # must not raise

    assert set(calls) == {"flaky:latest", "other:latest"}


def test_posts_to_the_configured_base_url_and_timeout(monkeypatch):
    calls = []
    monkeypatch.setattr(
        module.requests, "post",
        lambda url, json, timeout: calls.append({"url": url, "timeout": timeout}),
    )

    models = _models(rag="m1:latest", chat="m1:latest",
                      contextual="m1:latest", recomp="m1:latest")
    OllamaModelUnloader(models, base_url="http://example:9999", timeout=7).unload_all_except(None)

    assert calls[0]["url"] == "http://example:9999/api/generate"
    assert calls[0]["timeout"] == 7


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
