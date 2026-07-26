"""Unit tests for monkeygrab.adapters.embedding.ollama_embedder.OllamaEmbedder.

Stubs ollama.embeddings entirely -- no Ollama server, no network -- so these
run in milliseconds.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from monkeygrab.adapters.embedding import ollama_embedder as module
from monkeygrab.adapters.embedding.ollama_embedder import OllamaEmbedder
from monkeygrab.config.models import ModelsConfig


def test_embed_uses_the_injected_model_not_a_frozen_default(monkeypatch):
    """Two embedders built from different config must call Ollama with different models."""
    calls = []

    def fake_embeddings(model, prompt):
        calls.append((model, prompt))
        return {"embedding": [0.1, 0.2, 0.3]}

    monkeypatch.setattr(module.ollama, "embeddings", fake_embeddings)

    embedder_a = OllamaEmbedder(ModelsConfig(embedding="model-a:latest"))
    embedder_b = OllamaEmbedder(ModelsConfig(embedding="model-b:latest"))

    embedder_a.embed("search_document: hello")
    embedder_b.embed("search_document: hello")

    assert calls == [
        ("model-a:latest", "search_document: hello"),
        ("model-b:latest", "search_document: hello"),
    ]


def test_embed_returns_the_vector_from_ollama(monkeypatch):
    monkeypatch.setattr(module.ollama, "embeddings", lambda model, prompt: {"embedding": [1.0, 2.0]})

    result = OllamaEmbedder(ModelsConfig(embedding="m:latest")).embed("text")

    assert result == [1.0, 2.0]


def test_embed_does_not_prepend_any_prefix_itself(monkeypatch):
    """Task-prefix handling is the caller's job (see the Embedder port docstring):
    the adapter must send exactly the text it was given, untouched."""
    seen_prompts = []
    monkeypatch.setattr(
        module.ollama, "embeddings",
        lambda model, prompt: seen_prompts.append(prompt) or {"embedding": []},
    )

    OllamaEmbedder(ModelsConfig(embedding="m:latest")).embed("search_query: raw text")

    assert seen_prompts == ["search_query: raw text"]


def test_embed_retries_once_on_overlong_input_then_succeeds(monkeypatch):
    calls = []

    def fake_embeddings(model, prompt):
        calls.append(prompt)
        if len(calls) == 1:
            raise Exception("Ollama error: context length exceeded (500)")
        return {"embedding": [9.0]}

    monkeypatch.setattr(module.ollama, "embeddings", fake_embeddings)

    result = OllamaEmbedder(ModelsConfig(embedding="m:latest")).embed("x" * 5000)

    assert result == [9.0]
    assert len(calls) == 2
    assert calls[1] == ("x" * 5000)[:1000]


def test_embed_hard_fails_on_a_non_length_error_without_retrying(monkeypatch):
    calls = []

    def fake_embeddings(model, prompt):
        calls.append(prompt)
        raise ConnectionError("Ollama server unreachable")

    monkeypatch.setattr(module.ollama, "embeddings", fake_embeddings)

    with pytest.raises(RuntimeError, match="Ollama server unreachable"):
        OllamaEmbedder(ModelsConfig(embedding="m:latest")).embed("text")

    assert len(calls) == 1  # no silent retry for an unrelated failure


def test_embed_hard_fails_when_the_retry_also_fails(monkeypatch):
    """The port explicitly allows one truncate-and-retry for overlong input,
    but a second failure must still raise -- never a silent skip."""
    calls = []

    def fake_embeddings(model, prompt):
        calls.append(prompt)
        raise Exception("context length exceeded")

    monkeypatch.setattr(module.ollama, "embeddings", fake_embeddings)

    with pytest.raises(RuntimeError, match="even after truncating"):
        OllamaEmbedder(ModelsConfig(embedding="m:latest")).embed("y" * 5000)

    assert len(calls) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
