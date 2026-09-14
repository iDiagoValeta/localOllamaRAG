"""Each model role is built on the backend its config names.

Imports ``rag.engine.wiring`` (the engine), so the no-infrastructure CI job
skips this file via tests/conftest.py's collect_ignore; the engine job runs it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pytest

import rag.chat_pdfs  # noqa: F401 -- import before rag.engine.wiring, see its module docstring
from monkeygrab.adapters.chat.ollama_chat import OllamaChatModel
from monkeygrab.adapters.chat.openai_compat_chat import OpenAICompatChatModel
from monkeygrab.config import AppConfig
from rag.engine import wiring

_VARS = ("MONKEYGRAB_RAG_BACKEND", "MONKEYGRAB_CHAT_BACKEND", "MONKEYGRAB_CONTEXTUAL_BACKEND",
         "MONKEYGRAB_RECOMP_BACKEND", "MONKEYGRAB_OPENAI_BASE_URL", "MONKEYGRAB_OPENAI_API_KEY")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _VARS:
        monkeypatch.delenv(name, raising=False)


def test_openai_options_keep_only_what_the_api_understands():
    translated = wiring.openai_options_from_ollama(
        {"temperature": 0.15, "top_p": 0.9, "repeat_penalty": 1.15, "repeat_last_n": 64,
         "num_predict": 4096, "stop": ["\n\n\n"], "num_ctx": 16384}
    )
    assert translated == {"temperature": 0.15, "top_p": 0.9, "max_tokens": 4096, "stop": ["\n\n\n"]}


def test_openai_options_drop_an_unbounded_num_predict():
    assert wiring.openai_options_from_ollama({"num_predict": -1, "temperature": 0.1}) == {"temperature": 0.1}


def test_every_role_builds_ollama_by_default():
    config = AppConfig.from_env()
    assert isinstance(wiring.rag_chat_model(config), OllamaChatModel)
    assert isinstance(wiring.recomp_chat_model(config), OllamaChatModel)
    assert isinstance(wiring.query_decomposer(config), OllamaChatModel)


def test_rag_role_builds_the_openai_adapter_when_configured(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_RAG_BACKEND", "openai")
    monkeypatch.setenv("MONKEYGRAB_OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    config = AppConfig.from_env()
    model = wiring.rag_chat_model(config)
    assert isinstance(model, OpenAICompatChatModel)
    assert model.url() == "http://127.0.0.1:9/v1/chat/completions"
    # The other roles stay where they were.
    assert isinstance(wiring.query_decomposer(config), OllamaChatModel)


def test_chat_model_for_role_rejects_an_unknown_role():
    config = AppConfig.from_env()
    with pytest.raises(ValueError, match="role"):
        wiring.chat_model_for_role(config, "vision", options={}, num_ctx=1, keep_alive=0,
                                   generation_deadline=0)
