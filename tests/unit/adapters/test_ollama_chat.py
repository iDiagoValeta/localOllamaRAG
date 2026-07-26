"""Unit tests for monkeygrab.adapters.chat.ollama_chat.OllamaChatModel.

Stubs ollama.chat and requests.post entirely -- no Ollama server, no network
-- so these run in milliseconds.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from monkeygrab.adapters.chat import ollama_chat as module
from monkeygrab.adapters.chat.ollama_chat import OllamaChatModel


# ─────────────────────────────────────────────
# generate()
# ─────────────────────────────────────────────


def test_generate_uses_the_injected_model_and_num_ctx_not_frozen_defaults(monkeypatch):
    calls = []
    monkeypatch.setattr(
        module.ollama, "chat",
        lambda **kwargs: calls.append(kwargs) or {"message": {"content": "answer"}},
    )

    chat_model = OllamaChatModel(
        "my-role-model:latest", num_ctx=1234, options={"temperature": 0.5}
    )
    result = chat_model.generate("hello")

    assert result == "answer"
    assert calls[0]["model"] == "my-role-model:latest"
    assert calls[0]["options"]["num_ctx"] == 1234
    assert calls[0]["options"]["temperature"] == 0.5


def test_generate_always_disables_thinking(monkeypatch):
    calls = []
    monkeypatch.setattr(
        module.ollama, "chat",
        lambda **kwargs: calls.append(kwargs) or {"message": {"content": ""}},
    )

    OllamaChatModel("m", num_ctx=100).generate("hello")

    assert calls[0]["think"] is False


def test_generate_puts_the_system_prompt_first(monkeypatch):
    calls = []
    monkeypatch.setattr(
        module.ollama, "chat",
        lambda **kwargs: calls.append(kwargs) or {"message": {"content": ""}},
    )

    OllamaChatModel("m", num_ctx=100).generate("question", system="be concise")

    assert calls[0]["messages"] == [
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "question"},
    ]


def test_generate_without_system_sends_a_single_user_message(monkeypatch):
    calls = []
    monkeypatch.setattr(
        module.ollama, "chat",
        lambda **kwargs: calls.append(kwargs) or {"message": {"content": ""}},
    )

    OllamaChatModel("m", num_ctx=100).generate("question")

    assert calls[0]["messages"] == [{"role": "user", "content": "question"}]


def test_generate_base64_encodes_images(monkeypatch):
    import base64

    calls = []
    monkeypatch.setattr(
        module.ollama, "chat",
        lambda **kwargs: calls.append(kwargs) or {"message": {"content": ""}},
    )

    OllamaChatModel("m", num_ctx=100).generate("describe this", images=[b"\x89PNG raw bytes"])

    sent_images = calls[0]["messages"][0]["images"]
    assert sent_images == [base64.b64encode(b"\x89PNG raw bytes").decode("utf-8")]


def test_generate_hard_fails_on_ollama_error(monkeypatch):
    monkeypatch.setattr(
        module.ollama, "chat",
        lambda **kwargs: (_ for _ in ()).throw(ConnectionError("ollama down")),
    )

    with pytest.raises(RuntimeError, match="ollama down"):
        OllamaChatModel("m", num_ctx=100).generate("hello")


# ─────────────────────────────────────────────
# stream()
# ─────────────────────────────────────────────


class _FakeStreamResponse:
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            error = module.requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def iter_lines(self):
        return iter(self._lines)


def test_stream_uses_the_injected_model_and_num_ctx(monkeypatch):
    calls = []
    import json as jsonlib

    def fake_post(url, json, stream, timeout):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeStreamResponse([jsonlib.dumps({"response": "hi", "done": True}).encode()])

    monkeypatch.setattr(module.requests, "post", fake_post)

    chat_model = OllamaChatModel("streamed-model", num_ctx=777, request_timeout=42)
    tokens = list(chat_model.stream("prompt"))

    assert tokens == ["hi"]
    assert calls[0]["json"]["model"] == "streamed-model"
    assert calls[0]["json"]["options"]["num_ctx"] == 777
    assert calls[0]["timeout"] == 42
    assert calls[0]["json"]["think"] is False


def test_stream_yields_every_response_chunk_in_order(monkeypatch):
    import json as jsonlib

    lines = [
        jsonlib.dumps({"response": "Hel"}).encode(),
        jsonlib.dumps({"response": "lo"}).encode(),
        jsonlib.dumps({"response": "", "done": True}).encode(),
    ]
    monkeypatch.setattr(module.requests, "post", lambda **kwargs: _FakeStreamResponse(lines))

    tokens = list(OllamaChatModel("m", num_ctx=100).stream("prompt"))

    assert tokens == ["Hel", "lo"]


def test_stream_retries_once_on_a_5xx_error_then_succeeds(monkeypatch):
    import json as jsonlib

    attempts = []

    def fake_post(**kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            return _FakeStreamResponse([], status_code=503)
        return _FakeStreamResponse([jsonlib.dumps({"response": "ok", "done": True}).encode()])

    monkeypatch.setattr(module.requests, "post", fake_post)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)

    tokens = list(
        OllamaChatModel("m", num_ctx=100, generate_retries=2, generate_retry_delay=0).stream("prompt")
    )

    assert tokens == ["ok"]
    assert len(attempts) == 2


def test_stream_hard_fails_immediately_on_a_4xx_error_without_retrying(monkeypatch):
    attempts = []

    def fake_post(**kwargs):
        attempts.append(1)
        return _FakeStreamResponse([], status_code=400)

    monkeypatch.setattr(module.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="stream generation failed"):
        list(OllamaChatModel("m", num_ctx=100, generate_retries=3).stream("prompt"))

    assert len(attempts) == 1  # a client error must not be retried


def test_stream_hard_fails_after_exhausting_5xx_retries(monkeypatch):
    attempts = []

    def fake_post(**kwargs):
        attempts.append(1)
        return _FakeStreamResponse([], status_code=500)

    monkeypatch.setattr(module.requests, "post", fake_post)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="stream generation failed"):
        list(OllamaChatModel("m", num_ctx=100, generate_retries=3, generate_retry_delay=0).stream("prompt"))

    assert len(attempts) == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
