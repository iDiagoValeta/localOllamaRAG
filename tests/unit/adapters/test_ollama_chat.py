"""Unit tests for monkeygrab.adapters.chat.ollama_chat.OllamaChatModel.

Stubs ollama.chat and requests.post entirely -- no Ollama server, no network
-- so these run in milliseconds.
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from monkeygrab.adapters.chat import ollama_chat as module
from monkeygrab.adapters.chat.ollama_chat import OllamaChatModel
from monkeygrab.domain.generation_chunk import GenerationChunk


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
    chunks = list(chat_model.stream("prompt"))

    assert [c.text for c in chunks] == ["hi"]
    assert calls[0]["json"]["model"] == "streamed-model"
    assert calls[0]["json"]["options"]["num_ctx"] == 777
    assert calls[0]["timeout"] == 42
    assert calls[0]["json"]["think"] is False


def test_stream_yields_a_generation_chunk_per_line_including_the_empty_done_line(monkeypatch):
    """Unlike the pre-migration adapter (which filtered out any line with an
    empty "response", silently dropping the done=True stats line), every
    parsed line must reach the caller -- matching _ollama_generate_stream,
    which yields the raw dict for every line without filtering."""
    import json as jsonlib

    lines = [
        jsonlib.dumps({"response": "Hel"}).encode(),
        jsonlib.dumps({"response": "lo"}).encode(),
        jsonlib.dumps({"response": "", "done": True, "done_reason": "stop"}).encode(),
    ]
    monkeypatch.setattr(module.requests, "post", lambda **kwargs: _FakeStreamResponse(lines))

    chunks = list(OllamaChatModel("m", num_ctx=100).stream("prompt"))

    assert [c.text for c in chunks] == ["Hel", "lo", ""]
    assert [c.done for c in chunks] == [False, False, True]


def test_stream_final_chunk_carries_the_full_generation_metadata(monkeypatch):
    """The debug dump (guardar_debug_rag) and the tokens-per-second derivation
    in generar_respuesta both read these fields off the done chunk -- losing
    any of them here is exactly the bug this port change fixes."""
    import json as jsonlib

    done_line = {
        "response": "", "done": True, "done_reason": "stop", "model": "m",
        "total_duration": 123, "load_duration": 45, "prompt_eval_count": 10,
        "prompt_eval_duration": 67, "eval_count": 8, "eval_duration": 900,
    }
    lines = [jsonlib.dumps(done_line).encode()]
    monkeypatch.setattr(module.requests, "post", lambda **kwargs: _FakeStreamResponse(lines))

    [chunk] = list(OllamaChatModel("m", num_ctx=100).stream("prompt"))

    assert chunk == GenerationChunk(
        text="", done=True, model="m", done_reason="stop",
        total_duration=123, load_duration=45, prompt_eval_count=10,
        prompt_eval_duration=67, eval_count=8, eval_duration=900,
    )


def test_stream_logs_a_warning_when_generation_stops_early(monkeypatch, caplog):
    import json as jsonlib

    lines = [jsonlib.dumps({"response": "", "done": True, "done_reason": "length"}).encode()]
    monkeypatch.setattr(module.requests, "post", lambda **kwargs: _FakeStreamResponse(lines))

    with caplog.at_level(logging.WARNING):
        list(OllamaChatModel("truncated-model", num_ctx=100).stream("prompt"))

    assert any("truncated-model" in r.message and "length" in r.message for r in caplog.records)


def test_stream_does_not_warn_when_generation_stops_normally(monkeypatch, caplog):
    import json as jsonlib

    lines = [jsonlib.dumps({"response": "", "done": True, "done_reason": "stop"}).encode()]
    monkeypatch.setattr(module.requests, "post", lambda **kwargs: _FakeStreamResponse(lines))

    with caplog.at_level(logging.WARNING):
        list(OllamaChatModel("m", num_ctx=100).stream("prompt"))

    assert caplog.records == []


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

    chunks = list(
        OllamaChatModel("m", num_ctx=100, generate_retries=2, generate_retry_delay=0).stream("prompt")
    )

    assert [c.text for c in chunks] == ["ok"]
    assert len(attempts) == 2


class _FakeModelUnloader:
    def __init__(self):
        self.calls = []

    def unload_all_except(self, keep=None):
        self.calls.append(keep)


def test_stream_without_a_model_unloader_never_calls_one(monkeypatch):
    """model_unloader defaults to None -- must be a true no-op, not an error."""
    import json as jsonlib

    lines = [jsonlib.dumps({"response": "hi", "done": True}).encode()]
    monkeypatch.setattr(module.requests, "post", lambda **kwargs: _FakeStreamResponse(lines))

    list(OllamaChatModel("m", num_ctx=100).stream("prompt"))  # must not raise


def test_stream_unloads_every_other_model_before_the_initial_request(monkeypatch):
    import json as jsonlib

    lines = [jsonlib.dumps({"response": "hi", "done": True}).encode()]
    monkeypatch.setattr(module.requests, "post", lambda **kwargs: _FakeStreamResponse(lines))
    unloader = _FakeModelUnloader()

    list(OllamaChatModel("my-model", num_ctx=100, model_unloader=unloader).stream("prompt"))

    assert unloader.calls == ["my-model"]


def test_stream_unloads_everything_before_retrying_a_5xx(monkeypatch):
    import json as jsonlib

    attempts = []

    def fake_post(**kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            return _FakeStreamResponse([], status_code=503)
        return _FakeStreamResponse([jsonlib.dumps({"response": "ok", "done": True}).encode()])

    monkeypatch.setattr(module.requests, "post", fake_post)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    unloader = _FakeModelUnloader()

    list(
        OllamaChatModel(
            "my-model", num_ctx=100, generate_retries=2, generate_retry_delay=0,
            model_unloader=unloader,
        ).stream("prompt")
    )

    # Initial call keeps "my-model" loaded; the retry unloads everything (keep=None).
    assert unloader.calls == ["my-model", None]


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
