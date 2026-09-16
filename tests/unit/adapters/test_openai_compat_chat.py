"""Unit tests for OpenAICompatChatModel. ``requests.post`` is stubbed: no server."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import requests

from monkeygrab.adapters.chat import openai_compat_chat as module
from monkeygrab.adapters.chat.openai_compat_chat import OpenAICompatChatModel


class _Response:
    def __init__(self, *, status=200, body=None, lines=()):
        self.status_code = status
        self._body = body
        self._lines = list(lines)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)

    def json(self):
        return self._body

    def iter_lines(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _stub_post(monkeypatch, calls, response):
    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return response
    monkeypatch.setattr(module.requests, "post", fake_post)


def _model(**overrides):
    kwargs = {"base_url": "http://127.0.0.1:8000/v1/", "options": {"temperature": 0.15, "max_tokens": 4096}}
    kwargs.update(overrides)
    return OpenAICompatChatModel("qwen3-vl-30b", **kwargs)


def test_generate_posts_openai_shape_and_returns_content(monkeypatch):
    calls = []
    _stub_post(monkeypatch, calls, _Response(body={"choices": [{"message": {"content": "hola"}}]}))
    assert _model(api_key="k").generate("pregunta", system="sé breve") == "hola"
    call = calls[0]
    assert call["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer k"
    assert call["timeout"] == 900
    payload = call["json"]
    assert payload["model"] == "qwen3-vl-30b"
    assert payload["stream"] is False
    assert payload["temperature"] == 0.15 and payload["max_tokens"] == 4096
    assert payload["messages"] == [
        {"role": "system", "content": "sé breve"},
        {"role": "user", "content": "pregunta"},
    ]


def test_generate_without_api_key_sends_no_authorization_header(monkeypatch):
    calls = []
    _stub_post(monkeypatch, calls, _Response(body={"choices": [{"message": {"content": ""}}]}))
    _model().generate("x")
    assert "Authorization" not in calls[0]["headers"]


def test_generate_sends_images_as_data_uri_parts(monkeypatch):
    calls = []
    _stub_post(monkeypatch, calls, _Response(body={"choices": [{"message": {"content": "una figura"}}]}))
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 8
    jpeg = b"\xff\xd8\xff" + b"0" * 8
    _model().generate("describe", images=[png, jpeg])
    content = calls[0]["json"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_generate_sends_response_format_as_json_schema(monkeypatch):
    calls = []
    _stub_post(monkeypatch, calls, _Response(body={"choices": [{"message": {"content": "[]"}}]}))
    schema = {"type": "array", "items": {"type": "string"}}
    _model().generate("quiz", response_format=schema)
    assert calls[0]["json"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "reply", "schema": schema},
    }


def test_generate_raises_runtime_error_on_http_failure(monkeypatch):
    _stub_post(monkeypatch, [], _Response(status=503))
    with pytest.raises(RuntimeError, match="qwen3-vl-30b"):
        _model().generate("x")


def test_generate_raises_runtime_error_on_malformed_body(monkeypatch):
    _stub_post(monkeypatch, [], _Response(body={"unexpected": True}))
    with pytest.raises(RuntimeError):
        _model().generate("x")


def _sse(obj):
    return ("data: " + json.dumps(obj)).encode("utf-8")


def test_stream_yields_deltas_then_one_done_chunk_with_usage(monkeypatch):
    calls = []
    lines = [
        _sse({"model": "served-model", "choices": [{"delta": {"content": "Ho"}, "finish_reason": None}]}),
        b"",
        _sse({"choices": [{"delta": {"content": "la"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 2}}),
        b"data: [DONE]",
    ]
    _stub_post(monkeypatch, calls, _Response(lines=lines))
    chunks = list(_model().stream("pregunta", system="sys"))
    assert [c.text for c in chunks] == ["Ho", "la", ""]
    assert [c.done for c in chunks] == [False, False, True]
    final = chunks[-1]
    assert final.model == "served-model"
    assert final.done_reason == "stop"
    assert final.prompt_eval_count == 12 and final.eval_count == 2
    assert final.eval_duration is None
    payload = calls[0]["json"]
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert calls[0]["stream"] is True


def test_stream_logs_early_stop_and_reports_it(monkeypatch, caplog):
    lines = [
        _sse({"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "length"}]}),
        b"data: [DONE]",
    ]
    _stub_post(monkeypatch, [], _Response(lines=lines))
    with caplog.at_level("WARNING"):
        chunks = list(_model().stream("q"))
    assert chunks[-1].done_reason == "length"
    assert "finish_reason=length" in caplog.text


def test_stream_raises_runtime_error_on_http_failure(monkeypatch):
    _stub_post(monkeypatch, [], _Response(status=500))
    with pytest.raises(RuntimeError, match="qwen3-vl-30b"):
        list(_model().stream("q"))


def test_stream_raises_when_the_connection_breaks_mid_stream(monkeypatch):
    def broken():
        yield _sse({"choices": [{"delta": {"content": "a"}, "finish_reason": None}]})
        raise requests.ConnectionError("reset")

    response = _Response()
    response.iter_lines = broken
    _stub_post(monkeypatch, [], response)
    with pytest.raises(RuntimeError, match="reset"):
        list(_model().stream("q"))
