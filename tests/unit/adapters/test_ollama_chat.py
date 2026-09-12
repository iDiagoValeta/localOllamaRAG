"""Unit tests for monkeygrab.adapters.chat.ollama_chat.OllamaChatModel.

Stubs the ollama client and requests.post entirely -- no Ollama server, no
network -- so these run in milliseconds. The one exception is the request
timeout regression test, which talks to a real loopback socket (still no
Ollama server, no external network) because the deadline it pins lives in
the real ollama.Client -> httpx.Client path a Python-level stub bypasses.
"""

import logging
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from monkeygrab.adapters.chat import ollama_chat as module
from monkeygrab.adapters.chat.ollama_chat import OllamaChatModel
from monkeygrab.domain.generation_chunk import GenerationChunk


def _stub_chat(monkeypatch, calls, content="", error=None):
    """Replace the cached client factory with one recording every chat call.

    Each recorded call carries the ``host`` and ``timeout`` its client was
    built for, so a test can assert which server the adapter would have
    talked to, and with what deadline.
    """
    class _FakeClient:
        def __init__(self, host, timeout=None):
            self._host = host
            self._timeout = timeout

        def chat(self, **kwargs):
            calls.append({**kwargs, "host": self._host, "client_timeout": self._timeout})
            if error is not None:
                raise error
            return {"message": {"content": content}}

    monkeypatch.setattr(module, "ollama_client_for", _FakeClient)


def test_generate_uses_the_injected_model_and_num_ctx_not_frozen_defaults(monkeypatch):
    calls = []
    _stub_chat(monkeypatch, calls, content="answer")

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
    _stub_chat(monkeypatch, calls)

    OllamaChatModel("m", num_ctx=100).generate("hello")

    assert calls[0]["think"] is False


def test_generate_talks_to_the_configured_base_url(monkeypatch):
    """Single-shot generation (RECOMP, query decomposition, contextual
    enrichment) must reach the same server streaming does -- it used to go
    wherever the client's ambient OLLAMA_HOST pointed."""
    calls = []
    _stub_chat(monkeypatch, calls)

    OllamaChatModel("m", num_ctx=100, base_url="http://gpu-box:11434").generate("hello")

    assert calls[0]["host"] == "http://gpu-box:11434"


def test_generate_builds_its_client_with_the_configured_request_timeout(monkeypatch):
    """generate() used to build its client with no timeout at all (issue #232):
    stream() applied request_timeout via requests.post, generate() dropped it
    on the floor. The ollama client takes no per-call timeout, only one baked
    in at construction, so this is the only place generate() can hand it over."""
    calls = []
    _stub_chat(monkeypatch, calls)

    OllamaChatModel("m", num_ctx=100, request_timeout=42).generate("hello")

    assert calls[0]["client_timeout"] == 42


def test_generate_puts_the_system_prompt_first(monkeypatch):
    calls = []
    _stub_chat(monkeypatch, calls)

    OllamaChatModel("m", num_ctx=100).generate("question", system="be concise")

    assert calls[0]["messages"] == [
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "question"},
    ]


def test_generate_without_system_sends_a_single_user_message(monkeypatch):
    calls = []
    _stub_chat(monkeypatch, calls)

    OllamaChatModel("m", num_ctx=100).generate("question")

    assert calls[0]["messages"] == [{"role": "user", "content": "question"}]


def test_generate_base64_encodes_images(monkeypatch):
    import base64

    calls = []
    _stub_chat(monkeypatch, calls)

    OllamaChatModel("m", num_ctx=100).generate("describe this", images=[b"\x89PNG raw bytes"])

    sent_images = calls[0]["messages"][0]["images"]
    assert sent_images == [base64.b64encode(b"\x89PNG raw bytes").decode("utf-8")]


def test_generate_hard_fails_on_ollama_error(monkeypatch):
    _stub_chat(monkeypatch, [], error=ConnectionError("ollama down"))

    with pytest.raises(RuntimeError, match="ollama down"):
        OllamaChatModel("m", num_ctx=100).generate("hello")


def test_generate_raises_within_its_deadline_against_a_server_that_never_answers():
    """Doubles a real, unresponsive server (not a Python-level stub) to pin
    the issue #232 regression: generate() had no deadline of its own, so a
    resident-but-stuck Ollama server left it blocked forever -- exactly what
    made a 26-minute gate case (#229) impossible to bound from inside the
    adapter. Nothing here monkeypatches ollama_client_for, so this exercises
    the real ollama.Client -> httpx.Client path generate() actually takes.

    A plain TCP listener that accepts the connection and then never writes
    back is enough: the client's request write succeeds against the kernel's
    receive buffer regardless of whether anything ever reads it, so the
    adapter blocks on the response instead, which is the wait request_timeout
    must bound.
    """
    request_timeout = 0.5

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    stop = threading.Event()

    def _accept_and_stay_silent():
        try:
            conn, _ = server.accept()
        except OSError:
            return  # server closed during teardown before a connection arrived
        stop.wait(request_timeout + 5)  # hold the connection open, reply never sent
        conn.close()

    thread = threading.Thread(target=_accept_and_stay_silent, daemon=True)
    thread.start()

    # A distinct cache key per (base_url, timeout): nothing else in this
    # module builds a real client against this port, but clearing keeps the
    # real client this test builds from lingering in the cache afterwards.
    module.ollama_client_for.cache_clear()
    chat_model = OllamaChatModel(
        "m", num_ctx=100, base_url=f"http://127.0.0.1:{port}", request_timeout=request_timeout
    )

    start = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="Ollama generate failed"):
            chat_model.generate("hello")
        elapsed = time.monotonic() - start
    finally:
        stop.set()
        server.close()
        thread.join(timeout=2)
        module.ollama_client_for.cache_clear()

    # Bounded by request_timeout, not by the server (which never answers) or
    # by the suite's own patience -- and not near-instant either, which would
    # mean the raise came from something other than the deadline expiring.
    assert request_timeout <= elapsed < 5.0


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


def test_stream_posts_to_the_configured_base_url(monkeypatch):
    calls = []
    import json as jsonlib

    def fake_post(url, json, stream, timeout):
        calls.append(url)
        return _FakeStreamResponse([jsonlib.dumps({"response": "hi", "done": True}).encode()])

    monkeypatch.setattr(module.requests, "post", fake_post)

    list(OllamaChatModel("m", num_ctx=100, base_url="http://gpu-box:11434").stream("prompt"))

    assert calls == ["http://gpu-box:11434/api/generate"]


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


# generation_deadline (issue #249)


class _NeverEndingStreamResponse(_FakeStreamResponse):
    """A server that keeps sending tokens and never says done -- the shape of
    a runaway generation. Records whether the adapter let go of it."""

    def __init__(self, delay=0.01):
        super().__init__(lines=[])
        self._delay = delay
        self.closed = False

    def __exit__(self, *exc_info):
        self.closed = True
        return False

    def iter_lines(self):
        while True:
            time.sleep(self._delay)
            yield b'{"response": "x", "done": false}'


def test_stream_closes_a_runaway_request_when_its_deadline_passes(monkeypatch):
    """The evaluation budget abandons the thread consuming this stream, and
    without a deadline of its own the request kept the server's single slot
    for as long as the model cared to generate (issue #249: 96,000 tokens).
    Leaving the ``with requests.post(...)`` block is what closes the
    connection, and the disconnect is what makes Ollama cancel the task."""
    response = _NeverEndingStreamResponse()
    monkeypatch.setattr(module.requests, "post", lambda **kwargs: response)
    model = OllamaChatModel("m", num_ctx=8, generation_deadline=0.05)

    start = time.perf_counter()
    with pytest.raises(RuntimeError, match="deadline"):
        list(model.stream("prompt"))

    assert time.perf_counter() - start < 1.0
    assert response.closed


def test_stream_without_a_deadline_lets_a_slow_stream_finish(monkeypatch):
    # Zero is the product default and means what it always meant: the read
    # timeout is the only bound.
    lines = [b'{"response": "a", "done": false}', b'{"response": "", "done": true}']

    class _Slow(_FakeStreamResponse):
        def iter_lines(self):
            for line in self._lines:
                time.sleep(0.02)
                yield line

    monkeypatch.setattr(module.requests, "post", lambda **kwargs: _Slow(lines))
    model = OllamaChatModel("m", num_ctx=8, generation_deadline=0)
    assert [c.text for c in model.stream("prompt")] == ["a", ""]


def test_generate_bounds_its_client_timeout_by_the_deadline(monkeypatch):
    """generate() is not streamed, so nothing arrives before the answer is
    complete and the client's read timeout is, in effect, its total deadline.
    A deadline shorter than request_timeout must therefore win there."""
    calls = []
    _stub_chat(monkeypatch, calls, content="ok")
    OllamaChatModel("m", num_ctx=8, request_timeout=900, generation_deadline=10).generate("p")
    OllamaChatModel("m", num_ctx=8, request_timeout=900, generation_deadline=0).generate("p")
    OllamaChatModel("m", num_ctx=8, request_timeout=5, generation_deadline=10).generate("p")
    assert [c["client_timeout"] for c in calls] == [10, 900, 5]
