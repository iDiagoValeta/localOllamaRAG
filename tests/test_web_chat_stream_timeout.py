"""_chat_stream must not block forever on a wedged Ollama server (issue #237).

_chat_stream is CHAT mode's own streaming path: it calls ollama_client_for
directly instead of going through OllamaChatModel, so #232's fix to that
adapter (passing request_timeout into the cached client) never reached it.
Before this fix the call carried no timeout at all -- httpx's own "no
timeout" -- so a stalled server left the SSE generator, and the Flask worker
thread serving it, blocked indefinitely.

Doubles a real, unresponsive TCP server rather than monkeypatching the ollama
client: the deadline lives inside ollama.Client -> httpx.Client, which a
Python-level stub would bypass entirely (same approach as
tests/unit/adapters/test_ollama_chat.py's own timeout regression test).
"""

import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from monkeygrab.adapters.chat import ollama_chat as ollama_chat_module
from rag.web import app as web_app  # noqa: E402


@pytest.fixture
def _silent_server():
    """Bind a loopback socket that accepts a connection and never replies."""
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

    yield f"http://127.0.0.1:{port}", request_timeout

    stop.set()
    server.close()
    thread.join(timeout=2)


def test_chat_stream_raises_within_its_deadline_against_a_server_that_never_answers(
    monkeypatch, _silent_server
):
    base_url, request_timeout = _silent_server
    monkeypatch.setattr(web_app, "OLLAMA_BASE_URL", base_url)
    monkeypatch.setattr(web_app.rag_engine, "OLLAMA_REQUEST_TIMEOUT", request_timeout)

    # A distinct (base_url, timeout) cache key means this test builds its own
    # real client; clearing avoids leaking it into later tests.
    ollama_chat_module.ollama_client_for.cache_clear()
    try:
        start = time.monotonic()
        with pytest.raises(Exception):  # noqa: B017 -- exact type is ollama/httpx's, not ours to pin
            list(web_app._chat_stream("hola"))
        elapsed = time.monotonic() - start
    finally:
        ollama_chat_module.ollama_client_for.cache_clear()

    # Bounded by request_timeout, not by the server (which never answers) or
    # by the suite's own patience -- and not near-instant either, which would
    # mean the raise came from something other than the deadline expiring.
    assert request_timeout <= elapsed < 5.0
