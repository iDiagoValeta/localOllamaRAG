"""Unit tests for monkeygrab.adapters.embedding.jina_clip_embedder.JinaClipEmbedder.

Doubles subprocess.Popen entirely -- no real process, no torch, no CUDA, no
model download -- so these run in milliseconds, in the fast/architecture CI
gate. The fake worker is a small queue-backed line-JSON echo server that
lives entirely in this file; it exercises the SAME background-thread /
timeout code path the real worker talks to, just without a real pipe.
"""

import json
import queue
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from monkeygrab.adapters.embedding import jina_clip_embedder as module
from monkeygrab.adapters.embedding.jina_clip_embedder import JinaClipEmbedder

_DIM = module._TRUNCATE_DIM  # 512, kept as the module's own constant, not re-hardcoded here


def _vector(fill=0.1, n=_DIM):
    return [fill] * n


class _FakeStream:
    """Iterable line stream fed from a queue -- stands in for a Popen pipe.

    `for line in process.stdout` on a real pipe blocks until a line arrives
    and raises StopIteration on EOF; this reproduces exactly that so the
    adapter's background reader threads behave identically against it.
    """

    def __init__(self):
        self._q: "queue.Queue" = queue.Queue()

    def push(self, line: str):
        self._q.put(line)

    def close(self):
        self._q.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._q.get()
        if item is None:
            raise StopIteration
        return item


class _FakeStdin:
    def __init__(self, on_write):
        self.lines = []
        self._on_write = on_write

    def write(self, data):
        self.lines.append(data)
        self._on_write(data)

    def flush(self):
        pass

    def close(self):
        pass


class _FakeProcess:
    """Stands in for subprocess.Popen's return value.

    `behavior(process, request_dict)` is called for every line written to
    stdin (after JSON-decoding it) and decides what, if anything, the fake
    worker "replies" with by pushing onto `process.stdout`.

    `startup_message` controls what's on stdout before any request is ever
    sent (mirrors the real worker's ready/fatal handshake):
      - "ready" (default): push {"event": "ready", "dim": _DIM}.
      - "eof": close stdout immediately, i.e. the process died before
        signaling readiness.
      - None: push nothing -- the test drives stdout manually.
      - any dict: pushed as-is (e.g. a "fatal" startup message).
    """

    def __init__(self, behavior, *, startup_message="ready"):
        self.stdout = _FakeStream()
        self.stderr = _FakeStream()
        self._returncode = None
        self.killed = False
        self.stdin = _FakeStdin(on_write=lambda data: self._on_write(data))
        self._behavior = behavior

        if startup_message == "ready":
            self.stdout.push(json.dumps({"event": "ready", "dim": _DIM}) + "\n")
        elif startup_message == "eof":
            self.stdout.close()
        elif startup_message is not None:
            self.stdout.push(json.dumps(startup_message) + "\n")
        # startup_message is None: caller drives stdout manually.

    def _on_write(self, data: str):
        request = json.loads(data)
        self._behavior(self, request)

    def poll(self):
        return self._returncode

    def kill(self):
        self.killed = True
        self._returncode = -9
        self.stdout.close()
        self.stderr.close()

    def wait(self, timeout=None):
        return self._returncode


def _echo_ok(process: _FakeProcess, request: dict):
    """Default behavior: reply ok=True with a filler vector of the right shape."""
    process.stdout.push(
        json.dumps({"id": request["id"], "ok": True, "vector": _vector()}) + "\n"
    )


def _make_popen(behavior=_echo_ok, *, startup_message="ready"):
    """Return (fake_popen_callable, calls) -- calls records every FakeProcess made."""
    calls = []

    def fake_popen(cmd, **kwargs):
        process = _FakeProcess(behavior, startup_message=startup_message)
        calls.append(process)
        return process

    return fake_popen, calls


def _embedder(monkeypatch, behavior=_echo_ok, *, startup_message="ready", **kwargs):
    fake_popen, calls = _make_popen(behavior, startup_message=startup_message)
    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    embedder = JinaClipEmbedder("fake-python", worker_script="fake-worker.py", **kwargs)
    return embedder, calls


def test_embed_sends_a_text_request_and_returns_the_worker_vector(monkeypatch):
    embedder, calls = _embedder(monkeypatch)

    result = embedder.embed("hello world")

    assert result == _vector()
    sent = json.loads(calls[0].stdin.lines[0])
    assert sent == {"id": 1, "op": "text", "text": "hello world"}


def test_request_ids_increment_across_calls(monkeypatch):
    embedder, calls = _embedder(monkeypatch)

    embedder.embed("first")
    embedder.embed("second")

    ids = [json.loads(line)["id"] for line in calls[0].stdin.lines]
    assert ids == [1, 2]


def test_worker_is_not_started_at_construction(monkeypatch):
    fake_popen, calls = _make_popen()
    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)

    JinaClipEmbedder("fake-python", worker_script="fake-worker.py")

    assert calls == []


def test_worker_starts_once_and_is_reused_across_calls(monkeypatch):
    embedder, calls = _embedder(monkeypatch)

    embedder.embed("one")
    embedder.embed("two")
    embedder.embed("three")

    assert len(calls) == 1


def test_close_before_ever_starting_is_a_no_op(monkeypatch):
    embedder, _ = _embedder(monkeypatch)

    embedder.close()  # must not raise


def test_close_terminates_the_worker_and_a_later_call_starts_a_fresh_one(monkeypatch):
    embedder, calls = _embedder(monkeypatch)
    embedder.embed("one")
    assert len(calls) == 1

    embedder.close()
    calls[0]._returncode = 0  # simulate the OS having reaped the closed process

    embedder.embed("two")

    assert len(calls) == 2


def test_worker_that_dies_before_ready_raises(monkeypatch):
    embedder, _ = _embedder(monkeypatch, startup_message="eof")

    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        embedder.embed("hello")


def test_worker_startup_fatal_message_raises_with_the_reported_reason(monkeypatch):
    embedder, _ = _embedder(
        monkeypatch, startup_message={"event": "fatal", "error": "CUDA not available"}
    )

    with pytest.raises(RuntimeError, match="CUDA not available"):
        embedder.embed("hello")


def test_invalid_json_from_worker_raises(monkeypatch):
    def behavior(process, request):
        process.stdout.push("not-json-at-all\n")

    embedder, _ = _embedder(monkeypatch, behavior)

    with pytest.raises(RuntimeError, match="non-JSON"):
        embedder.embed("hello")


def test_worker_reported_error_raises_with_the_message(monkeypatch):
    def behavior(process, request):
        process.stdout.push(
            json.dumps({"id": request["id"], "ok": False, "error": "boom"}) + "\n"
        )

    embedder, _ = _embedder(monkeypatch, behavior)

    with pytest.raises(RuntimeError, match="boom"):
        embedder.embed("hello")


def test_unexpected_vector_dimension_raises(monkeypatch):
    def behavior(process, request):
        process.stdout.push(
            json.dumps({"id": request["id"], "ok": True, "vector": [0.1, 0.2]}) + "\n"
        )

    embedder, _ = _embedder(monkeypatch, behavior)

    with pytest.raises(RuntimeError, match="unexpected shape"):
        embedder.embed("hello")


def test_mismatched_response_id_raises(monkeypatch):
    def behavior(process, request):
        process.stdout.push(
            json.dumps({"id": 9999, "ok": True, "vector": _vector()}) + "\n"
        )

    embedder, _ = _embedder(monkeypatch, behavior)

    with pytest.raises(RuntimeError, match="desynchronized"):
        embedder.embed("hello")


def test_worker_dying_mid_request_raises_instead_of_hanging(monkeypatch):
    def behavior(process, request):
        process.stdout.close()  # EOF: simulate a crash instead of a reply

    embedder, _ = _embedder(monkeypatch, behavior)

    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        embedder.embed("hello")


def test_worker_that_never_responds_times_out_instead_of_hanging(monkeypatch):
    def behavior(process, request):
        pass  # never reply

    embedder, _ = _embedder(monkeypatch, behavior, request_timeout_s=0.05)

    with pytest.raises(RuntimeError, match="did not respond within"):
        embedder.embed("hello")


def test_worker_marked_dead_refuses_further_calls_without_a_new_process(monkeypatch):
    embedder, calls = _embedder(monkeypatch, startup_message="eof")

    with pytest.raises(RuntimeError):
        embedder.embed("first")

    with pytest.raises(RuntimeError, match="no longer usable"):
        embedder.embed("second")

    # Only one process was ever attempted -- death does not trigger a silent respawn.
    assert len(calls) == 1


def test_embed_image_without_caption_sends_only_an_image_request(monkeypatch):
    def behavior(process, request):
        assert request["op"] == "image"
        process.stdout.push(
            json.dumps({"id": request["id"], "ok": True, "vector": _vector(0.2)}) + "\n"
        )

    embedder, calls = _embedder(monkeypatch, behavior)
    monkeypatch.setattr(module.os.path, "isfile", lambda path: True)

    result = embedder.embed_image("figure.png")

    ops = [json.loads(line)["op"] for line in calls[0].stdin.lines]
    assert ops == ["image"]
    # A lone image vector is just re-normalized (already unit-norm here), so
    # it should equal the L2-normalized filler vector.
    assert result == pytest.approx(module._l2_normalize(_vector(0.2)))


@pytest.mark.parametrize("caption", ["", "   ", "\t\n"])
def test_embed_image_treats_blank_caption_as_no_caption(monkeypatch, caption):
    def behavior(process, request):
        process.stdout.push(
            json.dumps({"id": request["id"], "ok": True, "vector": _vector(0.3)}) + "\n"
        )

    embedder, calls = _embedder(monkeypatch, behavior)
    monkeypatch.setattr(module.os.path, "isfile", lambda path: True)

    embedder.embed_image("figure.png", caption=caption)

    ops = [json.loads(line)["op"] for line in calls[0].stdin.lines]
    assert ops == ["image"]  # never a "text" request for a blank caption


def test_embed_image_raises_when_the_file_does_not_exist(monkeypatch):
    embedder, calls = _embedder(monkeypatch)
    monkeypatch.setattr(module.os.path, "isfile", lambda path: False)

    with pytest.raises(RuntimeError, match="not found"):
        embedder.embed_image("missing.png")

    assert calls == []  # never even tried to talk to a worker


def test_embed_image_with_caption_is_a_normalized_sum_not_an_unnormalized_mean(monkeypatch):
    # Two orthogonal-ish unit vectors so sum-then-normalize is easy to hand-compute
    # and clearly distinguishable from a raw (unnormalized) arithmetic mean.
    image_raw = [1.0, 0.0] + [0.0] * (_DIM - 2)
    caption_raw = [0.0, 1.0] + [0.0] * (_DIM - 2)

    def behavior(process, request):
        vector = image_raw if request["op"] == "image" else caption_raw
        process.stdout.push(
            json.dumps({"id": request["id"], "ok": True, "vector": vector}) + "\n"
        )

    embedder, calls = _embedder(monkeypatch, behavior)
    monkeypatch.setattr(module.os.path, "isfile", lambda path: True)

    result = embedder.embed_image("figure.png", caption="a real caption")

    ops = [json.loads(line)["op"] for line in calls[0].stdin.lines]
    assert ops == ["image", "text"]  # both vectors were actually requested

    expected_sum_normalized = module._l2_normalize(
        [a + b for a, b in zip(image_raw, caption_raw)]
    )
    unnormalized_mean = [(a + b) / 2 for a, b in zip(image_raw, caption_raw)]

    assert result == pytest.approx(expected_sum_normalized)
    assert result != pytest.approx(unnormalized_mean)  # the mean has sub-unit norm
    assert sum(component * component for component in result) == pytest.approx(1.0)


def test_l2_normalize_returns_a_unit_vector():
    result = module._l2_normalize([3.0, 4.0])

    assert result == pytest.approx([0.6, 0.8])


def test_l2_normalize_rejects_a_zero_vector():
    with pytest.raises(RuntimeError, match="zero vector"):
        module._l2_normalize([0.0, 0.0, 0.0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
