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
import threading
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


def test_concurrent_embed_calls_each_get_their_own_vector_and_the_worker_survives(monkeypatch):
    """Two overlapping embed() calls must not race on the shared response queue (issue #24).

    Without a lock around the whole request round-trip (worker start, stdin
    write, and the matching queue pop), a second caller's write can land
    while a first caller's reply is still unclaimed; the shared queue does
    not know whose reply is whose, so a caller can pop the OTHER caller's
    message. The id check then reports "protocol desynchronized" and
    _fail_dead kills the worker -- an availability bug, not a wrong-vector
    one, but a real one: every later call fails until the process restarts.

    This is made deterministic with events instead of sleeps: call-a's fake
    reply handler waits (bounded) to see whether call-b's write ever lands
    while call-a's own request is still open. Only an unsynchronized
    caller can make that observable within the window; under the fix,
    call-b is blocked on the lock before it ever reaches the write, so the
    wait always times out and both calls simply succeed in isolation. If
    call-b's write DOES land early, call-a's handler queues only call-a's
    reply and lets call-b's write() return -- so call-b's very next step,
    popping the queue, steals it. That is exactly how the real bug
    manifests, reproduced without hoping two OS threads happen to collide.
    """
    a_write_seen = threading.Event()
    b_write_seen = threading.Event()
    b_may_return = threading.Event()
    vector_a = _vector(0.11)
    vector_b = _vector(0.22)

    def behavior(process, request):
        text = request.get("text")
        if text == "call-a":
            a_write_seen.set()
            interleaved = b_write_seen.wait(timeout=0.3)
            process.stdout.push(
                json.dumps({"id": request["id"], "ok": True, "vector": vector_a}) + "\n"
            )
            if interleaved:
                b_may_return.set()  # let call-b's write() return -- it will steal this reply
        elif text == "call-b":
            b_write_seen.set()
            if b_may_return.wait(timeout=0.3):
                return  # a's reply is already queued for us to steal; nothing to push
            process.stdout.push(
                json.dumps({"id": request["id"], "ok": True, "vector": vector_b}) + "\n"
            )
        else:
            # The priming call below, or the final aliveness check -- plain echo.
            process.stdout.push(
                json.dumps({"id": request["id"], "ok": True, "vector": _vector()}) + "\n"
            )

    embedder, _ = _embedder(monkeypatch, behavior, request_timeout_s=2.0)
    embedder.embed("prime the worker")  # start the worker before the race, out of band

    results = {}
    errors = {}

    def run(label, text):
        try:
            results[label] = embedder.embed(text)
        except Exception as exc:  # noqa: BLE001 -- captured for the assertion below
            errors[label] = exc

    t_a = threading.Thread(target=run, args=("a", "call-a"))
    t_b = threading.Thread(target=run, args=("b", "call-b"))
    t_a.start()
    t_b.start()
    t_a.join(timeout=5)
    t_b.join(timeout=5)

    assert not errors, f"concurrent embed() calls raised: {errors}"
    assert results["a"] == vector_a
    assert results["b"] == vector_b

    assert embedder._dead_reason is None
    assert embedder.embed("still alive") == _vector()


def _recording_thread_class(created_threads):
    """Return a threading.Thread subclass that records every instance it makes.

    Each entry is (thread, target, args) captured from the constructor
    kwargs, not read back off the Thread object afterwards -- so a caller
    can later identify a specific thread by which bound method and which
    worker it was started for, instead of trusting creation order (which
    silently breaks if any other thread gets created in between).
    """
    real_thread = threading.Thread

    class _RecordingThread(real_thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created_threads.append((self, kwargs.get("target"), kwargs.get("args", ())))

    return _RecordingThread


def _find_pump_thread(created_threads, target, worker_process):
    """Identify the recorded Thread for `target` bound to `worker_process`.

    Matches on the target function and on `args[0] is worker_process`
    (every pump thread's first argument is the process it belongs to)
    rather than on position in `created_threads`, so the test still finds
    the right thread even if unrelated threads are created around it.
    """
    matches = [
        thread
        for thread, recorded_target, args in created_threads
        if recorded_target == target and args and args[0] is worker_process
    ]
    assert len(matches) == 1, (
        f"expected exactly one recorded thread for {target!r} on {worker_process!r}, "
        f"found {len(matches)}"
    )
    return matches[0]


def test_a_stale_pump_thread_cannot_poison_a_fresh_workers_response_queue(monkeypatch):
    """Regression test for #47.

    _start_worker reassigns self._responses to a brand-new Queue every time
    it (re)spawns a worker, but _pump_stdout used to resolve self._responses
    dynamically through self instead of a value captured when the thread was
    created. So a pump thread left over from a worker that was close()'d --
    still blocked reading its own (dead) process's stdout -- could reach its
    `finally: self._responses.put(_EOF)` only AFTER a fresh worker had
    already been started, and end up dropping the EOF sentinel into the NEW
    worker's queue instead of its own. The next real call against the new,
    perfectly healthy worker then popped that stale EOF and failed with
    "exited unexpectedly", permanently killing a live process.

    Made deterministic by recording the actual Thread object the first
    worker's stdout pump runs on and join()-ing it -- so the test knows for
    certain that pump has finished running (including its finally block)
    before the second worker is asked to handle a real request, without
    relying on a sleep or guessing at internal state.
    """
    created_threads = []
    monkeypatch.setattr(module.threading, "Thread", _recording_thread_class(created_threads))

    embedder, calls = _embedder(monkeypatch)
    embedder.embed("one")  # starts worker A (calls[0])
    assert len(calls) == 1

    embedder.close()  # worker A is "closed" from the adapter's point of view, but this
    calls[0]._returncode = 0  # fake never closes its stdout pipe on its own -- the test
    # drives that closure explicitly below, once a new worker already exists, to pin down
    # exactly the interleaving the bug depends on.

    embedder.embed("two")  # starts a fresh worker B (calls[1]); self._responses now
    assert len(calls) == 2  # points at a brand-new Queue.

    worker_a_stdout_pump = _find_pump_thread(created_threads, embedder._pump_stdout, calls[0])

    calls[0].stdout.close()  # worker A's stale pump thread unblocks and runs its finally.
    worker_a_stdout_pump.join(timeout=2.0)
    assert not worker_a_stdout_pump.is_alive(), "stale pump thread from worker A never finished"

    # Worker B is alive and was never touched -- a real call against it must succeed.
    result = embedder.embed("three")

    assert result == _vector()
    assert len(calls) == 2  # no extra worker was spawned to recover
    assert embedder._dead_reason is None  # not permanently marked dead (issue #24's failure mode)


def test_a_stale_pump_thread_does_not_leak_its_stderr_into_a_fresh_workers_tail(monkeypatch):
    """Regression test for #47 (the _stderr_tail half of the same bug).

    Same reassignment hazard as _responses, but for _stderr_tail: pre-fix, a
    stale worker's stderr pump appended into whatever self._stderr_tail
    pointed at, so a DEAD worker's stderr lines could end up reported by
    _format_stderr() inside a LIVE worker's eventual failure message --
    misleading diagnostics on exactly the permanent-outage path this
    adapter's hard-fail policy exists to serve.
    """
    created_threads = []
    monkeypatch.setattr(module.threading, "Thread", _recording_thread_class(created_threads))

    embedder, calls = _embedder(monkeypatch)
    embedder.embed("one")  # starts worker A (calls[0])
    assert len(calls) == 1

    embedder.close()
    calls[0]._returncode = 0

    embedder.embed("two")  # starts a fresh worker B (calls[1]); self._stderr_tail now
    assert len(calls) == 2  # points at a brand-new list.

    worker_a_stderr_pump = _find_pump_thread(created_threads, embedder._pump_stderr, calls[0])

    calls[0].stderr.push("stale worker A traceback\n")
    calls[0].stderr.close()  # unblocks the stale pump so it finishes deterministically
    worker_a_stderr_pump.join(timeout=2.0)
    assert not worker_a_stderr_pump.is_alive(), "stale stderr pump from worker A never finished"

    # Worker B's own tail must stay untouched by worker A's stale line.
    assert embedder._stderr_tail == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
