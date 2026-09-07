"""Tests for replacing a dead jina-clip worker instead of poisoning the process (#191).

``JinaClipEmbedder`` deliberately never respawns its own worker: a crash loop
must stay a visible, repeated failure rather than hide behind what looks like
uninterrupted operation. Its class docstring names the way out -- "a caller
that wants a fresh worker constructs a new JinaClipEmbedder" -- and this is
that caller. Before this, ``wiring.embedder`` cached one instance for the
life of the process, so a worker killed by the OOM killer made every
subsequent query fail until the whole server was restarted.

The adapter's own policy is unchanged and still pinned by
tests/unit/adapters/test_jina_clip_embedder.py.
"""

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag.chat_pdfs  # noqa: E402,F401
from monkeygrab.config.app_config import AppConfig  # noqa: E402
from rag.engine import wiring  # noqa: E402


class _FakeEmbedder:
    """Stands in for JinaClipEmbedder: only the liveness contract matters."""

    def __init__(self):
        self.unusable = False
        self.closed = False

    @property
    def is_unusable(self) -> bool:
        return self.unusable

    def close(self):
        self.closed = True


def _reset_cache():
    with wiring._embedder_cache_lock:
        wiring._embedder_cache["embedder"] = None


def _patch_builder(monkeypatch):
    built = []

    def _build(_config):
        instance = _FakeEmbedder()
        built.append(instance)
        return instance

    monkeypatch.setattr(wiring, "build_embedder", _build)
    return built


def test_a_live_embedder_is_reused(monkeypatch):
    _reset_cache()
    built = _patch_builder(monkeypatch)
    config = AppConfig()

    first = wiring.embedder(config)
    second = wiring.embedder(config)

    assert first is second
    assert len(built) == 1


def test_a_dead_embedder_is_replaced_on_the_next_call(monkeypatch):
    # The issue's shape: the worker is killed (OOM killer picks it first,
    # being the largest resident process after the model server) and every
    # later query fails with "no longer usable" until app.py is restarted.
    _reset_cache()
    built = _patch_builder(monkeypatch)
    config = AppConfig()

    first = wiring.embedder(config)
    first.unusable = True

    second = wiring.embedder(config)

    assert second is not first
    assert len(built) == 2
    assert second.is_unusable is False


def test_the_replaced_embedder_is_closed_not_leaked(monkeypatch):
    # A dead worker's process object is gone, but close() also releases the
    # pipes and pump threads that otherwise pin the instance alive (#46).
    _reset_cache()
    _patch_builder(monkeypatch)
    config = AppConfig()

    first = wiring.embedder(config)
    first.unusable = True
    wiring.embedder(config)

    assert first.closed is True


def test_release_embedder_closes_the_worker_and_empties_the_slot(monkeypatch):
    # Issue #191's other half: an indexing run that raised left the worker
    # alive holding 1.7 GiB, so the *next* attempt failed on memory the
    # previous failure was holding -- blaming a PID with no visible
    # connection to anything the user did.
    _reset_cache()
    built = _patch_builder(monkeypatch)
    config = AppConfig()

    first = wiring.embedder(config)
    wiring.release_embedder()

    assert first.closed is True
    assert wiring._embedder_cache["embedder"] is None

    second = wiring.embedder(config)
    assert second is not first
    assert len(built) == 2


def test_release_embedder_with_nothing_built_is_a_no_op():
    _reset_cache()
    wiring.release_embedder()
    assert wiring._embedder_cache["embedder"] is None


def test_a_failing_close_still_empties_the_slot(monkeypatch):
    # Releasing VRAM is best-effort; refusing to clear the slot because
    # close() threw would strand the very instance we are replacing.
    _reset_cache()
    monkeypatch.setattr(wiring, "build_embedder", lambda _c: _RefusingEmbedder())

    wiring.embedder(AppConfig())
    wiring.release_embedder()

    assert wiring._embedder_cache["embedder"] is None


class _RefusingEmbedder(_FakeEmbedder):
    def close(self):
        raise OSError("pipe already gone")


def test_replacement_under_concurrency_builds_exactly_one_new_embedder(monkeypatch):
    # Same race the double-checked locking in this module exists to close
    # (#46): two threads observing the same dead instance must not each
    # construct a replacement, since each duplicate loads jina-clip onto the
    # same 8GB card.
    _reset_cache()
    built = _patch_builder(monkeypatch)
    config = AppConfig()

    dead = wiring.embedder(config)
    dead.unusable = True

    start = threading.Barrier(4)
    seen = []

    def _worker():
        start.wait()
        seen.append(wiring.embedder(config))

    threads = [threading.Thread(target=_worker) for _ in range(3)]
    for t in threads:
        t.start()
    start.wait()
    for t in threads:
        t.join()

    assert len(built) == 2, "one initial build plus exactly one replacement"
    assert len({id(e) for e in seen}) == 1
