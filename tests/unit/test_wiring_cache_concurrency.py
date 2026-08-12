"""Regression tests for the module-level caches in rag/engine/wiring.py (#46).

Each of ``embedder``/``vector_store``/``lexical_index``/``reranker`` used an
unguarded check-then-set: read the cache slot, and if empty, build. Two
threads racing the first call can both see the slot empty and both construct
the expensive object -- for the embedder and reranker specifically, that
means a second worker subprocess or a second copy of model weights competing
for the same 8GB card, with the loser orphaned (see the issue for why
``__del__`` never runs).

Each test below forces exactly that race deterministically: the constructor
double blocks the first caller inside the "build" step on a
``threading.Event``, gives a second caller time to attempt entry, then
releases. Against the pre-fix check-then-set, both threads observe an empty
slot and both call the constructor (2 builds). Against double-checked
locking, the second thread blocks on the lock and, once let through,
observes the slot already populated (1 build).
"""

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

import rag.chat_pdfs  # noqa: F401 -- import before rag.engine.wiring, see its module docstring
from monkeygrab.config.app_config import AppConfig
from rag.engine import wiring


class BlockingBuilder:
    """Constructor double that counts calls and blocks until released.

    Simulates the slow, VRAM-hungry construction (JinaClipEmbedder,
    CrossEncoderReranker, ...) that makes the race in #46 costly: the
    ``threading.Event.wait()`` releases the GIL, so a second thread gets a
    real chance to run concurrently while the first is still "building".
    """

    def __init__(self):
        self.calls = 0
        self._count_lock = threading.Lock()
        self.release_event = threading.Event()

    def __call__(self, *args, **kwargs):
        with self._count_lock:
            self.calls += 1
        self.release_event.wait(timeout=5)
        return object()


def _race_two_callers(target, args, builder):
    """Call ``target(*args)`` from two threads, racing a first-build window.

    ``builder`` is the ``BlockingBuilder`` the caller already patched into
    ``wiring`` in place of the real constructor. Returns the results each
    thread got back; ``builder.calls`` tells the caller how many times the
    underlying constructor actually ran.
    """
    results = [None, None]

    def call(i):
        results[i] = target(*args)

    t1 = threading.Thread(target=call, args=(0,))
    t2 = threading.Thread(target=call, args=(1,))

    t1.start()
    deadline = time.time() + 5
    while builder.calls < 1 and time.time() < deadline:
        time.sleep(0.001)
    assert builder.calls == 1, "first thread never entered the constructor"

    t2.start()
    # Give t2 a real chance to reach the constructor too (buggy code) or to
    # start blocking on the lock (fixed code) before we let t1 finish.
    time.sleep(0.05)

    builder.release_event.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert not t1.is_alive() and not t2.is_alive(), "a caller thread hung"

    return results


@pytest.fixture(autouse=True)
def _reset_wiring_caches():
    """Isolate each test from cache state left by others in the same process."""
    wiring._embedder_cache.update(embedder=None)
    wiring._store_cache.update(key=None, store=None)
    wiring._lexical_cache.update(key=None, index=None)
    wiring._reranker_cache.update(reranker=None)
    yield
    wiring._embedder_cache.update(embedder=None)
    wiring._store_cache.update(key=None, store=None)
    wiring._lexical_cache.update(key=None, index=None)
    wiring._reranker_cache.update(reranker=None)


def test_concurrent_first_calls_build_embedder_exactly_once(monkeypatch):
    builder = BlockingBuilder()
    monkeypatch.setattr(wiring, "build_embedder", builder)

    config = AppConfig()
    results = _race_two_callers(wiring.embedder, (config,), builder)

    assert builder.calls == 1
    assert results[0] is results[1]


def test_concurrent_first_calls_build_vector_store_exactly_once(monkeypatch):
    builder = BlockingBuilder()
    monkeypatch.setattr(wiring, "build_vector_store", builder)

    config = AppConfig()
    results = _race_two_callers(wiring.vector_store, (config,), builder)

    assert builder.calls == 1
    assert results[0] is results[1]


def test_concurrent_first_calls_build_lexical_index_exactly_once(monkeypatch):
    builder = BlockingBuilder()
    monkeypatch.setattr(wiring, "Bm25LexicalIndex", builder)

    config = AppConfig()
    store = object()  # cache key uses id(store); a real VectorStore is not needed
    results = _race_two_callers(wiring.lexical_index, (store, config), builder)

    assert builder.calls == 1
    assert results[0] is results[1]


def test_concurrent_first_calls_build_reranker_exactly_once(monkeypatch):
    builder = BlockingBuilder()
    monkeypatch.setattr(wiring, "CrossEncoderReranker", builder)

    config = AppConfig()
    results = _race_two_callers(wiring.reranker, (config,), builder)

    assert builder.calls == 1
    assert results[0] is results[1]
