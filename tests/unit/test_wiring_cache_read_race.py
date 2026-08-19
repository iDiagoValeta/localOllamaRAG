"""Regression tests for the cache-read race in rag/engine/wiring.py (#57).

PR #53 put double-checked locking around the check-then-*build* half of
``vector_store``/``lexical_index``, so two concurrent first callers can no
longer both construct the cached object. But the *read* that follows a
cache hit was left outside any lock: the key comparison and the value
fetch are two separate dict lookups, so a thread can validate its own key,
lose the CPU, have a second thread republish the cache slot under a
*different* key, and then hand back the wrong key's value on resume.

For ``vector_store`` this means a query answered against one corpus can be
served the FAISS store of whatever corpus a concurrent
``POST /api/stores/select`` (or BM25 retune, for ``lexical_index``) just
switched to -- a well-formed answer sourced from the wrong documents, with
no error and no log line.

``PausingDict`` below forces this interleaving deterministically without
assuming which internal keys wiring.py's cache reads: it pauses a chosen
thread the first time it subscripts the dict (whatever key that read
happens to be), after the value has already been fetched but before
control returns to the caller. That reproduces the exact race regardless
of whether the cache stores a key and a value as two separate slots or as
one atomically-read tuple, so the same test is valid both before and after
the fix.
"""

import os
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
from monkeygrab.config.paths import RAG_BASE_DIR
from rag.engine import wiring


class PausingDict(dict):
    """Dict whose chosen thread pauses mid-lookup on its first subscript read.

    The value is fetched from the underlying storage (mirroring exactly
    what a real, uninstrumented read would already have captured) and only
    then does the thread block on ``resume_event`` -- so a second thread is
    free to overwrite this dict's contents before the first thread resumes
    and finishes using the value it already fetched.
    """

    def __init__(self, *args, thread_name, ready_event, resume_event, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_name = thread_name
        self._ready_event = ready_event
        self._resume_event = resume_event
        self._paused = False

    def __getitem__(self, item):
        value = super().__getitem__(item)
        if not self._paused and threading.current_thread().name == self._thread_name:
            self._paused = True
            self._ready_event.set()
            assert self._resume_event.wait(timeout=5), "test never resumed the paused thread"
        return value


@pytest.fixture(autouse=True)
def _isolate_wiring_caches():
    """Snapshot/restore the module-level caches around each test.

    A plain content copy rather than a hardcoded shape, so this keeps
    working whether a cache is two separate key/value slots or a single
    atomically-read tuple slot (see #57).
    """
    store_snapshot = dict(wiring._store_cache)
    lexical_snapshot = dict(wiring._lexical_cache)
    yield
    wiring._store_cache.clear()
    wiring._store_cache.update(store_snapshot)
    wiring._lexical_cache.clear()
    wiring._lexical_cache.update(lexical_snapshot)


def test_vector_store_read_does_not_return_a_different_keys_store(monkeypatch):
    """T1 validates the 'en' store is still cached, then a corpus switch to
    'es' republishes the slot before T1's own read finishes -- T1 must get
    back the 'en' store it checked for, never the 'es' store T2 published.
    """
    store_en = object()
    store_es = object()
    config_en = AppConfig()
    config_es = config_en.with_overrides(
        **{"paths.docs_folder": os.path.join(RAG_BASE_DIR, "docs", "es")}
    )

    # Warm the cache through the public API so priming matches whatever
    # internal shape wiring.py currently uses -- the test does not assume it.
    monkeypatch.setattr(wiring, "build_vector_store", lambda cfg: store_en)
    wiring.vector_store(config_en)

    ready = threading.Event()
    resume = threading.Event()
    paused = PausingDict(
        wiring._store_cache, thread_name="T1", ready_event=ready, resume_event=resume
    )
    monkeypatch.setattr(wiring, "_store_cache", paused)

    result = {}
    t1 = threading.Thread(
        target=lambda: result.__setitem__("t1", wiring.vector_store(config_en)), name="T1"
    )
    t1.start()
    assert ready.wait(timeout=5), "T1 never reached its cache read"

    # T2 switches the active store to 'es' while T1 is paused mid-read.
    monkeypatch.setattr(wiring, "build_vector_store", lambda cfg: store_es)
    t2 = threading.Thread(target=wiring.vector_store, args=(config_es,), name="T2")
    t2.start()
    t2.join(timeout=5)
    assert not t2.is_alive(), "T2 never completed the corpus switch"

    resume.set()
    t1.join(timeout=5)
    assert not t1.is_alive(), "T1 hung after resuming"

    assert result["t1"] is store_en, (
        "T1 asked for the 'en' store but got back the 'es' store T2 "
        "published after T1's own key check -- wrong-corpus answer (#57)"
    )


def test_lexical_index_read_does_not_return_a_different_keys_index(monkeypatch):
    """Same race for the BM25 cache: T1's collection check must not be
    answered with the index a concurrent BM25 retune just published.
    """
    store_a = object()
    store_b = object()
    index_a = object()
    index_b = object()
    config = AppConfig()

    monkeypatch.setattr(wiring, "Bm25LexicalIndex", lambda store, retrieval: index_a)
    wiring.lexical_index(store_a, config)

    ready = threading.Event()
    resume = threading.Event()
    paused = PausingDict(
        wiring._lexical_cache, thread_name="T1", ready_event=ready, resume_event=resume
    )
    monkeypatch.setattr(wiring, "_lexical_cache", paused)

    result = {}
    t1 = threading.Thread(
        target=lambda: result.__setitem__("t1", wiring.lexical_index(store_a, config)),
        name="T1",
    )
    t1.start()
    assert ready.wait(timeout=5), "T1 never reached its cache read"

    # T2 retunes BM25 for a different collection while T1 is paused mid-read.
    monkeypatch.setattr(wiring, "Bm25LexicalIndex", lambda store, retrieval: index_b)
    t2 = threading.Thread(target=wiring.lexical_index, args=(store_b, config), name="T2")
    t2.start()
    t2.join(timeout=5)
    assert not t2.is_alive(), "T2 never completed the retune"

    resume.set()
    t1.join(timeout=5)
    assert not t1.is_alive(), "T1 hung after resuming"

    assert result["t1"] is index_a, (
        "T1 asked for store_a's lexical index but got back store_b's index "
        "that T2 published after T1's own key check (#57)"
    )


def test_reset_during_in_flight_build_serializes_on_the_lock(monkeypatch):
    """Coverage gap noted in #57: reset_vector_store_cache() shares
    _store_cache_lock with vector_store(), so a reset racing an in-flight
    build cannot observe (or interleave with) a half-published slot -- it
    runs entirely before the build starts or entirely after it publishes.
    Started once the build is already in progress, it must run after
    publication and leave the cache empty.
    """
    calls = {"count": 0}
    release = threading.Event()

    def blocking_build(config):
        calls["count"] += 1
        release.wait(timeout=5)
        return object()

    monkeypatch.setattr(wiring, "build_vector_store", blocking_build)
    config = AppConfig()

    t_build = threading.Thread(target=wiring.vector_store, args=(config,))
    t_build.start()

    deadline = time.time() + 5
    while calls["count"] < 1 and time.time() < deadline:
        time.sleep(0.001)
    assert calls["count"] == 1, "builder never entered construction"

    t_reset = threading.Thread(target=wiring.reset_vector_store_cache)
    t_reset.start()

    release.set()
    t_build.join(timeout=5)
    t_reset.join(timeout=5)
    assert not t_build.is_alive() and not t_reset.is_alive(), "a thread hung"

    cached_key, cached_store = wiring._store_cache["entry"]
    assert cached_key is None and cached_store is None
