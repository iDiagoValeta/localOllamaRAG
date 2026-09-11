"""Bm25LexicalIndex._ensure_index() has no lock against a concurrent corpus
mutation (#238).

rag/engine/wiring.py's lexical_index() caches one Bm25LexicalIndex per
(store, k1, b) and hands it to every concurrent Flask request thread
(threaded=True). Before this fix, a rebuild was three unsynchronized writes
(entries, then bm25, then cache_key): a thread scheduled out mid-rebuild --
a slow BM25Okapi() call, or plain GIL contention under load -- could resume
and publish its own snapshot, read *before* a concurrent delete_source(),
*after* a second thread's rebuild (triggered by that same delete) had
already published a fresher one. The slower write landed last and won for
all three fields at once, silently reverting the index to cite content the
caller had just deleted.

Reproduced deterministically -- no sleep, no luck -- with the same trick the
issue itself used: delay only the FIRST BM25Okapi() construction behind a
threading.Event, standing in for a thread being scheduled out mid-rebuild,
while a second, concurrent search (triggered by the delete landing in that
same window) races ahead and rebuilds correctly.

Inspects Bm25LexicalIndex._index directly rather than only asserting on a
further search() call: the cache_key mismatch this bug leaves behind is
exactly what makes the *next* search() call self-heal by rebuilding again,
which would silently mask the regression this test exists to catch. The
same "inspect the shared cache's own state, not just its public return
value" approach tests/unit/test_wiring_cache_concurrency.py already uses for
wiring's caches.
"""

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rank_bm25 import BM25Okapi as RealBM25Okapi

from monkeygrab.adapters.lexical import bm25_index as module
from monkeygrab.adapters.lexical.bm25_index import Bm25LexicalIndex
from monkeygrab.config.retrieval import RetrievalConfig
from monkeygrab.domain.chunk_metadata import ChunkMetadata
from monkeygrab.domain.fragment import Fragment


def _fragment(text, source, chunk=0):
    return Fragment(doc=text, metadata=ChunkMetadata(source=source, page=0, chunk=chunk))


class _MutableStore:
    """VectorStore double whose corpus can be mutated between calls --
    standing in for a real FAISS store racing a concurrent delete_source()."""

    def __init__(self, fragments):
        self._fragments = list(fragments)

    def count(self):
        return len(self._fragments)

    def get_page(self, limit, offset):
        return list(self._fragments)

    def delete_source(self, source):
        self._fragments = [f for f in self._fragments if f.metadata.source != source]


def test_a_slower_rebuild_never_overwrites_a_fresher_one_racing_a_delete(monkeypatch):
    config = RetrievalConfig()
    store = _MutableStore(
        [
            _fragment(
                "confidential quarterly salary figures for the finance team",
                "doc0.pdf",
            ),
            _fragment("unrelated public marketing copy", "public.pdf", chunk=1),
            _fragment("another unrelated public paragraph", "public.pdf", chunk=2),
        ]
    )
    index = Bm25LexicalIndex(store, config)

    entered_slow_build = threading.Event()
    release_slow_build = threading.Event()
    call_count = {"n": 0}

    def _delayed_bm25okapi(corpus_tokens, k1, b):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Thread A: signal it has already read the pre-delete corpus and
            # is now inside the slow constructor -- the exact window issue
            # #238 shows a concurrent delete landing in.
            entered_slow_build.set()
            assert release_slow_build.wait(timeout=5), "test deadlocked: never released"
        return RealBM25Okapi(corpus_tokens, k1=k1, b=b)

    monkeypatch.setattr(module, "BM25Okapi", _delayed_bm25okapi)

    slow_results = []
    thread_a = threading.Thread(
        target=lambda: slow_results.append(index.search("confidential", top_n=5))
    )
    thread_a.start()
    assert entered_slow_build.wait(timeout=5), "thread A never reached the slow constructor"

    # The delete that races thread A's in-flight rebuild.
    store.delete_source("doc0.pdf")

    # Thread B: a second, concurrent request triggered by that same delete.
    # Against the unlocked adapter it races ahead and rebuilds immediately;
    # against the locked one it blocks until thread A releases the rebuild
    # lock below -- either way it must end up rebuilding from the post-delete
    # corpus (call_count reaches 2 without ever waiting on
    # release_slow_build, since only the first call delays), so it is not
    # joined with an expectation of either timing here.
    fast_results = []
    thread_b = threading.Thread(
        target=lambda: fast_results.append(index.search("confidential", top_n=5))
    )
    thread_b.start()

    # Let thread A's stale build finish and attempt to publish.
    release_slow_build.set()
    thread_a.join(timeout=5)
    assert not thread_a.is_alive(), "thread A hung instead of publishing"
    thread_b.join(timeout=5)
    assert not thread_b.is_alive(), "thread B hung instead of publishing"

    # Thread B's own rebuild read the corpus after the delete, so its own
    # result must already be correct regardless of the race below.
    assert fast_results[0] == []

    # The decisive check: once both requests have settled, the SHARED index
    # must still reflect thread B's fresher rebuild, not thread A's stale
    # one publishing last in wall-clock time. Checked directly on the cache
    # rather than via one more search() call: a mismatched cache_key left
    # behind by the bug would make that next call rebuild again and quietly
    # self-heal, hiding exactly the regression under test.
    cache_key, entries, _bm25 = index._index
    assert cache_key == (2, config.bm25_k1, config.bm25_b), (
        f"stale rebuild overwrote the fresher one: cache_key={cache_key}"
    )
    assert all(f.metadata.source != "doc0.pdf" for f in entries), (
        "the shared index still cites the deleted document after both "
        f"requests settled: {[f.metadata.source for f in entries]}"
    )
