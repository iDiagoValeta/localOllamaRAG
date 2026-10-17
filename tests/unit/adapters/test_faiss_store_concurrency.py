"""FaissVectorStore has no instance lock against a concurrent delete (#393).

rag/engine/wiring.py hands one FaissVectorStore to every concurrent Flask
request thread (threaded=True), and indexing/deleting a document can run in
one of them while another is answering a query. ``query`` reads
``self._index`` and ``self._rows`` in several unsynchronized steps: it takes
the candidate positions from the search it started, then indexes
``self._rows`` with them. A ``delete_source`` landing between those two steps
replaces both with shorter ones, so the positions the query obtained no
longer address the rows it reads -- an IndexError, or (when the shrunken
index still covers the position) a fragment from the wrong chunk.

Reproduced deterministically, with no sleep and no luck: the real FAISS
index is wrapped in a proxy whose ``search`` signals it is in flight and then
waits, standing in for the query thread being scheduled out mid-query. The
delete runs meanwhile. Under the fix, ``delete_source`` blocks on the same
RLock until the query leaves, so the interleaving is impossible; without it,
the query resumes against a row list that no longer matches the positions it
just received.

The delete is driven from its own thread, not the main one, because with the
lock in place it must block until the paused query finishes. Waiting on its
completion before releasing the query uses a generous timeout purely as
orchestration -- the assertion below never depends on which of the two
outcomes happened, only that the query was not corrupted.
"""

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monkeygrab.adapters.vectorstore.faiss_store import FaissVectorStore
from monkeygrab.config.paths import PathsConfig
from monkeygrab.domain.chunk import Chunk
from monkeygrab.domain.chunk_metadata import ChunkMetadata


def _chunk(source, page, chunk_idx, text="body"):
    return Chunk(text=text, metadata=ChunkMetadata(source=source, page=page, chunk=chunk_idx))


def _paths(tmp_path):
    return PathsConfig(path_db=str(tmp_path / "db"), collection_name="store")


class _PausingIndex:
    """Forwarding proxy over a FAISS index whose ``search`` pauses first.

    Only ``search`` is intercepted; ``d``, ``ntotal`` and ``reconstruct``
    (which ``delete_source`` uses) fall through to the wrapped index, so the
    delete still runs against the pre-delete vectors and produces the same
    shorter index it would without this test in the way.
    """

    def __init__(self, inner, entered, release):
        self._inner = inner
        self._entered = entered
        self._release = release

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def search(self, vector, k):
        self._entered.set()
        assert self._release.wait(timeout=5), "test deadlocked: query was never released"
        return self._inner.search(vector, k)


def test_query_never_reads_rows_from_a_different_state_than_its_search(tmp_path):
    store = FaissVectorStore(_paths(tmp_path))
    # The exact-match victim is not the last row, so a post-delete row list
    # of length one cannot accidentally satisfy the old positions.
    store.add(_chunk("keep.pdf", 0, 0, "kept"), [1.0, 0.0])
    store.add(_chunk("doomed.pdf", 0, 0, "doomed"), [0.0, 1.0])
    store.add(_chunk("doomed.pdf", 0, 1, "doomed"), [0.0, 1.0])

    entered_search = threading.Event()
    release_search = threading.Event()
    store._index = _PausingIndex(store._index, entered_search, release_search)

    query_results = []
    query_errors = []

    def _run_query():
        try:
            query_results.append(store.query([0.0, 1.0], n_results=3))
        except BaseException as e:  # noqa: BLE001 - the race is what is under test
            query_errors.append(e)

    query_thread = threading.Thread(target=_run_query)
    query_thread.start()
    assert entered_search.wait(timeout=5), "query never reached FAISS search"

    delete_return = []
    delete_done = threading.Event()

    def _run_delete():
        try:
            delete_return.append(store.delete_source("doomed.pdf"))
        finally:
            delete_done.set()

    delete_thread = threading.Thread(target=_run_delete)
    delete_thread.start()

    # Without the lock the delete finishes now; with it, it is blocked behind
    # the paused query and this wait simply times out before the release.
    delete_done.wait(timeout=1.0)
    release_search.set()

    query_thread.join(timeout=5)
    delete_thread.join(timeout=5)
    assert not query_thread.is_alive(), "query hung instead of finishing"
    assert not delete_thread.is_alive(), "delete hung behind the query"

    assert query_errors == [], (
        f"a concurrent delete_source broke query: {query_errors!r}"
    )
    assert len(query_results[0]) == 3
    assert delete_return == [2]
    assert store.count() == 1
    assert [f.metadata.source for f in store.query([0.0, 1.0], n_results=3)] == ["keep.pdf"]
