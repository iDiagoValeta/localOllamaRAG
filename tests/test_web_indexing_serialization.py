"""Routes that mutate the index are serialized against each other (issue #394).

``_state["indexing"]`` used to be checked outside ``_indexing_lock``, so two
reindexes could both pass the check; ``DELETE /api/docs``
resolved the file and the collection at different instants; and a
``/api/clear`` during a streaming chat was undone when the generator persisted
the turn it had just deleted.

Each test drives the real route with engine doubles and observes the guarantee
the route now owes: one mutation at a time, and the folder/store pair resolved
once at entry. Loose test: importing ``rag.web.app`` needs the engine stack.
"""

import io
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.web import app as web


class _FakeStore:
    """Vector-store double that records deletes and holds indexed documents."""

    def __init__(self, documents=()):
        self.documents = list(documents)
        self.deleted = []

    def count(self):
        return len(self.documents)

    def delete_source(self, source):
        self.deleted.append(source)
        self.documents = [name for name in self.documents if name != source]
        return 1


class _FlagLock:
    """RLock wrapper that records whether the caller holds it."""

    def __init__(self):
        self._lock = threading.RLock()
        self.held = 0

    def __enter__(self):
        self._lock.acquire()
        self.held += 1
        return self

    def __exit__(self, *_exc):
        self.held -= 1
        self._lock.release()


@pytest.fixture
def docs_folder(tmp_path, monkeypatch):
    folder = tmp_path / "docs"
    monkeypatch.setattr(web.rag_engine, "CARPETA_DOCS", str(folder))
    for key, value in (
        ("indexing", False),
        ("indexing_failed", False),
        ("indexing_error", None),
        ("indexing_done_empty", False),
        ("indexing_progress", None),
        ("indexing_failed_files", []),
    ):
        monkeypatch.setitem(web._state, key, value)
    return folder


@pytest.fixture
def client(docs_folder):
    web.app.config["TESTING"] = True
    with web.app.test_client() as c:
        yield c


@pytest.fixture
def history(monkeypatch):
    """Capture the history I/O without touching the user's real file."""
    monkeypatch.setitem(web._state, "historial_chat", [])
    calls = []
    monkeypatch.setattr(
        web.rag_engine, "limpiar_historial", lambda entries: calls.append(("clear", list(entries)))
    )
    monkeypatch.setattr(
        web.rag_engine, "guardar_historial", lambda entries: calls.append(("save", list(entries)))
    )
    return calls


def _pdf(filename):
    return {"file": (io.BytesIO(b"%PDF-1.4\n"), filename)}


def test_reindex_while_a_run_is_in_progress_starts_nothing(client, monkeypatch):
    """A reindex already running is reported (202) and clears nothing."""
    monkeypatch.setitem(web._state, "indexing", True)
    monkeypatch.setattr(web, "_reset_db", lambda: pytest.fail("a running index must not be cleared"))

    response = client.post("/api/reindex")

    assert response.status_code == 202
    assert response.json["indexing"] is True


def test_upload_while_a_run_is_in_progress_is_rejected_and_writes_nothing(
    client, docs_folder, monkeypatch
):
    """An upload cannot index beside another run: 409, and no PDF lands."""
    monkeypatch.setitem(web._state, "indexing", True)

    response = client.post(
        "/api/docs", data=_pdf("paper.pdf"), content_type="multipart/form-data"
    )

    assert response.status_code == 409
    assert response.json["code"] == "indexing_in_progress"
    assert not docs_folder.exists() or list(docs_folder.iterdir()) == []


def test_a_second_store_switch_is_rejected_while_the_first_runs(
    client, docs_folder, monkeypatch
):
    """The claim must be visible before the switch finishes.

    Without it both requests pass the ``indexing`` check and the second
    switches the store while the first is still resolving its collection.
    """
    entered = threading.Event()
    release = threading.Event()
    switched = []
    store = _FakeStore(documents=["paper.pdf"])

    def _slow_collection():
        entered.set()
        release.wait(5)
        return store

    monkeypatch.setattr(web, "_get_collection", _slow_collection)
    monkeypatch.setattr(web.rag_engine, "resolver_carpeta_store", lambda name: f"/stores/{name}")
    monkeypatch.setattr(web.rag_engine, "guardar_ajustes_persistidos", lambda: None)
    monkeypatch.setattr(web.rag_engine, "reset_vector_store_cache", lambda: None)
    monkeypatch.setattr(
        web.rag_engine,
        "set_docs_folder_runtime",
        lambda folder: switched.append(folder),
    )
    monkeypatch.setattr(web.rag_engine, "obtener_documentos_indexados", lambda coll: [])
    monkeypatch.setattr(web, "_collection_document_details", lambda coll: [])

    first = {}

    def _first_request():
        with web.app.test_client() as c:
            first["response"] = c.post("/api/stores/select", json={"name": "en"})

    thread = threading.Thread(target=_first_request)
    thread.start()
    try:
        assert entered.wait(5), "the first switch never reached the store"
        response = client.post("/api/stores/select", json={"name": "es"})
    finally:
        release.set()
        thread.join(5)

    assert response.status_code == 409
    assert response.json["error"] == "indexing_in_progress"
    assert first["response"].status_code == 200
    assert switched == ["/stores/en"]


def test_delete_doc_deletes_the_chunks_while_holding_the_lock(client, tmp_path, monkeypatch):
    """File and collection must be resolved under the same claim."""
    folder = tmp_path / "docs"
    folder.mkdir()
    pdf = folder / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(web.rag_engine, "CARPETA_DOCS", str(folder))

    lock = _FlagLock()
    monkeypatch.setattr(web, "_indexing_lock", lock)

    store = _FakeStore(documents=["paper.pdf"])
    held = []
    store.delete_source = lambda source: held.append(lock.held)
    monkeypatch.setattr(web, "_get_collection", lambda: store)
    monkeypatch.setattr(web.rag_engine, "obtener_documentos_indexados", lambda coll: [])

    response = client.delete("/api/docs/paper.pdf")

    assert response.status_code == 200
    assert held == [1]
    assert not pdf.exists()



def test_delete_doc_is_refused_while_an_index_run_holds_the_turn(client, tmp_path, monkeypatch):
    """The background indexer adds chunks outside the lock (issue #453)."""
    folder = tmp_path / "docs"
    folder.mkdir()
    pdf = folder / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(web.rag_engine, "CARPETA_DOCS", str(folder))
    store = _FakeStore(documents=["paper.pdf"])
    deleted = []
    store.delete_source = deleted.append
    monkeypatch.setattr(web, "_get_collection", lambda: store)
    monkeypatch.setitem(web._state, "indexing", True)

    response = client.delete("/api/docs/paper.pdf")

    assert response.status_code == 409
    assert response.get_json()["error"] == "indexing_in_progress"
    assert deleted == [] and pdf.exists()

def test_clear_bumps_the_history_generation(client, history):
    before = web._state["history_generation"]

    response = client.post("/api/clear")

    assert response.status_code == 200
    assert web._state["history_generation"] == before + 1


def test_a_turn_streaming_across_a_clear_is_not_persisted(client, history, monkeypatch):
    def _stream(_pregunta):
        yield "hola"
        # The user clears the chat while the model is still answering.
        with web.app.test_request_context("/api/clear", method="POST"):
            web.api_clear()

    monkeypatch.setattr(web, "_chat_stream", _stream)

    response = client.post("/api/chat", json={"message": "¿Qué tal?"})

    assert response.status_code == 200
    assert "event: done" in response.get_data(as_text=True)
    assert web._state["historial_chat"] == []
    assert history == [("clear", [])]


def test_a_completed_turn_is_still_persisted(client, history, monkeypatch):
    monkeypatch.setattr(web, "_chat_stream", lambda _pregunta: iter(["hola"]))

    response = client.post("/api/chat", json={"message": "¿Qué tal?"})

    assert response.status_code == 200
    response.get_data()
    assert web._state["historial_chat"] == [
        {"role": "user", "content": "¿Qué tal?"},
        {"role": "assistant", "content": "hola"},
    ]
    assert history == [("save", web._state["historial_chat"])]
