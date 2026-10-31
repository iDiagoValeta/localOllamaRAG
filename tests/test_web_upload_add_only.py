"""``POST /api/docs`` indexes only the uploaded files (issue #554).

Adding a PDF used to go through ``/api/reindex``, which cleared the store and
re-embedded (and, with contextual retrieval on, re-contextualized with the LLM)
every document already indexed. These tests drive the real route against a
store double and pin what the user is owed: documents already indexed keep
their chunks and are not handed to the indexer again, and a re-uploaded name
replaces that document instead of duplicating it.

Doubles the engine: what is under test is which store calls the route makes
and the response shape, not extraction or embedding. Loose test: importing
``rag.web.app`` needs the engine stack.
"""

import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.web import app as web


class _FakeStore:
    """Vector-store double holding ``{source: chunks}`` and recording its calls."""

    def __init__(self, chunks_by_source=None):
        self.chunks = dict(chunks_by_source or {})
        self.events = []

    def count(self):
        return sum(self.chunks.values())

    def clear(self):
        self.events.append(("clear",))
        self.chunks.clear()

    def delete_source(self, source):
        self.events.append(("delete", source))
        return self.chunks.pop(source, 0)


@pytest.fixture
def docs_folder(tmp_path, monkeypatch):
    folder = tmp_path / "docs"
    monkeypatch.setattr(web.rag_engine, "CARPETA_DOCS", str(folder))
    for key, value in (
        ("indexing", False),
        ("indexing_done_empty", False),
        ("indexing_failed_files", []),
    ):
        monkeypatch.setitem(web._state, key, value)
    return folder


@pytest.fixture
def client(docs_folder):
    web.app.config["TESTING"] = True
    with web.app.test_client() as c:
        yield c


def _pdfs(*names):
    return {"file": [(io.BytesIO(b"%PDF-1.4\n"), name) for name in names]}


def _wire(monkeypatch, store, failed=()):
    """Double the engine; return the list of ``solo_archivos`` each pass received."""
    passes = []

    def _index(folder, coll, solo_archivos=None, silent=False, **kwargs):
        passes.append(list(solo_archivos))
        for name in solo_archivos:
            if name in failed:
                kwargs["on_file_failed"](name, RuntimeError("extraction failed"))
            else:
                coll.chunks[name] = 3
        return 3 * len(solo_archivos)

    monkeypatch.setattr(web, "_get_collection", lambda: store)
    monkeypatch.setattr(web.rag_engine, "indexar_documentos", _index)
    monkeypatch.setattr(
        web.rag_engine, "obtener_documentos_indexados", lambda coll: sorted(coll.chunks)
    )
    return passes


def _upload(client, *names):
    return client.post("/api/docs", data=_pdfs(*names), content_type="multipart/form-data")


def test_a_second_upload_keeps_the_first_documents_chunks(client, docs_folder, monkeypatch):
    store = _FakeStore({"first.pdf": 3})
    passes = _wire(monkeypatch, store)

    response = _upload(client, "second.pdf")

    assert response.status_code == 200
    assert response.json["ok"] is True
    assert ("clear",) not in store.events
    assert store.chunks == {"first.pdf": 3, "second.pdf": 3}
    assert passes == [["second.pdf"]], "an indexed document must not reach the indexer again"
    assert response.json["documents"] == ["first.pdf", "second.pdf"]
    assert response.json["total_fragments"] == 6
    assert (docs_folder / "second.pdf").is_file()


def test_a_reuploaded_name_replaces_that_document_instead_of_duplicating_it(
    client, monkeypatch
):
    store = _FakeStore({"report.pdf": 5, "other.pdf": 3})
    passes = _wire(monkeypatch, store)

    response = _upload(client, "report.pdf")

    assert response.status_code == 200
    # Old chunks go first: ids are positional, so adding over them would trip
    # the store's duplicate-id guard and leave the stale text answering.
    assert store.events == [("delete", "report.pdf")]
    assert passes == [["report.pdf"]]
    assert store.chunks == {"report.pdf": 3, "other.pdf": 3}


def test_the_same_name_twice_in_one_request_is_indexed_once(client, monkeypatch):
    store = _FakeStore()
    passes = _wire(monkeypatch, store)

    response = _upload(client, "twin.pdf", "twin.pdf")

    assert response.status_code == 200
    assert passes == [["twin.pdf"]]


def test_a_file_that_did_not_index_is_reported_not_listed_as_added(client, monkeypatch):
    store = _FakeStore({"kept.pdf": 3})
    _wire(monkeypatch, store, failed=["bad.pdf"])

    response = _upload(client, "good.pdf", "bad.pdf")

    assert response.status_code == 500
    assert response.json["ok"] is False
    assert response.json["code"] == "upload_index_failed"
    assert response.json["files"] == ["good.pdf"]
    assert response.json["failed_files"] == ["bad.pdf"]
    assert response.json["documents"] == ["good.pdf", "kept.pdf"]
    assert web._state["indexing_failed_files"] == ["bad.pdf"]


def test_an_indexer_that_fails_every_file_still_reports_the_store_it_left(client, monkeypatch):
    """The replaced document's chunks are gone, so the answer must not list it."""
    store = _FakeStore({"report.pdf": 5, "other.pdf": 3})
    _wire(monkeypatch, store, failed=["report.pdf"])

    def _all_fail(folder, coll, solo_archivos=None, silent=False, **kwargs):
        for name in solo_archivos:
            kwargs["on_file_failed"](name, RuntimeError("extraction failed"))
        raise RuntimeError("Indexing failed on all 1 file(s)")

    monkeypatch.setattr(web.rag_engine, "indexar_documentos", _all_fail)

    response = _upload(client, "report.pdf")

    assert response.status_code == 500
    assert response.json["ok"] is False
    assert response.json["code"] == "upload_failed"
    assert response.json["failed_files"] == ["report.pdf"]
    assert response.json["documents"] == ["other.pdf"]
    assert response.json["total_fragments"] == 3
    assert web._state["indexing_failed_files"] == ["report.pdf"]
    assert web._state["indexing"] is False


def test_an_indexer_that_breaks_midway_reports_what_did_not_land(client, monkeypatch):
    store = _FakeStore({"report.pdf": 5})
    _wire(monkeypatch, store)

    def _breaks(folder, coll, solo_archivos=None, silent=False, **kwargs):
        coll.chunks["landed.pdf"] = 3
        raise RuntimeError("boom")

    monkeypatch.setattr(web.rag_engine, "indexar_documentos", _breaks)

    response = _upload(client, "report.pdf", "landed.pdf")

    assert response.status_code == 500
    assert response.json["code"] == "upload_failed"
    assert response.json["failed_files"] == ["report.pdf"]
    assert response.json["files"] == ["landed.pdf"]
    assert response.json["documents"] == ["landed.pdf"]


def test_a_run_merges_into_the_earlier_failed_files_warning(client, monkeypatch):
    monkeypatch.setitem(web._state, "indexing_failed_files", ["old.pdf", "retried.pdf"])
    _wire(monkeypatch, _FakeStore(), failed=["bad.pdf"])

    response = _upload(client, "retried.pdf", "bad.pdf")

    assert response.status_code == 500
    # old.pdf is still missing and stays reported; retried.pdf landed now.
    assert web._state["indexing_failed_files"] == ["old.pdf", "bad.pdf"]


def test_a_clean_upload_keeps_the_warning_about_other_files(client, monkeypatch):
    monkeypatch.setitem(web._state, "indexing_failed_files", ["old.pdf"])
    _wire(monkeypatch, _FakeStore())

    response = _upload(client, "fresh.pdf")

    assert response.status_code == 200
    assert web._state["indexing_failed_files"] == ["old.pdf"]


def test_the_first_upload_into_an_empty_store_clears_the_empty_marker(client, monkeypatch):
    monkeypatch.setitem(web._state, "indexing_done_empty", True)
    _wire(monkeypatch, _FakeStore())

    response = _upload(client, "first.pdf")

    assert response.status_code == 200
    assert web._state["indexing_done_empty"] is False


def test_the_indexing_slot_is_free_again_after_an_upload(client, monkeypatch):
    _wire(monkeypatch, _FakeStore())

    _upload(client, "a.pdf")

    assert web._state["indexing"] is False
