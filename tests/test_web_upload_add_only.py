"""Tests for ``POST /api/upload?add_only=1`` replacing an already-indexed file.

Issue #324: the route overwrote the PDF on disk and re-indexed it without
removing the previous chunks. Chunk ids are positional
(``{source}_pag{p}_chunk{c}``), so ``FaissVectorStore.add`` rejected the
duplicate ids, the per-file handler swallowed that and retrieval kept citing
the stale chunks.

Doubles the engine: what is under test is the order of the store calls and the
response shape, not extraction or embedding.
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
    """Vector-store double that records deletes and holds indexed documents."""

    def __init__(self, documents=()):
        self.documents = list(documents)
        self.events = []

    def count(self):
        return len(self.documents)

    def delete_source(self, source):
        self.events.append(("delete", source))
        before = len(self.documents)
        self.documents = [name for name in self.documents if name != source]
        return before - len(self.documents)


@pytest.fixture
def docs_folder(tmp_path, monkeypatch):
    folder = tmp_path / "docs"
    monkeypatch.setattr(web.rag_engine, "CARPETA_DOCS", str(folder))
    monkeypatch.setitem(web._state, "indexing", False)
    monkeypatch.setitem(web._state, "indexing_failed_files", [])
    return folder


@pytest.fixture
def client(docs_folder):
    web.app.config["TESTING"] = True
    with web.app.test_client() as c:
        yield c


def _pdf(filename):
    return {"file": (io.BytesIO(b"%PDF-1.4\n"), filename)}


def _wire_indexer(monkeypatch, store, failed=()):
    def _index(folder, coll, solo_archivos=None, silent=False, progress_callback=None, **kwargs):
        coll.events.append(("index", list(solo_archivos)))
        on_file_failed = kwargs.get("on_file_failed")
        for name in failed:
            if on_file_failed is not None:
                on_file_failed(name, RuntimeError("extraction failed"))
        coll.documents.extend(name for name in solo_archivos if name not in failed)
        return 2 * len(solo_archivos)

    monkeypatch.setattr(web, "_get_collection", lambda: store)
    monkeypatch.setattr(web.rag_engine, "indexar_documentos", _index)
    monkeypatch.setattr(
        web.rag_engine, "obtener_documentos_indexados", lambda coll: list(coll.documents)
    )
    monkeypatch.setattr(web, "_collection_document_details", lambda coll: [])


def test_add_only_deletes_the_old_chunks_before_reindexing(client, monkeypatch):
    store = _FakeStore(documents=["report.pdf"])
    _wire_indexer(monkeypatch, store)

    response = client.post(
        "/api/upload?add_only=1", data=_pdf("report.pdf"), content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert store.events == [("delete", "report.pdf"), ("index", ["report.pdf"])]
    assert response.json["files"] == ["report.pdf"]
    assert response.json["failed_files"] == []


def test_add_only_reports_a_file_that_did_not_index(client, monkeypatch):
    store = _FakeStore(documents=["bad.pdf"])
    _wire_indexer(monkeypatch, store, failed=["bad.pdf"])

    response = client.post(
        "/api/upload?add_only=1", data=_pdf("bad.pdf"), content_type="multipart/form-data"
    )

    # The file is on disk but its chunks did not land: it must not be listed as
    # uploaded, and the names go back to the caller and into the web state.
    assert response.status_code == 200
    assert response.json["ok"] is False
    assert response.json["files"] == []
    assert response.json["failed_files"] == ["bad.pdf"]
    assert web._state["indexing_failed_files"] == ["bad.pdf"]
