"""Upload keeps the client's filename, within the same rules as DELETE/GET.

Issues #308 and #309. ``werkzeug.utils.secure_filename`` transliterates or drops
everything outside ASCII, so ``documentación.pdf`` arrived as
``documentacion.pdf`` and ``报告.pdf`` lost its stem entirely. Uploads now go
through ``_safe_pdf_basename``, the validator the delete/serve routes already
used: Unicode survives, path components do not. ``POST /api/docs`` is the
route the UI adds PDFs through.

Doubles the engine: what is under test is the name that reaches disk and the
status code each rejection returns, not indexing.
"""

import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.web import app as web


@pytest.fixture
def docs_folder(tmp_path, monkeypatch):
    folder = tmp_path / "docs"
    monkeypatch.setattr(web.rag_engine, "CARPETA_DOCS", str(folder))
    monkeypatch.setitem(web._state, "indexing", False)
    return folder


@pytest.fixture
def client(docs_folder):
    web.app.config["TESTING"] = True
    with web.app.test_client() as c:
        yield c


def _pdf(filename):
    return {"file": (io.BytesIO(b"%PDF-1.4\n"), filename)}


class _Store:
    """Vector-store double: the name tests never look at its contents."""

    def delete_source(self, _source):
        return 0

    def count(self):
        return 0


@pytest.fixture
def no_index(monkeypatch):
    """Record whether an indexing pass would have started."""
    started = []
    monkeypatch.setattr(web, "_get_collection", lambda: _Store())
    monkeypatch.setattr(
        web.rag_engine, "indexar_documentos", lambda *a, **k: started.append(1) or 0
    )
    monkeypatch.setattr(web.rag_engine, "obtener_documentos_indexados", lambda _coll: [])
    return started


def _upload(client, data):
    return client.post("/api/docs", data=data, content_type="multipart/form-data")


def test_upload_keeps_a_non_latin_name(client, docs_folder, no_index):
    response = _upload(client, _pdf("报告.pdf"))

    assert response.status_code == 200
    assert (docs_folder / "报告.pdf").is_file()


def test_upload_rejects_path_components(client, docs_folder, no_index):
    response = _upload(client, _pdf("../x.pdf"))

    assert response.status_code == 400
    assert response.json["error"] == "Nombre de archivo inválido"
    assert no_index == [], "an invalid name must not start indexing"


def test_upload_keeps_a_latin_accented_name(client, docs_folder, no_index):
    response = _upload(client, _pdf("documentación.pdf"))

    assert response.status_code == 200
    assert (docs_folder / "documentación.pdf").is_file()


def test_upload_rejects_a_file_that_is_not_a_pdf(client, docs_folder, no_index):
    response = _upload(client, _pdf("notes.txt"))

    assert response.status_code == 400
    assert response.json["error"] == "Nombre de archivo inválido"
    assert not docs_folder.exists() or list(docs_folder.iterdir()) == []
    assert no_index == []


def test_upload_validates_every_name_before_writing_any(client, docs_folder, no_index):
    # Saving the valid half first would leave a file behind for a request the
    # server rejected as a whole.
    response = _upload(
        client,
        {"file": [
            (io.BytesIO(b"%PDF-1.4\n"), "valid.pdf"),
            (io.BytesIO(b"%PDF-1.4\n"), "../escape.pdf"),
        ]},
    )

    assert response.status_code == 400
    assert not docs_folder.exists() or list(docs_folder.iterdir()) == []
    assert no_index == []
