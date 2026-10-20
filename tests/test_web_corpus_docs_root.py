"""/api/corpus resolves the preset folder under the configurable docs root (issue #450).

Since #344 the docs root is MONKEYGRAB_DOCS_ROOT (the packaged app points it at
a writable copy); /api/corpus built its path from the source tree instead.

Loose test: importing rag.web.app needs the engine stack.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.web import app as web  # noqa: E402


class _Store:
    def count(self):
        return 3


def test_corpus_switch_uses_the_docs_root(monkeypatch, tmp_path):
    (tmp_path / "ca").mkdir()
    selected = []
    monkeypatch.setattr(web, "_DOCS_ROOT", str(tmp_path))
    monkeypatch.setattr(web.rag_engine, "set_docs_folder_runtime", selected.append)
    monkeypatch.setattr(web.rag_engine, "reset_vector_store_cache", lambda: None)
    monkeypatch.setattr(web, "_get_collection", lambda: _Store())
    monkeypatch.setattr(web.rag_engine, "obtener_documentos_indexados", lambda coll: [])
    monkeypatch.setattr(web, "_collection_document_details", lambda coll: [])
    monkeypatch.setitem(web._state, "indexing", False)

    response = web.app.test_client().post("/api/corpus", json={"preset": "ca"})

    assert response.status_code == 200
    assert selected == [str(tmp_path / "ca")]
