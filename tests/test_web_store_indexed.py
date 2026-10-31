"""A store counts as indexed only when it holds rows (issue #451).

Loose test: importing rag.web.app needs the engine stack.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.web import app as web


def _entry(monkeypatch, tmp_path):
    folder = tmp_path / "docs" / "ca"
    folder.mkdir(parents=True)
    path_db = tmp_path / "vector_db" / "ca_eg2"
    monkeypatch.setattr(web.rag_engine, "_derivar_paths_db", lambda f: (str(path_db), "docs_ca"))
    return path_db / "docs_ca", lambda: web._store_entry("ca", "Valencià", str(folder))


def test_an_opened_but_empty_store_is_not_indexed(monkeypatch, tmp_path):
    store_dir, entry = _entry(monkeypatch, tmp_path)
    store_dir.mkdir(parents=True)  # what FaissVectorStore.__init__ leaves behind

    assert entry()["indexed"] is False


def test_a_store_with_rows_is_indexed(monkeypatch, tmp_path):
    store_dir, entry = _entry(monkeypatch, tmp_path)
    store_dir.mkdir(parents=True)
    (store_dir / "meta.jsonl").write_text('{"id": "a"}\n', encoding="utf-8")

    assert entry()["indexed"] is True
