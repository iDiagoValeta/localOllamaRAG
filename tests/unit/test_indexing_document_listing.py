"""``listar_documentos`` accepts PDF and text suffixes and nothing else."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rag.chat_pdfs  # noqa: F401 -- breaks the rag.engine.wiring/rag.chat_pdfs circular import
from rag.engine.indexing import SUPPORTED_DOCUMENT_SUFFIXES, listar_documentos


def test_lists_pdf_and_text_files_sorted_and_ignores_the_rest(tmp_path):
    for name in ("b.pdf", "a.md", "c.TXT", "notes.markdown", "image.png", "deck.pptx"):
        (tmp_path / name).write_bytes(b"x")
    assert listar_documentos(str(tmp_path)) == ["a.md", "b.pdf", "c.TXT", "notes.markdown"]


def test_missing_folder_is_created_and_empty(tmp_path):
    folder = tmp_path / "new"
    assert listar_documentos(str(folder)) == []
    assert folder.is_dir()


def test_supported_suffixes_are_pdf_plus_text():
    assert SUPPORTED_DOCUMENT_SUFFIXES == (".pdf", ".txt", ".md", ".markdown")
