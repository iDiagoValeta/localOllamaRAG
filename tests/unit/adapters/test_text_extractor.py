"""TextFileExtractor: a text file becomes one zero-based page."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from monkeygrab.adapters.extraction.text_extractor import TEXT_SUFFIXES, TextFileExtractor


def test_markdown_file_is_one_page_with_its_text(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Título\n\nCuerpo con acentos: canción.\n", encoding="utf-8")
    pages = TextFileExtractor().extract(str(path))
    assert len(pages) == 1
    assert pages[0].page == 0
    assert pages[0].text == "# Título\n\nCuerpo con acentos: canción.\n"


def test_utf8_bom_is_stripped(tmp_path):
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbfhola")
    assert TextFileExtractor().extract(str(path))[0].text == "hola"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TextFileExtractor().extract(str(tmp_path / "nope.txt"))


def test_undecodable_bytes_raise(tmp_path):
    path = tmp_path / "bin.txt"
    path.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(UnicodeDecodeError):
        TextFileExtractor().extract(str(path))


def test_non_text_suffix_is_refused(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF")
    with pytest.raises(ValueError, match=".pdf"):
        TextFileExtractor().extract(str(path))


def test_suffixes_are_the_three_documented():
    assert TEXT_SUFFIXES == (".txt", ".md", ".markdown")
