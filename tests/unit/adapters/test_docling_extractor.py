"""DoclingExtractor against a fake Docling interpreter: contract, cache and hard-fail."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from monkeygrab.adapters.extraction.docling_extractor import DoclingExtractor  # noqa: E402

FAKE_SHIM = """
import json, sys
calls = sys.argv[2] + ".calls"
open(calls, "a").write("x")
json.dump(PAGES, open(sys.argv[2], "w"))
"""


def _extractor(tmp_path, pages):
    shim = tmp_path / "shim.py"
    shim.write_text(f"PAGES = {json.dumps(pages)}\n{FAKE_SHIM}", encoding="utf-8")
    return DoclingExtractor(python=sys.executable, cache_dir=tmp_path / "cache", shim=shim)


def _pdf(tmp_path, name="doc.pdf", body=b"%PDF-1.4 fake"):
    pdf = tmp_path / name
    pdf.write_bytes(body)
    return pdf


def test_pages_come_back_zero_based_with_their_text(tmp_path):
    pages = [
        {"page": 1, "text": "Uno"},
        {"page": 2, "text": "<table><tr><td>28.4</td></tr></table>"},
    ]
    out = _extractor(tmp_path, pages).extract(str(_pdf(tmp_path)))
    assert [(p.page, p.text) for p in out] == [
        (0, "Uno"),
        (1, "<table><tr><td>28.4</td></tr></table>"),
    ]


def test_an_unchanged_pdf_is_not_extracted_twice(tmp_path):
    extractor = _extractor(tmp_path, [{"page": 1, "text": "Uno"}])
    pdf = _pdf(tmp_path)
    extractor.extract(str(pdf))
    extractor.extract(str(pdf))
    calls = list((tmp_path / "cache").glob("*.calls"))
    assert len(calls) == 1 and calls[0].read_text() == "x"


def test_no_text_at_all_is_a_failure_not_an_empty_document(tmp_path):
    with pytest.raises(RuntimeError, match="no text"):
        _extractor(tmp_path, [{"page": 1, "text": "  "}]).extract(str(_pdf(tmp_path)))
    assert not list((tmp_path / "cache").glob("*.json"))


def test_a_failing_shim_raises_with_its_error(tmp_path):
    shim = tmp_path / "shim.py"
    shim.write_text(
        "import sys\nsys.stderr.write('layout model missing')\nsys.exit(3)\n", encoding="utf-8"
    )
    extractor = DoclingExtractor(python=sys.executable, cache_dir=tmp_path / "cache", shim=shim)
    with pytest.raises(RuntimeError, match="exited 3.*layout model missing"):
        extractor.extract(str(_pdf(tmp_path)))


def test_without_the_isolated_interpreter_it_says_how_to_install_it(tmp_path):
    extractor = DoclingExtractor(
        python=str(tmp_path / "nope" / "python"), cache_dir=tmp_path / "cache"
    )
    with pytest.raises(RuntimeError, match=".venv-docling"):
        extractor.extract(str(_pdf(tmp_path)))


def test_a_missing_pdf_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        _extractor(tmp_path, []).extract(str(tmp_path / "missing.pdf"))
