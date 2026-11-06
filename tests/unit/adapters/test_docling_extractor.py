"""Docling adapters against a fake Docling interpreter: contract, shared cache and failure policy."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from PIL import Image

from monkeygrab.adapters.extraction.docling_extractor import DoclingExtractor, DoclingImageExtractor

# Writes RESULT (and any FIGURES, {name: (w, h)}) into the output directory,
# and counts its own runs next to it, outside the cache entries.
FAKE_SHIM = """
import json, sys
from pathlib import Path
out = Path(sys.argv[2])
with open(out.parent / "calls", "a") as calls:
    calls.write("x")
(out / "figures").mkdir(parents=True, exist_ok=True)
for name, size in FIGURES.items():
    from PIL import Image
    Image.new("RGB", tuple(size)).save(out / name)
(out / "result.json").write_text(json.dumps(RESULT))
"""


def _shim(tmp_path, pages, figures=(), files=None):
    shim = tmp_path / "shim.py"
    result = {"pages": pages, "figures": list(figures)}
    shim.write_text(
        f"RESULT = {json.dumps(result)}\nFIGURES = {json.dumps(files or {})}\n{FAKE_SHIM}",
        encoding="utf-8",
    )
    return shim


def _extractors(tmp_path, pages, figures=(), files=None):
    shim = _shim(tmp_path, pages, figures, files)
    kwargs = {"python": _python(tmp_path), "cache_dir": tmp_path / "cache", "shim": shim}
    return DoclingExtractor(**kwargs), DoclingImageExtractor(**kwargs)


def _python(tmp_path, version="2.1.0"):
    """An interpreter laid out like a venv with docling installed next to it."""
    env = tmp_path / "venv"
    python = env / "bin" / "python"
    if not python.exists():
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        (env / "lib" / "python3.12" / "site-packages" / f"docling-{version}.dist-info").mkdir(parents=True)
    return str(python)


def _pdf(tmp_path, name="doc.pdf", body=b"%PDF-1.4 fake"):
    pdf = tmp_path / name
    pdf.write_bytes(body)
    return pdf


def _entries(tmp_path):
    return [p for p in (tmp_path / "cache").iterdir() if p.is_dir()]


def test_pages_come_back_zero_based_with_their_text(tmp_path):
    pages = [
        {"page": 1, "text": "Uno"},
        {"page": 2, "text": "<table><tr><td>28.4</td></tr></table>"},
    ]
    text, _ = _extractors(tmp_path, pages)
    assert [(p.page, p.text) for p in text.extract(str(_pdf(tmp_path)))] == [
        (0, "Uno"),
        (1, "<table><tr><td>28.4</td></tr></table>"),
    ]


def test_figures_come_back_zero_based_with_pixels_and_caption(tmp_path):
    figures = [
        {"page": 2, "file": "figures/0.png", "caption": " Figure 1: the model "},
        {"page": 2, "file": "figures/1.png", "caption": ""},
    ]
    files = {"figures/0.png": [40, 30], "figures/1.png": [8, 8]}
    _, images = _extractors(tmp_path, [{"page": 1, "text": "Uno"}], figures, files)
    by_page = images.extract(str(_pdf(tmp_path)))
    assert list(by_page) == [1]
    first, second = by_page[1]
    assert (first.width, first.height, first.ext, first.caption) == (40, 30, "png", "Figure 1: the model")
    assert Image.open(__import__("io").BytesIO(first.image_bytes)).size == (40, 30)
    assert (second.width, second.caption) == (8, "")


def test_text_and_figures_share_one_docling_run(tmp_path):
    text, images = _extractors(
        tmp_path, [{"page": 1, "text": "Uno"}],
        [{"page": 1, "file": "figures/0.png", "caption": ""}], {"figures/0.png": [4, 4]},
    )
    pdf = _pdf(tmp_path)
    text.extract(str(pdf))
    images.extract(str(pdf))
    text.extract(str(pdf))
    assert (tmp_path / "cache" / "calls").read_text() == "x"


def test_no_text_at_all_is_a_failure_not_an_empty_document(tmp_path):
    text, _ = _extractors(tmp_path, [{"page": 1, "text": "  "}])
    with pytest.raises(RuntimeError, match="no text"):
        text.extract(str(_pdf(tmp_path)))
    assert not _entries(tmp_path)


def test_a_failing_shim_raises_for_text_and_yields_no_figures(tmp_path):
    shim = tmp_path / "shim.py"
    shim.write_text("import sys\nsys.stderr.write('layout model missing')\nsys.exit(3)\n", encoding="utf-8")
    kwargs = {"python": _python(tmp_path), "cache_dir": tmp_path / "cache", "shim": shim}
    with pytest.raises(RuntimeError, match="exited 3.*layout model missing"):
        DoclingExtractor(**kwargs).extract(str(_pdf(tmp_path)))
    assert DoclingImageExtractor(**kwargs).extract(str(_pdf(tmp_path))) == {}
    assert not _entries(tmp_path)


def test_without_the_isolated_interpreter_it_says_how_to_install_it(tmp_path):
    extractor = DoclingExtractor(python=str(tmp_path / "nope" / "python"), cache_dir=tmp_path / "cache")
    with pytest.raises(RuntimeError, match=".venv-docling"):
        extractor.extract(str(_pdf(tmp_path)))


def test_an_interpreter_without_docling_is_reported_as_such(tmp_path):
    extractor = DoclingExtractor(python=sys.executable, cache_dir=tmp_path / "cache", shim=_shim(tmp_path, []))
    with pytest.raises(RuntimeError, match="Docling is not installed"):
        extractor.extract(str(_pdf(tmp_path)))


def test_a_missing_pdf_raises_file_not_found(tmp_path):
    text, _ = _extractors(tmp_path, [])
    with pytest.raises(FileNotFoundError):
        text.extract(str(tmp_path / "missing.pdf"))


def test_a_malformed_result_is_not_cached(tmp_path):
    shim = tmp_path / "shim.py"
    shim.write_text(
        "import sys, pathlib\n(pathlib.Path(sys.argv[2]) / 'result.json').write_text('{\"pages\": ')\n",
        encoding="utf-8",
    )
    extractor = DoclingExtractor(python=_python(tmp_path), cache_dir=tmp_path / "cache", shim=shim)
    with pytest.raises(RuntimeError, match="unreadable"):
        extractor.extract(str(_pdf(tmp_path)))
    assert not _entries(tmp_path)


def test_a_figure_path_outside_the_output_is_skipped(tmp_path):
    (tmp_path / "secret.png").write_bytes(b"not for the index")
    _, images = _extractors(
        tmp_path, [{"page": 1, "text": "Uno"}], [{"page": 1, "file": "../../secret.png", "caption": ""}]
    )
    assert images.extract(str(_pdf(tmp_path))) == {}


def test_upgrading_docling_misses_the_cache(tmp_path):
    text, _ = _extractors(tmp_path, [{"page": 1, "text": "Uno"}])
    pdf = _pdf(tmp_path)
    text.extract(str(pdf))
    site = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    (site / "docling-2.1.0.dist-info").rename(site / "docling-2.2.0.dist-info")
    text.extract(str(pdf))
    assert len(_entries(tmp_path)) == 2
