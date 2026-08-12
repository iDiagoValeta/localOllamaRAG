"""Unit tests for monkeygrab.adapters.extraction.mineru_extractor.

Stubs ``subprocess.run`` entirely -- no real MinerU CLI, no GPU, no network --
so these run in milliseconds. The fake ``subprocess.run`` inspects the ``-o``
argument it is given and writes the directory structure MinerU would have
produced, so tests never need to predict the adapter's internal cache-key
hash.

The failure-mode tests are the important ones here: this adapter's whole
reason for existing (issue #6) is refusing to silently degrade the way the
previous fallback chain did, so every way MinerU can fail must
surface as a raised exception with an actionable message, never as empty or
partial output.
"""

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from monkeygrab.adapters.extraction import mineru_extractor as module  # noqa: E402
from monkeygrab.adapters.extraction.mineru_extractor import (  # noqa: E402
    MineruExtractor,
    MineruImageExtractor,
)
from monkeygrab.domain.extracted_image import ExtractedImage  # noqa: E402
from monkeygrab.domain.extracted_page import ExtractedPage  # noqa: E402

# A minimal-but-valid 1x1 transparent PNG, small enough to inline. Used as a
# stand-in for a MinerU-cropped figure -- Pillow can read real
# dimensions from it, unlike arbitrary bytes.
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _dest_from_cmd(cmd):
    """Extract the ``-o`` argument MinerU would have written into."""
    return Path(cmd[cmd.index("-o") + 1])


def _write_valid_output(dest: Path, pdf_stem: str, blocks=None, md_text="# Heading\n\nSome text."):
    """Write the directory tree MinerU produces for a successful extraction."""
    auto = dest / pdf_stem / "auto"
    auto.mkdir(parents=True, exist_ok=True)
    (auto / f"{pdf_stem}.md").write_text(md_text, encoding="utf-8")
    (auto / f"{pdf_stem}_content_list.json").write_text(
        json.dumps(blocks if blocks is not None else []), encoding="utf-8"
    )
    return auto


def _make_pdf(tmp_path: Path, name="paper.pdf") -> Path:
    pdf = tmp_path / name
    pdf.write_bytes(b"%PDF-1.4 fake pdf bytes")
    return pdf


def _make_bin(tmp_path: Path, name="mineru.exe") -> Path:
    bin_path = tmp_path / name
    bin_path.write_text("fake binary", encoding="utf-8")
    return bin_path


def test_extract_builds_the_correct_mineru_command(monkeypatch, tmp_path):
    pdf = _make_pdf(tmp_path)
    bin_path = _make_bin(tmp_path)
    cache_dir = tmp_path / "cache"
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        _write_valid_output(_dest_from_cmd(cmd), pdf.stem)
        return _FakeCompleted()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    MineruExtractor(mineru_bin=str(bin_path), cache_dir=str(cache_dir)).extract(str(pdf))

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == str(bin_path)
    assert cmd[cmd.index("-p") + 1] == str(pdf)
    assert cmd[cmd.index("-b") + 1] == "pipeline"
    assert Path(cmd[cmd.index("-o") + 1]).parent == cache_dir


def test_model_source_defaults_to_local(monkeypatch, tmp_path):
    pdf = _make_pdf(tmp_path)
    bin_path = _make_bin(tmp_path)
    seen_env = {}

    def fake_run(cmd, **kwargs):
        seen_env.update(kwargs["env"])
        _write_valid_output(_dest_from_cmd(cmd), pdf.stem)
        return _FakeCompleted()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.delenv("MINERU_MODEL_SOURCE", raising=False)

    MineruExtractor(mineru_bin=str(bin_path), cache_dir=str(tmp_path / "cache")).extract(str(pdf))

    assert seen_env["MINERU_MODEL_SOURCE"] == "local"


def test_model_source_is_configurable(monkeypatch, tmp_path):
    pdf = _make_pdf(tmp_path)
    bin_path = _make_bin(tmp_path)
    seen_env = {}

    def fake_run(cmd, **kwargs):
        seen_env.update(kwargs["env"])
        _write_valid_output(_dest_from_cmd(cmd), pdf.stem)
        return _FakeCompleted()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    MineruExtractor(
        mineru_bin=str(bin_path), cache_dir=str(tmp_path / "cache"), model_source="modelscope",
    ).extract(str(pdf))

    assert seen_env["MINERU_MODEL_SOURCE"] == "modelscope"


def test_a_preexisting_ambient_model_source_env_var_is_not_overridden(monkeypatch, tmp_path):
    """setdefault semantics: the caller's shell environment wins over this
    adapter's own default, matching the reference MinerU spike's behavior."""
    pdf = _make_pdf(tmp_path)
    bin_path = _make_bin(tmp_path)
    seen_env = {}

    def fake_run(cmd, **kwargs):
        seen_env.update(kwargs["env"])
        _write_valid_output(_dest_from_cmd(cmd), pdf.stem)
        return _FakeCompleted()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setenv("MINERU_MODEL_SOURCE", "modelscope")  # ambient, set before construction

    MineruExtractor(mineru_bin=str(bin_path), cache_dir=str(tmp_path / "cache")).extract(str(pdf))

    assert seen_env["MINERU_MODEL_SOURCE"] == "modelscope"


def test_default_mineru_bin_reads_the_mineru_bin_env_var(monkeypatch, tmp_path):
    bin_path = _make_bin(tmp_path, "custom-mineru")
    monkeypatch.setenv("MINERU_BIN", str(bin_path))

    assert module._default_mineru_bin() == str(bin_path)


def test_default_mineru_bin_falls_back_to_the_windows_venv_layout(monkeypatch, tmp_path):
    """Windows layout is probed first, mirroring composition._isolated_python."""
    monkeypatch.delenv("MINERU_BIN", raising=False)
    monkeypatch.setattr(module, "_PROJECT_ROOT", str(tmp_path))
    windows_bin = tmp_path / ".venv-mineru" / "Scripts" / "mineru.exe"
    windows_bin.parent.mkdir(parents=True)
    windows_bin.write_text("fake binary", encoding="utf-8")

    assert module._default_mineru_bin() == str(windows_bin)


def test_default_mineru_bin_falls_back_to_the_posix_venv_layout(monkeypatch, tmp_path):
    """Non-Windows layout: only ``.venv-mineru/bin/mineru`` exists.

    Driven by creating (or not creating) files under a monkeypatched
    ``_PROJECT_ROOT`` rather than the host OS, since this suite runs on
    Windows -- the Windows candidate is deliberately absent so the function
    must fall through to the POSIX one.
    """
    monkeypatch.delenv("MINERU_BIN", raising=False)
    monkeypatch.setattr(module, "_PROJECT_ROOT", str(tmp_path))
    posix_bin = tmp_path / ".venv-mineru" / "bin" / "mineru"
    posix_bin.parent.mkdir(parents=True)
    posix_bin.write_text("fake binary", encoding="utf-8")

    assert module._default_mineru_bin() == str(posix_bin)


def test_default_mineru_bin_raises_when_neither_venv_layout_exists(monkeypatch, tmp_path):
    monkeypatch.delenv("MINERU_BIN", raising=False)
    monkeypatch.setattr(module, "_PROJECT_ROOT", str(tmp_path))

    with pytest.raises(RuntimeError, match="MinerU binary not found"):
        module._default_mineru_bin()


def test_missing_binary_raises_an_actionable_error(tmp_path):
    pdf = _make_pdf(tmp_path)
    extractor = MineruExtractor(
        mineru_bin=str(tmp_path / "does-not-exist-anywhere"), cache_dir=str(tmp_path / "cache"),
    )

    with pytest.raises(RuntimeError, match="MinerU CLI not found"):
        extractor.extract(str(pdf))


def test_missing_pdf_raises_file_not_found(tmp_path):
    bin_path = _make_bin(tmp_path)
    extractor = MineruExtractor(mineru_bin=str(bin_path), cache_dir=str(tmp_path / "cache"))

    with pytest.raises(FileNotFoundError):
        extractor.extract(str(tmp_path / "missing.pdf"))


def test_nonzero_exit_code_raises_with_the_cli_stderr(monkeypatch, tmp_path):
    pdf = _make_pdf(tmp_path)
    bin_path = _make_bin(tmp_path)

    def fake_run(cmd, **kwargs):
        return _FakeCompleted(returncode=1, stderr="model cache not found: PDF-Extract-Kit-1.0")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    extractor = MineruExtractor(mineru_bin=str(bin_path), cache_dir=str(tmp_path / "cache"))

    with pytest.raises(RuntimeError, match="exit code 1") as excinfo:
        extractor.extract(str(pdf))
    assert "model cache not found" in str(excinfo.value)


def test_no_output_directory_raises(monkeypatch, tmp_path):
    pdf = _make_pdf(tmp_path)
    bin_path = _make_bin(tmp_path)

    def fake_run(cmd, **kwargs):
        return _FakeCompleted(returncode=0)  # exits clean but writes nothing

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    extractor = MineruExtractor(mineru_bin=str(bin_path), cache_dir=str(tmp_path / "cache"))

    with pytest.raises(RuntimeError, match="no output directory"):
        extractor.extract(str(pdf))


def test_empty_markdown_raises_instead_of_falling_back(monkeypatch, tmp_path):
    """An empty extraction must remain visible."""
    pdf = _make_pdf(tmp_path)
    bin_path = _make_bin(tmp_path)

    def fake_run(cmd, **kwargs):
        _write_valid_output(_dest_from_cmd(cmd), pdf.stem, blocks=[], md_text="   \n  ")
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    extractor = MineruExtractor(mineru_bin=str(bin_path), cache_dir=str(tmp_path / "cache"))

    with pytest.raises(RuntimeError, match="empty Markdown"):
        extractor.extract(str(pdf))


def test_missing_content_list_json_raises(monkeypatch, tmp_path):
    pdf = _make_pdf(tmp_path)
    bin_path = _make_bin(tmp_path)

    def fake_run(cmd, **kwargs):
        auto = _dest_from_cmd(cmd) / pdf.stem / "auto"
        auto.mkdir(parents=True)
        (auto / f"{pdf.stem}.md").write_text("real content", encoding="utf-8")
        # deliberately no *_content_list.json
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    extractor = MineruExtractor(mineru_bin=str(bin_path), cache_dir=str(tmp_path / "cache"))

    with pytest.raises(RuntimeError, match="content-list JSON"):
        extractor.extract(str(pdf))


def test_launch_failure_raises_runtime_error(monkeypatch, tmp_path):
    """subprocess.run itself raising (e.g. the resolved path stops being executable)."""
    pdf = _make_pdf(tmp_path)
    bin_path = _make_bin(tmp_path)

    def fake_run(cmd, **kwargs):
        raise OSError("Exec format error")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    extractor = MineruExtractor(mineru_bin=str(bin_path), cache_dir=str(tmp_path / "cache"))

    with pytest.raises(RuntimeError, match="Failed to launch MinerU CLI"):
        extractor.extract(str(pdf))


def test_image_extractor_swallows_cli_failure_and_returns_empty_mapping(tmp_path):
    """Unlike MineruExtractor, MineruImageExtractor follows the ImageExtractor
    port's documented soft-fail carve-out."""
    pdf = _make_pdf(tmp_path)
    extractor = MineruImageExtractor(
        mineru_bin=str(tmp_path / "does-not-exist-anywhere"), cache_dir=str(tmp_path / "cache"),
    )

    assert extractor.extract(str(pdf)) == {}


def test_second_extraction_of_the_same_pdf_reuses_the_cache(monkeypatch, tmp_path):
    pdf = _make_pdf(tmp_path)
    bin_path = _make_bin(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        _write_valid_output(_dest_from_cmd(cmd), pdf.stem)
        return _FakeCompleted()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    extractor = MineruExtractor(mineru_bin=str(bin_path), cache_dir=str(tmp_path / "cache"))

    extractor.extract(str(pdf))
    extractor.extract(str(pdf))

    assert len(calls) == 1  # second call was a cache hit, no subprocess spawned


def test_the_pdf_extractor_and_image_extractor_share_one_cached_run(monkeypatch, tmp_path):
    """IndexCorpus calls PdfExtractor then ImageExtractor on the same PDF --
    the second must not re-run the (expensive) CLI."""
    pdf = _make_pdf(tmp_path)
    bin_path = _make_bin(tmp_path)
    cache_dir = tmp_path / "cache"
    calls = []

    blocks = [
        {"type": "text", "text": "Title", "text_level": 1, "page_idx": 0},
        {
            "type": "image",
            "img_path": "images/fig.png",
            "image_caption": ["Figure 1: a diagram"],
            "page_idx": 0,
        },
    ]

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        auto = _write_valid_output(_dest_from_cmd(cmd), pdf.stem, blocks=blocks)
        (auto / "images").mkdir()
        (auto / "images" / "fig.png").write_bytes(_TINY_PNG)
        return _FakeCompleted()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    pages = MineruExtractor(mineru_bin=str(bin_path), cache_dir=str(cache_dir)).extract(str(pdf))
    images = MineruImageExtractor(mineru_bin=str(bin_path), cache_dir=str(cache_dir)).extract(str(pdf))

    assert len(calls) == 1
    assert pages == [ExtractedPage(page=0, text="# Title\n\nFigure 1: a diagram")]
    assert images == {0: [ExtractedImage(
        image_bytes=_TINY_PNG, width=1, height=1, ext="png", caption="Figure 1: a diagram",
    )]}


def test_a_corrupted_cache_entry_is_discarded_and_reextracted(monkeypatch, tmp_path):
    pdf = _make_pdf(tmp_path)
    bin_path = _make_bin(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        _write_valid_output(_dest_from_cmd(cmd), pdf.stem)
        return _FakeCompleted()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    extractor = MineruExtractor(mineru_bin=str(bin_path), cache_dir=str(tmp_path / "cache"))

    extractor.extract(str(pdf))
    assert len(calls) == 1

    # Simulate a previous run that was interrupted mid-write: the cache
    # directory exists but its Markdown is now empty/corrupt.
    dest = _dest_from_cmd(calls[0])
    (dest / pdf.stem / "auto" / f"{pdf.stem}.md").write_text("", encoding="utf-8")

    extractor.extract(str(pdf))

    assert len(calls) == 2  # bad cache was discarded, MinerU ran again


def test_parses_a_representative_mineru_output_including_a_table_and_a_figure(monkeypatch, tmp_path):
    pdf = _make_pdf(tmp_path)
    bin_path = _make_bin(tmp_path)

    table_html = "<table><tr><td>Model</td><td>BLEU</td></tr><tr><td>Base</td><td>27.3</td></tr></table>"
    blocks = [
        # Page 0: a heading, a paragraph, and page furniture that must be dropped.
        {"type": "text", "text": "Attention Is All You Need", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "We propose a new architecture.", "page_idx": 0},
        {"type": "footer", "text": "NeurIPS 2017", "page_idx": 0},
        {"type": "page_number", "text": "1", "page_idx": 0},
        # Page 1: a table -- the whole point of this adapter.
        {
            "type": "table",
            "img_path": "images/table1.png",
            "table_caption": ["Table 1: Results"],
            "table_body": table_html,
            "page_idx": 1,
        },
        # Page 2: a figure with a caption, plus a page-footnote to drop.
        {
            "type": "image",
            "img_path": "images/fig1.png",
            "image_caption": ["Figure 1: model architecture"],
            "page_idx": 2,
        },
        {"type": "page_footnote", "text": "Work performed at Google Brain.", "page_idx": 2},
    ]

    def fake_run(cmd, **kwargs):
        auto = _write_valid_output(_dest_from_cmd(cmd), pdf.stem, blocks=blocks)
        images_dir = auto / "images"
        images_dir.mkdir()
        (images_dir / "table1.png").write_bytes(_TINY_PNG)
        (images_dir / "fig1.png").write_bytes(_TINY_PNG)
        return _FakeCompleted()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    cache_dir = str(tmp_path / "cache")

    pages = MineruExtractor(mineru_bin=str(bin_path), cache_dir=cache_dir).extract(str(pdf))
    images = MineruImageExtractor(mineru_bin=str(bin_path), cache_dir=cache_dir).extract(str(pdf))

    pages_by_number = {p.page: p.text for p in pages}

    # Page 0: heading rendered as Markdown, paragraph kept, furniture dropped.
    assert pages_by_number[0] == "# Attention Is All You Need\n\nWe propose a new architecture."

    # Page 1: the table survives as literal HTML -- identifiable as a table
    # by any downstream code that looks for a "<table" tag in the chunk text.
    assert "<table>" in pages_by_number[1]
    assert table_html in pages_by_number[1]
    assert "Table 1: Results" in pages_by_number[1]

    # Page 2: only the figure's caption contributes to page text (its pixels
    # go through the image port instead); the footnote is dropped.
    assert pages_by_number[2] == "Figure 1: model architecture"
    assert "Google Brain" not in pages_by_number[2]

    # The figure itself comes back through the image port, not the text one.
    assert images[2] == [ExtractedImage(
        image_bytes=_TINY_PNG, width=1, height=1, ext="png", caption="Figure 1: model architecture",
    )]
    # Tables are never surfaced as images (indexed as HTML text instead).
    assert 1 not in images


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
