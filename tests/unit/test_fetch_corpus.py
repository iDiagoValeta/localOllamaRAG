"""``tools/fetch_corpus.py`` must not bless a truncated PDF as present.

Issue #386. ``_is_valid_pdf`` only checked a 10 kB floor and the ``%PDF``
header, so a download cut off mid-file passed ``--check`` and
``packaging/build_exe.py`` (#345) froze a bundle whose corpus member could not
be read. A finished PDF ends with a ``%%EOF`` trailer inside its last KiB; that
is the cheapest local proof that the file is whole.

``tools/fetch_corpus.py`` imports ``requests`` at module scope even though
``_is_valid_pdf`` never uses it. The fast gate's architecture job installs no
project dependencies, so the module is loaded with a placeholder ``requests``
only when the real one is absent, and the placeholder is removed afterwards.
"""

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "tools" / "fetch_corpus.py"


def _load_fetch_corpus():
    """Load the tool by path, standing in for ``requests`` when it is absent."""
    fake = None
    if importlib.util.find_spec("requests") is None:
        fake = types.ModuleType("requests")
        sys.modules["requests"] = fake
    try:
        spec = importlib.util.spec_from_file_location("fetch_corpus_under_test", _SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if fake is not None:
            del sys.modules["requests"]


fetch_corpus = _load_fetch_corpus()


def _write_pdf(path, *, trailer=b"%%EOF\n", trailing=b"", filler=b"x" * 12_000):
    """Write a plausible PDF-sized file with the header, filler and trailer."""
    path.write_bytes(b"%PDF-1.7\n" + filler + trailer + trailing)
    return path


class TestIsValidPdf:
    def test_a_complete_pdf_is_accepted(self, tmp_path):
        assert fetch_corpus._is_valid_pdf(str(_write_pdf(tmp_path / "ok.pdf")))

    def test_a_download_cut_off_before_the_trailer_is_rejected(self, tmp_path):
        """The regression: a truncated file kept the header and cleared 10 kB."""
        path = _write_pdf(tmp_path / "truncated.pdf", trailer=b"")
        assert not fetch_corpus._is_valid_pdf(str(path))

    def test_the_trailer_must_sit_inside_the_last_kibibyte(self, tmp_path):
        path = _write_pdf(tmp_path / "stale-trailer.pdf", trailing=b"x" * 2_000)
        assert not fetch_corpus._is_valid_pdf(str(path))

    def test_trailing_whitespace_after_the_trailer_is_still_complete(self, tmp_path):
        path = _write_pdf(tmp_path / "newline.pdf", trailing=b"\r\n")
        assert fetch_corpus._is_valid_pdf(str(path))

    def test_a_file_below_the_size_floor_is_rejected_even_with_a_trailer(self, tmp_path):
        path = _write_pdf(tmp_path / "tiny.pdf", filler=b"x" * 100)
        assert not fetch_corpus._is_valid_pdf(str(path))

    def test_a_non_pdf_is_rejected_even_with_a_trailer(self, tmp_path):
        path = tmp_path / "error-page.pdf"
        path.write_bytes(b"<!DOCTYPE html>" + b"x" * 12_000 + b"%%EOF\n")
        assert not fetch_corpus._is_valid_pdf(str(path))

    def test_a_missing_file_is_rejected(self, tmp_path):
        assert not fetch_corpus._is_valid_pdf(str(tmp_path / "absent.pdf"))
