"""DoclingExtractor -- a PdfExtractor adapter over Docling (MIT).

An alternative to ``MineruExtractor`` for the same port, chosen with
``MONKEYGRAB_PDF_EXTRACTOR=docling`` (see ``composition.build_extractor``).
Docling runs out of its own interpreter (``.venv-docling``) through
``tools/docling_extract.py``, the same isolation MinerU gets: its torch and
model pins never enter the product's venv.

Same contract as MinerU's adapter: one ``ExtractedPage`` per PDF page,
zero-based, tables as HTML, and a hard-fail on every failure mode (no
interpreter, non-zero exit, empty output). Results are cached per PDF content
and shim version, so re-indexing an unchanged PDF does not run Docling again.
"""

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional

from monkeygrab.domain.extracted_page import ExtractedPage

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SHIM = _PROJECT_ROOT / "tools" / "docling_extract.py"
_DEFAULT_TIMEOUT_SECONDS = 1800


def _default_python() -> Optional[Path]:
    for candidate in (
        _PROJECT_ROOT / ".venv-docling" / "bin" / "python",
        _PROJECT_ROOT / ".venv-docling" / "Scripts" / "python.exe",
    ):
        if candidate.is_file():
            return candidate
    return None


def _default_cache_dir() -> Path:
    return Path(os.environ.get("DOCLING_CACHE_DIR", _PROJECT_ROOT / "rag" / "cache" / "docling"))


def _cache_key(pdf_path: Path, shim: Path) -> str:
    digest = hashlib.sha256()
    digest.update(pdf_path.read_bytes())
    digest.update(shim.read_bytes())
    digest.update(os.environ.get("DOCLING_OCR", "0").encode())
    return digest.hexdigest()[:16]


class DoclingExtractor:
    """``PdfExtractor`` over the Docling CLI shim."""

    def __init__(
        self,
        python: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        shim: Path = _SHIM,
    ):
        configured = python or os.environ.get("DOCLING_PYTHON")
        self._python = Path(configured) if configured else _default_python()
        self._cache_dir = cache_dir or _default_cache_dir()
        self._timeout = timeout_seconds
        self._shim = shim

    def extract(self, pdf_path: str) -> List[ExtractedPage]:
        """Extract every page of ``pdf_path``.

        Raises:
            FileNotFoundError: the PDF does not exist.
            RuntimeError: no Docling interpreter, the shim failed or timed
                out, or it produced no text at all.
        """
        pdf = Path(pdf_path)
        if not pdf.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf}")
        if self._python is None or not self._python.is_file():
            raise RuntimeError(
                "Docling interpreter not found. Create the isolated venv "
                "(.venv-docling) and install docling in it, or set DOCLING_PYTHON."
            )

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        out = self._cache_dir / f"{pdf.stem}-{_cache_key(pdf, self._shim)}.json"
        if not out.is_file():
            partial = out.with_suffix(".partial")
            try:
                result = subprocess.run(
                    [str(self._python), str(self._shim), str(pdf), str(partial)],
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                partial.unlink(missing_ok=True)
                raise RuntimeError(f"Docling timed out after {self._timeout}s on {pdf}") from exc
            if result.returncode != 0:
                partial.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Docling exited {result.returncode} on {pdf}: {result.stderr.strip()[-500:]}"
                )
            partial.replace(out)

        raw = json.loads(out.read_text(encoding="utf-8"))
        pages = [
            ExtractedPage(page=int(entry["page"]) - 1, text=str(entry["text"])) for entry in raw
        ]
        if not any(page.text.strip() for page in pages):
            out.unlink(missing_ok=True)
            raise RuntimeError(f"Docling extracted no text from {pdf}")
        return pages
