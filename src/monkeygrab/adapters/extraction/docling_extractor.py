"""DoclingExtractor / DoclingImageExtractor -- PdfExtractor and ImageExtractor adapters over Docling (MIT).

The alternative to the MinerU adapters for the same two ports, chosen with
``MONKEYGRAB_PDF_EXTRACTOR=docling`` (``ChunkingConfig.pdf_extractor``, see
``composition.build_extractor``). Docling runs out of its own interpreter
(``.venv-docling``) through ``tools/docling_extract.py``, the same isolation
MinerU gets: its torch and model pins never enter the product's venv.

Both adapters share one Docling run per PDF (``_run_docling``), as the MinerU
adapters share one CLI run: the shim writes the page text and the cropped
figures together, so asking for each port separately costs a single
conversion, and choosing Docling never starts MinerU.

``DoclingExtractor`` keeps MinerU's text contract: one ``ExtractedPage`` per PDF
page, zero-based, tables as HTML, and a hard-fail on every failure mode (no
interpreter, non-zero exit, unreadable or empty output). ``DoclingImageExtractor``
follows the ``ImageExtractor`` port's carve-out instead: a failed run is logged
and yields ``{}``, an unreadable figure is skipped.

Results are cached per PDF content, shim, OCR switch and installed Docling
version, so re-indexing an unchanged PDF does not run Docling again and an
upgrade re-extracts.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from monkeygrab.domain.extracted_image import ExtractedImage
from monkeygrab.domain.extracted_page import ExtractedPage

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SHIM = _PROJECT_ROOT / "tools" / "docling_extract.py"
_RESULT = "result.json"
_DEFAULT_TIMEOUT_SECONDS = 1800


def _default_python() -> Optional[Path]:
    configured = os.environ.get("DOCLING_PYTHON")
    if configured:
        return Path(configured)
    for candidate in (
        _PROJECT_ROOT / ".venv-docling" / "bin" / "python",
        _PROJECT_ROOT / ".venv-docling" / "Scripts" / "python.exe",
    ):
        if candidate.is_file():
            return candidate
    return None


def _default_cache_dir() -> Path:
    return Path(os.environ.get("DOCLING_CACHE_DIR", _PROJECT_ROOT / ".docling_cache"))


def installed_version(python: Optional[Path] = None) -> str:
    """The installed docling distributions next to ``python``, for the cache key and the index fingerprint.

    Raises:
        RuntimeError: No Docling interpreter, or no docling installed next to it.
    """
    python = python or _default_python()
    if python is None or not python.is_file():
        raise RuntimeError("Docling interpreter not found (.venv-docling or DOCLING_PYTHON)")
    env = python.parent.parent
    found = sorted(
        d.name
        for pattern in ("lib/python*/site-packages/docling*.dist-info", "Lib/site-packages/docling*.dist-info")
        for d in env.glob(pattern)
    )
    if not found:
        raise RuntimeError(f"No docling distribution installed next to {python}")
    return ",".join(found)


def _cache_key(pdf_path: Path, shim: Path, python: Path) -> str:
    digest = hashlib.sha256()
    digest.update(pdf_path.read_bytes())
    digest.update(shim.read_bytes())
    digest.update(os.environ.get("DOCLING_OCR", "0").encode())
    digest.update(installed_version(python).encode())
    return digest.hexdigest()[:16]


def _read_result(out: Path) -> Dict[str, Any]:
    """The shim's result.json; raises ValueError if it is not ``{pages: [...], figures: [...]}``."""
    raw = json.loads((out / _RESULT).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("pages"), list) or not isinstance(raw.get("figures"), list):
        raise ValueError("expected {pages: [...], figures: [...]}")
    return raw


def _run_docling(pdf: Path, python: Optional[Path], shim: Path, cache_dir: Path, timeout: int) -> Dict[str, Any]:
    """Run the shim once per PDF (or reuse its cached output) and return the parsed result plus its directory.

    Raises:
        FileNotFoundError: the PDF does not exist.
        RuntimeError: no Docling interpreter, the shim failed or timed out, or
            its result is unreadable. A broken result is removed, never cached.
    """
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    if python is None or not python.is_file():
        raise RuntimeError(
            "Docling interpreter not found. Create the isolated venv "
            "(.venv-docling) and install docling in it, or set DOCLING_PYTHON."
        )
    try:
        key = _cache_key(pdf, shim, python)
    except RuntimeError as exc:
        raise RuntimeError(f"Docling is not installed next to {python}: {exc}") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{pdf.stem}-{key}"
    if not (out / _RESULT).is_file():
        partial = out.with_name(out.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir()
        try:
            result = subprocess.run(
                [str(python), str(shim), str(pdf), str(partial)],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            shutil.rmtree(partial, ignore_errors=True)
            raise RuntimeError(f"Docling timed out after {timeout}s on {pdf}") from exc
        if result.returncode != 0 or not (partial / _RESULT).is_file():
            shutil.rmtree(partial, ignore_errors=True)
            raise RuntimeError(f"Docling exited {result.returncode} on {pdf}: {result.stderr.strip()[-500:]}")
        shutil.rmtree(out, ignore_errors=True)
        partial.rename(out)

    try:
        raw = _read_result(out)
    except (ValueError, KeyError, TypeError) as exc:
        # A malformed result must not stay cached: every later call would fail the same way.
        shutil.rmtree(out, ignore_errors=True)
        raise RuntimeError(f"Docling wrote an unreadable result for {pdf}: {exc}") from exc
    raw["dir"] = out
    return raw


class DoclingExtractor:
    """``PdfExtractor`` over the Docling shim; hard-fails like ``MineruExtractor``."""

    def __init__(
        self,
        python: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        shim: Path = _SHIM,
    ):
        self._python = Path(python) if python else _default_python()
        self._cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        self._timeout = timeout_seconds
        self._shim = shim

    def extract(self, pdf_path: str) -> List[ExtractedPage]:
        """Extract every page of ``pdf_path``.

        Raises:
            FileNotFoundError: the PDF does not exist.
            RuntimeError: no Docling interpreter, the shim failed or timed
                out, or it produced no text at all.
        """
        raw = _run_docling(Path(pdf_path), self._python, self._shim, self._cache_dir, self._timeout)
        try:
            pages = [ExtractedPage(page=int(entry["page"]) - 1, text=str(entry["text"])) for entry in raw["pages"]]
        except (ValueError, KeyError, TypeError) as exc:
            shutil.rmtree(raw["dir"], ignore_errors=True)
            raise RuntimeError(f"Docling wrote an unreadable result for {pdf_path}: {exc}") from exc
        if not any(page.text.strip() for page in pages):
            shutil.rmtree(raw["dir"], ignore_errors=True)
            raise RuntimeError(f"Docling extracted no text from {pdf_path}")
        return pages


class DoclingImageExtractor:
    """``ImageExtractor`` over the same Docling run as ``DoclingExtractor``."""

    def __init__(
        self,
        python: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        shim: Path = _SHIM,
    ):
        self._python = Path(python) if python else _default_python()
        self._cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        self._timeout = timeout_seconds
        self._shim = shim

    def extract(self, pdf_path: str) -> Dict[int, List[ExtractedImage]]:
        """Every figure of ``pdf_path``, keyed by zero-based page; ``{}`` if Docling could not run."""
        try:
            raw = _run_docling(Path(pdf_path), self._python, self._shim, self._cache_dir, self._timeout)
        except Exception as exc:
            logger.warning("Docling image extraction failed for %s: %s", pdf_path, exc)
            return {}

        out: Path = raw["dir"]
        images_by_page: Dict[int, List[ExtractedImage]] = {}
        for figure in raw["figures"]:
            try:
                path = (out / figure["file"]).resolve()
                # The file name comes from a JSON file on disk: one that escapes
                # the output directory would read any file into the index.
                if not path.is_relative_to(out.resolve()):
                    raise ValueError(f"figure path {figure['file']!r} escapes {out}")
                image_bytes = path.read_bytes()
                with Image.open(BytesIO(image_bytes)) as image:
                    width, height = image.size
                page = int(figure["page"]) - 1
            except Exception as exc:
                logger.warning("Skipping unreadable Docling figure in %s: %s", pdf_path, exc)
                continue
            images_by_page.setdefault(page, []).append(
                ExtractedImage(
                    image_bytes=image_bytes,
                    width=width,
                    height=height,
                    ext=path.suffix.lstrip(".").lower() or "png",
                    caption=str(figure.get("caption") or "").strip(),
                )
            )
        return images_by_page
