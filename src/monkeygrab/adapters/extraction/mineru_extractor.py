"""MineruExtractor / MineruImageExtractor -- PdfExtractor and ImageExtractor
adapters over the MinerU CLI (`-b pipeline` backend).

MinerU is never a Python dependency of this product (see rag/requirements.txt):
it is an external CLI, invoked through ``subprocess`` against its own isolated
venv, so its pins never collide with the product's. Both adapters in this
module share one CLI invocation per PDF (``_run_mineru``) -- MinerU emits
Markdown, a structured ``*_content_list.json`` and cropped figures in a single
run, so calling it twice (once per port) would double the cost for nothing.

Two adapters, not one, because ``PdfExtractor`` and ``ImageExtractor`` are two
separate ``Protocol``s, each requiring its own ``extract(pdf_path)`` method
with a different return type -- a single class cannot satisfy both under the
same method name.

**Tables are this adapter's reason for existing** (issue #3): MinerU reports
each table as a block with a `table_body` HTML string, which SECTION 2 copies
into the page text verbatim, wrapped in its `<table>`/`</table>` tags. Nothing
downstream needs to know it came from MinerU to notice it -- an HTML table
tag inside a chunk's text is grep-able as "this chunk is a table" on its own,
without depending on any adapter-specific signal. Images are never rendered
as HTML/base64 inline text: they go through ``MineruImageExtractor`` instead,
matching this project's multimodal image pipeline (docs/design/2026-07-26-
monkeygrab-v2.md, section 3: "Las tablas se indexan como texto HTML, no como
imagenes").

**Hard-fail policy** (issue #6, this adapter's other reason for existing): the
previous fallback chain hid the fact that MinerU integration
had never actually worked, for months, by silently degrading to a lower-
fidelity extractor and producing plausible-looking output. ``_run_mineru``
raises with an actionable message on every failure mode (CLI missing, exit
code != 0, no output directory, empty Markdown) instead. ``MineruExtractor``
(the ``PdfExtractor`` adapter) always lets that exception propagate, per the
port's hard-fail contract. ``MineruImageExtractor`` (the ``ImageExtractor``
adapter) is the one documented exception in this codebase's hard-fail policy:
its port's own docstring specifies that extraction failures are swallowed and
logged, matching the image port's per-file failure carve-out --
see SECTION 4 for why that is not a contradiction of the above.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from monkeygrab.domain.extracted_image import ExtractedImage
from monkeygrab.domain.extracted_page import ExtractedPage

# Not subclassed from the ports: Protocol conformance here is structural
# (duck typing), the same contract every other adapter in this package
# satisfies without inheriting its port.

logger = logging.getLogger(__name__)

# Repo root, computed the same way monkeygrab.config.paths derives
# RAG_BASE_DIR: this file lives at <root>/src/monkeygrab/adapters/extraction/
# mineru_extractor.py, so four `dirname` calls from its own directory land on
# <root>.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR))))

# Bumped whenever SECTION 2's block-to-text/image mapping changes shape.
# Folded into the cache key so every cached extraction from a previous
# version of this adapter is treated as a miss instead of being replayed
# under parsing logic it was never produced for.
_CACHE_VERSION = "v1"

# MinerU block types that carry no document content -- running headers,
# footers, page numbers and marginal footnotes. Structural (by MinerU's own
# `type` field), not tied to any one document's layout.
_DISCARDED_BLOCK_TYPES = frozenset({"page_number", "footer", "page_footnote", "aside_text"})


# CLI INVOCATION + CACHE


def _default_mineru_bin() -> str:
    """Resolve the configured MinerU binary: ``MINERU_BIN`` env var, else the
    project's isolated venv, probing both layouts the same way
    ``composition._isolated_python`` probes both ``python.exe`` and
    ``python`` -- whichever of the two conventional paths actually exists
    wins, so ``MINERU_BIN`` is only needed as an override, not as a
    non-Windows requirement.

    Raises:
        RuntimeError: Neither conventional path exists and ``MINERU_BIN`` is
            unset.
    """
    configured = os.getenv("MINERU_BIN", "").strip()
    if configured:
        return configured

    windows_path = os.path.join(_PROJECT_ROOT, ".venv-mineru", "Scripts", "mineru.exe")
    posix_path = os.path.join(_PROJECT_ROOT, ".venv-mineru", "bin", "mineru")
    for candidate in (windows_path, posix_path):
        if os.path.isfile(candidate):
            return candidate

    raise RuntimeError(
        f"MinerU binary not found at {windows_path!r} or {posix_path!r}. "
        "Install MinerU in the isolated venv (.venv-mineru) or set MINERU_BIN "
        "to its executable."
    )


def _default_cache_dir() -> Path:
    """Resolve the extraction cache directory: ``MINERU_CACHE_DIR`` env var,
    else ``<repo_root>/.mineru_cache`` (gitignored -- see .gitignore)."""
    configured = os.getenv("MINERU_CACHE_DIR", "").strip()
    if configured:
        return Path(configured)
    return Path(_PROJECT_ROOT) / ".mineru_cache"


def _resolve_runnable_bin(configured: str) -> Optional[str]:
    """Turn a configured MinerU binary (path or bare command) into something
    ``subprocess.run`` can actually execute, or ``None`` if it cannot be found.
    """
    if os.path.isfile(configured):
        return configured
    return shutil.which(configured)


def _cache_key(pdf_path: Path, mineru_bin: str) -> str:
    """Cache key for one (PDF, MinerU binary, parsing-logic version) triple.

    Includes the PDF's size and mtime rather than hashing its bytes -- MinerU
    extraction is expensive enough (seconds to minutes) that the cost of a
    false cache hit on a byte-identical-but-touched file would be trivial
    compared to hashing every PDF's full content on every indexing run.
    """
    stat = pdf_path.stat()
    raw = f"{pdf_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{mineru_bin}|{_CACHE_VERSION}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _find_output_dir(dest: Path, pdf_stem: str) -> Path:
    """The directory holding this extraction's Markdown, whatever it is called.

    MinerU nests its output one level deeper than the ``-o`` directory it is
    given. 2.x always named that level ``auto/``; 3.x names it after the
    backend that ran -- ``hybrid_auto/``, ``vlm_auto/``, ``pipeline_auto/`` --
    so the hardcoded name meant a fresh install extracted a PDF successfully
    and had the result rejected (issue #118).

    The directory is therefore located by what it CONTAINS (``<stem>.md``)
    rather than by its name: no version detection, no list of backend names to
    keep current as MinerU adds them, and 2.x's ``auto/`` keeps working
    untouched.

    Raises:
        RuntimeError: No candidate, or more than one. Two output directories
            side by side -- a stale one from an earlier run next to a fresh
            one -- is ambiguous, and picking whichever sorted first would
            index a mixture of two extractions while reporting one. That is
            the failure this adapter's no-silent-fallback policy exists to
            prevent, so it is a hard failure rather than a heuristic.
    """
    parent = dest / pdf_stem
    candidates = (
        sorted(child for child in parent.iterdir() if (child / f"{pdf_stem}.md").is_file())
        if parent.is_dir()
        else []
    )

    if not candidates:
        raise RuntimeError(
            f"MinerU produced no output directory containing {pdf_stem}.md under {parent}. "
            "Check the CLI's own stderr/stdout for the real cause -- with MinerU 3.x, "
            "MINERU_MODEL_SOURCE=local requires a local models config file and crashes "
            "without one; MINERU_MODEL_SOURCE=huggingface downloads them instead "
            "(issue #118)."
        )
    if len(candidates) > 1:
        names = ", ".join(child.name for child in candidates)
        raise RuntimeError(
            f"MinerU left more than one output directory under {parent}: {names}. "
            "This adapter will not guess which extraction is current -- delete the "
            "stale one (or clear the extraction cache) and re-run."
        )
    return candidates[0]


def _validate_output(dest: Path, pdf_stem: str) -> Tuple[Path, Path]:
    """Locate and sanity-check MinerU's output inside ``dest``.

    The output directory is found by ``_find_output_dir`` (its name depends on
    the MinerU version and backend); this checks that what is inside it is
    usable -- the Markdown and the sibling ``*_content_list.json`` carrying
    the structured block list this adapter parses (SECTION 2).

    Returns:
        ``(md_path, content_list_path)``.

    Raises:
        RuntimeError: If the output directory, the Markdown file or the
            content-list JSON is missing, or the Markdown is empty --
            the three "MinerU silently produced nothing usable" failure
            modes this adapter must never paper over (see module docstring).
    """
    auto_dir = _find_output_dir(dest, pdf_stem)

    md_path = auto_dir / f"{pdf_stem}.md"
    if not md_path.is_file() or not md_path.read_text(encoding="utf-8").strip():
        raise RuntimeError(
            f"MinerU produced an empty Markdown file: {md_path}. "
            "This adapter does not degrade; fix the MinerU "
            "installation or its model cache (MINERU_MODEL_SOURCE)."
        )

    content_list_path = auto_dir / f"{pdf_stem}_content_list.json"
    if not content_list_path.is_file():
        raise RuntimeError(f"MinerU produced no content-list JSON: {content_list_path}")

    return md_path, content_list_path


def _run_mineru(
    pdf_path: Path,
    mineru_bin: str,
    cache_dir: Path,
    model_source: str,
    timeout_seconds: int,
) -> Tuple[Path, Path]:
    """Extract ``pdf_path`` with MinerU, reusing a cached prior run if valid.

    Args:
        pdf_path: PDF to extract.
        mineru_bin: Configured MinerU binary -- a path or a bare command to
            resolve on PATH.
        cache_dir: Root directory extraction output is cached under, one
            subdirectory per (PDF, binary, parsing-version) triple.
        model_source: Value for the ``MINERU_MODEL_SOURCE`` env var the CLI
            reads (``"local"`` uses the already-downloaded HuggingFace cache
            and never attempts a download).
        timeout_seconds: Passed straight to ``subprocess.run``.

    Returns:
        ``(md_path, content_list_path)`` -- see ``_validate_output``.

    Raises:
        RuntimeError: MinerU binary not found, CLI exited non-zero, or its
            output failed validation (see ``_validate_output``).
        FileNotFoundError: ``pdf_path`` does not exist.
    """
    resolved_bin = _resolve_runnable_bin(mineru_bin)
    if resolved_bin is None:
        raise RuntimeError(
            f"MinerU CLI not found (looked for {mineru_bin!r} on disk and on "
            "PATH). Install MinerU in an isolated venv and either add it to "
            "PATH or set MINERU_BIN to the executable, e.g. "
            r".venv-mineru\Scripts\mineru.exe (Windows) or "
            ".venv-mineru/bin/mineru (Linux/Mac)."
        )

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{pdf_path.stem}-{_cache_key(pdf_path, resolved_bin)}"

    if dest.exists():
        try:
            return _validate_output(dest, pdf_path.stem)
        except RuntimeError:
            # A previous run was interrupted or produced a partial result --
            # do not replay it, and do not leave a permanently-stuck bad
            # cache entry either. Re-extract from scratch.
            logger.warning("Discarding incomplete MinerU cache at %s, re-extracting", dest)
            shutil.rmtree(dest, ignore_errors=True)

    dest.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MINERU_MODEL_SOURCE", model_source)

    cmd = [resolved_bin, "-p", str(pdf_path), "-o", str(dest), "-b", "pipeline"]
    logger.info("Running MinerU: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=timeout_seconds, check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to launch MinerU CLI ({resolved_bin!r}): {exc}") from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"MinerU CLI failed (exit code {proc.returncode}) for {pdf_path.name}:\n"
            f"{(proc.stderr or proc.stdout)[-4000:]}"
        )

    return _validate_output(dest, pdf_path.stem)


# CONTENT-LIST PARSING


def _render_block_text(block: Dict[str, Any]) -> str:
    """Render one MinerU content-list block as page text.

    Tables and equations keep their MinerU-native markup verbatim (HTML,
    LaTeX) -- see the module docstring for why that matters for tables.
    Figures/charts contribute only their caption here; their pixels are
    ``MineruImageExtractor``'s job, not this text path's.
    """
    block_type = block.get("type", "")

    if block_type in _DISCARDED_BLOCK_TYPES:
        return ""

    if block_type == "text":
        text = block.get("text", "")
        level = block.get("text_level")
        # MinerU marks headings with a numeric `text_level` instead of
        # Markdown `#` syntax; reproduce the Markdown convention so a
        # section_header lookup downstream still recognizes it as one.
        if level and text:
            return f"{'#' * int(level)} {text}"
        return text

    if block_type == "equation":
        return block.get("text", "")

    if block_type == "table":
        caption = " ".join(block.get("table_caption") or []).strip()
        body = block.get("table_body", "")
        if caption and body:
            return f"{caption}\n{body}"
        return body or caption

    if block_type == "list":
        return "\n".join(block.get("list_items") or [])

    if block_type in ("image", "chart"):
        caption_key = "image_caption" if block_type == "image" else "chart_caption"
        return " ".join(block.get(caption_key) or []).strip()

    # Forward-compatible fallback: an unrecognized block type still
    # contributes its "text" field if present, so a future MinerU release
    # adding new block kinds degrades to plain inclusion instead of silently
    # dropping content.
    return block.get("text", "")


def _content_list_to_pages(blocks: List[Dict[str, Any]]) -> List[ExtractedPage]:
    """Group MinerU content-list blocks into per-page text.

    Args:
        blocks: Parsed ``*_content_list.json`` -- a flat list of blocks, each
            carrying a zero-based ``page_idx`` (MinerU's own convention,
            already matching ``ExtractedPage.page``'s -- no offset needed,
            without an offset).

    Returns:
        One ``ExtractedPage`` per ``page_idx`` that produced at least one
        non-empty rendered block, in ascending page order. A page whose only
        blocks are page furniture (all discarded, see
        ``_DISCARDED_BLOCK_TYPES``) is omitted rather than emitted empty.
    """
    by_page: Dict[int, List[str]] = {}
    for block in blocks:
        if "page_idx" not in block:
            continue
        rendered = _render_block_text(block)
        if rendered:
            by_page.setdefault(int(block["page_idx"]), []).append(rendered)

    return [
        ExtractedPage(page=page_idx, text="\n\n".join(parts))
        for page_idx, parts in sorted(by_page.items())
    ]


def _content_list_to_images(
    blocks: List[Dict[str, Any]], auto_dir: Path
) -> Dict[int, List[ExtractedImage]]:
    """Group MinerU content-list ``image``/``chart`` blocks into per-page images.

    Tables are deliberately excluded here even though MinerU also gives them
    an ``img_path`` (a rendering of the table as a picture) -- this adapter
    indexes tables as HTML text (SECTION 2's ``_render_block_text``), not as
    images, per the module docstring.

    Args:
        blocks: Parsed ``*_content_list.json``.
        auto_dir: The ``auto/`` directory the content-list JSON lives in --
            every block's ``img_path`` is relative to it (MinerU already
            includes the ``images/`` prefix in the value).

    Returns:
        Mapping of zero-based page number to the images found on that page.
        A referenced image file that is missing or unreadable is skipped
        (logged), not fatal -- matching the ``ImageExtractor`` port's
        documented per-image failure carve-out.
    """
    images_by_page: Dict[int, List[ExtractedImage]] = {}

    for block in blocks:
        block_type = block.get("type")
        if block_type not in ("image", "chart"):
            continue

        rel_path = block.get("img_path")
        page_idx = block.get("page_idx")
        if not rel_path or page_idx is None:
            continue

        image_path = auto_dir / rel_path
        try:
            image_bytes = image_path.read_bytes()
            from io import BytesIO

            with Image.open(BytesIO(image_bytes)) as image:
                width, height = image.size
        except Exception as exc:
            logger.warning("Skipping unreadable MinerU figure %s: %s", image_path, exc)
            continue

        caption_key = "image_caption" if block_type == "image" else "chart_caption"
        caption = " ".join(block.get(caption_key) or []).strip()
        ext = image_path.suffix.lstrip(".").lower() or "jpg"

        images_by_page.setdefault(int(page_idx), []).append(
            ExtractedImage(image_bytes=image_bytes, width=width, height=height, ext=ext, caption=caption)
        )

    return images_by_page


# MINERUEXTRACTOR (PdfExtractor)


class MineruExtractor:
    """Extracts PDF pages as text via the MinerU CLI, tables kept as HTML.

    Hard-fail, per the ``PdfExtractor`` port and this adapter's whole reason
    for existing (module docstring, issue #6): never falls back.
    """

    def __init__(
        self,
        mineru_bin: Optional[str] = None,
        cache_dir: Optional[str] = None,
        model_source: str = "local",
        timeout_seconds: int = 1800,
    ):
        """Args:
            mineru_bin: Path to the MinerU executable, or a bare command to
                resolve on PATH. Defaults to the ``MINERU_BIN`` env var, then
                the project's isolated venv (``.venv-mineru``), probing both
                the Windows (``Scripts\\mineru.exe``) and POSIX
                (``bin/mineru``) layouts.
            cache_dir: Directory cached extractions are stored under.
                Defaults to the ``MINERU_CACHE_DIR`` env var, then
                ``<repo_root>/.mineru_cache``.
            model_source: ``MINERU_MODEL_SOURCE`` value passed to the CLI
                (``"local"`` never attempts a model download).
            timeout_seconds: Maximum time to wait for one PDF's extraction.
        """
        self._mineru_bin = mineru_bin if mineru_bin is not None else _default_mineru_bin()
        self._cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        self._model_source = model_source
        self._timeout_seconds = timeout_seconds

    def extract(self, pdf_path: str) -> List[ExtractedPage]:
        """Extract every page of ``pdf_path`` via MinerU.

        Args:
            pdf_path: Absolute or relative path to the PDF file.

        Returns:
            One ``ExtractedPage`` per page that produced extractable text,
            in page order, 0-based.

        Raises:
            RuntimeError: MinerU CLI missing, failed, or produced no/empty
                output. Never falls back to a different extractor.
            FileNotFoundError: ``pdf_path`` does not exist.
        """
        _, content_list_path = _run_mineru(
            Path(pdf_path), self._mineru_bin, self._cache_dir, self._model_source, self._timeout_seconds,
        )
        blocks = json.loads(content_list_path.read_text(encoding="utf-8"))
        return _content_list_to_pages(blocks)


# MINERUIMAGEEXTRACTOR (ImageExtractor)


class MineruImageExtractor:
    """Extracts figures/charts (not tables) from a PDF via the MinerU CLI.

    Shares ``_run_mineru``'s cache with ``MineruExtractor``: indexing a PDF
    through both adapters (text then images, as ``IndexCorpus`` does) runs
    the CLI only once.

    Per the ``ImageExtractor`` port's documented carve-out from this
    project's hard-fail default, CLI-level failures during ``extract`` (a
    configured binary that turns out to be missing at run time, exit code
    != 0, empty output) are logged and swallowed here, returning ``{}`` --
    exactly like the image port's "cannot open the file" case. Resolving the
    *default* binary (``mineru_bin=None``) happens eagerly at construction,
    via ``_default_mineru_bin``, and raises there instead -- before this
    carve-out ever applies. This does not reopen the hard-fail question
    SECTION 3 exists to close: ``IndexCorpus`` always calls the
    ``PdfExtractor`` first, so a broken MinerU installation already aborts
    the whole PDF via ``MineruExtractor`` before this adapter would ever run
    in production.
    """

    def __init__(
        self,
        mineru_bin: Optional[str] = None,
        cache_dir: Optional[str] = None,
        model_source: str = "local",
        timeout_seconds: int = 1800,
    ):
        """Args: same as ``MineruExtractor.__init__``."""
        self._mineru_bin = mineru_bin if mineru_bin is not None else _default_mineru_bin()
        self._cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        self._model_source = model_source
        self._timeout_seconds = timeout_seconds

    def extract(self, pdf_path: str) -> Dict[int, List[ExtractedImage]]:
        """Extract every figure/chart of ``pdf_path``, grouped by page.

        Args:
            pdf_path: Absolute or relative path to the PDF file.

        Returns:
            Mapping of zero-based page number to the figures found on that
            page. ``{}`` if MinerU could not extract the file at all.
        """
        try:
            _, content_list_path = _run_mineru(
                Path(pdf_path), self._mineru_bin, self._cache_dir, self._model_source, self._timeout_seconds,
            )
        except Exception as exc:
            logger.warning("MinerU image extraction failed for %s: %s", pdf_path, exc)
            return {}

        blocks = json.loads(content_list_path.read_text(encoding="utf-8"))
        return _content_list_to_images(blocks, content_list_path.parent)
