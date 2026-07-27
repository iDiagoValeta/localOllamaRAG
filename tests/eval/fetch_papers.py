"""fetch_papers -- idempotent arXiv PDF downloader for the gold eval corpus.

Downloads arXiv papers by numeric id into a local cache directory that is
NOT committed to git (see ``.gitignore``): the id already pins the exact
paper version (arXiv ids are immutable once assigned, and a bare id always
resolves to latest version, so re-running months later still fetches the
same PDF content), so nothing is lost by keeping the binary out of version
control -- only the id needs to be reproducible, and that lives in
``gold_cases.jsonl``.

Every file already in the cache is checked for a valid ``%PDF`` header
before being trusted; a previous run that was interrupted mid-download (or
an arXiv rate-limit HTML error page saved with a ``.pdf`` name) does not
silently pass as cached and does not stay cached -- it is deleted and
re-fetched.

Usage:
    python tests/eval/fetch_papers.py 1512.03385 1810.04805 2010.11929
    python tests/eval/fetch_papers.py --gold-file tests/eval/gold_cases.jsonl
    python tests/eval/fetch_papers.py --dest C:/tmp/papers 1512.03385

Dependencies: requests (already a pinned dependency of rag/requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List

import requests

# CONSTANTS

# Default cache location: alongside the gold cases, gitignored (see root
# .gitignore). Kept inside tests/eval/ so the whole eval corpus -- code,
# case definitions and downloaded PDFs -- lives under one directory.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "papers_cache"

# arXiv's "abs" id (e.g. "1512.03385" or "1512.03385v1") resolves to the
# newest version if no vN suffix is given; the export mirror is requested
# per arXiv's own robots/API guidance to keep automated traffic off the
# main site.
ARXIV_PDF_URL_TEMPLATE = "https://export.arxiv.org/pdf/{arxiv_id}"

# arXiv's terms of use ask automated clients to identify themselves and to
# throttle requests (https://info.arxiv.org/help/api/tou.html). The contact
# URL lets a maintainer reach us if this script misbehaves.
USER_AGENT = (
    "MonkeyGrab-eval-fetcher/1.0 "
    "(+https://github.com/iDiagoValeta/localOllamaRAG; eval corpus builder)"
)

# Minimum pause between successive downloads in one run (seconds). Not
# applied around a cache hit -- only real HTTP requests are throttled.
REQUEST_INTERVAL_SECONDS = 3.0

REQUEST_TIMEOUT_SECONDS = 60

# A real arXiv paper PDF is at least a few tens of KB; a rate-limit or
# "not found" HTML error page saved with a .pdf extension is typically a
# few KB. This is a cheap sanity floor, not a content check.
MIN_VALID_PDF_BYTES = 10_000

_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


class PaperDownloadError(RuntimeError):
    """Raised when an arXiv id cannot be resolved to a valid, complete PDF."""


# VALIDATION


def _looks_like_pdf(path: Path) -> bool:
    """Check the ``%PDF-`` magic header and a minimum size.

    A truncated download or an HTML error page (arXiv returns these with a
    200 status on some throttling/maintenance responses, not just 4xx/5xx)
    must not be mistaken for a cached paper on the next run.
    """
    if not path.is_file() or path.stat().st_size < MIN_VALID_PDF_BYTES:
        return False
    with path.open("rb") as fh:
        header = fh.read(5)
    return header == b"%PDF-"


# DOWNLOAD


def download_paper(arxiv_id: str, dest_dir: Path) -> Path:
    """Fetch one arXiv paper's PDF into ``dest_dir``, unless already cached.

    Args:
        arxiv_id: arXiv identifier, e.g. ``"1512.03385"`` (optionally with
            a ``vN`` version suffix). The bare (unversioned) form is what
            makes the corpus reproducible without pinning a specific
            revision: arXiv keeps the numeric id fixed and a bare id always
            resolves to the same content once a paper stops being revised,
            which is true for every paper old enough to be a stable gold
            case.
        dest_dir: Cache directory; created if missing.

    Returns:
        Path to the validated PDF on disk (``<dest_dir>/<arxiv_id>.pdf``).

    Raises:
        PaperDownloadError: If the id is malformed, the HTTP request fails,
            or the downloaded bytes do not look like a real PDF.
    """
    if not _ARXIV_ID_RE.match(arxiv_id):
        raise PaperDownloadError(
            f"{arxiv_id!r} does not look like an arXiv id (expected e.g. '1512.03385')"
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{arxiv_id}.pdf"

    if _looks_like_pdf(dest_path):
        return dest_path  # already cached and valid -- no network call

    if dest_path.exists():
        dest_path.unlink()  # stale/corrupt leftover from an interrupted run

    url = ARXIV_PDF_URL_TEMPLATE.format(arxiv_id=arxiv_id)
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise PaperDownloadError(f"request to {url} failed: {exc}") from exc

    if response.status_code != 200:
        raise PaperDownloadError(
            f"arXiv returned HTTP {response.status_code} for {arxiv_id!r} ({url})"
        )

    dest_path.write_bytes(response.content)

    if not _looks_like_pdf(dest_path):
        dest_path.unlink()
        raise PaperDownloadError(
            f"downloaded content for {arxiv_id!r} is not a valid PDF "
            f"(missing '%PDF-' header or too small) -- check the id is correct"
        )

    return dest_path


def download_papers(arxiv_ids: Iterable[str], dest_dir: Path) -> List[Path]:
    """Fetch several papers, pausing between real (non-cached) downloads.

    Args:
        arxiv_ids: arXiv identifiers to fetch.
        dest_dir: Cache directory shared by all downloads.

    Returns:
        Paths to each validated PDF, in the same order as ``arxiv_ids``.
    """
    paths: List[Path] = []
    first_real_download = True
    for arxiv_id in arxiv_ids:
        was_cached = _looks_like_pdf(dest_dir / f"{arxiv_id}.pdf")
        if not was_cached and not first_real_download:
            time.sleep(REQUEST_INTERVAL_SECONDS)
        paths.append(download_paper(arxiv_id, dest_dir))
        if not was_cached:
            first_real_download = False
    return paths


# DISCOVERY


def arxiv_ids_from_gold_file(gold_file: Path) -> List[str]:
    """Collect the distinct ``arxiv_id`` values referenced by a gold-cases file.

    Cases whose ``source`` is ``"corpus"`` ship their PDF in ``rag/docs/``
    already and have no ``arxiv_id`` -- only ``source: "arxiv"`` cases need
    fetching. Order is preserved (first appearance) so repeated runs fetch
    in a stable order.

    Args:
        gold_file: Path to a ``gold_cases.jsonl`` file.

    Returns:
        Distinct arXiv ids in first-seen order.
    """
    seen: List[str] = []
    for line in gold_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        case = json.loads(line)
        arxiv_id = case.get("arxiv_id")
        if arxiv_id and arxiv_id not in seen:
            seen.append(arxiv_id)
    return seen


# CLI


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "arxiv_ids", nargs="*", help="arXiv ids to fetch, e.g. 1512.03385"
    )
    parser.add_argument(
        "--dest", type=Path, default=DEFAULT_CACHE_DIR,
        help=f"cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--gold-file", type=Path,
        default=Path(__file__).resolve().parent / "gold_cases.jsonl",
        help="fetch every arxiv_id referenced by this file when no ids are given on the CLI",
    )
    args = parser.parse_args(argv)

    ids = args.arxiv_ids or arxiv_ids_from_gold_file(args.gold_file)
    if not ids:
        print("nothing to fetch: no ids given and none found in gold file", file=sys.stderr)
        return 1

    try:
        paths = download_papers(ids, args.dest)
    except PaperDownloadError as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1

    for arxiv_id, path in zip(ids, paths):
        print(f"{arxiv_id} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
