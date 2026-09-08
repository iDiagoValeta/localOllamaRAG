"""fetch_corpus -- idempotent downloader for the product's three PDF corpora.

Reads ``rag/docs/corpus_manifest.json`` and fills ``rag/docs/<lang>/`` from two
sources: arXiv by immutable id, and Wikipedia's PDF export by article title.

**Why the PDFs are not versioned.** The same reasoning as
``tests/eval/fetch_papers.py``: what needs to be reproducible is the
*identity* of each document, and that is small enough to live in git. The
binaries are not, and a corpus of ~50 papers and articles would add most of
a hundred megabytes to a repository that already carries 400. Note that a
``.gitignore`` rule never applies to a file git already tracks, so the
documents committed before this script existed stay versioned exactly as they
were; only what this script adds is ignored.

**Why Wikipedia revids are recorded.** arXiv ids pin content forever: a bare
id always resolves to the latest version of a paper that, by policy, is never
rewritten in place. Wikipedia has no such guarantee -- an article can be
rewritten between one run and the next, and the export API only ever serves
the current revision. A gold case whose answer was verified against an
article that has since changed does not fail loudly; it fails as if the
pipeline had regressed, which is the worst way for a benchmark to rot. So the
manifest stores the revision each document was fetched and verified at, and
this script reports every article that has moved past it. It does not refuse
to download -- the newer text is still a perfectly good corpus document; it
refuses to let the drift go unnoticed.

Usage:
    python tools/fetch_corpus.py              # fill every corpus
    python tools/fetch_corpus.py --lang es    # one corpus
    python tools/fetch_corpus.py --check      # report only, download nothing

Dependencies: requests (already pinned in rag/requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(REPO_ROOT, "rag", "docs", "corpus_manifest.json")
DOCS_ROOT = os.path.join(REPO_ROOT, "rag", "docs")

ARXIV_PDF_URL = "https://arxiv.org/pdf/{id}"
WIKI_PDF_URL = "https://{lang}.wikipedia.org/api/rest_v1/page/pdf/{title}"
WIKI_API_URL = "https://{lang}.wikipedia.org/w/api.php"

# Both hosts ask for a descriptive agent and throttle anonymous bursts. One
# second between documents is enough for a corpus this size and keeps the
# script well inside what either service considers polite.
USER_AGENT = "MonkeyGrab-corpus-fetcher/1.0 (local RAG evaluation corpus)"
REQUEST_TIMEOUT_S = 120
PAUSE_BETWEEN_DOWNLOADS_S = 4.0

# Wikipedia renders each export on demand rather than serving a cached file,
# so it rate-limits far more aggressively than a static download: a one-second
# pause got 17 documents through and then returned 429 for the next 18. On a
# 429 the server's own Retry-After is honoured when present, and otherwise the
# wait doubles. Failing the whole corpus over a throttle would be the script
# reporting a network condition as if it were a missing document.
RATE_LIMIT_ATTEMPTS = 5
RATE_LIMIT_BACKOFF_S = 20.0

# A truncated download or an HTML error page saved under a .pdf name must not
# pass as cached on the next run -- the same guard fetch_papers.py applies.
PDF_MAGIC = b"%PDF"
MIN_PLAUSIBLE_PDF_BYTES = 10_000


def _load_manifest() -> Dict[str, Any]:
    """Return the manifest, minus its documentation key.

    Raises:
        SystemExit: The manifest is missing or is not valid JSON.
    """
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {MANIFEST_PATH}: {exc}") from exc
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _wiki_filename(title: str) -> str:
    """Filename for a Wikipedia article, matching the corpus's existing style."""
    return title.replace(" ", "_") + ".pdf"


def _is_valid_pdf(path: str) -> bool:
    """Whether ``path`` looks like a complete PDF rather than a stub."""
    try:
        if os.path.getsize(path) < MIN_PLAUSIBLE_PDF_BYTES:
            return False
        with open(path, "rb") as handle:
            return handle.read(len(PDF_MAGIC)) == PDF_MAGIC
    except OSError:
        return False


def _download(url: str, dest: str) -> Tuple[bool, str]:
    """Fetch ``url`` into ``dest``, writing only a validated PDF.

    Downloads to a temporary name first so an interrupted run never leaves a
    half-written file that the next run would trust.

    Returns:
        ``(ok, detail)``; ``detail`` carries the byte count or the failure.
    """
    tmp = dest + ".part"
    try:
        wait = RATE_LIMIT_BACKOFF_S
        for attempt in range(RATE_LIMIT_ATTEMPTS):
            response = requests.get(
                url,
                timeout=REQUEST_TIMEOUT_S,
                headers={"User-Agent": USER_AGENT},
                allow_redirects=True,
            )
            if response.status_code != 429:
                break
            if attempt == RATE_LIMIT_ATTEMPTS - 1:
                return False, "rate-limited (429) after every retry"
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if (retry_after or "").isdigit() else wait
            print(f"       rate-limited, waiting {delay:.0f}s", flush=True)
            time.sleep(delay)
            wait *= 2
        response.raise_for_status()
        with open(tmp, "wb") as handle:
            handle.write(response.content)
        if not _is_valid_pdf(tmp):
            os.remove(tmp)
            return False, "response was not a usable PDF"
        os.replace(tmp, dest)
        return True, f"{os.path.getsize(dest):,} bytes"
    except Exception as exc:  # noqa: BLE001 -- reported per document, never fatal
        if os.path.exists(tmp):
            os.remove(tmp)
        return False, str(exc)


def _current_revid(lang: str, title: str) -> Optional[int]:
    """Live revision id for an article, or ``None`` if it cannot be read."""
    try:
        response = requests.get(
            WIKI_API_URL.format(lang=lang),
            params={
                "action": "query",
                "prop": "revisions",
                "titles": title,
                "rvprop": "ids",
                "format": "json",
                "redirects": "1",
            },
            timeout=REQUEST_TIMEOUT_S,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        pages = response.json()["query"]["pages"]
        page = next(iter(pages.values()))
        return int(page["revisions"][0]["revid"])
    except Exception:  # noqa: BLE001 -- a drift check must never end the run
        return None


def _fetch_corpus(lang: str, spec: Dict[str, Any], check_only: bool) -> Dict[str, int]:
    """Download one corpus's missing documents and report revision drift."""
    dest_dir = os.path.join(DOCS_ROOT, lang)
    os.makedirs(dest_dir, exist_ok=True)
    tally = {"present": 0, "fetched": 0, "failed": 0, "drifted": 0}

    for entry in spec.get("arxiv", []):
        dest = os.path.join(dest_dir, entry["filename"])
        if _is_valid_pdf(dest):
            tally["present"] += 1
            continue
        if check_only:
            print(f"  [{lang}] MISSING {entry['filename']}")
            tally["failed"] += 1
            continue
        ok, detail = _download(ARXIV_PDF_URL.format(id=entry["id"]), dest)
        print(f"  [{lang}] {'OK  ' if ok else 'FAIL'} {entry['filename']} -- {detail}")
        tally["fetched" if ok else "failed"] += 1
        time.sleep(PAUSE_BETWEEN_DOWNLOADS_S)

    for entry in spec.get("wikipedia", []):
        title = entry["title"]
        dest = os.path.join(dest_dir, _wiki_filename(title))
        recorded = entry.get("revid")
        live = _current_revid(lang, title)
        if recorded and live and live != recorded:
            tally["drifted"] += 1
            print(
                f"  [{lang}] DRIFT   {title}: manifest records r{recorded}, "
                f"live article is r{live} -- any gold case verified against the "
                f"old text needs re-checking"
            )
        if _is_valid_pdf(dest):
            tally["present"] += 1
            continue
        if check_only:
            print(f"  [{lang}] MISSING {_wiki_filename(title)}")
            tally["failed"] += 1
            continue
        url = WIKI_PDF_URL.format(lang=lang, title=requests.utils.quote(title.replace(" ", "_")))
        ok, detail = _download(url, dest)
        print(f"  [{lang}] {'OK  ' if ok else 'FAIL'} {_wiki_filename(title)} -- {detail}")
        tally["fetched" if ok else "failed"] += 1
        time.sleep(PAUSE_BETWEEN_DOWNLOADS_S)

    return tally


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", choices=("en", "es", "ca"), help="only this corpus")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what is missing or has drifted; download nothing",
    )
    args = parser.parse_args(argv)

    manifest = _load_manifest()
    langs = [args.lang] if args.lang else list(manifest)
    totals = {"present": 0, "fetched": 0, "failed": 0, "drifted": 0}

    for lang in langs:
        print(f"[{lang}]")
        for key, value in _fetch_corpus(lang, manifest.get(lang, {}), args.check).items():
            totals[key] += value

    print(
        f"\npresent {totals['present']}  fetched {totals['fetched']}  "
        f"failed {totals['failed']}  drifted {totals['drifted']}"
    )
    # Drift is reported but never fatal: the corpus is still usable, and the
    # decision about affected gold cases belongs to a person.
    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
