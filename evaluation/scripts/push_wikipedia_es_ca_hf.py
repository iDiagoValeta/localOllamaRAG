"""
Push merged MonkeyGrab ES+CA eval JSON to a Hugging Face Dataset repo.

Reads ``evaluation/datasets/local/dataset_eval_{es,ca}.json``, concatenates rows,
uploads as a single ``train`` split via ``datasets``, and refreshes the dataset README.

Usage (from repo root, with HF token in env or ``.env``)::

    python evaluation/scripts/push_wikipedia_es_ca_hf.py
    python evaluation/scripts/push_wikipedia_es_ca_hf.py --repo-id nadiva1243/wikipediaEs-Ca4RAG --private

Dependencies:
    pip install -r evaluation/requirements.txt
"""

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
# CONFIGURATION
# +-- 1. Imports
# +-- 2. Helpers (repo root, JSON load)
#
# BUSINESS LOGIC
# +-- 3. README template
# +-- 4. main()
#
# ENTRY
# +-- 5. __main__
#
# ─────────────────────────────────────────────

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected JSON array in {path}")
    return data


README_TEMPLATE = """---
license: mit
language:
  - es
  - ca
pretty_name: Wikipedia ES/CA for RAG evaluation (MonkeyGrab)
tags:
  - rag
  - retrieval
  - wikipedia
---

# Wikipedia ES/CA for RAG evaluation

Merged evaluation split used in the **MonkeyGrab** TFG project (UPV ETSINF):
Spanish (`es`) and Catalan (`ca`) question–answer pairs with short contexts
extracted from Wikipedia articles.

## Splits

| Split | Description |
|-------|-------------|
| `train` | All rows from `dataset_eval_es.json` + `dataset_eval_ca.json` (field `language`: `es` or `ca`) |

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Stable sample id (e.g. `wiki_es_001`) |
| `language` | string | `es` or `ca` |
| `source_url` | string | Wikipedia article URL |
| `context` | string | Retrieved passage used as RAG context |
| `question` | string | User question |
| `ground_truth` | string | Reference answer |
| `source_type` | string | e.g. `wikipedia` |

## Source and license

Contexts and questions were built from **Wikipedia** content; each row cites the
article URL in `source_url`. Respect Wikipedia's [Terms of use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use)
and [licensing](https://en.wikipedia.org/wiki/Wikipedia:Copyrights) when redistributing or
deriving new works. This repository release is marked **MIT** for the packaging and
metadata; the underlying text remains subject to Wikipedia's CC BY-SA where applicable.

## Project repository

Full source code (RAG pipeline, CLI, training scripts, evaluation workflows):

> **[https://github.com/iDiagoValeta/localOllamaRAG](https://github.com/iDiagoValeta/localOllamaRAG)**

## Citation (project)

If you use this dataset, cite the MonkeyGrab / TFG work and link this dataset on the Hub:

```bibtex
@misc{monkeygrab_wikipedia_es_ca,
  title        = {Wikipedia ES/CA for RAG evaluation (MonkeyGrab)},
  author       = {nadiva1243},
  year         = {2026},
  howpublished = {Hugging Face Datasets: \\url{https://huggingface.co/datasets/nadiva1243/wikipediaEs-Ca4RAG}},
  note         = {Source: https://github.com/iDiagoValeta/localOllamaRAG}
}
```
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--repo-id",
        default="nadiva1243/wikipediaEs-Ca4RAG",
        help="Hugging Face dataset repo id (org/name)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update as a private dataset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only merge and print row counts; do not upload",
    )
    args = parser.parse_args()

    root = _repo_root()
    local_dir = root / "evaluation" / "datasets" / "local"
    es_path = local_dir / "dataset_eval_es.json"
    ca_path = local_dir / "dataset_eval_ca.json"
    for p in (es_path, ca_path):
        if not p.is_file():
            print(f"ERROR: missing file: {p}", file=sys.stderr)
            return 1

    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass

    es_rows = _load_json(es_path)
    ca_rows = _load_json(ca_path)
    merged = es_rows + ca_rows
    print(f"Merged {len(es_rows)} (es) + {len(ca_rows)} (ca) = {len(merged)} rows")

    if args.dry_run:
        return 0

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print(
            "ERROR: set HF_TOKEN or HUGGINGFACE_HUB_TOKEN (or huggingface-cli login).",
            file=sys.stderr,
        )
        return 1

    from datasets import Dataset
    from huggingface_hub import HfApi

    ds = Dataset.from_list(merged)
    ds.push_to_hub(args.repo_id, split="train", private=args.private, token=token)
    print(f"Pushed split 'train' to https://huggingface.co/datasets/{args.repo_id}")

    api = HfApi(token=token)
    readme_bytes = README_TEMPLATE.encode("utf-8")
    api.upload_file(
        path_or_fileobj=io.BytesIO(readme_bytes),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="dataset card: add github repo link, citation bibtex",
    )
    print("Uploaded README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
