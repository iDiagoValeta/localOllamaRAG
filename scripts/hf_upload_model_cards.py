# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# MODULE MAP -- Section index
# ------------------------------------------------------------
#
# CONFIGURATION
# +-- 1. Imports
# +-- 2. Constants
#
# ENTRY
# +-- 3. main()
#
# ------------------------------------------------------------
"""Upload MonkeyGrab model cards + reproduction/ snapshots to Hugging Face Hub.

Usage (from repo root, with HUGGINGFACE_HUB_TOKEN set):
    python scripts/hf_upload_model_cards.py
    python scripts/hf_upload_model_cards.py --upload-qwen-q4-gguf

``--upload-qwen-q4-gguf`` uploads only ``models/gguf-output/qwen-3/Qwen3-14B-Q4_K_M.gguf``
(~9 GB); use when the binary is ready after local conversion.
"""
from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QWEN_GGUF_REPO = "nadiva1243/qwen3RAG"
QWEN_GGUF_LOCAL = os.path.join(
    ROOT, "models", "gguf-output", "qwen-3", "Qwen3-14B-Q4_K_M.gguf"
)
QWEN_GGUF_REMOTE = "Qwen3-14B-Q4_K_M.gguf"


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload model cards / GGUF to Hugging Face Hub.")
    parser.add_argument(
        "--upload-qwen-q4-gguf",
        action="store_true",
        help="Upload only the Qwen3 Q4_K_M GGUF to nadiva1243/qwen3RAG (large file).",
    )
    args = parser.parse_args()

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("Missing HUGGINGFACE_HUB_TOKEN (or HF_TOKEN).", file=sys.stderr)
        return 1

    api = HfApi(token=token)

    def up(local_path: str, repo_id: str, path_in_repo: str) -> None:
        abs_path = os.path.join(ROOT, local_path.replace("/", os.sep))
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(abs_path)
        api.upload_file(
            path_or_fileobj=abs_path,
            path_in_repo=path_in_repo.replace("\\", "/"),
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  OK {repo_id} <- {path_in_repo}")

    if args.upload_qwen_q4_gguf:
        if not os.path.isfile(QWEN_GGUF_LOCAL):
            print(f"Missing GGUF: {QWEN_GGUF_LOCAL}", file=sys.stderr)
            return 1
        size_gb = os.path.getsize(QWEN_GGUF_LOCAL) / (1024**3)
        print(f"Uploading {QWEN_GGUF_REMOTE} ({size_gb:.2f} GiB) to {QWEN_GGUF_REPO} …")
        up(
            "models/gguf-output/qwen-3/Qwen3-14B-Q4_K_M.gguf",
            QWEN_GGUF_REPO,
            QWEN_GGUF_REMOTE,
        )
        print("GGUF upload done.")
        return 0

    print("Uploading nadiva1243/phi4RAG …")
    up("models/gguf-output/phi-4/README.md", "nadiva1243/phi4RAG", "README.md")
    up("models/gguf-output/phi-4/Modelfile", "nadiva1243/phi4RAG", "Modelfile")
    up("models/gguf-output/phi-4/LICENSE", "nadiva1243/phi4RAG", "LICENSE")
    up("models/gguf-output/phi-4/CONVERSION.md", "nadiva1243/phi4RAG", "reproduction/CONVERSION.md")
    up("scripts/training/train-phi4.py", "nadiva1243/phi4RAG", "reproduction/train-phi4.py")
    up("scripts/conversion/merge_lora.py", "nadiva1243/phi4RAG", "reproduction/merge_lora.py")
    up("training-output/phi-4/evaluation_comparison.json", "nadiva1243/phi4RAG", "reproduction/evaluation_comparison.json")
    up("training-output/phi-4/training_stats.json", "nadiva1243/phi4RAG", "reproduction/training_stats.json")

    print("Uploading nadiva1243/qwen3RAG …")
    up("models/gguf-output/qwen-3/README.md", "nadiva1243/qwen3RAG", "README.md")
    up("models/gguf-output/qwen-3/Modelfile", "nadiva1243/qwen3RAG", "Modelfile")
    up("models/gguf-output/qwen-3/LICENSE", "nadiva1243/qwen3RAG", "LICENSE")
    up("models/gguf-output/qwen-3/CONVERSION.md", "nadiva1243/qwen3RAG", "reproduction/CONVERSION.md")
    up("scripts/training/train-qwen3.py", "nadiva1243/qwen3RAG", "reproduction/train-qwen3.py")
    up("scripts/conversion/merge_lora.py", "nadiva1243/qwen3RAG", "reproduction/merge_lora.py")
    up("training-output/qwen-3/evaluation_comparison.json", "nadiva1243/qwen3RAG", "reproduction/evaluation_comparison.json")
    up("training-output/qwen-3/training_stats.json", "nadiva1243/qwen3RAG", "reproduction/training_stats.json")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
