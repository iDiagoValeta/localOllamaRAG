"""Shared setup: import the product and the gate from the checkout training/ lives in."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# training/ lives inside the MonkeyGrab checkout whose product and gate it imports.
REPO = os.path.dirname(ROOT)
os.environ.setdefault("MONKEYGRAB_DATA_DIR", os.path.join(ROOT, "data", "store"))
for p in (REPO, os.path.join(REPO, "src"), os.path.join(REPO, "tests", "eval")):
    if p not in sys.path:
        sys.path.insert(0, p)
TEACHER = "qwen3:4b-instruct-2507-q4_K_M"
LANGS = ("en", "es", "ca")

def docs_dir(lang):
    from pathlib import Path
    return Path(ROOT) / "docs" / lang

# One held-out document per language that no training set may contain: the
# fast dev evaluation (i_deveval.py) scores on them, so the gate stays an exam.
DEV_DOCS = {"en": "lora.pdf", "es": "Volcán.pdf", "ca": "Pirineus.pdf"}
