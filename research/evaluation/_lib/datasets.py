"""Dataset loading, normalization, and corpus/path resolution.

# ─────────────────────────────────────────────
# MODULE MAP
# ─────────────────────────────────────────────
#
#  1. Paths and corpus constants
#  2. Dataset I/O
#  3. Path helpers per corpus
#
# ─────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

# ─────────────────────────────────────────────
# SECTION 1: PATHS AND CORPUS CONSTANTS
# ─────────────────────────────────────────────

EVAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ_ROOT = os.path.dirname(os.path.dirname(EVAL_DIR))

DATASETS_DIR = os.path.join(EVAL_DIR, "datasets")
LOCAL_DATASETS_DIR = os.path.join(DATASETS_DIR, "local")
RAGBENCH_DATASETS_DIR = os.path.join(DATASETS_DIR, "ragbench")
RAGBENCH_PREPARED_DIR = os.path.join(RAGBENCH_DATASETS_DIR, "prepared")
RUNS_DIR = os.path.join(EVAL_DIR, "runs")
RAGAS_RUNS_DIR = os.path.join(RUNS_DIR, "ragas")
SINGLE_RUNS_DIR = os.path.join(RAGAS_RUNS_DIR, "single")
COMPARISON_RUNS_DIR = os.path.join(RAGAS_RUNS_DIR, "comparisons")
RAGBENCH_RUNS_DIR = os.path.join(RAGAS_RUNS_DIR, "ragbench")
RAGBENCH_VISUAL_RUNS_DIR = os.path.join(RAGAS_RUNS_DIR, "ragbench_visual")
TMP_DIR = os.path.join(EVAL_DIR, "tmp")

SUPPORTED_CORPORA = ("es", "ca", "en", "mix")


# ─────────────────────────────────────────────
# SECTION 2: DATASET I/O
# ─────────────────────────────────────────────

def cargar_dataset(ruta: str) -> pd.DataFrame:
    """Load a dataset from a JSON, CSV, or Excel file into a DataFrame."""
    ext = os.path.splitext(ruta)[1].lower()
    if ext == ".json":
        with open(ruta, encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(ruta)
    if ext == ".csv":
        return pd.read_csv(ruta, encoding="utf-8")
    raise ValueError(f"Unsupported format: {ext}")


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to a canonical ``question`` / ``ground_truth`` schema."""
    mapeo = {
        "pregunta": "question",
        "question": "question",
        "preguntas": "question",
        "ground_truth": "ground_truth",
        "respuesta_esperada": "ground_truth",
        "respuesta_referencia": "ground_truth",
        "reference": "ground_truth",
    }
    cols = {c.lower(): c for c in df.columns}
    out = {}
    for orig, target in mapeo.items():
        if orig in cols:
            out[target] = df[cols[orig]].tolist()
    if "question" not in out:
        raise ValueError("Dataset must have a 'question' or 'pregunta' column")
    if "ground_truth" not in out:
        out["ground_truth"] = [""] * len(out["question"])
    return pd.DataFrame(out)


def resolver_ruta_dataset(ruta: str) -> str:
    """Resolve a dataset path, falling back to bundled ``datasets/local/``."""
    expanded = os.path.abspath(os.path.expanduser(ruta))
    if os.path.isfile(expanded):
        return expanded
    name = os.path.basename(expanded)
    if name.startswith("dataset_eval_") and name.lower().endswith(".json"):
        alt = os.path.join(LOCAL_DATASETS_DIR, name)
        if os.path.isfile(alt):
            print(
                f"   Note: dataset path resolved to {alt} (input was {ruta!r}; "
                "use research/evaluation/datasets/… to silence this message)"
            )
            return alt
    raise FileNotFoundError(
        f"Dataset file not found: {expanded}. "
        f"Bundled datasets are under {DATASETS_DIR!r}, "
        f"e.g. {os.path.join(LOCAL_DATASETS_DIR, 'dataset_eval_es.json')}"
    )


# ─────────────────────────────────────────────
# SECTION 3: PATH HELPERS PER CORPUS
# ─────────────────────────────────────────────

def default_dataset_for_corpus(eval_corpus: str) -> str:
    """Default bundled JSON path for a local evaluation corpus."""
    if eval_corpus not in SUPPORTED_CORPORA:
        valid = ", ".join(SUPPORTED_CORPORA)
        raise ValueError(f"Unsupported corpus {eval_corpus!r}. Valid: {valid}")
    name = f"dataset_eval_{eval_corpus}.json"
    return os.path.join(LOCAL_DATASETS_DIR, name)


def default_docs_dir_for_corpus(eval_corpus: str) -> str | None:
    """Default PDF folder for a corpus, or ``None`` to use the RAG module default."""
    corpus_to_folder: dict[str, str | None] = {
        "es": None,
        "ca": os.path.join(PROJ_ROOT, "rag", "docs", "ca"),
        "en": os.path.join(PROJ_ROOT, "rag", "docs", "en"),
        "mix": None,
        "ragbench": os.path.join(PROJ_ROOT, "rag", "docs", "en"),
    }
    return corpus_to_folder.get(eval_corpus)


def artifact_suffix(eval_corpus: str) -> str:
    """Filename suffix for scores/debug/checkpoints (always includes language tag)."""
    suffix_map: dict[str, str] = {
        "es": "_es",
        "ca": "_ca",
        "en": "_en",
        "mix": "_mix",
        "ragbench": "_en",
    }
    return suffix_map.get(eval_corpus, f"_{eval_corpus}")


def slugify(value: str) -> str:
    """Convert a free-form label into a filesystem-friendly slug."""
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "eval"


def safe_tag(value: str) -> str:
    """Convert an arbitrary string to a safe filesystem tag (preserves case)."""
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def build_output_stem(dataset_path: str) -> str:
    """Create a stable stem for outputs derived from the dataset filename."""
    return slugify(Path(dataset_path).stem)


def single_run_dir(dataset_path: str, suffix: str = "") -> str:
    return os.path.join(SINGLE_RUNS_DIR, f"{build_output_stem(dataset_path)}{suffix}")


def default_output_path(dataset_path: str, suffix: str = "") -> str:
    return os.path.join(single_run_dir(dataset_path, suffix), "scores.csv")


def default_debug_path(dataset_path: str, suffix: str = "") -> str:
    return os.path.join(single_run_dir(dataset_path, suffix), "debug.json")


def build_run_slug(dataset_path: str, label: str | None, eval_corpus: str) -> str:
    """Build a stable folder slug for a comparison batch."""
    dataset_stem = Path(dataset_path).stem
    suffix = label.strip().replace(" ", "_") if label else f"{dataset_stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    if eval_corpus == "ca":
        suffix = f"{suffix}_ca"
    return suffix


def guardar_json(path: str, payload: dict[str, Any]) -> None:
    """Write a JSON payload ensuring the parent directory exists."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
