"""Aggregate RAGAS per-question scores by dataset subset (source_type, language, …).

Reads the RAGAS debug JSONs that ``evaluar_respuestas_con_ragas`` writes for
each ablation variant of a comparison run, aligns each result to its dataset
row by ``index``, and produces nested means: variant → subset → metric.

# ─────────────────────────────────────────────
# MODULE MAP
# ─────────────────────────────────────────────
#
#  1. Constants and dataset resolution
#  2. Subset key extraction
#  3. Aggregation core
#  4. Output (JSON + CSV)
#  5. High-level helper for evaluate.py
#
# ─────────────────────────────────────────────
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .datasets import DATASETS_DIR


# ─────────────────────────────────────────────
# SECTION 1: CONSTANTS AND DATASET RESOLUTION
# ─────────────────────────────────────────────

# RAGAS display names (debug JSON) → Spanish labels for the TFG report
METRICA_EN_A_ES: dict[str, str] = {
    "Factual Correctness": "Corrección factual",
    "Faithfulness": "Fidelidad",
    "Response Relevancy": "Relevancia de la respuesta",
    "Context Precision": "Precisión del contexto",
    "Context Recall": "Cobertura del contexto",
}

SUPPORTED_GROUP_BY = ("source_type", "language", "source_type_language", "id_prefix")


def _resolver_dataset(ruta: str | Path) -> Path:
    """Resolve a dataset path, falling back to common locations under datasets/."""
    p = Path(ruta)
    if p.is_file():
        return p.resolve()
    candidates = [
        Path(DATASETS_DIR) / "local" / p.name,
        Path(DATASETS_DIR) / "ragbench" / "prepared" / "en_eval" / p.name,
        Path(DATASETS_DIR) / "ragbench" / "prepared" / "dev_frozen" / p.name,
        Path(DATASETS_DIR) / "ragbench" / "prepared" / "visual" / p.name,
        Path(DATASETS_DIR) / p.name,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"No se encuentra el dataset: {ruta}")


# ─────────────────────────────────────────────
# SECTION 2: SUBSET KEY EXTRACTION
# ─────────────────────────────────────────────

def _conjunto_key(row: dict[str, Any], group_by: str) -> str:
    """Return subset label for one dataset row."""
    if group_by == "source_type":
        v = row.get("source_type")
        return str(v).strip() if v is not None and str(v).strip() else "unknown"
    if group_by == "language":
        v = row.get("language")
        return str(v).strip().lower() if v is not None and str(v).strip() else "unknown"
    if group_by == "source_type_language":
        st = row.get("source_type") or "unknown"
        lang = row.get("language") or "unknown"
        return f"{st}_{lang}".lower()
    if group_by == "id_prefix":
        rid = row.get("id")
        if rid is None:
            return "unknown"
        s = str(rid).strip()
        m = re.match(r"^([a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)?)_", s)
        if m:
            return m.group(1).lower()
        parts = s.split("_")
        return parts[0].lower() if parts else "unknown"
    raise ValueError(f"group_by no soportado: {group_by}")


# ─────────────────────────────────────────────
# SECTION 3: AGGREGATION CORE
# ─────────────────────────────────────────────

def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _mean_dict(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        return {}
    keys: set[str] = set()
    for d in values:
        keys.update(d.keys())
    out: dict[str, float] = {}
    for k in sorted(keys):
        nums = [d[k] for d in values if k in d and d[k] is not None]
        if nums:
            out[k] = sum(nums) / len(nums)
    return out


def _traducir_claves_metricas(d: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        out[METRICA_EN_A_ES.get(k, k)] = v
    return out


def _aplicar_etiquetas_es(report: dict[str, Any]) -> None:
    for vb in report.get("variants") or []:
        for block in (vb.get("by_conjunto") or {}).values():
            ms = block.get("mean_scores")
            if isinstance(ms, dict):
                block["mean_scores"] = _traducir_claves_metricas(ms)


def aggregate_variants(
    variant_debug_paths: list[tuple[str, Path]],
    dataset_path: Path,
    group_by: str,
) -> dict[str, Any]:
    """Build nested structure: variant → subset → mean_scores + n.

    Args:
        variant_debug_paths: ``[(variant_name, debug_json_path), ...]``. Each
            JSON must contain a ``results`` list with per-question ``index``
            (1-based) and ``scores`` (RAGAS display names → float|None).
        dataset_path: Source dataset JSON. Rows are aligned by 1-based index.
        group_by: One of ``SUPPORTED_GROUP_BY``.

    Returns:
        Report dict ready to be JSON-serialized. Mutate with
        ``_aplicar_etiquetas_es`` for Spanish metric labels.
    """
    if group_by not in SUPPORTED_GROUP_BY:
        valid = ", ".join(SUPPORTED_GROUP_BY)
        raise ValueError(f"group_by {group_by!r} no soportado. Válidos: {valid}")

    with dataset_path.open(encoding="utf-8") as f:
        dataset: list[dict[str, Any]] = json.load(f)

    report: dict[str, Any] = {
        "dataset_path": str(dataset_path.resolve()),
        "group_by": group_by,
        "variants": [],
    }

    for variant_name, json_path in variant_debug_paths:
        data = _load_json(json_path)
        results = data.get("results") or []
        bucket_scores: dict[str, list[dict[str, float]]] = defaultdict(list)

        for entry in results:
            idx = int(entry.get("index", 0)) - 1
            if idx < 0 or idx >= len(dataset):
                continue
            row = dataset[idx]
            label = _conjunto_key(row, group_by)
            scores = entry.get("scores") or {}
            numeric: dict[str, float] = {}
            for mk, mv in scores.items():
                if mv is None:
                    continue
                try:
                    numeric[str(mk)] = float(mv)
                except (TypeError, ValueError):
                    continue
            if numeric:
                bucket_scores[label].append(numeric)

        variant_block: dict[str, Any] = {
            "variant": variant_name,
            "source_file": str(json_path),
            "by_conjunto": {},
        }
        for label in sorted(bucket_scores.keys()):
            rows = bucket_scores[label]
            variant_block["by_conjunto"][label] = {
                "n": len(rows),
                "mean_scores": _mean_dict(rows),
            }
        report["variants"].append(variant_block)

    return report


# ─────────────────────────────────────────────
# SECTION 4: OUTPUT (JSON + CSV)
# ─────────────────────────────────────────────

def write_csv(report: dict[str, Any], csv_path: Path) -> None:
    """Write a long-form CSV (one row per variant × subset)."""
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("pandas is required to write the aggregation CSV.") from e

    rows: list[dict[str, Any]] = []
    for vb in report["variants"]:
        v = vb["variant"]
        for conjunto, block in vb["by_conjunto"].items():
            base = {"variant": v, "conjunto": conjunto, "n": block["n"]}
            for mk, mv in block["mean_scores"].items():
                base[mk] = mv
            rows.append(base)
    if not rows:
        return
    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")


# ─────────────────────────────────────────────
# SECTION 5: HIGH-LEVEL HELPER FOR evaluate.py
# ─────────────────────────────────────────────

def aggregate_comparison_run(
    variant_debug_paths: list[tuple[str, Path]],
    dataset_path: Path,
    out_dir: Path,
    group_by_list: list[str],
    etiquetas_es: bool = False,
    write_csv_too: bool = True,
) -> list[Path]:
    """Run aggregation for one comparison label across one or more group-by keys.

    Writes ``by_conjunto_<group_by>.json`` (or ``_metricas_es.json`` variants)
    plus a sibling ``resumen_por_conjunto.csv`` when ``write_csv_too`` is True.

    Returns the list of JSON paths written.
    """
    out_paths: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = _resolver_dataset(dataset_path)

    for group_by in group_by_list:
        report = aggregate_variants(variant_debug_paths, dataset_path, group_by)
        if etiquetas_es:
            _aplicar_etiquetas_es(report)
            report["metric_labels"] = "es"

        json_name = (
            f"by_conjunto_{group_by}_metricas_es.json"
            if etiquetas_es
            else f"by_conjunto_{group_by}.json"
        )
        out_json = out_dir / json_name
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        out_paths.append(out_json)
        print(f"  Aggregation report: {out_json}")

        if write_csv_too:
            try:
                csv_path = out_dir / f"resumen_por_conjunto_{group_by}.csv"
                write_csv(report, csv_path)
                print(f"  Aggregation CSV:    {csv_path}")
            except RuntimeError as exc:
                print(f"  [warn] no se escribió el CSV de agregación: {exc}")

    return out_paths
