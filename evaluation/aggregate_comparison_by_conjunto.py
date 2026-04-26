# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
# CONFIGURATION
# +-- 1. Imports and paths
# +-- 2. CLI
#
# BUSINESS LOGIC
# +-- 3. Dataset loading and grouping keys
# +-- 4. Per-variant aggregation
# +-- 5. Export JSON / CSV
#
# ENTRY
# +-- 6. main()
#
# ─────────────────────────────────────────────
"""
aggregate_comparison_by_conjunto -- Agrupa métricas RAGAS por subconjunto del dataset.

Lee los JSON de debug de un directorio de comparación (p. ej.
``evaluation/runs/ragas/comparisons/todas_ablacion``), alinea cada muestra con
la fila del dataset por ``index`` y calcula medias por ``source_type``,
``language``, etc.

Usage:
    python evaluation/aggregate_comparison_by_conjunto.py \\
        --dir evaluation/runs/ragas/comparisons/todas_ablacion

    python evaluation/aggregate_comparison_by_conjunto.py \\
        --dir evaluation/runs/ragas/comparisons/todas_ablacion \\
        --group-by language --output evaluation/runs/ragas/comparisons/todas_ablacion/aggregates/por_idioma.json

Dependencies:
    - stdlib + optional pandas for CSV (--csv)

Documentation (presets, flags, TFG notes):
    ``evaluation/EVALUACIONES_PIPELINE.md`` — sección *Agregación por conjunto*.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parent
RAGAS_RUNS_DIR = EVAL_DIR / "runs" / "ragas"
DEFAULT_COMPARISONS_DIR = RAGAS_RUNS_DIR / "comparisons"
SUMMARY_NAME = "comparison_summary.json"

# RAGAS display names (debug JSON) -> etiquetas en castellano para informes
METRICA_EN_A_ES: dict[str, str] = {
    "Factual Correctness": "Corrección factual",
    "Faithfulness": "Fidelidad",
    "Response Relevancy": "Relevancia de la respuesta",
    "Context Precision": "Precisión del contexto",
    "Context Recall": "Cobertura del contexto",
}


def _resolver_dataset(ruta: str | None) -> Path:
    """Resolve dataset path; try ``evaluation/datasets/<basename>`` if missing."""
    if not ruta:
        raise ValueError("dataset path is empty")
    p = Path(ruta)
    if p.is_file():
        return p.resolve()
    candidates = [
        EVAL_DIR / "datasets" / "local" / p.name,
        EVAL_DIR / "datasets" / "ragbench" / "prepared" / "en_eval" / p.name,
        EVAL_DIR / "datasets" / "ragbench" / "prepared" / "dev_frozen" / p.name,
        EVAL_DIR / "datasets" / "ragbench" / "prepared" / "visual" / p.name,
        EVAL_DIR / "datasets" / p.name,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"No se encuentra el dataset: {ruta} (tampoco {cand})")


def _resolve_comparison_input(path: Path) -> tuple[Path, Path, Path]:
    """Return comparison root, debug dir and aggregate dir for new or legacy layouts."""
    if (path / "debug").is_dir():
        return path, path / "debug", path / "aggregates"
    return path, path, path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


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


def _discover_variant_files(
    debug_dir: Path,
    ignore_comparison_summary: bool = False,
) -> tuple[list[tuple[str, Path]], Path | None, str | None]:
    """Return (variant_name, json_path) list, optional summary path, dataset_path from summary.

    If ``ignore_comparison_summary`` is True, every ``*.json`` in the folder is used except
    ``comparison_summary.json`` and aggregated reports ``by_conjunto_*.json``. Use this when
    the summary's ``runs`` list is from a partial ``compare`` but full per-variant debug files
    are already on disk.
    """
    summary_path = debug_dir / SUMMARY_NAME
    if not summary_path.is_file() and (debug_dir.parent / SUMMARY_NAME).is_file():
        summary_path = debug_dir.parent / SUMMARY_NAME
    dataset_from_summary: str | None = None
    out: list[tuple[str, Path]] = []

    if summary_path.is_file():
        summary = _load_json(summary_path)
        dataset_from_summary = summary.get("dataset_path")
        if not ignore_comparison_summary:
            for run in summary.get("runs") or []:
                v = run.get("variant")
                dp = run.get("debug_path")
                if not v or not dp:
                    continue
                p = Path(dp)
                if not p.is_file():
                    p = debug_dir / f"{v}.json"
                if p.is_file():
                    out.append((str(v), p.resolve()))

    if not out:
        for p in sorted(debug_dir.glob("*.json")):
            if p.name == SUMMARY_NAME:
                continue
            if p.name.startswith("by_conjunto_"):
                continue
            out.append((p.stem, p.resolve()))

    return out, summary_path if summary_path.is_file() else None, dataset_from_summary


def _traducir_claves_metricas(d: dict[str, float]) -> dict[str, float]:
    """Rename known English RAGAS labels to Spanish; unknown keys pass through."""
    out: dict[str, float] = {}
    for k, v in d.items():
        out[METRICA_EN_A_ES.get(k, k)] = v
    return out


def _aplicar_etiquetas_es_al_informe(report: dict[str, Any]) -> None:
    """Mutate report in place: mean_scores keys -> Spanish where mapped."""
    for vb in report.get("variants") or []:
        for block in (vb.get("by_conjunto") or {}).values():
            ms = block.get("mean_scores")
            if isinstance(ms, dict):
                block["mean_scores"] = _traducir_claves_metricas(ms)


def _mean_dict(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        return {}
    keys = set()
    for d in values:
        keys.update(d.keys())
    out: dict[str, float] = {}
    for k in sorted(keys):
        nums = [d[k] for d in values if k in d and d[k] is not None]
        if nums:
            out[k] = sum(nums) / len(nums)
    return out


def aggregate_folder(
    debug_dir: Path,
    dataset_path: Path,
    group_by: str,
    ignore_comparison_summary: bool = False,
) -> dict[str, Any]:
    """Build nested structure: variant -> subset -> mean_scores + n."""
    with dataset_path.open(encoding="utf-8") as f:
        dataset: list[dict[str, Any]] = json.load(f)

    variants, summary_path, _ = _discover_variant_files(
        debug_dir, ignore_comparison_summary=ignore_comparison_summary
    )
    if not variants:
        raise FileNotFoundError(f"No hay JSON de variantes en {debug_dir}")

    report: dict[str, Any] = {
        "debug_dir": str(debug_dir.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "group_by": group_by,
        "comparison_summary": str(summary_path.resolve()) if summary_path else None,
        "variant_discovery": "all_json_in_folder" if ignore_comparison_summary else "comparison_summary_runs",
        "variants": [],
    }

    for variant_name, json_path in variants:
        data = _load_json(json_path)
        results = data.get("results") or []
        # scores per (subset) list of metric dicts
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


def _write_csv(report: dict[str, Any], csv_path: Path) -> None:
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("Instala pandas para usar --csv: pip install pandas") from e

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agrega métricas por conjunto (subconjunto del dataset) a partir de los JSON de comparación.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=str(DEFAULT_COMPARISONS_DIR / "todas_ablacion"),
        help="Carpeta con baseline_*.json / no_*.json y opcionalmente comparison_summary.json",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Ruta al JSON del dataset. Por defecto: dataset_path del comparison_summary.json",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        choices=("source_type", "language", "source_type_language", "id_prefix"),
        default="source_type",
        help="Campo o regla para definir el conjunto (subconjunto) de cada pregunta.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta del JSON de salida. Por defecto: <dir>/by_conjunto_<group_by>.json",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Si se indica, escribe también un CSV largo (variante × conjunto × métricas).",
    )
    parser.add_argument(
        "--etiquetas-es",
        action="store_true",
        help="Nombres de métricas en castellano en mean_scores y columnas del CSV.",
    )
    parser.add_argument(
        "--ignore-comparison-summary",
        action="store_true",
        help="Usar todos los <variante>.json de la carpeta (excepto comparison_summary y by_conjunto_*), "
        "no solo las variantes listadas en comparison_summary.json (útil tras un compare parcial).",
    )
    args = parser.parse_args()

    comparison_root, debug_dir, aggregates_dir = _resolve_comparison_input(Path(args.dir).resolve())
    if not debug_dir.is_dir():
        raise SystemExit(f"No es un directorio: {debug_dir}")

    _, _, ds_summary = _discover_variant_files(debug_dir, ignore_comparison_summary=False)
    dataset_arg = args.dataset or ds_summary
    if not dataset_arg:
        raise SystemExit(
            "No hay dataset_path en comparison_summary.json; pasa --dataset explícitamente.",
        )
    dataset_path = _resolver_dataset(dataset_arg)

    report = aggregate_folder(
        debug_dir,
        dataset_path,
        args.group_by,
        ignore_comparison_summary=args.ignore_comparison_summary,
    )
    if args.etiquetas_es:
        _aplicar_etiquetas_es_al_informe(report)
        report["metric_labels"] = "es"

    out_json = (
        Path(args.output)
        if args.output
        else aggregates_dir
        / (
            f"by_conjunto_{args.group_by}_metricas_es.json"
            if args.etiquetas_es
            else f"by_conjunto_{args.group_by}.json"
        )
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Escrito: {out_json}")

    if args.csv:
        csv_path = Path(args.csv)
        _write_csv(report, csv_path)
        print(f"Escrito: {csv_path.resolve()}")
    elif (comparison_root / "scores").is_dir():
        csv_path = comparison_root / "scores" / "resumen_por_conjunto.csv"
        _write_csv(report, csv_path)
        print(f"Escrito: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
