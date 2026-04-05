# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
# CONFIGURATION
# +-- 1. Imports
# +-- 2. Constants, formatting
#
# BUSINESS LOGIC
# +-- 3. Load JSON, dataset / model ordering
# +-- 4. CSV writers
# +-- 5. Markdown tables
# +-- 6. Figures (matplotlib)
#
# ENTRY
# +-- 7. main()
#
# ─────────────────────────────────────────────
"""
generate_reports.py -- Rebuild baseline CSV + Markdown reports from baseline_evaluation.json.

Reads the aggregate output of ``scripts/evaluation/evaluate_baselines.py`` and writes the
same artefacts as ``reports/`` (per-metric CSVs and ``model_dataset_metric_tables.md``).

Usage:
    cd training-output/baseline && python generate_reports.py
    python training-output/baseline/generate_reports.py
    python generate_reports.py --input baseline_evaluation.json --out-dir reports

If you add datasets or splits, unknown dataset names are appended (sorted) before
``aggregate``; split sections are emitted for every key found under each mode
(e.g. ``dev``, ``test``).

PNG figures (requires matplotlib) are written under ``reports/figures/``:
heatmap grid per mode/split, and horizontal bar charts using the aggregate column.
Models are ordered **best → worst** per figure using the ``aggregate`` row for that
metric (see ``SORT_METRIC_DESCENDING``; response length treats shorter as better).
CSVs / Markdown keep alphabetical model order.

Use ``--no-figures`` to skip plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# ─────────────────────────────────────────────
# SECTION 2: CONSTANTS, FORMATTING
# ─────────────────────────────────────────────

# Column order in existing reports (matches evaluate_baselines dataset names).
DATASET_ORDER: List[str] = [
    "Aina-CA",
    "Aina-EN",
    "Aina-ES",
    "Dolly QA",
    "Neural-Bridge RAG",
    "aggregate",
]

METRICS_ORDER: List[str] = [
    "Token_F1",
    "ROUGE_L_F1",
    "BERTScore_F1",
    "Context_Faithfulness_Pct",
    "Sentence_Completeness_Pct",
    "Avg_Response_Length_Words",
]

MODES: Tuple[str, ...] = ("with_context", "without_context")

# Figure ordering: sort by aggregate[metric]. True => higher score = better (top).
# Avg_Response_Length_Words: False => fewer words = better (concise).
SORT_METRIC_DESCENDING: Dict[str, bool] = {
    "Token_F1": True,
    "ROUGE_L_F1": True,
    "BERTScore_F1": True,
    "Context_Faithfulness_Pct": True,
    "Sentence_Completeness_Pct": True,
    "Avg_Response_Length_Words": False,
}

# ─────────────────────────────────────────────
# SECTION 3: DATASET / MODEL ORDERING
# ─────────────────────────────────────────────


def short_model_name(full_id: str) -> str:
    """HF-style id -> last path segment (e.g. Qwen/Qwen3-14B -> Qwen3-14B)."""
    return full_id.split("/")[-1]


def ordered_dataset_columns(split_block: Mapping[str, Any]) -> List[str]:
    """Stable column order: known datasets first, then extras (sorted), aggregate last."""
    keys = set(split_block.keys())
    known_non_agg = [k for k in DATASET_ORDER if k != "aggregate" and k in keys]
    known_set = set(DATASET_ORDER)
    extras = sorted(k for k in keys if k not in known_set and k != "aggregate")
    out = known_non_agg + extras
    if "aggregate" in keys:
        out.append("aggregate")
    return out


def fmt_cell(value: Any) -> str:
    """Format a number for CSV/Markdown tables (2 decimal places)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return str(value)


def get_metric(
    dataset_block: Mapping[str, Any],
    metric: str,
) -> Any:
    """Return metric value or None if missing."""
    if metric not in dataset_block:
        return None
    return dataset_block[metric]


def iter_models_sorted(results: Mapping[str, Any]) -> List[str]:
    """Model keys sorted by display name (same ordering as legacy reports)."""
    return sorted(results.keys(), key=lambda k: short_model_name(k))


def discover_splits(results: Mapping[str, Any]) -> List[str]:
    """Union of split names (e.g. dev, test) under both modes."""
    splits: set[str] = set()
    for _mid, modes in results.items():
        for mode in MODES:
            block = modes.get(mode)
            if not isinstance(block, dict):
                continue
            splits.update(block.keys())
    # Stable: dev first if present, then alphabetical
    ordered = [s for s in ("dev", "test") if s in splits]
    for s in sorted(splits):
        if s not in ordered:
            ordered.append(s)
    return ordered


# ─────────────────────────────────────────────
# SECTION 4: CSV WRITERS
# ─────────────────────────────────────────────


def write_metric_csv(
    *,
    path: str,
    results: Mapping[str, Any],
    mode: str,
    split: str,
    metric: str,
    dataset_cols: List[str],
    model_ids: Sequence[str],
) -> None:
    """Write one CSV: Model + one column per dataset."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model"] + dataset_cols)
        for mid in model_ids:
            row = [short_model_name(mid)]
            modes = results.get(mid, {})
            split_block = (modes.get(mode) or {}).get(split) or {}
            for ds in dataset_cols:
                ds_block = split_block.get(ds)
                if not isinstance(ds_block, dict):
                    row.append("")
                    continue
                v = get_metric(ds_block, metric)
                row.append(fmt_cell(v) if v is not None else "")
            w.writerow(row)


# ─────────────────────────────────────────────
# SECTION 5: MARKDOWN TABLES
# ─────────────────────────────────────────────


def build_markdown(
    results: Mapping[str, Any],
    splits: List[str],
) -> str:
    lines: List[str] = [
        "# Comparativa directa de modelos por metrica",
        "",
    ]
    model_ids = iter_models_sorted(results)

    for mode in MODES:
        for split in splits:
            # Split block sample for dataset columns
            sample_split: Dict[str, Any] = {}
            for mid in model_ids:
                sp = (results.get(mid, {}).get(mode) or {}).get(split)
                if isinstance(sp, dict) and sp:
                    sample_split = sp
                    break
            if not sample_split:
                continue
            dataset_cols = ordered_dataset_columns(sample_split)

            lines.append(f"## Mode: `{mode}` | split: `{split}`")
            lines.append("")

            for metric in METRICS_ORDER:
                lines.append(f"### Metrica: `{metric}`")
                lines.append("")
                header = "| Model | " + " | ".join(dataset_cols) + " |"
                sep = "| --- | " + " | ".join(["---"] * len(dataset_cols)) + " |"
                lines.append(header)
                lines.append(sep)
                for mid in model_ids:
                    split_block = (results.get(mid, {}).get(mode) or {}).get(split) or {}
                    cells = []
                    for ds in dataset_cols:
                        ds_block = split_block.get(ds)
                        if not isinstance(ds_block, dict):
                            cells.append("")
                            continue
                        v = get_metric(ds_block, metric)
                        cells.append(fmt_cell(v) if v is not None else "")
                    name = short_model_name(mid)
                    line = "| " + name + " | " + " | ".join(cells) + " |"
                    lines.append(line)
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ─────────────────────────────────────────────
# SECTION 6: FIGURES (MATPLOTLIB)
# ─────────────────────────────────────────────


def _aggregate_value(
    results: Mapping[str, Any],
    mid: str,
    mode: str,
    split: str,
    metric: str,
) -> float:
    """Aggregate metric for one model, or NaN if missing."""
    agg = ((results.get(mid, {}).get(mode) or {}).get(split) or {}).get("aggregate")
    if not isinstance(agg, dict):
        return float("nan")
    v = get_metric(agg, metric)
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def sort_models_for_figure_metric(
    results: Mapping[str, Any],
    model_ids: Sequence[str],
    mode: str,
    split: str,
    metric: str,
) -> List[str]:
    """Order model ids best-first for this metric (uses aggregate row). NaN last."""
    import math

    descending = SORT_METRIC_DESCENDING.get(metric, True)
    scored = [(mid, _aggregate_value(results, mid, mode, split, metric)) for mid in model_ids]

    def sort_key(item: Tuple[str, float]) -> Tuple[int, float, str]:
        mid, s = item
        if math.isnan(s):
            return (1, 0.0, short_model_name(mid))
        if descending:
            return (0, -s, short_model_name(mid))
        return (0, s, short_model_name(mid))

    return [mid for mid, _ in sorted(scored, key=sort_key)]


def _figure_basename(mode: str, split: str, stem: str, splits: List[str]) -> str:
    if len(splits) == 1:
        return f"{mode}__{stem}.png"
    return f"{mode}__{split}__{stem}.png"


def _matrix_for_metric(
    results: Mapping[str, Any],
    model_ids: Sequence[str],
    mode: str,
    split: str,
    metric: str,
    dataset_cols: List[str],
) -> Any:
    import numpy as np

    rows = []
    for mid in model_ids:
        split_block = (results.get(mid, {}).get(mode) or {}).get(split) or {}
        r = []
        for ds in dataset_cols:
            ds_block = split_block.get(ds)
            if not isinstance(ds_block, dict):
                r.append(np.nan)
                continue
            v = get_metric(ds_block, metric)
            r.append(float(v) if v is not None else np.nan)
        rows.append(r)
    return np.array(rows, dtype=float)


def write_figure_heatmap_grid(
    *,
    out_path: str,
    results: Mapping[str, Any],
    model_ids: Sequence[str],
    mode: str,
    split: str,
    dataset_cols: List[str],
) -> None:
    """2x3 panel of heatmaps (models x datasets), one panel per metric."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, METRICS_ORDER):
        ordered = sort_models_for_figure_metric(results, model_ids, mode, split, metric)
        short_labels = [short_model_name(m) for m in ordered]
        data = _matrix_for_metric(results, ordered, mode, split, metric, dataset_cols)
        im = ax.imshow(data, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_title(metric, fontsize=10)
        ax.set_xticks(range(len(dataset_cols)))
        ax.set_xticklabels(dataset_cols, rotation=38, ha="right", fontsize=7)
        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels(short_labels, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    fig.suptitle(f"Baseline metrics (heatmap) — {mode} / {split}", fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_figure_aggregate_bars(
    *,
    out_path: str,
    results: Mapping[str, Any],
    model_ids: Sequence[str],
    mode: str,
    split: str,
) -> None:
    """2x3 horizontal bar charts using only the aggregate row per metric."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = os.path.dirname(out_path) or "."
    os.makedirs(fig_dir, exist_ok=True)

    if "aggregate" not in (
        (results.get(model_ids[0], {}).get(mode) or {}).get(split) or {}
    ):
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, "No aggregate row in JSON for this split", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, METRICS_ORDER):
        ordered = sort_models_for_figure_metric(results, model_ids, mode, split, metric)
        short_labels = [short_model_name(m) for m in ordered]
        y = np.arange(len(ordered))
        vals: List[float] = []
        for mid in ordered:
            agg = (
                (results.get(mid, {}).get(mode) or {}).get(split) or {}
            ).get("aggregate")
            if not isinstance(agg, dict):
                vals.append(float("nan"))
                continue
            v = get_metric(agg, metric)
            vals.append(float(v) if v is not None else float("nan"))
        ax.barh(y, vals, color="#2ca02c", height=0.65)
        ax.set_yticks(y)
        ax.set_yticklabels(short_labels, fontsize=7)
        ax.set_title(f"{metric} (aggregate)", fontsize=9)
        ax.set_xlabel("value", fontsize=8)
        ax.invert_yaxis()

    fig.suptitle(f"Baseline — aggregate by model — {mode} / {split}", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_figure_single_metric_heatmap(
    *,
    out_path: str,
    results: Mapping[str, Any],
    model_ids: Sequence[str],
    mode: str,
    split: str,
    metric: str,
    dataset_cols: List[str],
) -> None:
    """One heatmap (models x datasets) for a single metric."""
    import matplotlib.pyplot as plt
    import numpy as np

    ordered = sort_models_for_figure_metric(results, model_ids, mode, split, metric)
    data = _matrix_for_metric(results, ordered, mode, split, metric, dataset_cols)
    short_labels = [short_model_name(m) for m in ordered]

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(dataset_cols)), max(5, 0.35 * len(ordered))))
    im = ax.imshow(data, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_title(f"{metric} — {mode} / {split}", fontsize=11)
    ax.set_xticks(range(len(dataset_cols)))
    ax.set_xticklabels(dataset_cols, rotation=38, ha="right", fontsize=8)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(short_labels, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_figure_grouped_bars_by_dataset(
    *,
    out_path: str,
    results: Mapping[str, Any],
    model_ids: Sequence[str],
    mode: str,
    split: str,
    metric: str,
    dataset_cols: List[str],
) -> None:
    """Grouped bars: x = dataset (excluding aggregate), one series per model."""
    import matplotlib.pyplot as plt
    import numpy as np

    ds_only = [c for c in dataset_cols if c != "aggregate"]
    if not ds_only:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No per-dataset columns (only aggregate)", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    ordered = sort_models_for_figure_metric(results, model_ids, mode, split, metric)
    n_m = len(ordered)
    n_d = len(ds_only)
    x = np.arange(n_d, dtype=float)
    bar_w = min(0.8 / max(n_m, 1), 0.12)
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_m, 10)))

    fig, ax = plt.subplots(figsize=(max(9, 1.1 * n_d), 5))
    for i, mid in enumerate(ordered):
        vals: List[float] = []
        split_block = (results.get(mid, {}).get(mode) or {}).get(split) or {}
        for ds in ds_only:
            ds_block = split_block.get(ds)
            if not isinstance(ds_block, dict):
                vals.append(float("nan"))
                continue
            v = get_metric(ds_block, metric)
            vals.append(float(v) if v is not None else float("nan"))
        offset = (i - (n_m - 1) / 2.0) * bar_w
        ax.bar(
            x + offset,
            vals,
            bar_w,
            label=short_model_name(mid),
            color=colors[i % len(colors)],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ds_only, rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("value", fontsize=9)
    ax.set_title(f"{metric} — by dataset (grouped by model) — {mode} / {split}", fontsize=10)
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_figure_assets(
    results: Mapping[str, Any],
    *,
    out_dir: str,
    splits: List[str],
    model_ids: Sequence[str],
) -> int:
    """Write PNGs under ``out_dir/figures``. Returns number of PNG files written."""
    n = 0
    for mode in MODES:
        for split in splits:
            sample_split: Dict[str, Any] = {}
            for mid in model_ids:
                sp = (results.get(mid, {}).get(mode) or {}).get(split)
                if isinstance(sp, dict) and sp:
                    sample_split = sp
                    break
            if not sample_split:
                continue
            dataset_cols = ordered_dataset_columns(sample_split)
            fig_dir = os.path.join(out_dir, "figures")
            base = _figure_basename(mode, split, "heatmaps_metrics", splits)
            p1 = os.path.join(fig_dir, base)
            write_figure_heatmap_grid(
                out_path=p1,
                results=results,
                model_ids=model_ids,
                mode=mode,
                split=split,
                dataset_cols=dataset_cols,
            )
            n += 1
            base2 = _figure_basename(mode, split, "aggregate_bars", splits)
            p2 = os.path.join(fig_dir, base2)
            write_figure_aggregate_bars(
                out_path=p2,
                results=results,
                model_ids=model_ids,
                mode=mode,
                split=split,
            )
            n += 1

            for metric in METRICS_ORDER:
                hm = os.path.join(
                    fig_dir,
                    _figure_basename(mode, split, f"heatmap__{metric}", splits),
                )
                write_figure_single_metric_heatmap(
                    out_path=hm,
                    results=results,
                    model_ids=model_ids,
                    mode=mode,
                    split=split,
                    metric=metric,
                    dataset_cols=dataset_cols,
                )
                n += 1
                gb = os.path.join(
                    fig_dir,
                    _figure_basename(mode, split, f"grouped_bars__{metric}", splits),
                )
                write_figure_grouped_bars_by_dataset(
                    out_path=gb,
                    results=results,
                    model_ids=model_ids,
                    mode=mode,
                    split=split,
                    metric=metric,
                    dataset_cols=dataset_cols,
                )
                n += 1
    return n


def generate_figure_assets_safe(
    results: Mapping[str, Any],
    *,
    out_dir: str,
    splits: List[str],
    model_ids: Sequence[str],
) -> Tuple[int, Optional[str]]:
    """Try matplotlib figures; on failure return (0, error message)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError as e:
        return 0, f"matplotlib not installed: {e}"
    try:
        written = generate_figure_assets(
            results, out_dir=out_dir, splits=splits, model_ids=model_ids
        )
        return written, None
    except Exception as e:
        return 0, str(e)


# ─────────────────────────────────────────────
# SECTION 7: main()
# ─────────────────────────────────────────────


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.join(here, "baseline_evaluation.json")
    default_out = os.path.join(here, "reports")

    parser = argparse.ArgumentParser(description="Generate baseline reports from baseline_evaluation.json")
    parser.add_argument("--input", "-i", default=default_in, help="Path to baseline_evaluation.json")
    parser.add_argument("--out-dir", "-o", default=default_out, help="Output directory (CSVs + MD + figures/)")
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip matplotlib PNGs under <out-dir>/figures/",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        results: Dict[str, Any] = json.load(f)

    splits = discover_splits(results)
    if not splits:
        raise SystemExit("No splits found under with_context/without_context.")

    model_ids = iter_models_sorted(results)
    os.makedirs(args.out_dir, exist_ok=True)

    def csv_filename(mode: str, split: str, metric: str) -> str:
        # Legacy layout: no split in the name when only one split exists (matches reports/).
        if len(splits) == 1:
            return f"{mode}__{metric}.csv"
        return f"{mode}__{split}__{metric}.csv"

    csv_count = 0
    for mode in MODES:
        for split in splits:
            sample_split: Dict[str, Any] = {}
            for mid in model_ids:
                sp = (results.get(mid, {}).get(mode) or {}).get(split)
                if isinstance(sp, dict) and sp:
                    sample_split = sp
                    break
            if not sample_split:
                continue
            dataset_cols = ordered_dataset_columns(sample_split)
            for metric in METRICS_ORDER:
                fname = csv_filename(mode, split, metric)
                out_path = os.path.join(args.out_dir, fname)
                write_metric_csv(
                    path=out_path,
                    results=results,
                    mode=mode,
                    split=split,
                    metric=metric,
                    dataset_cols=dataset_cols,
                    model_ids=model_ids,
                )
                csv_count += 1

    md_path = os.path.join(args.out_dir, "model_dataset_metric_tables.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown(results, splits))

    print(f"Input:  {args.input}")
    print(f"Output: {args.out_dir}")
    print(f"  - {csv_count} CSV file(s)")
    print("  - model_dataset_metric_tables.md")

    if args.no_figures:
        print("  - figures/ (skipped: --no-figures)")
    else:
        n_fig, err = generate_figure_assets_safe(
            results,
            out_dir=args.out_dir,
            splits=splits,
            model_ids=model_ids,
        )
        if err:
            print(f"  - figures/: {err}")
        else:
            print(f"  - figures/: {n_fig} PNG file(s)")


if __name__ == "__main__":
    main()
