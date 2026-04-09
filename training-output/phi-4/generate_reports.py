# ------------------------------------------------------------
# MODULE MAP -- Section index
# ------------------------------------------------------------
#
# CONFIGURATION
# +-- 1. Imports
# +-- 2. Constants
#
# BUSINESS LOGIC
# +-- 3. Generic helpers
# +-- 4. Eval report builders
# +-- 5. Train report builders
# +-- 6. Figure writers
#
# ENTRY
# +-- 7. main()
#
# ------------------------------------------------------------
"""
generate_reports.py -- Build train/eval reports for LoRA fine-tuning artifacts.

Copy lives under ``training-output/<model-slug>/`` (e.g. ``qwen-3``, ``phi-4``,
``gemma-3``); defaults resolve paths relative to the script directory.

Output layout (same for every model folder):
    plots/eval   — per-metric CSVs, markdown tables, comparison figures
    plots/train  — training/eval history CSVs, markdown tables, curve figures

Eval inputs: ``evaluation_comparison.json``. Train inputs: ``training_stats.json``
or the latest ``checkpoint-*/trainer_state.json`` (``log_history``).

Usage:
    python training-output/<model-slug>/generate_reports.py
    python training-output/<model-slug>/generate_reports.py --no-figures
    python training-output/<model-slug>/generate_reports.py --plots-dir training-output/<model-slug>/plots
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# ------------------------------------------------------------
# SECTION 2: CONSTANTS
# ------------------------------------------------------------

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

SPLIT_PREFERRED_ORDER: Tuple[str, ...] = ("dev", "test")

VARIANTS_ORDER: Tuple[str, ...] = (
    "base",
    "adapted",
    "delta_pp",
    "delta_rel_pct",
)


# ------------------------------------------------------------
# SECTION 3: GENERIC HELPERS
# ------------------------------------------------------------


def _safe_float(value: Any) -> Optional[float]:
    """Convert value to float; return None if conversion is not possible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_number(value: Any) -> str:
    """Format numeric cell as two decimals; empty string for missing values."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return ""
        return f"{v:.2f}"
    return str(value)


def _read_json(path: str) -> Dict[str, Any]:
    """Load one JSON file as dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return obj


def _ordered_splits(per_split: Mapping[str, Any]) -> List[str]:
    """Keep dev/test first, then any additional split in alphabetical order."""
    keys = set(per_split.keys())
    ordered = [s for s in SPLIT_PREFERRED_ORDER if s in keys]
    for split in sorted(keys):
        if split not in ordered:
            ordered.append(split)
    return ordered


def _ordered_datasets(split_block: Mapping[str, Any]) -> List[str]:
    """Return dataset columns in baseline order + append unknown datasets sorted."""
    keys = set(split_block.keys())
    known_non_agg = [k for k in DATASET_ORDER if k != "aggregate" and k in keys]
    known_set = set(DATASET_ORDER)
    extras = sorted(k for k in keys if k not in known_set)
    ordered = known_non_agg + extras
    ordered.append("aggregate")
    return ordered


def _ensure_dir(path: str) -> None:
    """Create destination directory if missing."""
    os.makedirs(path, exist_ok=True)


def _find_latest_checkpoint_trainer_state(model_dir: str) -> Optional[str]:
    """Find latest checkpoint-XXXX/trainer_state.json by numeric step."""
    best_step = -1
    best_path: Optional[str] = None
    for name in os.listdir(model_dir):
        full = os.path.join(model_dir, name)
        if not os.path.isdir(full):
            continue
        if not name.startswith("checkpoint-"):
            continue
        suffix = name.split("checkpoint-", 1)[-1]
        if not suffix.isdigit():
            continue
        step = int(suffix)
        state_path = os.path.join(full, "trainer_state.json")
        if step > best_step and os.path.isfile(state_path):
            best_step = step
            best_path = state_path
    return best_path


def _weighted_average(values: Sequence[Tuple[Optional[float], int]]) -> Optional[float]:
    """Compute weighted average ignoring missing values and non-positive weights."""
    acc = 0.0
    total_w = 0
    for v, w in values:
        if v is None:
            continue
        if w <= 0:
            continue
        acc += v * w
        total_w += w
    if total_w <= 0:
        return None
    return acc / total_w


# ------------------------------------------------------------
# SECTION 4: EVAL REPORT BUILDERS
# ------------------------------------------------------------


def _dataset_metric(
    split_block: Mapping[str, Any],
    dataset_name: str,
    variant: str,
    metric: str,
) -> Optional[float]:
    """Get one dataset metric for base/adapted/delta variants."""
    ds = split_block.get(dataset_name)
    if not isinstance(ds, dict):
        return None

    if variant == "base":
        return _safe_float((ds.get("base") or {}).get(metric))
    if variant == "adapted":
        return _safe_float((ds.get("adapted") or {}).get(metric))
    if variant == "delta_pp":
        return _safe_float((ds.get("deltas") or {}).get(metric))
    if variant == "delta_rel_pct":
        return _safe_float((ds.get("deltas_rel_pct") or {}).get(metric))
    return None


def _aggregate_metric(
    split_block: Mapping[str, Any],
    metric: str,
    variant: str,
    dataset_names: Sequence[str],
) -> Optional[float]:
    """Compute weighted aggregate per variant for one split and metric."""
    if variant in ("base", "adapted"):
        weighted: List[Tuple[Optional[float], int]] = []
        for ds_name in dataset_names:
            ds = split_block.get(ds_name)
            if not isinstance(ds, dict):
                continue
            n_samples = int((ds.get("base") or {}).get("n_samples") or 0)
            value = _dataset_metric(split_block, ds_name, variant, metric)
            weighted.append((value, n_samples))
        return _weighted_average(weighted)

    if variant == "delta_pp":
        base_v = _aggregate_metric(split_block, metric, "base", dataset_names)
        adapted_v = _aggregate_metric(split_block, metric, "adapted", dataset_names)
        if base_v is None or adapted_v is None:
            return None
        return adapted_v - base_v

    if variant == "delta_rel_pct":
        base_v = _aggregate_metric(split_block, metric, "base", dataset_names)
        adapted_v = _aggregate_metric(split_block, metric, "adapted", dataset_names)
        if base_v is None or adapted_v is None or base_v == 0:
            return None
        return ((adapted_v - base_v) / base_v) * 100.0

    return None


def write_eval_metric_csv(
    *,
    out_path: str,
    split_block: Mapping[str, Any],
    dataset_cols: Sequence[str],
    metric: str,
) -> None:
    """Write one CSV file per split/metric with base/adapted/deltas rows."""
    ds_no_agg = [d for d in dataset_cols if d != "aggregate"]
    _ensure_dir(os.path.dirname(out_path) or ".")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Variant"] + list(dataset_cols))
        for variant in VARIANTS_ORDER:
            row = [variant]
            for ds_name in dataset_cols:
                if ds_name == "aggregate":
                    value = _aggregate_metric(split_block, metric, variant, ds_no_agg)
                else:
                    value = _dataset_metric(split_block, ds_name, variant, metric)
                row.append(_fmt_number(value))
            w.writerow(row)


def build_eval_markdown(
    *,
    per_split: Mapping[str, Any],
    ordered_splits: Sequence[str],
) -> str:
    """Build markdown tables for all eval metrics and splits."""
    lines: List[str] = [
        "# Phi-4 eval report tables",
        "",
        "Rows represent base/adapted system and delta rows.",
        "",
    ]

    for split in ordered_splits:
        split_block = per_split.get(split)
        if not isinstance(split_block, dict) or not split_block:
            continue

        dataset_cols = _ordered_datasets(split_block)
        ds_no_agg = [d for d in dataset_cols if d != "aggregate"]
        lines.append(f"## Split: `{split}`")
        lines.append("")

        for metric in METRICS_ORDER:
            lines.append(f"### Metric: `{metric}`")
            lines.append("")
            header = "| Variant | " + " | ".join(dataset_cols) + " |"
            sep = "| --- | " + " | ".join(["---"] * len(dataset_cols)) + " |"
            lines.append(header)
            lines.append(sep)
            for variant in VARIANTS_ORDER:
                cells: List[str] = []
                for ds_name in dataset_cols:
                    if ds_name == "aggregate":
                        value = _aggregate_metric(split_block, metric, variant, ds_no_agg)
                    else:
                        value = _dataset_metric(split_block, ds_name, variant, metric)
                    cells.append(_fmt_number(value))
                lines.append("| " + variant + " | " + " | ".join(cells) + " |")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ------------------------------------------------------------
# SECTION 5: TRAIN REPORT BUILDERS
# ------------------------------------------------------------


def _extract_train_eval_rows(log_history: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split trainer log_history into train rows and eval rows."""
    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []

    for entry in log_history:
        if not isinstance(entry, dict):
            continue
        if "loss" in entry:
            train_rows.append(
                {
                    "step": entry.get("step"),
                    "epoch": entry.get("epoch"),
                    "loss": entry.get("loss"),
                    "grad_norm": entry.get("grad_norm"),
                    "learning_rate": entry.get("learning_rate"),
                }
            )
        if "eval_loss" in entry:
            eval_rows.append(
                {
                    "step": entry.get("step"),
                    "epoch": entry.get("epoch"),
                    "eval_loss": entry.get("eval_loss"),
                    "eval_runtime": entry.get("eval_runtime"),
                    "eval_samples_per_second": entry.get("eval_samples_per_second"),
                    "eval_steps_per_second": entry.get("eval_steps_per_second"),
                }
            )

    train_rows.sort(key=lambda r: (_safe_float(r.get("step")) or 0.0))
    eval_rows.sort(key=lambda r: (_safe_float(r.get("step")) or 0.0))
    return train_rows, eval_rows


def _write_rows_csv(path: str, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Write generic row dictionaries into CSV with selected columns."""
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(columns))
        for row in rows:
            w.writerow([_fmt_number(row.get(c)) for c in columns])


def _best_eval_row(eval_rows: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    """Return eval row with minimum eval_loss."""
    best: Optional[Mapping[str, Any]] = None
    best_loss = float("inf")
    for row in eval_rows:
        loss = _safe_float(row.get("eval_loss"))
        if loss is None:
            continue
        if loss < best_loss:
            best_loss = loss
            best = row
    return best


def build_train_markdown(
    *,
    train_source: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Create markdown summary for train dynamics and convergence."""
    min_train = min((_safe_float(r.get("loss")) for r in train_rows if _safe_float(r.get("loss")) is not None), default=None)
    max_train = max((_safe_float(r.get("loss")) for r in train_rows if _safe_float(r.get("loss")) is not None), default=None)
    last_train = _safe_float(train_rows[-1].get("loss")) if train_rows else None
    last_eval = _safe_float(eval_rows[-1].get("eval_loss")) if eval_rows else None
    best_eval = _best_eval_row(eval_rows)

    lines = [
        "# Phi-4 train report",
        "",
        "## Run summary",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| model_name | {_fmt_number(train_source.get('model_name')) or ''} |",
        f"| version | {_fmt_number(train_source.get('version')) or ''} |",
        f"| total_steps | {_fmt_number(train_source.get('total_steps')) or ''} |",
        f"| dataset_size | {_fmt_number(train_source.get('dataset_size')) or ''} |",
        f"| effective_batch | {_fmt_number(train_source.get('effective_batch')) or ''} |",
        f"| final_eval_loss | {_fmt_number(train_source.get('eval_loss'))} |",
        f"| perplexity | {_fmt_number(train_source.get('perplexity'))} |",
        f"| train_points | {_fmt_number(len(train_rows))} |",
        f"| eval_points | {_fmt_number(len(eval_rows))} |",
        f"| min_train_loss | {_fmt_number(min_train)} |",
        f"| max_train_loss | {_fmt_number(max_train)} |",
        f"| last_train_loss | {_fmt_number(last_train)} |",
        f"| last_eval_loss | {_fmt_number(last_eval)} |",
    ]

    if best_eval is not None:
        lines.extend(
            [
                f"| best_eval_step | {_fmt_number(best_eval.get('step'))} |",
                f"| best_eval_loss | {_fmt_number(best_eval.get('eval_loss'))} |",
            ]
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- train_history.csv contains optimization points logged during training steps.",
            "- eval_history.csv contains periodic evaluation losses and throughput stats.",
            "",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


# ------------------------------------------------------------
# SECTION 6: FIGURE WRITERS
# ------------------------------------------------------------


def _import_matplotlib() -> Tuple[Any, Any]:
    """Import matplotlib in headless mode."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, plt


def write_eval_figures(
    *,
    out_dir: str,
    per_split: Mapping[str, Any],
    ordered_splits: Sequence[str],
) -> int:
    """Write eval heatmaps + grouped bars + aggregate chart."""
    _, plt = _import_matplotlib()
    figure_dir = os.path.join(out_dir, "figures")
    _ensure_dir(figure_dir)

    count = 0
    for split in ordered_splits:
        split_block = per_split.get(split)
        if not isinstance(split_block, dict) or not split_block:
            continue

        dataset_cols = _ordered_datasets(split_block)
        ds_no_agg = [d for d in dataset_cols if d != "aggregate"]

        for metric in METRICS_ORDER:
            # Heatmap rows: base, adapted, delta_pp
            heat_rows = [
                [
                    _aggregate_metric(split_block, metric, "base", ds_no_agg)
                    if ds_name == "aggregate"
                    else _dataset_metric(split_block, ds_name, "base", metric)
                    for ds_name in dataset_cols
                ],
                [
                    _aggregate_metric(split_block, metric, "adapted", ds_no_agg)
                    if ds_name == "aggregate"
                    else _dataset_metric(split_block, ds_name, "adapted", metric)
                    for ds_name in dataset_cols
                ],
                [
                    _aggregate_metric(split_block, metric, "delta_pp", ds_no_agg)
                    if ds_name == "aggregate"
                    else _dataset_metric(split_block, ds_name, "delta_pp", metric)
                    for ds_name in dataset_cols
                ],
            ]

            heat_path = os.path.join(figure_dir, f"heatmap__{metric}__{split}.png")
            fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(dataset_cols)), 4.0))
            im = ax.imshow(heat_rows, aspect="auto", cmap="viridis", interpolation="nearest")
            ax.set_title(f"{metric} -- {split}", fontsize=11)
            ax.set_xticks(range(len(dataset_cols)))
            ax.set_xticklabels(dataset_cols, rotation=30, ha="right", fontsize=8)
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(["base", "adapted", "delta_pp"], fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            fig.tight_layout()
            fig.savefig(heat_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            count += 1

            # Grouped bars by dataset (without aggregate): base vs adapted
            bar_path = os.path.join(figure_dir, f"grouped_bars__{metric}__{split}.png")
            fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(ds_no_agg)), 5.0))
            x = list(range(len(ds_no_agg)))
            width = 0.38
            base_vals = [_dataset_metric(split_block, ds_name, "base", metric) or 0.0 for ds_name in ds_no_agg]
            adapted_vals = [_dataset_metric(split_block, ds_name, "adapted", metric) or 0.0 for ds_name in ds_no_agg]
            x_base = [xi - width / 2.0 for xi in x]
            x_adapted = [xi + width / 2.0 for xi in x]
            ax.bar(x_base, base_vals, width=width, label="base")
            ax.bar(x_adapted, adapted_vals, width=width, label="adapted")
            ax.set_xticks(x)
            ax.set_xticklabels(ds_no_agg, rotation=25, ha="right", fontsize=8)
            ax.set_ylabel("value", fontsize=9)
            ax.set_title(f"{metric} -- by dataset -- {split}", fontsize=10)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(bar_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            count += 1

        # Aggregate by metric chart for this split
        agg_path = os.path.join(figure_dir, f"aggregate_metrics__{split}.png")
        fig, ax = plt.subplots(figsize=(10, 5.2))
        x = list(range(len(METRICS_ORDER)))
        width = 0.38
        agg_base = [_aggregate_metric(split_block, m, "base", ds_no_agg) or 0.0 for m in METRICS_ORDER]
        agg_adapted = [_aggregate_metric(split_block, m, "adapted", ds_no_agg) or 0.0 for m in METRICS_ORDER]
        x_base = [xi - width / 2.0 for xi in x]
        x_adapted = [xi + width / 2.0 for xi in x]
        ax.bar(x_base, agg_base, width=width, label="base")
        ax.bar(x_adapted, agg_adapted, width=width, label="adapted")
        ax.set_xticks(x)
        ax.set_xticklabels(METRICS_ORDER, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("value", fontsize=9)
        ax.set_title(f"Aggregate metrics by split -- {split}", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(agg_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        count += 1

    return count


def _moving_average(values: Sequence[float], window: int) -> List[float]:
    """Simple moving average for visual smoothing."""
    if window <= 1 or len(values) <= 1:
        return list(values)
    out: List[float] = []
    running = 0.0
    q: List[float] = []
    for v in values:
        q.append(v)
        running += v
        if len(q) > window:
            running -= q.pop(0)
        out.append(running / float(len(q)))
    return out


def write_train_figures(
    *,
    out_dir: str,
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
) -> int:
    """Write train dynamics figures from train/eval histories."""
    _, plt = _import_matplotlib()
    figure_dir = os.path.join(out_dir, "figures")
    _ensure_dir(figure_dir)

    count = 0
    if train_rows:
        train_steps = [(_safe_float(r.get("step")) or 0.0) for r in train_rows]
        train_loss = [(_safe_float(r.get("loss")) or 0.0) for r in train_rows]
        train_loss_smooth = _moving_average(train_loss, window=7)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_steps, train_loss, label="train_loss", alpha=0.35)
        ax.plot(train_steps, train_loss_smooth, label="train_loss_ma7", linewidth=2.0)
        ax.set_title("Train loss over steps", fontsize=11)
        ax.set_xlabel("step", fontsize=9)
        ax.set_ylabel("loss", fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(figure_dir, "train_loss_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        count += 1

        lr_vals = [(_safe_float(r.get("learning_rate")) or 0.0) for r in train_rows]
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.plot(train_steps, lr_vals, color="#ff7f0e")
        ax.set_title("Learning rate schedule", fontsize=11)
        ax.set_xlabel("step", fontsize=9)
        ax.set_ylabel("learning_rate", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(figure_dir, "learning_rate_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        count += 1

        grad_vals = [(_safe_float(r.get("grad_norm")) or 0.0) for r in train_rows]
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.plot(train_steps, grad_vals, color="#2ca02c")
        ax.set_title("Gradient norm over steps", fontsize=11)
        ax.set_xlabel("step", fontsize=9)
        ax.set_ylabel("grad_norm", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(figure_dir, "grad_norm_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        count += 1

    if train_rows and eval_rows:
        train_steps = [(_safe_float(r.get("step")) or 0.0) for r in train_rows]
        train_loss = [(_safe_float(r.get("loss")) or 0.0) for r in train_rows]
        train_loss_smooth = _moving_average(train_loss, window=7)
        eval_steps = [(_safe_float(r.get("step")) or 0.0) for r in eval_rows]
        eval_loss = [(_safe_float(r.get("eval_loss")) or 0.0) for r in eval_rows]

        fig, ax = plt.subplots(figsize=(10, 5.2))
        ax.plot(train_steps, train_loss_smooth, label="train_loss_ma7", linewidth=2.0)
        ax.plot(eval_steps, eval_loss, marker="o", linestyle="-", label="eval_loss")
        ax.set_title("Train vs eval loss", fontsize=11)
        ax.set_xlabel("step", fontsize=9)
        ax.set_ylabel("loss", fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(figure_dir, "train_vs_eval_loss.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        count += 1

    return count


# ------------------------------------------------------------
# SECTION 7: MAIN
# ------------------------------------------------------------


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Generate train/eval report artifacts for a LoRA model output folder"
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory with training_stats.json / evaluation_comparison.json (default: script directory)",
    )
    parser.add_argument(
        "--phi-dir",
        default=None,
        help=argparse.SUPPRESS,  # legacy alias; prefer --model-dir
    )
    parser.add_argument(
        "--eval-input",
        default=None,
        help="Path to evaluation_comparison.json (default: <model-dir>/evaluation_comparison.json)",
    )
    parser.add_argument(
        "--train-input",
        default=None,
        help="Path to training_stats.json (default: <model-dir>/training_stats.json)",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Output plots root directory (default: <model-dir>/plots)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip matplotlib figures",
    )
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir or args.phi_dir or here)
    eval_input = os.path.abspath(args.eval_input or os.path.join(model_dir, "evaluation_comparison.json"))
    train_input = os.path.abspath(args.train_input or os.path.join(model_dir, "training_stats.json"))
    plots_dir = os.path.abspath(args.plots_dir or os.path.join(model_dir, "plots"))

    eval_obj = _read_json(eval_input)
    per_split = eval_obj.get("per_split")
    if not isinstance(per_split, dict) or not per_split:
        raise SystemExit("evaluation_comparison.json has no 'per_split' content")

    # Load train source with fallback to latest checkpoint trainer_state.
    train_obj: Dict[str, Any]
    if os.path.isfile(train_input):
        train_obj = _read_json(train_input)
    else:
        fallback = _find_latest_checkpoint_trainer_state(model_dir)
        if fallback is None:
            raise SystemExit(
                "Could not find training_stats.json or checkpoint-*/trainer_state.json"
            )
        train_obj = _read_json(fallback)
        print(f"Warning: training input not found, using checkpoint state: {fallback}")

    log_history = train_obj.get("log_history")
    if not isinstance(log_history, list):
        raise SystemExit("train source has no 'log_history' list")

    eval_dir = os.path.join(plots_dir, "eval")
    train_dir = os.path.join(plots_dir, "train")
    _ensure_dir(eval_dir)
    _ensure_dir(train_dir)

    splits = _ordered_splits(per_split)

    # --- Eval CSV and markdown ---
    eval_csv_count = 0
    for split in splits:
        split_block = per_split.get(split)
        if not isinstance(split_block, dict) or not split_block:
            continue
        dataset_cols = _ordered_datasets(split_block)
        for metric in METRICS_ORDER:
            csv_path = os.path.join(eval_dir, f"{split}__{metric}.csv")
            write_eval_metric_csv(
                out_path=csv_path,
                split_block=split_block,
                dataset_cols=dataset_cols,
                metric=metric,
            )
            eval_csv_count += 1

    eval_md = build_eval_markdown(per_split=per_split, ordered_splits=splits)
    eval_md_path = os.path.join(eval_dir, "model_dataset_metric_tables.md")
    with open(eval_md_path, "w", encoding="utf-8") as f:
        f.write(eval_md)

    # --- Train CSV and markdown ---
    train_rows, eval_rows = _extract_train_eval_rows(log_history)
    train_csv_path = os.path.join(train_dir, "train_history.csv")
    eval_csv_path = os.path.join(train_dir, "eval_history.csv")

    _write_rows_csv(
        train_csv_path,
        ["step", "epoch", "loss", "grad_norm", "learning_rate"],
        train_rows,
    )
    _write_rows_csv(
        eval_csv_path,
        ["step", "epoch", "eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"],
        eval_rows,
    )

    train_md = build_train_markdown(
        train_source=train_obj,
        train_rows=train_rows,
        eval_rows=eval_rows,
    )
    train_md_path = os.path.join(train_dir, "training_tables.md")
    with open(train_md_path, "w", encoding="utf-8") as f:
        f.write(train_md)

    # --- Optional figures ---
    eval_fig_count = 0
    train_fig_count = 0
    if not args.no_figures:
        try:
            eval_fig_count = write_eval_figures(
                out_dir=eval_dir,
                per_split=per_split,
                ordered_splits=splits,
            )
            train_fig_count = write_train_figures(
                out_dir=train_dir,
                train_rows=train_rows,
                eval_rows=eval_rows,
            )
        except Exception as exc:
            print(f"Warning: figure generation failed: {exc}")

    print(f"Eval input:   {eval_input}")
    print(f"Train input:  {train_input if os.path.isfile(train_input) else 'fallback checkpoint'}")
    print(f"Output root:  {plots_dir}")
    print(f"  - eval CSV:      {eval_csv_count}")
    print("  - eval markdown: model_dataset_metric_tables.md")
    print("  - train CSV:     2")
    print("  - train markdown: training_tables.md")
    if args.no_figures:
        print("  - figures:       skipped (--no-figures)")
    else:
        print(f"  - eval figures:  {eval_fig_count}")
        print(f"  - train figures: {train_fig_count}")


if __name__ == "__main__":
    main()
