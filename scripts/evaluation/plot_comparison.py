"""
Base-vs-fine-tuned model comparison visualization.

Generates evaluation charts from ``evaluation_comparison.json`` and
``training_stats.json`` located in ``training-output/<model>/``.  Three
figures are produced and saved to ``training-output/<model>/plots/eval/``:

  - ``eval_metrics_by_dataset.png``  -- 4 metrics x 3 datasets (grouped bars)
  - ``eval_aggregate.png``           -- aggregate summary with annotated deltas
  - ``eval_sample_scatter.png``      -- scatter F1 base vs. adapted per dataset

Usage:
    python scripts/evaluation/plot_comparison.py [--model qwen-3|llama-3|gemma-3]

Dependencies:
    - matplotlib
    - numpy
"""

import argparse
import json
import os
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
except ImportError:
    print("Install dependencies: pip install matplotlib numpy")
    sys.exit(1)


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
VALID_MODELS  = ["qwen-3", "llama-3", "gemma-3"]
MODEL_DISPLAY = {"qwen-3": "Qwen3-FineTuned", "llama-3": "Llama3-FineTuned", "gemma-3": "Gemma3-FineTuned"}

# These globals are set by main() after resolving --model
DATA_DIR        = os.path.join(PROJECT_ROOT, "training-output", "qwen-3")
IMAGES_DIR      = os.path.join(DATA_DIR, "plots", "eval")
COMPARISON_FILE = os.path.join(DATA_DIR, "evaluation_comparison.json")
STATS_FILE      = os.path.join(DATA_DIR, "training_stats.json")
ADAPTED_LABEL   = MODEL_DISPLAY["qwen-3"]


# ─────────────────────────────────────────────
# PALETTE AND STYLES
# ─────────────────────────────────────────────

BASE_COLOR    = "#64748b"   # slate-500 -- base model
ADAPT_COLOR   = "#2563eb"   # blue-600  -- fine-tuned model
DELTA_COLOR   = "#16a34a"   # green-600 -- positive improvement
NEG_COLOR     = "#dc2626"   # red-600   -- regression
BG_COLOR      = "#f8fafc"

DATASET_LABELS = {
    "Neural-Bridge RAG": "Neural-Bridge",
    "Dolly QA":          "Dolly QA",
    "Aina RAG":          "Aina RAG",
}

METRIC_LABELS = {
    "Token_F1":                  "Token F1 (%)",
    "Context_Faithfulness_Pct":  "Faithfulness (%)",
    "Avg_Response_Length_Words": "Mean length (words)",
    "Sentence_Completeness_Pct": "Sentence completeness (%)",
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_json(path: str) -> dict:
    """Load a JSON file and return its contents as a dictionary.

    Args:
        path: Absolute or relative path to the JSON file.

    Returns:
        Parsed dictionary from the JSON file.
    """
    if not os.path.exists(path):
        print(f"Error: file not found {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def annotate_bar(ax, bar, value: float, fmt: str = "{:.1f}", color: str = "black",
                 fontsize: int = 8, offset: float = 0.5) -> None:
    """Draw a formatted value label above a single bar.

    Args:
        ax: Matplotlib axes containing the bar.
        bar: A single bar rectangle from ``ax.bar()``.
        value: Numeric value to display.
        fmt: Format string for the label.
        color: Text color.
        fontsize: Font size for the label.
        offset: Vertical offset above the bar top.
    """
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + offset,
        fmt.format(value),
        ha="center", va="bottom",
        fontsize=fontsize, color=color, fontweight="bold",
    )


def set_common_style(ax, title: str, ylabel: str, ylim: tuple | None = None) -> None:
    """Apply a consistent visual style to an axes.

    Args:
        ax: Matplotlib axes to style.
        title: Axes title text.
        ylabel: Y-axis label text.
        ylim: Optional ``(ymin, ymax)`` tuple to set the y-axis range.
    """
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylim:
        ax.set_ylim(*ylim)


# ─────────────────────────────────────────────
# FIGURE 1 -- Metrics by dataset
# ─────────────────────────────────────────────

def plot_metrics_by_dataset(data: dict) -> None:
    """Generate a 2x2 grid of grouped bar charts (base vs. fine-tuned) for
    each metric, broken down by dataset plus an aggregate bar.

    Args:
        data: Parsed contents of ``evaluation_comparison.json``.
    """
    per_ds   = data["per_dataset"]
    agg      = data["aggregate"]
    datasets = list(DATASET_LABELS.values())

    metrics = [
        ("Token_F1",                  "Token F1 (%)",             (0, 100)),
        ("Context_Faithfulness_Pct",  "Context Faithfulness (%)", (0, 105)),
        ("Avg_Response_Length_Words", "Mean response length (words)", None),
        ("Sentence_Completeness_Pct", "Sentence completeness (%)",   (0, 105)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Comparativa base vs. {ADAPTED_LABEL} por dataset",
                 fontsize=14, fontweight="bold", y=1.01)
    axes_flat = axes.flatten()

    agg_map = {
        "Token_F1":                  ("Base_Token_F1",       "Adapted_Token_F1"),
        "Context_Faithfulness_Pct":  ("Base_Faithfulness",   "Adapted_Faithfulness"),
        "Avg_Response_Length_Words": (None, None),
        "Sentence_Completeness_Pct": (None, None),
    }

    for idx, (metric, ylabel, ylim) in enumerate(metrics):
        ax = axes_flat[idx]

        ds_names = datasets + ["Agregado"]
        x        = np.arange(len(ds_names))
        width    = 0.35

        base_vals  = []
        adapt_vals = []
        for ds_key, ds_label in DATASET_LABELS.items():
            ds_data = per_ds[ds_key]
            base_vals.append(ds_data["base"].get(metric, 0))
            adapt_vals.append(ds_data["adapted"].get(metric, 0))

        # Aggregate value from precomputed keys or fallback to mean
        b_key, a_key = agg_map[metric]
        if b_key and b_key in agg:
            base_vals.append(agg[b_key])
            adapt_vals.append(agg[a_key])
        else:
            base_vals.append(np.mean(base_vals))
            adapt_vals.append(np.mean(adapt_vals))

        bars_b = ax.bar(x - width / 2, base_vals,  width, label="Base",
                        color=BASE_COLOR, alpha=0.85, zorder=3)
        bars_a = ax.bar(x + width / 2, adapt_vals, width, label="Fine-tuned",
                        color=ADAPT_COLOR, alpha=0.90, zorder=3)

        offset = max(max(base_vals), max(adapt_vals)) * 0.015
        for bar, val in zip(bars_b, base_vals):
            annotate_bar(ax, bar, val, color=BASE_COLOR, offset=offset)
        for bar, val in zip(bars_a, adapt_vals):
            annotate_bar(ax, bar, val, color=ADAPT_COLOR, offset=offset)

        ax.set_xticks(x)
        ax.set_xticklabels(ds_names, fontsize=9)
        ax.legend(fontsize=8)
        set_common_style(ax, ylabel, ylabel, ylim)

    plt.tight_layout()
    out_path = os.path.join(IMAGES_DIR, "eval_metrics_by_dataset.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"eval_metrics_by_dataset.png saved to: {out_path}")


# ─────────────────────────────────────────────
# FIGURE 2 -- Aggregate summary with deltas
# ─────────────────────────────────────────────

def plot_aggregate(data: dict, stats: dict) -> None:
    """Generate a summary panel with aggregate metric bars, a per-dataset
    delta table, and a training statistics info box.

    Args:
        data: Parsed contents of ``evaluation_comparison.json``.
        stats: Parsed contents of ``training_stats.json``.
    """
    agg = data["aggregate"]

    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(f"Resumen agregado — Modelo base vs. {ADAPTED_LABEL} (600 muestras)",
                 fontsize=13, fontweight="bold")

    # --- Left subplot: Token F1 and Faithfulness bars ---
    ax_bars = fig.add_subplot(1, 3, (1, 2))

    metrics_agg = [
        ("Token F1 (%)",        agg["Base_Token_F1"],    agg["Adapted_Token_F1"],    agg["Delta_Token_F1"]),
        ("Faithfulness (%)",    agg["Base_Faithfulness"], agg["Adapted_Faithfulness"], agg["Delta_Faithfulness"]),
    ]

    x      = np.arange(len(metrics_agg))
    width  = 0.30
    labels = [m[0] for m in metrics_agg]
    base_v = [m[1] for m in metrics_agg]
    adap_v = [m[2] for m in metrics_agg]
    delta_v= [m[3] for m in metrics_agg]

    bars_b = ax_bars.bar(x - width / 2, base_v,  width, label="Base",
                         color=BASE_COLOR, alpha=0.85, zorder=3)
    bars_a = ax_bars.bar(x + width / 2, adap_v,  width, label="Fine-tuned",
                         color=ADAPT_COLOR, alpha=0.90, zorder=3)

    for bar, val in zip(bars_b, base_v):
        annotate_bar(ax_bars, bar, val, offset=0.3, fontsize=9)
    for bar, val in zip(bars_a, adap_v):
        annotate_bar(ax_bars, bar, val, color=ADAPT_COLOR, offset=0.3, fontsize=9)

    # Delta annotations between bar pairs
    for xi, (bar_b, bar_a, delta) in enumerate(zip(bars_b, bars_a, delta_v)):
        mid_x   = (bar_b.get_x() + bar_b.get_width() / 2 +
                   bar_a.get_x() + bar_a.get_width() / 2) / 2
        top_y   = max(bar_b.get_height(), bar_a.get_height()) + 2.5
        color   = DELTA_COLOR if delta >= 0 else NEG_COLOR
        sign    = "+" if delta >= 0 else ""
        ax_bars.annotate(f"Δ {sign}{delta:.2f}",
                         xy=(mid_x, top_y),
                         ha="center", va="bottom",
                         fontsize=10, color=color, fontweight="bold")

    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(labels, fontsize=11)
    ax_bars.set_ylim(0, 105)
    ax_bars.legend(fontsize=9)
    set_common_style(ax_bars, "Métricas agregadas (n=600)", "Valor (%)")

    # --- Right subplot: training info table ---
    ax_info = fig.add_subplot(1, 3, 3)
    ax_info.axis("off")

    per_ds = data["per_dataset"]
    rows   = []
    for ds_key, ds_label in DATASET_LABELS.items():
        d = per_ds[ds_key]["deltas"]
        rows.append([
            ds_label,
            f"+{d['Token_F1']:.1f}" if d["Token_F1"] >= 0 else f"{d['Token_F1']:.1f}",
            f"+{d['Context_Faithfulness_Pct']:.1f}" if d["Context_Faithfulness_Pct"] >= 0
            else f"{d['Context_Faithfulness_Pct']:.1f}",
        ])
    rows.append([
        "Agregado",
        f"+{agg['Delta_Token_F1']:.2f}",
        f"+{agg['Delta_Faithfulness']:.2f}",
    ])

    col_labels = ["Dataset", "ΔToken F1", "ΔFaithfulness"]
    table = ax_info.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.6)

    # Color delta cells green for improvement, red for regression
    for row_idx in range(1, len(rows) + 1):
        for col_idx in [1, 2]:
            cell  = table[row_idx, col_idx]
            txt   = rows[row_idx - 1][col_idx]
            color = "#dcfce7" if not txt.startswith("-") else "#fee2e2"
            cell.set_facecolor(color)

    ax_info.set_title("Δ por dataset", fontsize=10, fontweight="bold", pad=12)

    # Training statistics info box
    ev_loss   = stats.get("eval_loss", "N/A")
    ppl       = stats.get("perplexity", "N/A")
    n_steps   = stats.get("total_steps", "N/A")
    ds_size   = stats.get("dataset_size", "N/A")
    info_text = (
        f"Training\n"
        f"  Steps:      {n_steps}\n"
        f"  Dataset:    {ds_size} samples\n"
        f"  Eval loss:  {ev_loss:.4f}\n"
        f"  Perplexity: {ppl:.4f}"
    )
    ax_info.text(0.5, 0.08, info_text, transform=ax_info.transAxes,
                 fontsize=8.5, verticalalignment="bottom", horizontalalignment="center",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#eff6ff", alpha=0.8),
                 family="monospace")

    plt.tight_layout()
    out_path = os.path.join(IMAGES_DIR, "eval_aggregate.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"eval_aggregate.png saved to: {out_path}")


# ─────────────────────────────────────────────
# FIGURE 3 -- Scatter F1 base vs. adapted per dataset
# ─────────────────────────────────────────────

def plot_sample_scatter(data: dict) -> None:
    """Generate per-dataset scatter plots where X = base F1 and Y = fine-tuned
    F1.  Points above the diagonal indicate improvement; points below indicate
    regression.

    Args:
        data: Parsed contents of ``evaluation_comparison.json``.
    """
    per_ds   = data["per_dataset"]
    datasets = list(DATASET_LABELS.items())

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Distribución F1: base vs. fine-tuned (muestras individuales)",
                 fontsize=13, fontweight="bold")

    ds_colors = ["#2563eb", "#16a34a", "#9333ea"]

    for ax, (ds_key, ds_label), color in zip(axes, datasets, ds_colors):
        pairs     = per_ds[ds_key].get("sample_pairs", [])
        if not pairs:
            ax.text(0.5, 0.5, "No samples", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            continue

        base_f1  = [p.get("base_f1",    p.get("base_faithfulness",    0)) for p in pairs]
        adap_f1  = [p.get("adapted_f1", p.get("adapted_faithfulness", 0)) for p in pairs]

        improved = [(b, a) for b, a in zip(base_f1, adap_f1) if a >= b]
        regress  = [(b, a) for b, a in zip(base_f1, adap_f1) if a  < b]

        if improved:
            ax.scatter([p[0] for p in improved], [p[1] for p in improved],
                       color=color, alpha=0.65, s=50, label="Improved", zorder=3)
        if regress:
            ax.scatter([p[0] for p in regress],  [p[1] for p in regress],
                       color=NEG_COLOR, alpha=0.65, s=50, marker="x", label="Regression", zorder=3)

        # Reference diagonal (y = x)
        lim = max(max(base_f1 + adap_f1), 0.01)
        ax.plot([0, lim], [0, lim], "--", color="#94a3b8", linewidth=1, zorder=1)
        ax.fill_between([0, lim], [0, lim], [lim, lim],
                        alpha=0.04, color=color)

        ax.set_xlabel("F1 base", fontsize=9)
        ax.set_ylabel("F1 fine-tuned", fontsize=9)
        ax.legend(fontsize=8)
        set_common_style(ax, ds_label, "F1 fine-tuned")
        ax.set_xlim(0, lim + 0.05)
        ax.set_ylim(0, lim + 0.05)
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    out_path = os.path.join(IMAGES_DIR, "eval_sample_scatter.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"eval_sample_scatter.png saved to: {out_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments, load data files, and generate all comparison
    plots."""
    global DATA_DIR, IMAGES_DIR, COMPARISON_FILE, STATS_FILE, ADAPTED_LABEL

    parser = argparse.ArgumentParser(description="Base vs. fine-tuned model comparison plots.")
    parser.add_argument(
        "--model", choices=VALID_MODELS, default="qwen-3",
        help="Model to visualize (default: qwen-3).",
    )
    args = parser.parse_args()

    DATA_DIR        = os.path.join(PROJECT_ROOT, "training-output", args.model)
    IMAGES_DIR      = os.path.join(DATA_DIR, "plots", "eval")
    COMPARISON_FILE = os.path.join(DATA_DIR, "evaluation_comparison.json")
    STATS_FILE      = os.path.join(DATA_DIR, "training_stats.json")
    ADAPTED_LABEL   = MODEL_DISPLAY[args.model]

    os.makedirs(IMAGES_DIR, exist_ok=True)

    comparison = load_json(COMPARISON_FILE)
    stats      = load_json(STATS_FILE)

    print("Generating eval_metrics_by_dataset.png ...")
    plot_metrics_by_dataset(comparison)

    print("Generating eval_aggregate.png ...")
    plot_aggregate(comparison, stats)

    print("Generating eval_sample_scatter.png ...")
    plot_sample_scatter(comparison)

    print(f"\nAll images saved to: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
