"""
Baseline evaluation results visualization.

Reads ``baseline_evaluation.json`` from ``training-output/baseline/`` and
generates nine publication-ready figures that cover token-level F1, context
faithfulness, grounding deltas, response verbosity, dev-vs-test stability,
per-dataset breakdowns, radar profiles, metric heatmaps, and an aggregate
overview.  All figures are saved to ``training-output/baseline/plots/``.

Usage:
    python scripts/evaluation/plot_baseline_results.py

Dependencies:
    - matplotlib
    - numpy
"""

import json
import os
import sys
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
EVAL_DIR     = os.path.join(ROOT_DIR, "training-output", "baseline")
EVAL_JSON    = os.path.join(EVAL_DIR, "baseline_evaluation.json")
OUTPUT_DIR   = os.path.join(EVAL_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
with open(EVAL_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# ─────────────────────────────────────────────
# VISUAL CONFIGURATION
# ─────────────────────────────────────────────
MODEL_LABELS = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Qwen/Qwen3-14B":                "Qwen3-14B",
    "Qwen/Qwen3.5-9B":               "Qwen3.5-9B",
    "Qwen/Qwen2.5-14B-Instruct":     "Qwen2.5-14B",
    "google/gemma-3-12b-it":         "Gemma-3-12B",
    "microsoft/phi-4":               "Phi-4",
}
MODELS = list(MODEL_LABELS.keys())
SHORT_LABELS = [MODEL_LABELS[m] for m in MODELS]

PALETTE = {
    "meta-llama/Llama-3.1-8B-Instruct": "#C44E52",
    "Qwen/Qwen3-14B":                "#4C72B0",
    "Qwen/Qwen3.5-9B":               "#DD8452",
    "Qwen/Qwen2.5-14B-Instruct":     "#55A868",
    "google/gemma-3-12b-it":         "#8172B2",
    "microsoft/phi-4":               "#937860",
}
COLORS = [PALETTE[m] for m in MODELS]

WITH_CTX_COLOR    = "#4C72B0"
WITHOUT_CTX_COLOR = "#C44E52"
DEV_COLOR         = "#5B8DB8"
TEST_COLOR         = "#E8825C"

DATASETS = ["Neural-Bridge RAG", "Dolly QA", "Aina RAG"]
DATASET_COLORS = ["#4C72B0", "#55A868", "#C44E52"]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def agg(model, mode, split, metric):
    """Return a metric value from the aggregate results, or NaN if missing.

    Args:
        model: Full model identifier string.
        mode: Evaluation mode (``"with_context"`` or ``"without_context"``).
        split: Data split (``"dev"`` or ``"test"``).
        metric: Metric key inside the aggregate dictionary.

    Returns:
        The numeric metric value, or ``float("nan")`` when the key path does
        not exist.
    """
    try:
        return data[model][mode][split]["aggregate"][metric]
    except (KeyError, TypeError):
        return float("nan")


def bar_label(ax, rects, fmt="{:.1f}", fontsize=8, pad=2):
    """Draw value labels above each bar in a bar chart.

    Args:
        ax: Matplotlib axes to annotate.
        rects: Sequence of bar rectangles returned by ``ax.bar()``.
        fmt: Format string applied to the bar height.
        fontsize: Font size for the label text.
        pad: Vertical padding between the bar top and the label.
    """
    for rect in rects:
        h = rect.get_height()
        if not math.isnan(h) and h > 0:
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                h + pad,
                fmt.format(h),
                ha="center", va="bottom",
                fontsize=fontsize, color="#333333",
            )


def save(fig, name):
    """Save a figure to the output directory and close it.

    Args:
        fig: Matplotlib figure object.
        name: Output filename (e.g. ``"01_token_f1.png"``).
    """
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Token F1: with vs without context (dev and test)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_token_f1():
    """Generate a side-by-side bar chart comparing Token F1 with and without
    context for both the dev and test splits."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("Token F1 — Con contexto vs Sin contexto", fontsize=13, fontweight="bold", y=1.01)

    x = np.arange(len(MODELS))
    w = 0.38

    for ax, split, title in zip(axes, ["dev", "test"], ["Split de Validación (dev)", "Split de Test (test)"]):
        with_vals    = [agg(m, "with_context",    split, "Token_F1") for m in MODELS]
        without_vals = [agg(m, "without_context", split, "Token_F1") for m in MODELS]

        r1 = ax.bar(x - w/2, with_vals,    w, label="Con contexto",    color=WITH_CTX_COLOR,    alpha=0.85)
        r2 = ax.bar(x + w/2, without_vals, w, label="Sin contexto",   color=WITHOUT_CTX_COLOR,  alpha=0.85)

        bar_label(ax, r1)
        bar_label(ax, r2)

        ax.set_title(title)
        ax.set_ylabel("Token F1 (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT_LABELS, rotation=15, ha="right")
        ax.set_ylim(0, 90)
        ax.axhline(50, color="gray", lw=0.8, linestyle="--", alpha=0.5)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, "01_token_f1_with_without_context.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Faithfulness: with vs without context
# ═══════════════════════════════════════════════════════════════════════════════
def fig_faithfulness():
    """Generate a side-by-side bar chart comparing Context Faithfulness with
    and without context for both the dev and test splits."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("Context Faithfulness — Con contexto vs Sin contexto", fontsize=13, fontweight="bold", y=1.01)

    x = np.arange(len(MODELS))
    w = 0.38

    for ax, split, title in zip(axes, ["dev", "test"], ["Split de Validación (dev)", "Split de Test (test)"]):
        with_vals    = [agg(m, "with_context",    split, "Context_Faithfulness_Pct") for m in MODELS]
        without_vals = [agg(m, "without_context", split, "Context_Faithfulness_Pct") for m in MODELS]

        r1 = ax.bar(x - w/2, with_vals,    w, label="Con contexto",    color=WITH_CTX_COLOR,    alpha=0.85)
        r2 = ax.bar(x + w/2, without_vals, w, label="Sin contexto",   color=WITHOUT_CTX_COLOR,  alpha=0.85)

        bar_label(ax, r1)
        bar_label(ax, r2)

        ax.set_title(title)
        ax.set_ylabel("Context Faithfulness (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT_LABELS, rotation=15, ha="right")
        ax.set_ylim(0, 105)
        ax.axhline(50, color="gray", lw=0.8, linestyle="--", alpha=0.5)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, "02_faithfulness_with_without_context.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Grounding delta (Faithfulness_with - Faithfulness_without)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_grounding_delta():
    """Generate a grouped bar chart showing the faithfulness improvement each
    model gains from having context, for both dev and test splits."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Δ Context Faithfulness (con − sin contexto)\n"
        "Cuánto contribuye el contexto al grounding de cada modelo",
        fontsize=12, fontweight="bold",
    )

    x = np.arange(len(MODELS))
    w = 0.35

    for i, (split, color, label) in enumerate(
        [("dev", DEV_COLOR, "Dev"), ("test", TEST_COLOR, "Test")]
    ):
        deltas = [
            agg(m, "with_context", split, "Context_Faithfulness_Pct") -
            agg(m, "without_context", split, "Context_Faithfulness_Pct")
            for m in MODELS
        ]
        offset = (i - 0.5) * w
        rects = ax.bar(x + offset, deltas, w, label=split.capitalize(), color=color, alpha=0.85)
        bar_label(ax, rects, fmt="+{:.1f}" if all(d >= 0 for d in deltas if not math.isnan(d)) else "{:.1f}")

    ax.set_ylabel("Δ Faithfulness (puntos porcentuales)")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS, rotation=15, ha="right")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 65)

    # Reference annotation line at 40 percentage points
    ax.axhline(40, color="green", lw=0.8, linestyle=":", alpha=0.6)
    ax.text(len(MODELS) - 1.1, 41, "40 pp", fontsize=8, color="green", ha="right")

    fig.tight_layout()
    save(fig, "03_context_grounding_delta.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Response length (verbosity with vs without context)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_response_length():
    """Generate a side-by-side bar chart comparing mean response length (in
    words) with and without context for both splits."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("Longitud Media de Respuesta (palabras) — Verbosidad", fontsize=13, fontweight="bold", y=1.01)

    x = np.arange(len(MODELS))
    w = 0.38

    for ax, split, title in zip(axes, ["dev", "test"], ["Split de Validación (dev)", "Split de Test (test)"]):
        with_vals    = [agg(m, "with_context",    split, "Avg_Response_Length_Words") for m in MODELS]
        without_vals = [agg(m, "without_context", split, "Avg_Response_Length_Words") for m in MODELS]

        r1 = ax.bar(x - w/2, with_vals,    w, label="Con contexto",    color=WITH_CTX_COLOR,    alpha=0.85)
        r2 = ax.bar(x + w/2, without_vals, w, label="Sin contexto",   color=WITHOUT_CTX_COLOR,  alpha=0.85)

        bar_label(ax, r1, fmt="{:.0f}")
        bar_label(ax, r2, fmt="{:.0f}")

        ax.set_title(title)
        ax.set_ylabel("Palabras por respuesta")
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT_LABELS, rotation=15, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, "04_response_length_verbosity.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Dev vs test stability (with context)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_dev_vs_test():
    """Generate paired bar charts showing dev-vs-test consistency for Token F1
    and Context Faithfulness, with connector lines between splits."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = [
        ("Token_F1",               "Token F1 (%)"),
        ("Context_Faithfulness_Pct", "Context Faithfulness (%)"),
    ]
    fig.suptitle("Estabilidad Dev vs Test — Con contexto", fontsize=13, fontweight="bold", y=1.01)

    x = np.arange(len(MODELS))
    w = 0.38

    for ax, (metric, ylabel) in zip(axes, metrics):
        dev_vals  = [agg(m, "with_context", "dev",  metric) for m in MODELS]
        test_vals = [agg(m, "with_context", "test", metric) for m in MODELS]

        r1 = ax.bar(x - w/2, dev_vals,  w, label="Dev",  color=DEV_COLOR,  alpha=0.85)
        r2 = ax.bar(x + w/2, test_vals, w, label="Test", color=TEST_COLOR, alpha=0.85)

        bar_label(ax, r1)
        bar_label(ax, r2)

        # Connector lines between dev and test for each model
        for i, (d, t) in enumerate(zip(dev_vals, test_vals)):
            if not (math.isnan(d) or math.isnan(t)):
                ax.plot(
                    [x[i] - w/2 + w/2, x[i] + w/2 - w/2],
                    [d, t],
                    "k--", lw=0.8, alpha=0.4,
                )

        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT_LABELS, rotation=15, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    fig.tight_layout()
    save(fig, "05_dev_vs_test_stability.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Llama-3.1-8B: breakdown by dataset
# ═══════════════════════════════════════════════════════════════════════════════
def fig_llama_per_dataset():
    """Generate a 2x2 grid breaking down Llama-3.1-8B performance across
    datasets, modes (with/without context), and splits (dev/test)."""
    model = "meta-llama/Llama-3.1-8B-Instruct"
    metrics = [
        ("Token_F1",               "Token F1 (%)"),
        ("Context_Faithfulness_Pct", "Faithfulness (%)"),
        ("Avg_Response_Length_Words", "Response length (words)"),
        ("Sentence_Completeness_Pct", "Sentence completeness (%)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Llama-3.1-8B-Instruct — Desglose por Dataset\n(dev y test, con y sin contexto)",
        fontsize=12, fontweight="bold",
    )

    x = np.arange(len(DATASETS))
    w = 0.2
    offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
    styles = [
        ("with_context",    "dev",  DEV_COLOR,         "Con ctx - Dev"),
        ("with_context",    "test", WITH_CTX_COLOR,    "Con ctx - Test"),
        ("without_context", "dev",  "#E07B54",         "Sin ctx - Dev"),
        ("without_context", "test", WITHOUT_CTX_COLOR, "Sin ctx - Test"),
    ]

    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        for offset, (mode, split, color, label) in zip(offsets, styles):
            vals = []
            for ds in DATASETS:
                try:
                    vals.append(data[model][mode][split][ds][metric])
                except (KeyError, TypeError):
                    vals.append(float("nan"))

            rects = ax.bar(x + offset, vals, w, label=label, color=color, alpha=0.82)
            bar_label(ax, rects, fmt="{:.0f}", fontsize=7, pad=1)

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS, rotation=10, ha="right")
        ax.legend(fontsize=7.5, ncol=2)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, "06_llama_per_dataset.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Radar: global overview (test, with context)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_radar():
    """Generate a radar chart of four normalized metrics (F1, Faithfulness,
    Compactness, Grounding Delta) using the test split with context."""
    raw = {}
    for m in MODELS:
        f1   = agg(m, "with_context",    "test", "Token_F1")
        faith = agg(m, "with_context",   "test", "Context_Faithfulness_Pct")
        # Compactness: inverse of length, normalized within the range
        length = agg(m, "with_context",  "test", "Avg_Response_Length_Words")
        delta  = (agg(m, "with_context", "test", "Context_Faithfulness_Pct") -
                  agg(m, "without_context", "test", "Context_Faithfulness_Pct"))
        raw[m] = [f1, faith, length, delta]

    # Min-max normalization (0-1) per dimension
    n_dim = 4
    all_vals = np.array([[raw[m][d] for m in MODELS] for d in range(n_dim)])
    mins = all_vals.min(axis=1)
    maxs = all_vals.max(axis=1)

    def normalize(vals):
        normed = []
        for i, v in enumerate(vals):
            span = maxs[i] - mins[i]
            if span < 1e-9:
                normed.append(0.5)
            else:
                # For length, shorter is better so we invert the scale
                if i == 2:
                    normed.append(1 - (v - mins[i]) / span)
                else:
                    normed.append((v - mins[i]) / span)
        return normed

    labels = ["Token F1", "Faithfulness", "Compacidad\n(inv. longitud)", "Δ Grounding"]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.suptitle("Perfil de Modelos — Test, Con contexto\n(métricas normalizadas 0-1)", fontsize=12, fontweight="bold")

    for m in MODELS:
        values = normalize(raw[m])
        values += values[:1]
        ax.plot(angles, values, "o-", lw=2, color=PALETTE[m], label=MODEL_LABELS[m])
        ax.fill(angles, values, alpha=0.10, color=PALETTE[m])

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))

    fig.tight_layout()
    save(fig, "07_radar_overview.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Full metrics heatmap
# ═══════════════════════════════════════════════════════════════════════════════
def fig_heatmap():
    """Generate a color-coded heatmap with rows for each model-mode pair and
    columns for each split-metric combination, normalized per column."""
    metrics_short = {
        "Token_F1": "F1",
        "Context_Faithfulness_Pct": "Faith.",
        "Avg_Response_Length_Words": "Length",
    }
    splits = ["dev", "test"]
    modes  = ["with_context", "without_context"]

    # Build rows and columns
    row_labels = []
    matrix     = []

    for m in MODELS:
        for mode in modes:
            row = []
            row_labels.append(f"{MODEL_LABELS[m]}\n({'C' if mode == 'with_context' else 'S'}/ctx)")
            for split in splits:
                for metric, mshort in metrics_short.items():
                    row.append(agg(m, mode, split, metric))
            matrix.append(row)

    col_labels = [f"{s.capitalize()}\n{mshort}" for s in splits for mshort in metrics_short.values()]

    mat = np.array(matrix, dtype=float)

    # Normalize per column for coloring
    col_min = np.nanmin(mat, axis=0)
    col_max = np.nanmax(mat, axis=0)
    mat_norm = (mat - col_min) / np.where(col_max - col_min < 1e-9, 1, col_max - col_min)

    # Invert Length columns (fewer words = better = green for low values)
    n_m = len(metrics_short)
    length_idx = list(metrics_short.keys()).index("Avg_Response_Length_Words")
    for s_idx in range(len(splits)):
        col = s_idx * n_m + length_idx
        mat_norm[:, col] = 1.0 - mat_norm[:, col]

    fig, ax = plt.subplots(figsize=(11, 10))
    fig.suptitle("Heatmap de Métricas — Todos los Modelos, Modos y Splits", fontsize=12, fontweight="bold")

    im = ax.imshow(mat_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0, ha="center")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Horizontal separator between with/without context groups (every 2 rows)
    for y in [1.5, 3.5, 5.5, 7.5, 9.5]:
        ax.axhline(y, color="white", lw=2)

    # Cell value annotations
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not math.isnan(v):
                text_color = "black" if 0.3 < mat_norm[i, j] < 0.75 else "white"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Normalizado por columna (0=peor, 1=mejor)")

    # Vertical separator between dev and test column groups
    ax.axvline(len(metrics_short) - 0.5, color="white", lw=2)
    ax.text(len(metrics_short) / 2 - 0.5,   -0.7, "Dev",  ha="center", fontsize=10, color="#555")
    ax.text(len(metrics_short) * 1.5 - 0.5, -0.7, "Test", ha="center", fontsize=10, color="#555")

    fig.tight_layout()
    save(fig, "08_performance_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Aggregate overview: 3 metrics in 3 subplots, test split
# ═══════════════════════════════════════════════════════════════════════════════
def fig_aggregate_overview():
    """Generate three side-by-side bar charts summarizing all models on the
    test split, comparing with-context and without-context modes."""
    metrics = [
        ("Token_F1",               "Token F1 (%)"),
        ("Context_Faithfulness_Pct", "Context Faithfulness (%)"),
        ("Avg_Response_Length_Words", "Longitud respuesta (palabras)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Resumen Agregado — Split Test, Todos los Modelos", fontsize=13, fontweight="bold", y=1.02)

    x = np.arange(len(MODELS))
    w = 0.38

    for ax, (metric, ylabel) in zip(axes, metrics):
        with_vals    = [agg(m, "with_context",    "test", metric) for m in MODELS]
        without_vals = [agg(m, "without_context", "test", metric) for m in MODELS]

        r1 = ax.bar(x - w/2, with_vals,    w, label="Con contexto",    color=WITH_CTX_COLOR,    alpha=0.85)
        r2 = ax.bar(x + w/2, without_vals, w, label="Sin contexto",   color=WITHOUT_CTX_COLOR,  alpha=0.85)

        fmt = "{:.0f}" if "Length" in metric else "{:.1f}"
        bar_label(ax, r1, fmt=fmt)
        bar_label(ax, r2, fmt=fmt)

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT_LABELS, rotation=15, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, "09_aggregate_test_overview.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\nGenerando visualizaciones en: {OUTPUT_DIR}\n")

    fig_token_f1()
    fig_faithfulness()
    fig_grounding_delta()
    fig_response_length()
    fig_dev_vs_test()
    fig_llama_per_dataset()
    fig_radar()
    fig_heatmap()
    fig_aggregate_overview()

    print(f"\nListo — {len(os.listdir(OUTPUT_DIR))} archivos en {OUTPUT_DIR}")
