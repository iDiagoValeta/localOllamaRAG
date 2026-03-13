"""
plot_bertscore.py — Visualización de resultados BERTScore (base vs. adaptado)
==============================================================================

Lee bertscore_summary.csv desde training-output/bertscore/
y genera las siguientes figuras en training-output/bertscore/plots/:

  01_bertscore_f1_by_model.png    — BERTScore F1 base vs adaptado por modelo/dataset
  02_bertscore_components.png     — Componentes P, R, F1 por modelo (base vs adaptado)
  03_improvement_delta.png        — Δ de todas las métricas (adaptado − base)
  04_all_metrics_comparison.png   — BERTScore F1 + Token F1 + Faithfulness comparados
  05_heatmap.png                  — Heatmap completo de métricas
  06_radar.png                    — Radar de perfiles de modelos (adaptado)

Uso:
    python scripts/training/plot_bertscore.py
"""

import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Rutas ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
EVAL_DIR    = os.path.join(ROOT_DIR, "training-output", "bertscore")
CSV_PATH    = os.path.join(EVAL_DIR, "bertscore_summary.csv")
OUTPUT_DIR  = os.path.join(EVAL_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Carga de datos ────────────────────────────────────────────────────────────
rows = []
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        for col in ("BERTScore_P", "BERTScore_R", "BERTScore_F1", "Token_F1", "Faithfulness"):
            row[col] = float(row[col])
        rows.append(row)

MODELS   = ["qwen-3", "llama-3", "gemma-3"]
DATASETS = ["Neural-Bridge RAG", "Dolly QA", "Aina RAG"]
METRICS  = ["BERTScore_P", "BERTScore_R", "BERTScore_F1", "Token_F1", "Faithfulness"]

MODEL_LABELS = {
    "qwen-3":  "Qwen3-14B",
    "llama-3": "Llama-3.1-8B",
    "gemma-3": "Gemma-3-12B",
}
PALETTE = {
    "qwen-3":  "#4C72B0",
    "llama-3": "#C44E52",
    "gemma-3": "#8172B2",
}
DATASET_COLORS = ["#4C72B0", "#55A868", "#C44E52"]

BASE_COLOR    = "#9DB0CE"
ADAPTED_COLOR = "#2D6A4F"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def get(model, dataset, variant, metric):
    for r in rows:
        if r["model"] == model and r["dataset"] == dataset and r["variant"] == variant:
            return r[metric]
    return float("nan")


def bar_label(ax, rects, fmt="{:.1f}", fontsize=8, pad=2):
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
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK  {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — BERTScore F1: base vs adaptado por modelo, desglosado por dataset
# ═══════════════════════════════════════════════════════════════════════════════
def fig_bertscore_f1_by_model():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(
        "BERTScore F1 — Base vs. Adaptado por Modelo y Dataset",
        fontsize=13, fontweight="bold", y=1.02,
    )

    x = np.arange(len(DATASETS))
    w = 0.38

    for ax, model in zip(axes, MODELS):
        base_vals    = [get(model, ds, "base",    "BERTScore_F1") for ds in DATASETS]
        adapted_vals = [get(model, ds, "adapted", "BERTScore_F1") for ds in DATASETS]

        r1 = ax.bar(x - w/2, base_vals,    w, label="Base",    color=BASE_COLOR,    alpha=0.88)
        r2 = ax.bar(x + w/2, adapted_vals, w, label="Adaptado", color=ADAPTED_COLOR, alpha=0.88)

        bar_label(ax, r1)
        bar_label(ax, r2)

        # Líneas de mejora
        for i, (b, a) in enumerate(zip(base_vals, adapted_vals)):
            if not (math.isnan(b) or math.isnan(a)):
                ax.annotate(
                    f"+{a-b:.1f}",
                    xy=(x[i] + w/2, a + 2),
                    fontsize=7.5, color="#2D6A4F", ha="center", fontweight="bold",
                )

        ax.set_title(MODEL_LABELS[model])
        ax.set_ylabel("BERTScore F1 (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS, rotation=12, ha="right")
        ax.set_ylim(0, 100)
        ax.axhline(50, color="gray", lw=0.8, linestyle="--", alpha=0.5)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, "01_bertscore_f1_by_model.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Componentes P, R, F1 por modelo: base vs adaptado
# ═══════════════════════════════════════════════════════════════════════════════
def fig_bertscore_components():
    components = [
        ("BERTScore_P",  "Precisión (P)"),
        ("BERTScore_R",  "Recall (R)"),
        ("BERTScore_F1", "F1"),
    ]

    fig, axes = plt.subplots(len(MODELS), 3, figsize=(14, 10), sharey="col")
    fig.suptitle(
        "Componentes BERTScore (P / R / F1) — Base vs. Adaptado",
        fontsize=13, fontweight="bold", y=1.01,
    )

    x = np.arange(len(DATASETS))
    w = 0.38
    comp_colors = ["#4C72B0", "#55A868", "#C44E52"]

    for row_idx, model in enumerate(MODELS):
        for col_idx, (metric, mlabel) in enumerate(components):
            ax = axes[row_idx][col_idx]
            base_vals    = [get(model, ds, "base",    metric) for ds in DATASETS]
            adapted_vals = [get(model, ds, "adapted", metric) for ds in DATASETS]

            r1 = ax.bar(x - w/2, base_vals,    w, label="Base",    color=BASE_COLOR,             alpha=0.85)
            r2 = ax.bar(x + w/2, adapted_vals, w, label="Adaptado", color=comp_colors[col_idx], alpha=0.85)

            bar_label(ax, r1, fontsize=7)
            bar_label(ax, r2, fontsize=7)

            if row_idx == 0:
                ax.set_title(mlabel, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\n(%)", fontsize=9)

            ax.set_xticks(x)
            ax.set_xticklabels(DATASETS, rotation=12, ha="right", fontsize=8)
            ax.set_ylim(0, 100)
            ax.axhline(50, color="gray", lw=0.6, linestyle="--", alpha=0.4)
            ax.grid(axis="y", alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8)

    fig.tight_layout()
    save(fig, "02_bertscore_components.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Δ de mejora por métrica y dataset (adaptado − base)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_improvement_delta():
    delta_metrics = [
        ("BERTScore_F1", "Δ BERTScore F1"),
        ("Token_F1",     "Δ Token F1"),
        ("Faithfulness", "Δ Faithfulness"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Mejora por Fine-tuning (Δ = Adaptado − Base) — por Modelo y Dataset",
        fontsize=12, fontweight="bold", y=1.02,
    )

    x = np.arange(len(DATASETS))
    w = 0.22

    for ax, (metric, ylabel) in zip(axes, delta_metrics):
        for i, (model, color) in enumerate(PALETTE.items()):
            deltas = [
                get(model, ds, "adapted", metric) - get(model, ds, "base", metric)
                for ds in DATASETS
            ]
            offset = (i - 1) * w
            rects = ax.bar(x + offset, deltas, w, label=MODEL_LABELS[model], color=color, alpha=0.85)
            bar_label(ax, rects, fmt="{:+.1f}", fontsize=7, pad=0.5)

        ax.set_title(ylabel)
        ax.set_ylabel("Puntos porcentuales (pp)")
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS, rotation=12, ha="right")
        ax.axhline(0, color="black", lw=0.8)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(
            min(ax.get_ylim()[0] - 2, -5),
            max(ax.get_ylim()[1] + 2, 5),
        )

    fig.tight_layout()
    save(fig, "03_improvement_delta.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Comparativa de métricas: BERTScore F1 + Token F1 + Faithfulness
# ═══════════════════════════════════════════════════════════════════════════════
def fig_all_metrics_comparison():
    metrics = [
        ("BERTScore_F1", "BERTScore F1 (%)"),
        ("Token_F1",     "Token F1 (%)"),
        ("Faithfulness", "Context Faithfulness (%)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(
        "Comparativa de Métricas — Agregado por Modelo (media de datasets)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    model_keys   = list(PALETTE.keys())
    model_labels = [MODEL_LABELS[m] for m in model_keys]
    x = np.arange(len(model_keys))
    w = 0.38

    for ax, (metric, ylabel) in zip(axes, metrics):
        base_vals    = [
            np.mean([get(m, ds, "base",    metric) for ds in DATASETS])
            for m in model_keys
        ]
        adapted_vals = [
            np.mean([get(m, ds, "adapted", metric) for ds in DATASETS])
            for m in model_keys
        ]

        r1 = ax.bar(x - w/2, base_vals,    w, label="Base",    color=BASE_COLOR,    alpha=0.88)
        r2 = ax.bar(x + w/2, adapted_vals, w, label="Adaptado", color=ADAPTED_COLOR, alpha=0.88)

        bar_label(ax, r1)
        bar_label(ax, r2)

        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=12, ha="right")
        ax.set_ylim(0, 100)
        ax.axhline(50, color="gray", lw=0.8, linestyle="--", alpha=0.5)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, "04_all_metrics_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Heatmap: modelos × datasets × variante
# ═══════════════════════════════════════════════════════════════════════════════
def fig_heatmap():
    show_metrics = [
        ("BERTScore_P",  "BS-P"),
        ("BERTScore_R",  "BS-R"),
        ("BERTScore_F1", "BS-F1"),
        ("Token_F1",     "Tok-F1"),
        ("Faithfulness", "Faith."),
    ]
    variants = [("base", "Base"), ("adapted", "Adapt.")]

    row_labels = []
    matrix     = []

    for model in MODELS:
        for variant, var_label in variants:
            for ds in DATASETS:
                row_labels.append(f"{MODEL_LABELS[model]} [{var_label}]\n{ds}")
                row = [get(model, ds, variant, m) for m, _ in show_metrics]
                matrix.append(row)

    col_labels = [ml for _, ml in show_metrics]
    mat = np.array(matrix, dtype=float)

    # Normalizar por columna
    col_min  = np.nanmin(mat, axis=0)
    col_max  = np.nanmax(mat, axis=0)
    span     = np.where(col_max - col_min < 1e-9, 1, col_max - col_min)
    mat_norm = (mat - col_min) / span

    fig, ax = plt.subplots(figsize=(10, 14))
    fig.suptitle(
        "Heatmap BERTScore — Todos los Modelos, Datasets y Variantes",
        fontsize=12, fontweight="bold",
    )

    im = ax.imshow(mat_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0, ha="center", fontweight="bold")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    # Separadores: cada 6 filas (2 variantes × 3 datasets) = 1 modelo
    for y in [5.5, 11.5]:
        ax.axhline(y, color="white", lw=3)
    # Separadores entre variantes dentro de cada modelo (cada 3 filas)
    for y in [2.5, 8.5, 14.5]:
        ax.axhline(y, color="white", lw=1.5)

    # Valores en las celdas
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not math.isnan(v):
                text_color = "black" if 0.25 < mat_norm[i, j] < 0.78 else "white"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Normalizado por columna (0=peor, 1=mejor)", shrink=0.6)
    fig.tight_layout()
    save(fig, "05_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Radar: perfil de modelos adaptados (media sobre datasets)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_radar():
    radar_metrics = [
        ("BERTScore_P",  "BS-Precisión"),
        ("BERTScore_R",  "BS-Recall"),
        ("BERTScore_F1", "BS-F1"),
        ("Token_F1",     "Token F1"),
        ("Faithfulness", "Faithfulness"),
    ]

    # Valores medios por variante
    raw = {variant: {} for variant in ("base", "adapted")}
    for variant in raw:
        for model in MODELS:
            raw[variant][model] = [
                np.mean([get(model, ds, variant, m) for ds in DATASETS])
                for m, _ in radar_metrics
            ]

    # Normalizar 0-1 usando el rango global (base + adapted juntos)
    all_vals = np.array(
        [raw[v][m] for v in ("base", "adapted") for m in MODELS]
    )
    mins = all_vals.min(axis=0)
    maxs = all_vals.max(axis=0)

    def normalize(vals):
        span = maxs - mins
        span = np.where(span < 1e-9, 1, span)
        return ((np.array(vals) - mins) / span).tolist()

    N      = len(radar_metrics)
    labels = [ml for _, ml in radar_metrics]
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(polar=True))
    fig.suptitle(
        "Radar de Métricas BERTScore — Base vs. Adaptado (media sobre datasets)",
        fontsize=12, fontweight="bold",
    )

    for ax, (variant, title) in zip(axes, [("base", "Modelos Base"), ("adapted", "Modelos Adaptados")]):
        ax.set_title(title, fontweight="bold", pad=15)
        for model in MODELS:
            values = normalize(raw[variant][model])
            values += values[:1]
            ax.plot(angles, values, "o-", lw=2, color=PALETTE[model], label=MODEL_LABELS[model])
            ax.fill(angles, values, alpha=0.10, color=PALETTE[model])

        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
        ax.grid(True, alpha=0.4)
        ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15))

    fig.tight_layout()
    save(fig, "06_radar.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\nGenerando visualizaciones en: {OUTPUT_DIR}\n")

    fig_bertscore_f1_by_model()
    fig_bertscore_components()
    fig_improvement_delta()
    fig_all_metrics_comparison()
    fig_heatmap()
    fig_radar()

    print(f"\nListo — {len(os.listdir(OUTPUT_DIR))} archivos en {OUTPUT_DIR}")
