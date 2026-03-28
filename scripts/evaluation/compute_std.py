"""
compute_std.py -- Per-sample mean and standard deviation from BERTScore CSVs.

Reads the per-sample CSV files produced by the evaluation pipeline:
  training-output/bertscore/bertscore_per_sample_{model}_{dataset}_{variant}.csv
  training-output/bertscore/metrics_per_sample_{model}_{dataset}_{variant}.csv

Computes mean +/- standard deviation for every cell
(model x dataset x metric x variant).  Prints a summary table to stdout
and generates ready-to-paste LaTeX snippets for the thesis.

Usage:
    python compute_std.py
    python compute_std.py --bertscore-dir training-output/bertscore
    python compute_std.py --latex-out sigma_table.tex

Dependencies:
    - Standard library only (argparse, csv, math, os, re)
"""

import argparse
import csv
import math
import os
import re

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Compute mean +/- std over per-sample CSVs.")
parser.add_argument(
    "--bertscore-dir",
    default=os.path.join("training-output", "bertscore"),
    help="Directory containing the per-sample CSVs (default: training-output/bertscore).",
)
parser.add_argument(
    "--latex-out",
    default=None,
    help="Output file for LaTeX snippets (optional; if omitted, stdout only).",
)
args = parser.parse_args()

BERTSCORE_DIR = args.bertscore_dir


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def mean(values):
    return sum(values) / len(values) if values else float("nan")


def std(values):
    if len(values) < 2:
        return float("nan")
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def read_csv_column(path, col):
    """Read a column from a CSV by header name and return a list of floats.

    Args:
        path: Path to the CSV file.
        col: Column header name to extract.

    Returns:
        List of float values from the specified column.
    """
    values = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[col]))
    return values


def slug_to_display(slug):
    """Convert an internal slug to a human-readable display name.

    Args:
        slug: Underscore-separated identifier (e.g. "qwen_3").

    Returns:
        Display-friendly string, or the original slug if unknown.
    """
    mapping = {
        "neural_bridge_rag": "Neural-Bridge RAG",
        "dolly_qa":          "Dolly QA",
        "aina_rag":          "Aina RAG",
        "qwen-3":            "Qwen3-14B",
        "llama-3":           "Llama-3.1-8B",
        "gemma-3":           "Gemma-3-12B",
    }
    return mapping.get(slug, slug)


# ─────────────────────────────────────────────
# PARSE FILENAMES AND COLLECT DATA
# ─────────────────────────────────────────────
# Expected filename patterns:
#   bertscore_per_sample_{model}_{dataset}_{variant}.csv
#   metrics_per_sample_{model}_{dataset}_{variant}.csv
# where model in {qwen-3, llama-3, gemma-3}, stored as slug with underscores

BS_PATTERN  = re.compile(r"^bertscore_per_sample_(.+)_(base|adapted)\.csv$")
AUX_PATTERN = re.compile(r"^metrics_per_sample_(.+)_(base|adapted)\.csv$")

DATASETS = ["neural_bridge_rag", "dolly_qa", "aina_rag"]
MODELS   = ["qwen-3", "llama-3", "gemma-3"]

data = {}   # data[model_slug][ds_slug][variant] = {metric: [float, ...]}


def ensure(model, ds, variant):
    data.setdefault(model, {}).setdefault(ds, {}).setdefault(variant, {})


if not os.path.isdir(BERTSCORE_DIR):
    print(f"[ERROR] Directory not found: {BERTSCORE_DIR}")
    print("  Run eval_bertscore.py first to generate the per-sample CSVs.")
    raise SystemExit(1)

found_any = False
for fname in sorted(os.listdir(BERTSCORE_DIR)):
    fpath = os.path.join(BERTSCORE_DIR, fname)

    m = BS_PATTERN.match(fname)
    if m:
        model_ds_slug, variant = m.group(1), m.group(2)
        # model_ds_slug = "{model}_{dataset}" -- split on known dataset slugs
        for ds_slug in DATASETS:
            if model_ds_slug.endswith("_" + ds_slug):
                model_slug = model_ds_slug[: -(len(ds_slug) + 1)]
                ensure(model_slug, ds_slug, variant)
                data[model_slug][ds_slug][variant]["bs_precision"] = read_csv_column(fpath, "bs_precision")
                data[model_slug][ds_slug][variant]["bs_recall"]    = read_csv_column(fpath, "bs_recall")
                data[model_slug][ds_slug][variant]["bs_f1"]        = read_csv_column(fpath, "bs_f1")
                found_any = True
                break

    m = AUX_PATTERN.match(fname)
    if m:
        model_ds_slug, variant = m.group(1), m.group(2)
        for ds_slug in DATASETS:
            if model_ds_slug.endswith("_" + ds_slug):
                model_slug = model_ds_slug[: -(len(ds_slug) + 1)]
                ensure(model_slug, ds_slug, variant)
                data[model_slug][ds_slug][variant]["token_f1"]    = read_csv_column(fpath, "token_f1")
                data[model_slug][ds_slug][variant]["faithfulness"] = read_csv_column(fpath, "faithfulness")
                found_any = True
                break

if not found_any:
    print("[ERROR] No per-sample CSVs found in:", BERTSCORE_DIR)
    print("  Run eval_bertscore.py first to generate them.")
    raise SystemExit(1)


# ─────────────────────────────────────────────
# COMPUTE STATISTICS
# ─────────────────────────────────────────────

def stats(values):
    """Returns (mean*100, std*100) as percentages, or (nan, nan)."""
    if not values:
        return float("nan"), float("nan")
    m = mean(values) * 100
    s = std(values)  * 100
    return m, s


# ─────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────────

print("\n" + "=" * 90)
print("  RESUMEN media ± σ por (modelo × dataset × métrica)")
print("=" * 90)

COL_METRICS = [
    ("bs_f1",        "BS-F1 (%)"),
    ("token_f1",     "Token F1 (%)"),
    ("faithfulness", "Faithfulness (%)"),
]

for model_slug in MODELS:
    if model_slug not in data:
        continue
    print(f"\n  Modelo: {slug_to_display(model_slug)}")
    print(f"  {'Dataset':<22} {'Variante':<10} " +
          "  ".join(f"{lbl:<22}" for _, lbl in COL_METRICS))
    print("  " + "-" * 85)
    for ds_slug in DATASETS:
        if ds_slug not in data[model_slug]:
            continue
        for variant in ["base", "adapted"]:
            if variant not in data[model_slug][ds_slug]:
                continue
            d = data[model_slug][ds_slug][variant]
            cells = []
            for metric, _ in COL_METRICS:
                vals = d.get(metric, [])
                m_val, s_val = stats(vals)
                cells.append(f"{m_val:.1f} ± {s_val:.1f}" if not math.isnan(m_val) else "N/A")
            ds_disp = slug_to_display(ds_slug)
            print(f"  {ds_disp:<22} {variant:<10}  " + "  ".join(f"{c:<22}" for c in cells))


# ─────────────────────────────────────────────
# GENERATE LATEX TABLE SNIPPETS
# ─────────────────────────────────────────────

latex_lines = []
latex_lines.append("% ============================================================")
latex_lines.append("% LaTeX snippet: tab:bertscore_full  (mean +/- std, n=200 per cell)")
latex_lines.append("% ============================================================")
latex_lines.append(r"\begin{tabular}{|l|l|cc|cc|cc|}")
latex_lines.append(r"\hline")
latex_lines.append(r"\textbf{Modelo} & \textbf{Dataset} & "
                   r"\textbf{BS-F1 base} & \textbf{BS-F1 adj.} & "
                   r"\textbf{Token F1 base} & \textbf{Token F1 adj.} & "
                   r"\textbf{Faith. base} & \textbf{Faith. adj.} \\")
latex_lines.append(r"\hline")

for model_slug in MODELS:
    if model_slug not in data:
        continue
    model_disp = slug_to_display(model_slug)
    n_datasets = sum(1 for ds in DATASETS if ds in data[model_slug])
    first = True
    for ds_slug in DATASETS:
        if ds_slug not in data[model_slug]:
            continue
        ds_disp = slug_to_display(ds_slug)
        model_cell = (r"\multirow{" + str(n_datasets) + r"}{*}{" + model_disp + "}") if first else ""
        first = False
        cells = []
        for metric in ["bs_f1", "token_f1", "faithfulness"]:
            for variant in ["base", "adapted"]:
                vals = data[model_slug][ds_slug].get(variant, {}).get(metric, [])
                m_val, s_val = stats(vals)
                cells.append(
                    f"${m_val:.1f} \\pm {s_val:.1f}$" if not math.isnan(m_val) else "---"
                )
        latex_lines.append(
            f"  {model_cell} & {ds_disp} & "
            + " & ".join(cells) + r" \\"
        )
    latex_lines.append(r"\hline")

latex_lines.append(r"\end{tabular}")
latex_lines.append("")

# tab:bertscore_global -- global averages across models
latex_lines.append("% ============================================================")
latex_lines.append("% LaTeX snippet: tab:bertscore_global  (global mean +/- std)")
latex_lines.append("% ============================================================")
latex_lines.append(r"\begin{tabular}{|l|cc|c|c|}")
latex_lines.append(r"\hline")
latex_lines.append(r"\textbf{Dataset} & \textbf{BS-F1 base} & \textbf{BS-F1 adj.} & "
                   r"\textbf{$\Delta$ BS-F1} & \textbf{$\Delta$ Faithfulness} \\")
latex_lines.append(r"\hline")

for ds_slug in DATASETS:
    ds_disp = slug_to_display(ds_slug)
    all_base_f1, all_adapted_f1 = [], []
    all_base_faith, all_adapted_faith = [], []
    for model_slug in MODELS:
        if model_slug not in data or ds_slug not in data[model_slug]:
            continue
        all_base_f1    += data[model_slug][ds_slug].get("base", {}).get("bs_f1", [])
        all_adapted_f1 += data[model_slug][ds_slug].get("adapted", {}).get("bs_f1", [])
        all_base_faith    += data[model_slug][ds_slug].get("base", {}).get("faithfulness", [])
        all_adapted_faith += data[model_slug][ds_slug].get("adapted", {}).get("faithfulness", [])
    mb, sb   = stats(all_base_f1)
    ma, sa   = stats(all_adapted_f1)
    delta_f1 = (ma - mb) if not math.isnan(ma) else float("nan")
    mbf, _   = stats(all_base_faith)
    maf, _   = stats(all_adapted_faith)
    delta_faith = (maf - mbf) if not math.isnan(maf) else float("nan")
    latex_lines.append(
        f"  {ds_disp} & ${mb:.1f} \\pm {sb:.1f}$ & ${ma:.1f} \\pm {sa:.1f}$ & "
        f"${delta_f1:+.1f}$~pp & ${delta_faith:+.1f}$~pp \\\\"
    )

latex_lines.append(r"\hline")
latex_lines.append(r"\end{tabular}")

latex_str = "\n".join(latex_lines)

print("\n\n" + "=" * 90)
print("  SNIPPETS LaTeX")
print("=" * 90)
print(latex_str)

if args.latex_out:
    with open(args.latex_out, "w", encoding="utf-8") as f:
        f.write(latex_str + "\n")
    print(f"\n--> LaTeX saved to: {args.latex_out}")
