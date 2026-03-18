"""
Training metrics visualization.
===============================================

Generates training curves from ``training_stats.json`` produced by
``train.py``.  The goal is to facilitate experimental analysis of:

- Training and validation loss.
- Learning rate schedule evolution.
- Gradient norm (when available in the training history).

Usage:
    python plot_training.py [path_to_training_stats.json] [--model qwen-3|llama-3]

Dependencies:
    matplotlib
"""

import argparse
import json
import sys
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    print("Install matplotlib: pip install matplotlib")
    sys.exit(1)


# ─────────────────────────────────────────────
# PATHS AND METRICS FILE LOOKUP
# ─────────────────────────────────────────────
# Defines candidate locations for locating ``training_stats.json`` when the
# user does not supply an explicit path via the command line.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
VALID_MODELS = ["qwen-3", "llama-3", "gemma-3"]


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with ``stats_file`` and ``model``.
    """
    parser = argparse.ArgumentParser(description="Generate training curves.")
    parser.add_argument(
        "stats_file", nargs="?", default=None,
        help="Explicit path to training_stats.json (optional).",
    )
    parser.add_argument(
        "--model", choices=VALID_MODELS, default="qwen-3",
        help="Model to visualize (default: qwen-3).",
    )
    return parser.parse_args()


def get_paths(model: str, stats_arg: str | None):
    """Resolve the statistics file path and image output directory.

    Args:
        model: Model identifier (e.g. ``"qwen-3"``).
        stats_arg: Optional explicit path provided by the user.

    Returns:
        Tuple of (stats_path, images_dir).
    """
    model_dir = os.path.join(PROJECT_ROOT, "training-output", model)
    default_stats = os.path.join(model_dir, "training_stats.json")
    stats_path = stats_arg if stats_arg else default_stats
    images_dir = os.path.join(model_dir, "plots", "train")
    return stats_path, images_dir


def find_stats_file(stats_path: str) -> str:
    """Validate that the training statistics file exists.

    Args:
        stats_path: Path resolved by ``get_paths()``.

    Returns:
        The same path if it exists; otherwise returns it so that ``main()``
        can report the error with context.
    """
    return stats_path


# ─────────────────────────────────────────────
# LOADING, PROCESSING AND PLOTTING METRICS
# ─────────────────────────────────────────────
# Executes the full analysis workflow:
# - Locates and validates ``training_stats.json``.
# - Extracts train/eval/lr/grad_norm series from ``log_history``.
# - Generates and saves a PNG with the curves for reporting and review.

def main():
    """Main entry point for generating training curves.

    The number of subplots adapts dynamically:
    - 2 subplots: loss and learning rate.
    - 3 subplots: adds grad_norm when valid data is available.
    """
    args = parse_args()
    stats_path, IMAGES_DIR = get_paths(args.model, args.stats_file)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    stats_path = find_stats_file(stats_path)
    if not os.path.exists(stats_path):
        print(f"Error: {stats_path} does not exist")
        print("Run training first or provide the path to the file.")
        sys.exit(1)

    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    log_history = data.get("log_history", [])
    if not log_history:
        print("No data found in log_history.")
        sys.exit(1)

    train_steps, train_loss, lr, grad_norm = [], [], [], []
    eval_steps, eval_loss = [], []
    seen_eval_steps = set()

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry and "train_runtime" not in entry:
            step = entry.get("step")
            if step is None:
                continue
            train_steps.append(step)
            train_loss.append(entry["loss"])
            lr.append(entry.get("learning_rate", 0))
            grad_norm.append(entry.get("grad_norm"))
        elif "eval_loss" in entry:
            step = entry.get("step")
            if step is not None and step not in seen_eval_steps:
                seen_eval_steps.add(step)
                eval_steps.append(step)
                eval_loss.append(entry["eval_loss"])

    n_plots = 3 if any(g is not None for g in grad_norm) else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=True)
    if n_plots == 2:
        axes = [axes[0], axes[1]]

    model_name = data.get("model_name", "Model")
    version = data.get("version", "")
    fig.suptitle(f"Training curves — {model_name} {version}",
                 fontsize=13, fontweight="bold", y=1.01)

    axes[0].plot(train_steps, train_loss, color="#2563eb", alpha=0.75,
                 linewidth=1.2, label="Train Loss")
    if eval_steps and eval_loss:
        axes[0].plot(eval_steps, eval_loss, color="#dc2626", marker="o",
                     markersize=6, linewidth=1.8, label="Eval Loss")
        # Final eval loss annotation
        axes[0].annotate(f"{eval_loss[-1]:.4f}",
                         xy=(eval_steps[-1], eval_loss[-1]),
                         xytext=(8, 4), textcoords="offset points",
                         fontsize=8, color="#dc2626")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and validation loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if lr:
        axes[1].plot(train_steps, lr, color="#16a34a", alpha=0.85, linewidth=1.2)
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate evolution")
    axes[1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axes[1].grid(True, alpha=0.3)

    if n_plots == 3 and grad_norm:
        valid_gn = [(s, g) for s, g in zip(train_steps, grad_norm) if g is not None]
        if valid_gn:
            axes[2].plot([x[0] for x in valid_gn], [x[1] for x in valid_gn],
                         color="#9333ea", alpha=0.7, linewidth=1.0)
    if n_plots == 3:
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Grad Norm")
        axes[2].set_title("Gradient norm")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[1].set_xlabel("Step")

    plt.tight_layout()
    out_path = os.path.join(IMAGES_DIR, "training_curves.png")  # noqa: F821 — set in main()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {out_path}")

    final_loss = data.get("eval_loss") or data.get("final_loss")
    loss_str = f"{final_loss:.4f}" if final_loss is not None else "N/A"
    print(f"\nSummary: {data.get('total_steps', 'N/A')} steps, Final eval loss: {loss_str}")
    if "perplexity" in data and data["perplexity"]:
        print(f"Perplexity: {data['perplexity']:.4f}")


# ─────────────────────────────────────────────
# SCRIPT EXECUTION
# ─────────────────────────────────────────────
# Allows invoking this file directly from the terminal.

if __name__ == "__main__":
    main()
