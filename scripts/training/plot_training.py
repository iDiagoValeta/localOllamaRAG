"""
Visualización de métricas de entrenamiento
===============================================

Este script genera curvas de entrenamiento a partir de `training_stats.json`
producido por `train.py`. El objetivo es facilitar el análisis experimental de:
- Pérdida de entrenamiento y validación.
- Evolución del learning rate.
- Norma del gradiente (si está disponible en el historial).

Uso:
    python plot_training.py [ruta_al_training_stats.json] [--model qwen-3|llama-3]
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
    print("Instala matplotlib: pip install matplotlib")
    sys.exit(1)


# =============================================================================
# SECCIÓN 1: RUTAS Y BÚSQUEDA DE ARCHIVO DE MÉTRICAS
# =============================================================================
# Define ubicaciones candidatas para localizar `training_stats.json` cuando el
# usuario no proporciona una ruta explícita por línea de comandos.
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
VALID_MODELS = ["qwen-3", "llama-3"]


def parse_args():
    parser = argparse.ArgumentParser(description="Genera curvas de entrenamiento.")
    parser.add_argument(
        "stats_file", nargs="?", default=None,
        help="Ruta explícita a training_stats.json (opcional).",
    )
    parser.add_argument(
        "--model", choices=VALID_MODELS, default="qwen-3",
        help="Modelo a visualizar (default: qwen-3).",
    )
    return parser.parse_args()


def get_paths(model: str, stats_arg: str | None):
    model_dir = os.path.join(PROJECT_ROOT, "training-output", model)
    default_stats = os.path.join(model_dir, "training_stats.json")
    stats_path = stats_arg if stats_arg else default_stats
    images_dir = os.path.join(model_dir, "plots", "train")
    return stats_path, images_dir


def find_stats_file(stats_path: str) -> str:
    """
    Valida que el archivo de estadísticas de entrenamiento existe.

    Args:
        stats_path: Ruta resuelta por get_paths().

    Returns:
        La misma ruta si existe; si no, la devuelve para que main() informe
        del error con contexto.
    """
    return stats_path


# =============================================================================
# SECCIÓN 2: CARGA, PROCESADO Y GRAFICADO DE MÉTRICAS
# =============================================================================
# Ejecuta el flujo completo de análisis:
# - Localiza y valida `training_stats.json`.
# - Extrae series de train/eval/lr/grad_norm desde `log_history`.
# - Genera y guarda un PNG con las curvas para informe y revisión del TFG.
# =============================================================================

def main():
    """
    Punto de entrada principal para generar curvas de entrenamiento.

    El número de subgráficas se adapta dinámicamente:
    - 2 subgráficas: loss y learning rate.
    - 3 subgráficas: añade grad_norm cuando existen datos válidos.
    """
    args = parse_args()
    stats_path, IMAGES_DIR = get_paths(args.model, args.stats_file)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    stats_path = find_stats_file(stats_path)
    if not os.path.exists(stats_path):
        print(f"Error: No existe {stats_path}")
        print("Ejecuta el entrenamiento primero o indica la ruta al archivo.")
        sys.exit(1)

    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    log_history = data.get("log_history", [])
    if not log_history:
        print("No hay datos en log_history.")
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
    fig.suptitle(f"Curvas de entrenamiento — {model_name} {version}",
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
    axes[0].set_title("Pérdida de entrenamiento y validación")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if lr:
        axes[1].plot(train_steps, lr, color="#16a34a", alpha=0.85, linewidth=1.2)
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Evolución del Learning Rate")
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
        axes[2].set_title("Norma del gradiente")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[1].set_xlabel("Step")

    plt.tight_layout()
    out_path = os.path.join(IMAGES_DIR, "training_curves.png")  # noqa: F821 — set in main()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Curvas de entrenamiento guardadas en: {out_path}")

    final_loss = data.get("eval_loss") or data.get("final_loss")
    loss_str = f"{final_loss:.4f}" if final_loss is not None else "N/A"
    print(f"\nResumen: {data.get('total_steps', 'N/A')} steps, Eval Loss final: {loss_str}")
    if "perplexity" in data and data["perplexity"]:
        print(f"Perplexity: {data['perplexity']:.4f}")


# =============================================================================
# SECCIÓN 3: EJECUCIÓN COMO SCRIPT
# =============================================================================
# Permite invocar el archivo directamente desde terminal.
# =============================================================================

if __name__ == "__main__":
    main()
