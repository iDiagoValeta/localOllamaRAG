"""
Visualización de métricas de entrenamiento
===============================================

Este script genera curvas de entrenamiento a partir de `training_stats.json`
producido por `train.py`. El objetivo es facilitar el análisis experimental de:
- Pérdida de entrenamiento y validación.
- Evolución del learning rate.
- Norma del gradiente (si está disponible en el historial).

Uso:
    python plot_training.py [ruta_al_training_stats.json]
"""

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
DEFAULT_PATHS = [
    os.path.join(PROJECT_ROOT, "training-output", "training_stats.json"),
    os.path.join(PROJECT_ROOT, "models", "training-output", "training_stats.json"),
    os.path.join(PROJECT_ROOT, "training_stats.json"),
]


def find_stats_file(path_arg):
    """
    Resuelve la ruta del archivo de estadísticas de entrenamiento.

    Prioridad de búsqueda:
    1) Ruta pasada como argumento.
    2) Rutas por defecto del proyecto.

    Args:
        path_arg: Ruta opcional recibida por línea de comandos.

    Returns:
        Ruta existente encontrada o, en su defecto, una ruta candidata para
        informar al usuario en el mensaje de error.
    """
    if path_arg and os.path.exists(path_arg):
        return path_arg
    for p in DEFAULT_PATHS:
        if os.path.exists(p):
            return p
    return path_arg or DEFAULT_PATHS[0]


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
    stats_path = find_stats_file(sys.argv[1] if len(sys.argv) > 1 else None)
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

    axes[0].plot(train_steps, train_loss, "b-", alpha=0.7, label="Train Loss")
    if eval_steps and eval_loss:
        axes[0].plot(eval_steps, eval_loss, "r-o", markersize=5, label="Eval Loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss durante el entrenamiento")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if lr:
        axes[1].plot(train_steps, lr, "g-", alpha=0.7)
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate")
    axes[1].grid(True, alpha=0.3)

    if n_plots == 3 and grad_norm:
        valid_gn = [(s, g) for s, g in zip(train_steps, grad_norm) if g is not None]
        if valid_gn:
            axes[2].plot([x[0] for x in valid_gn], [x[1] for x in valid_gn], "m-", alpha=0.7)
    if n_plots == 3:
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Grad Norm")
        axes[2].set_title("Norma del gradiente")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[1].set_xlabel("Step")

    plt.tight_layout()
    out_dir = os.path.dirname(stats_path)
    out_path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150)
    print(f"Gráficos guardados en: {out_path}")

    final_loss = data.get("final_loss")
    loss_str = f"{final_loss:.4f}" if final_loss is not None else "N/A"
    print(f"\nResumen: {data.get('total_steps', 'N/A')} steps, Loss final: {loss_str}")
    if "perplexity" in data and data["perplexity"]:
        print(f"Perplexity: {data['perplexity']:.2f}")


# =============================================================================
# SECCIÓN 3: EJECUCIÓN COMO SCRIPT
# =============================================================================
# Permite invocar el archivo directamente desde terminal.
# =============================================================================

if __name__ == "__main__":
    main()
