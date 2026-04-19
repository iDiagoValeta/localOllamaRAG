"""
run_eval_recomp_comparison.py -- Reindex once and compare RAGAS with/without RECOMP.

This helper script rebuilds the current ChromaDB collection from the PDFs under
``rag/pdfs/`` (or ``DOCS_FOLDER`` if overridden), then launches two evaluation
runs over the same dataset. If the collection is already indexed, you can reuse
it with ``--skip-reindex``. Each variant saves question-by-question checkpoints,
so rerunning with the same label resumes after the last completed question.

    1. RECOMP enabled
    2. RECOMP disabled

Each run stores its CSV/debug artifacts in organized subfolders under
``evaluation/scores/`` and ``evaluation/debug/``.

Usage:
    python evaluation/run_eval_recomp_comparison.py
    python evaluation/run_eval_recomp_comparison.py --dataset evaluation/datasets/dataset_eval_es.json --verbose
    python evaluation/run_eval_recomp_comparison.py --dataset evaluation/datasets/dataset_eval_es.json --label mi_eval --skip-reindex

Dependencies:
    - Same prerequisites as evaluation/run_eval.py
"""

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  CONFIGURATION
#  +-- 1. Imports            stdlib + shared evaluation runner
#  +-- 2. Paths and naming   output directories and run metadata
#
#  PIPELINE
#  +-- 3. Helpers            timestamping, manifest writing
#  +-- 4. Comparison run     RECOMP on/off orchestration
#
#  ENTRY
#  +-- 5. main()
#
# ─────────────────────────────────────────────

import os
import sys
import json
import time
import argparse
from pathlib import Path


EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(EVAL_DIR)
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


from evaluation.run_eval import ejecutar_evaluacion, SCORES_DIR, DEBUG_DIR, CHECKPOINTS_DIR


# ─────────────────────────────────────────────
# SECTION 2: PATHS AND NAMING
# ─────────────────────────────────────────────

DEFAULT_DATASET = os.path.join(EVAL_DIR, "datasets", "dataset_eval_es.json")
DEFAULT_SCORES_ROOT = os.path.join(SCORES_DIR, "comparison_runs")
DEFAULT_DEBUG_ROOT = os.path.join(DEBUG_DIR, "comparison_runs")


# ─────────────────────────────────────────────
# SECTION 3: HELPERS
# ─────────────────────────────────────────────

def _build_run_slug(dataset_path: str, label: str | None) -> str:
    """Build a stable slug for a comparison batch."""
    dataset_stem = Path(dataset_path).stem
    suffix = label.strip().replace(" ", "_") if label else f"{dataset_stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    return suffix


def _write_json(path: str, payload: dict) -> None:
    """Write a JSON file ensuring the parent directory exists."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# SECTION 4: COMPARISON RUN
# ─────────────────────────────────────────────

def ejecutar_comparativa_recomp(
    dataset_path: str,
    scores_root: str,
    debug_root: str,
    verbose: bool = False,
    save_debug: bool = True,
    label: str | None = None,
    skip_reindex: bool = False,
) -> dict:
    """Reindex once (or reuse current index) and execute paired evaluations.

    Args:
        dataset_path: Dataset file consumed by ``run_eval.py``.
        scores_root: Base directory where CSV files will be stored.
        debug_root: Base directory where debug files and checkpoints will be stored.
        verbose: Whether to print per-question progress during each run.
        save_debug: Whether to save debug JSON files for each run.
        label: Optional suffix for the batch folder name.
        skip_reindex: Reuse the existing ChromaDB collection without rebuilding it.

    Returns:
        Manifest dictionary describing the two generated runs.
    """
    run_slug = _build_run_slug(dataset_path=dataset_path, label=label)
    scores_dir = os.path.join(scores_root, run_slug)
    debug_dir = os.path.join(debug_root, run_slug)
    checkpoints_dir = os.path.join(CHECKPOINTS_DIR, "comparison_runs", run_slug)
    os.makedirs(scores_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    run_specs = [
        ("recomp_on", True, not skip_reindex),
        ("recomp_off", False, False),
    ]

    results = []
    for folder_name, recomp_enabled, force_reindex in run_specs:
        print("\n" + "=" * 70)
        print(
            f"Launching evaluation: {folder_name} "
            f"(RECOMP {'ON' if recomp_enabled else 'OFF'})"
        )
        print("=" * 70)

        result = ejecutar_evaluacion(
            dataset_path=dataset_path,
            output_path=os.path.join(scores_dir, f"{folder_name}.csv"),
            debug_path=os.path.join(debug_dir, f"{folder_name}.json"),
            checkpoint_path=os.path.join(checkpoints_dir, f"{folder_name}.json"),
            verbose=verbose,
            save_debug=save_debug,
            force_reindex=force_reindex,
            recomp_enabled=recomp_enabled,
        )
        result["variant"] = folder_name
        results.append(result)

    by_variant = {entry["variant"]: entry for entry in results}
    deltas = {}
    for metric_name, recomp_value in by_variant["recomp_on"]["mean_scores"].items():
        no_recomp_value = by_variant["recomp_off"]["mean_scores"].get(metric_name)
        if no_recomp_value is None:
            continue
        deltas[metric_name] = recomp_value - no_recomp_value

    manifest = {
        "dataset_path": os.path.abspath(dataset_path),
        "scores_dir": os.path.abspath(scores_dir),
        "debug_dir": os.path.abspath(debug_dir),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": results,
        "mean_score_deltas_recomp_minus_no_recomp": deltas,
    }
    _write_json(os.path.join(debug_dir, "comparison_summary.json"), manifest)
    return manifest


# ─────────────────────────────────────────────
# SECTION 5: MAIN
# ─────────────────────────────────────────────

def main() -> None:
    """Parse CLI args and launch the paired evaluation workflow."""
    parser = argparse.ArgumentParser(
        description="Reindex PDFs and compare RAGAS runs with and without RECOMP"
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Path to the dataset (JSON, CSV, or Excel)",
    )
    parser.add_argument(
        "--scores-root",
        default=DEFAULT_SCORES_ROOT,
        help="Base directory where comparison CSV files are stored",
    )
    parser.add_argument(
        "--debug-root",
        default=DEFAULT_DEBUG_ROOT,
        help="Base directory where comparison debug files are stored",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional suffix for the comparison folder name",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-question progress during each evaluation run",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Skip saving ragas_debug.json for both runs",
    )
    parser.add_argument(
        "--skip-reindex",
        action="store_true",
        help="Reuse the existing ChromaDB collection instead of rebuilding it first",
    )
    args = parser.parse_args()

    manifest = ejecutar_comparativa_recomp(
        dataset_path=args.dataset,
        scores_root=args.scores_root,
        debug_root=args.debug_root,
        verbose=args.verbose,
        save_debug=not args.no_debug,
        label=args.label,
        skip_reindex=args.skip_reindex,
    )

    print("\nComparison finished.")
    print(f"Scores folder: {manifest['scores_dir']}")
    print(f"Debug folder:  {manifest['debug_dir']}")
    print("Summary file: " + os.path.join(manifest["debug_dir"], "comparison_summary.json"))


if __name__ == "__main__":
    main()
