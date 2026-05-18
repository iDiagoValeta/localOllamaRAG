"""MonkeyGrab RAG inference CLI (no RAGAS).

Genera respuestas RAG sobre datasets de evaluación y persiste checkpoints que
``evaluate.py`` consume después con cualquier proveedor RAGAS.

Subcomandos:
    single             Inferencia con la variante baseline para un corpus local.
    compare            Comparación final o ablation legacy sobre un corpus local.
    ragbench-prepare   Prepara el corpus EN final RagBench (descarga + dataset + manifest).
    ragbench-eval      Inferencia sobre el manifest preparado RagBench EN.
    visual             Prepara + ejecuta inferencia RagBench visual (tablas/imágenes).
    list-variants      Lista las variantes ablation disponibles.

# ─────────────────────────────────────────────
# MODULE MAP
# ─────────────────────────────────────────────
#
#  1. Environment setup
#  2. Pipeline orchestration (single, compare, ragbench, visual)
#  3. CLI parser and dispatch
#
# ─────────────────────────────────────────────
"""

from __future__ import annotations

# ─────────────────────────────────────────────
# SECTION 1: ENVIRONMENT SETUP
# ─────────────────────────────────────────────

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

_this_file = Path(__file__).resolve()
_proj_root = _this_file.parent.parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

try:
    from dotenv import load_dotenv
    load_dotenv(_proj_root / ".env")
except ImportError:
    pass

from research.evaluation._lib.datasets import (
    COMPARISON_RUNS_DIR,
    RAGBENCH_RUNS_DIR,
    RAGBENCH_VISUAL_RUNS_DIR,
    SUPPORTED_CORPORA,
    artifact_suffix,
    build_run_slug,
    default_dataset_for_corpus,
    default_debug_path,
    default_output_path,
    guardar_json,
    resolver_ruta_dataset,
    safe_tag,
    single_run_dir,
)
from research.evaluation._lib.inference import generar_respuestas_rag
from research.evaluation._lib.pipeline_flags import (
    DEFAULT_VARIANT_SUITE,
    RAGBENCH_FINAL_PIPELINE_FLAGS,
    RAGBENCH_VISUAL_PIPELINE_FLAGS,
    VARIANT_SUITES,
    listar_variantes,
    seleccionar_variantes,
)
from research.evaluation._lib.ragbench import (
    RAGBENCH_EVAL_MANIFEST_PATH,
    RAGBENCH_DEV_DOC_IDS_PATH,
    RAGBENCH_EVAL_PDFS_DIR,
    RAGBENCH_VISUAL_PDFS_DIR,
    RAGBENCH_VISUAL_PREPARED_DIR,
    DEFAULT_VISUAL_SOURCES,
    cargar_manifest_ragbench_eval,
    exportar_resultados_inferencia,
    parse_visual_sources,
    preparar_ragbench_eval_en,
    preparar_ragbench_visual,
    selected_pdf_filenames_from_manifest,
)


VISUAL_RUN_DIR = os.path.join(RAGBENCH_VISUAL_RUNS_DIR, "inference")


# ─────────────────────────────────────────────
# SECTION 2: PIPELINE ORCHESTRATION
# ─────────────────────────────────────────────

def ejecutar_single(args: argparse.Namespace) -> None:
    """Run baseline RAG inference on a single corpus, no RAGAS."""
    eval_corpus = args.corpus
    dataset = args.dataset or default_dataset_for_corpus(eval_corpus)
    recomp_override = False if args.no_recomp else None

    generation = generar_respuestas_rag(
        dataset_path=dataset,
        output_path=args.output,
        debug_path=args.debug_output,
        checkpoint_path=args.checkpoint,
        verbose=args.verbose,
        force_reindex=args.force_reindex,
        recomp_enabled=recomp_override,
        eval_corpus=eval_corpus,
        docs_dir=args.docs_dir,
    )
    print("\nInference finished.")
    print(f"   Checkpoint: {generation['checkpoint_path']}")
    print(f"   Pipeline:   {generation['pipeline_seconds']:.1f}s")


def ejecutar_compare(args: argparse.Namespace) -> None:
    """Run an ablation comparison: one generation per variant."""
    eval_corpus = args.corpus
    dataset_arg = args.dataset or default_dataset_for_corpus(eval_corpus)
    dataset_path = resolver_ruta_dataset(dataset_arg)

    variants = seleccionar_variantes(args.suite, args.variants)
    baseline_variant = (
        "baseline_all_on"
        if any(v["name"] == "baseline_all_on" for v in variants)
        else variants[0]["name"]
    )

    run_slug = build_run_slug(dataset_path=dataset_path, label=args.label, eval_corpus=eval_corpus)
    run_dir = os.path.join(COMPARISON_RUNS_DIR, run_slug)
    scores_dir = os.path.join(run_dir, "scores")
    debug_dir = os.path.join(run_dir, "debug")
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(scores_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    generations: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        variant_name = variant["name"]
        should_reindex = args.reindex and index == 0
        print("\n" + "=" * 70)
        print(f"Launching RAG inference: {variant_name}")
        print(f"Variant: {variant['description']}")
        print("=" * 70)
        generation = generar_respuestas_rag(
            dataset_path=dataset_path,
            output_path=os.path.join(scores_dir, f"{variant_name}.csv"),
            debug_path=os.path.join(debug_dir, f"{variant_name}.json"),
            checkpoint_path=os.path.join(checkpoints_dir, f"{variant_name}.json"),
            verbose=args.verbose,
            force_reindex=should_reindex,
            pipeline_flags=variant["flags"],
            eval_corpus=eval_corpus,
            docs_dir=args.docs_dir,
        )
        generation["variant"] = variant_name
        generation["variant_description"] = variant["description"]
        generation["requested_pipeline_flags"] = dict(variant["flags"])
        generations.append(generation)

    manifest = {
        "dataset_path": os.path.abspath(dataset_path),
        "eval_corpus": eval_corpus,
        "suite": args.suite,
        "baseline_variant": baseline_variant,
        "selected_variants": [variant["name"] for variant in variants],
        "scores_dir": os.path.abspath(scores_dir),
        "debug_dir": os.path.abspath(debug_dir),
        "checkpoints_dir": os.path.abspath(checkpoints_dir),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": [
            {
                "variant": gen["variant"],
                "variant_description": gen["variant_description"],
                "requested_pipeline_flags": gen["requested_pipeline_flags"],
                "checkpoint_path": gen["checkpoint_path"],
                "questions_count": gen["questions_count"],
                "indexed_fragments": gen["indexed_fragments"],
                "pipeline_seconds": gen["pipeline_seconds"],
            }
            for gen in generations
        ],
    }
    summary_path = os.path.join(run_dir, "inference_summary.json")
    guardar_json(summary_path, manifest)

    print("\nComparison inference finished.")
    print(f"   Run dir:    {run_dir}")
    print(f"   Summary:    {summary_path}")
    print("   Next:       run evaluate.py over the checkpoints directory.")


def ejecutar_ragbench_prepare(args: argparse.Namespace) -> None:
    """Download the fixed RagBench EN final corpus and write its manifest."""
    manifest = preparar_ragbench_eval_en(
        source=args.source,
        n_papers=args.n_papers,
        max_q=args.max_q,
        skip_download=args.skip_download,
        docs_dir=args.docs_dir or RAGBENCH_EVAL_PDFS_DIR,
        manifest_path=args.manifest,
        excluded_doc_ids_path=args.excluded_doc_ids,
    )
    print("\nRagBench EN preparation finished.")
    print(f"   Dataset:   {manifest['dataset_path']}")
    print(f"   Docs dir:  {manifest['docs_dir']}")
    print(f"   Manifest:  {args.manifest}")


def ejecutar_ragbench_eval(args: argparse.Namespace) -> None:
    """Run RAG inference on the prepared RagBench EN manifest (no RAGAS)."""
    manifest = cargar_manifest_ragbench_eval(args.manifest)
    dataset_path = os.path.abspath(manifest["dataset_path"])
    docs_dir = os.path.abspath(manifest["docs_dir"])
    indexed_files = selected_pdf_filenames_from_manifest(manifest)
    if not indexed_files:
        raise SystemExit("ERROR: el manifiesto RagBench EN no contiene indexed_files.")

    tag = safe_tag(Path(dataset_path).stem)
    run_dir = os.path.join(RAGBENCH_RUNS_DIR, "en_eval", tag)
    os.makedirs(run_dir, exist_ok=True)

    print("\nEjecutando inferencia RagBench EN final:")
    print("   source=text, baseline pipeline flags=on")
    print(f"   dataset:    {dataset_path}")
    print(f"   docs_dir:   {docs_dir}")
    print(f"   files:      {len(indexed_files)}")
    print(f"   output dir: {run_dir}")

    generation = generar_respuestas_rag(
        dataset_path=dataset_path,
        output_path=os.path.join(run_dir, "scores.csv"),
        debug_path=os.path.join(run_dir, "debug.json"),
        checkpoint_path=os.path.join(run_dir, "checkpoint.json"),
        verbose=args.verbose,
        force_reindex=args.force_reindex,
        pipeline_flags=RAGBENCH_FINAL_PIPELINE_FLAGS,
        eval_corpus="ragbench",
        docs_dir=docs_dir,
        solo_archivos=indexed_files,
        add_missing_from_filter=True,
    )
    print("\nRagBench EN inference finished.")
    print(f"   Checkpoint: {generation['checkpoint_path']}")


def ejecutar_visual(args: argparse.Namespace) -> None:
    """Prepare + run RagBench visual inference (table/image, no RAGAS)."""
    try:
        sources = parse_visual_sources(args.sources)
    except ValueError as exc:
        raise SystemExit(str(exc))

    manifest = preparar_ragbench_visual(
        sources=sources,
        n_papers=args.n_papers,
        max_q=args.max_q,
        skip_download=args.skip_download,
        docs_dir=Path(args.docs_dir),
        debug_dir=Path(args.prepared_dir),
        only_doc=args.only_doc,
        excluded_doc_ids_path=Path(args.excluded_doc_ids),
    )

    dataset_path = str(Path(manifest["dataset_path"]).resolve())
    docs_dir = str(Path(manifest["docs_dir"]).resolve())
    indexed_files = [str(name) for name in manifest.get("indexed_files", [])]
    if not indexed_files:
        raise SystemExit("ERROR: el manifiesto visual no contiene indexed_files.")

    tag = safe_tag(Path(dataset_path).stem.replace("dataset_ragbench_visual_", ""))
    run_dir = Path(args.output_dir) / tag
    result_csv = run_dir / "results.csv"
    result_json = run_dir / "results.json"
    checkpoint_path = run_dir / "checkpoint.json"

    print("\nEjecutando inferencia RagBench visual sin RAGAS:")
    print(f"   sources={','.join(sources)}")
    print("   baseline pipeline flags=on")
    print(f"   output dir: {run_dir}")

    generation = generar_respuestas_rag(
        dataset_path=dataset_path,
        output_path=str(result_csv),
        debug_path=str(result_json),
        checkpoint_path=str(checkpoint_path),
        verbose=args.verbose,
        force_reindex=args.force_reindex,
        pipeline_flags=RAGBENCH_VISUAL_PIPELINE_FLAGS,
        eval_corpus="ragbench",
        docs_dir=docs_dir,
        solo_archivos=indexed_files,
        add_missing_from_filter=True,
    )
    exportar_resultados_inferencia(generation, manifest, result_csv, result_json)
    print("\nRagBench visual inference finished.")
    print(f"   Checkpoint: {checkpoint_path}")


# ─────────────────────────────────────────────
# SECTION 3: CLI PARSER AND DISPATCH
# ─────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MonkeyGrab RAG inference runner (no RAGAS).")
    sub = parser.add_subparsers(dest="command")

    # ── single ────────────────────────────────────────────
    p_single = sub.add_parser("single", help="Run baseline RAG inference on a local corpus")
    p_single.add_argument("--corpus", choices=SUPPORTED_CORPORA, default="es")
    p_single.add_argument("--dataset", default=None)
    p_single.add_argument("--docs-dir", default=None)
    p_single.add_argument("--output", default=None)
    p_single.add_argument("--debug-output", default=None)
    p_single.add_argument("--checkpoint", default=None)
    p_single.add_argument("--force-reindex", action="store_true")
    p_single.add_argument("--no-recomp", action="store_true", help="Disable RECOMP synthesis")
    p_single.add_argument("--verbose", action="store_true")
    p_single.set_defaults(func=ejecutar_single)

    # ── compare ───────────────────────────────────────────
    p_compare = sub.add_parser("compare", help="Run ablation comparison on a local corpus")
    p_compare.add_argument("--corpus", choices=SUPPORTED_CORPORA, default="es")
    p_compare.add_argument("--dataset", default=None)
    p_compare.add_argument("--docs-dir", default=None)
    p_compare.add_argument("--label", default=None, help="Folder suffix for this comparison batch")
    p_compare.add_argument("--suite", choices=sorted(VARIANT_SUITES), default=DEFAULT_VARIANT_SUITE)
    p_compare.add_argument("--variants", default=None, help="Comma-separated variant names")
    p_compare.add_argument("--reindex", action="store_true", help="Rebuild Chroma before the first variant")
    p_compare.add_argument("--verbose", action="store_true")
    p_compare.set_defaults(func=ejecutar_compare)

    # ── list-variants ─────────────────────────────────────
    p_list = sub.add_parser("list-variants", help="List available comparison variants")
    p_list.add_argument("--suite", choices=sorted(VARIANT_SUITES), default=DEFAULT_VARIANT_SUITE)
    p_list.set_defaults(func=lambda args: listar_variantes(args.suite))

    # ── ragbench-prepare ──────────────────────────────────
    p_prep = sub.add_parser("ragbench-prepare", help="Prepare the fixed RagBench EN evaluation corpus")
    p_prep.add_argument("--source", default="text", choices=["text"])
    p_prep.add_argument("--n-papers", type=int, default=40)
    p_prep.add_argument("--max-q", type=int, default=5)
    p_prep.add_argument("--skip-download", action="store_true")
    p_prep.add_argument("--docs-dir", default=RAGBENCH_EVAL_PDFS_DIR)
    p_prep.add_argument("--manifest", default=RAGBENCH_EVAL_MANIFEST_PATH)
    p_prep.add_argument("--excluded-doc-ids", default=RAGBENCH_DEV_DOC_IDS_PATH)
    p_prep.set_defaults(func=ejecutar_ragbench_prepare)

    # ── ragbench-eval ─────────────────────────────────────
    p_eval = sub.add_parser("ragbench-eval", help="Run inference on the prepared RagBench EN manifest")
    p_eval.add_argument("--manifest", default=RAGBENCH_EVAL_MANIFEST_PATH)
    p_eval.add_argument("--force-reindex", action="store_true")
    p_eval.add_argument("--verbose", action="store_true")
    p_eval.set_defaults(func=ejecutar_ragbench_eval)

    # ── visual ────────────────────────────────────────────
    p_vis = sub.add_parser("visual", help="Prepare and infer over RagBench visual (table/image)")
    p_vis.add_argument("--sources", default=",".join(DEFAULT_VISUAL_SOURCES))
    p_vis.add_argument("--n-papers", type=int, default=25)
    p_vis.add_argument("--max-q", type=int, default=5)
    p_vis.add_argument("--only-doc", default=None)
    p_vis.add_argument("--skip-download", action="store_true")
    p_vis.add_argument("--force-reindex", action="store_true")
    p_vis.add_argument("--verbose", action="store_true")
    p_vis.add_argument("--docs-dir", default=str(RAGBENCH_VISUAL_PDFS_DIR))
    p_vis.add_argument(
        "--prepared-dir",
        default=str(RAGBENCH_VISUAL_PREPARED_DIR),
        help="Directory for the prepared visual dataset and manifest.",
    )
    p_vis.add_argument("--output-dir", default=str(VISUAL_RUN_DIR))
    p_vis.add_argument("--excluded-doc-ids", default=RAGBENCH_DEV_DOC_IDS_PATH)
    p_vis.set_defaults(func=ejecutar_visual)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        raise SystemExit(1)
    args.func(args)


if __name__ == "__main__":
    main()
