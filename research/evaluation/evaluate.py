"""MonkeyGrab RAGAS evaluation CLI (reads from checkpoint, multi-provider).

Ejecuta RAGAS sobre los checkpoints generados por ``infer.py`` o por runs
previas, sin volver a generar respuestas. Soporta tres backends de juez:

    --provider google   Gemini 2.5 Flash + gemini-embedding-001 (GOOGLE_API_KEY)
    --provider aws      AWS Bedrock (langchain-aws, Anthropic/Titan)
    --provider nvidia   NVIDIA NIM via OpenAI-compatible API (NVIDIA_API_KEY)

Ejemplos:
    python evaluate.py --provider google --source-root runs/ragas/comparisons/mi_label
    python evaluate.py --provider aws --checkpoint path/to/checkpoint.json --dry-run
    python evaluate.py --provider nvidia --all-known --rate-limit-per-minute 40

# ─────────────────────────────────────────────
# MODULE MAP
# ─────────────────────────────────────────────
#
#  1. Environment setup
#  2. Provider routing + output path resolution
#  3. Per-checkpoint evaluation loop
#  4. CLI parser and dispatch
#
# ─────────────────────────────────────────────
"""

from __future__ import annotations

# ─────────────────────────────────────────────
# SECTION 1: ENVIRONMENT SETUP
# ─────────────────────────────────────────────

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

_this_file = Path(__file__).resolve()
_proj_root = _this_file.parent.parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

try:
    from dotenv import load_dotenv
    load_dotenv(_proj_root / ".env")
except ImportError:
    pass

from research.evaluation._lib.aggregation import (
    SUPPORTED_GROUP_BY,
    aggregate_comparison_run,
)
from research.evaluation._lib.checkpoints import (
    apply_limit,
    failed_score_indexes,
    generation_from_checkpoint,
    load_json_resilient,
    merge_retry_scores,
    subset_generation,
)
from research.evaluation._lib.datasets import RAGAS_RUNS_DIR
from research.evaluation._lib.ragas_runner import (
    METRIC_NAMES,
    evaluar_respuestas_con_ragas,
)
from research.evaluation._lib.providers import aws as aws_provider
from research.evaluation._lib.providers import nvidia as nvidia_provider
from research.evaluation._lib.providers.google import configurar_llm_evaluacion_google


DEFAULT_SOURCE_ROOT = Path(RAGAS_RUNS_DIR)
DEFAULT_OUTPUT_ROOT_BY_PROVIDER = {
    "google":  Path(RAGAS_RUNS_DIR).parent / "ragas_google_revaluation",
    "aws":     Path(RAGAS_RUNS_DIR).parent / "ragas_aws_revaluation",
    "nvidia":  Path(RAGAS_RUNS_DIR).parent / "ragas_nvidia_revaluation",
}


# ─────────────────────────────────────────────
# SECTION 2: PROVIDER ROUTING + OUTPUT PATHS
# ─────────────────────────────────────────────

def _build_configurator(args: argparse.Namespace) -> Callable:
    """Resolve ``--provider`` to a RAGAS LLM/embeddings configurator."""
    if args.provider == "google":
        return configurar_llm_evaluacion_google
    if args.provider == "aws":
        return aws_provider.build_aws_configurator(args)
    if args.provider == "nvidia":
        return nvidia_provider.build_nvidia_configurator(args)
    raise SystemExit(f"Unknown provider: {args.provider}")


def _output_paths_for(source_path: Path, output_root: Path) -> tuple[Path, Path, Path]:
    """Mirror a checkpoint's relative path under ``output_root``."""
    try:
        rel = source_path.resolve().relative_to(DEFAULT_SOURCE_ROOT.resolve())
    except ValueError:
        rel = Path(source_path.stem)

    parts = list(rel.parts)
    if "checkpoints" in parts:
        parts.remove("checkpoints")
    if parts and parts[-1].endswith(".json"):
        parts[-1] = Path(parts[-1]).stem
    tag_dir = output_root.joinpath(*parts)
    return tag_dir, tag_dir / "scores.csv", tag_dir / "debug.json"


def _discover_known_inputs(source_root: Path) -> list[Path]:
    """Discover all checkpoint JSON files under the standard ragas runs root."""
    candidates: set[Path] = set()
    for pattern in (
        "comparisons/*/checkpoints/*.json",
        "single/**/checkpoint*.json",
        "ragbench/**/checkpoint.json",
        "ragbench_visual/**/checkpoint.json",
        "ragbench_visual/**/results.json",
    ):
        candidates.update(path.resolve() for path in source_root.glob(pattern) if path.is_file())
    return sorted(candidates)


def _expand_input_path(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(child.resolve() for child in path.glob("*.json") if child.is_file())
    return [path.resolve()]


# ─────────────────────────────────────────────
# SECTION 3: PER-CHECKPOINT EVALUATION LOOP
# ─────────────────────────────────────────────

def evaluate_one(
    source_path: Path,
    args: argparse.Namespace,
    configurator: Callable,
) -> dict[str, Any] | None:
    """Evaluate one checkpoint and return its summary dict (or None when skipped)."""
    source_path = source_path.resolve()
    payload = load_json_resilient(source_path)
    generation = generation_from_checkpoint(payload, source_path)
    generation = apply_limit(generation, args.limit)

    run_dir, output_csv, debug_json = _output_paths_for(source_path, args.output_root)
    generation["output_path"] = str(output_csv.resolve())
    generation["debug_path"] = str(debug_json.resolve()) if args.save_debug else None

    retry_indexes: list[int] = []
    original_output_csv = output_csv
    original_debug_json = debug_json
    if args.retry_failed:
        if not output_csv.exists():
            raise FileNotFoundError(f"--retry-failed requires an existing scores CSV: {output_csv}")
        retry_indexes = failed_score_indexes(output_csv, METRIC_NAMES)
        if not retry_indexes:
            print(f"[skip] no NaN metric cells found: {output_csv}")
            return None
        generation = subset_generation(generation, retry_indexes)
        output_csv = output_csv.with_name(f"{output_csv.stem}.retry_failed.csv")
        debug_json = debug_json.with_name(f"{debug_json.stem}.retry_failed.json")
        generation["output_path"] = str(output_csv.resolve())
        generation["debug_path"] = str(debug_json.resolve()) if args.save_debug else None

    if output_csv.exists() and not args.overwrite and not args.retry_failed:
        print(f"[skip] exists: {output_csv}")
        return None

    print("\n" + "=" * 80)
    print(f"Input:  {source_path}")
    print(f"Rows:   {generation['questions_count']}")
    print(f"Output: {output_csv}")
    if args.save_debug:
        print(f"Debug:  {debug_json}")
    if args.retry_failed:
        listed = ", ".join(str(i + 1) for i in retry_indexes[:20])
        suffix = "..." if len(retry_indexes) > 20 else ""
        print(f"Retry failed rows: {listed}{suffix}")
    print("=" * 80)

    if args.dry_run:
        return {
            "checkpoint_path": str(source_path),
            "output_path": str(output_csv),
            "questions_count": generation["questions_count"],
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    result = evaluar_respuestas_con_ragas(
        generation=generation,
        save_debug=args.save_debug,
        ragas_timeout=args.ragas_timeout,
        ragas_max_retries=args.ragas_max_retries,
        ragas_max_wait=args.ragas_max_wait,
        ragas_max_workers=args.ragas_max_workers,
        ragas_batch_size=args.ragas_batch_size,
        ragas_metrics=args.metrics,
        google_timeout=args.google_timeout,
        google_retries=args.google_retries,
        raise_exceptions=args.raise_exceptions,
        llm_configurator=configurator,
    )
    if args.retry_failed:
        merge_summary = merge_retry_scores(original_output_csv, output_csv, retry_indexes, METRIC_NAMES)
        result["output_path"] = str(original_output_csv.resolve())
        result["retry_output_path"] = str(output_csv.resolve())
        result["retry_debug_path"] = str(debug_json.resolve()) if args.save_debug else None
        result["retry_failed_rows"] = [idx + 1 for idx in retry_indexes]
        result["retry_merge_summary"] = merge_summary
        print(f"\nMerged retry scores into: {original_output_csv}")
        print(
            f"Retry recovery: {len(merge_summary['recovered_rows'])}/{len(retry_indexes)} row(s) recovered; "
            f"remaining NaN metric cells={merge_summary['remaining_nan_cells']}"
        )
        if args.save_debug and original_debug_json.exists():
            print(f"Original debug remains at: {original_debug_json}")
    return result


# ─────────────────────────────────────────────
# SECTION 4: CLI PARSER AND DISPATCH
# ─────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RAGAS on stored RAG checkpoints (Google / AWS / NVIDIA).",
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=("google", "aws", "nvidia"),
        help="RAGAS LLM/embeddings backend.",
    )
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="Checkpoint JSON or visual results JSON. Repeatable.")
    parser.add_argument("--all-known", action="store_true",
                        help="Evaluate every known checkpoint under --source-root.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Defaults to runs/ragas_<provider>_revaluation/.")

    # Common RAGAS knobs
    parser.add_argument("--metrics", default=None, help="Comma-separated RAGAS metrics, or 'all'.")
    parser.add_argument("--ragas-timeout", type=int, default=600)
    parser.add_argument("--ragas-max-retries", type=int, default=5)
    parser.add_argument("--ragas-max-wait", type=int, default=120)
    parser.add_argument("--ragas-max-workers", type=int, default=3)
    parser.add_argument("--ragas-batch-size", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows.")
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--raise-exceptions", action="store_true")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-evaluate only rows with NaN metric cells in an existing scores.csv.")
    parser.add_argument("--dry-run", action="store_true")

    # Aggregation (subset summary over comparison runs)
    agg = parser.add_argument_group("aggregation")
    agg.add_argument(
        "--aggregate-group-by",
        default="source_type",
        help=(
            "Comma-separated subset keys for the per-subset aggregation written "
            "next to each comparison run. Choices: "
            + ",".join(SUPPORTED_GROUP_BY)
            + ". Default: source_type."
        ),
    )
    agg.add_argument(
        "--aggregate-etiquetas-es",
        action="store_true",
        help="Use Spanish RAGAS metric labels in the aggregation reports.",
    )
    agg.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip the automatic per-subset aggregation step.",
    )

    # Google-only
    google = parser.add_argument_group("google")
    google.add_argument("--google-timeout", type=int, default=None)
    google.add_argument("--google-retries", type=int, default=None)

    # AWS-only
    aws_group = parser.add_argument_group("aws")
    aws_provider.add_aws_args(aws_group)

    # NVIDIA-only
    nvidia_group = parser.add_argument_group("nvidia")
    nvidia_provider.add_nvidia_args(nvidia_group)

    return parser


def _resolve_provider_defaults(args: argparse.Namespace) -> None:
    """Fill provider-dependent defaults that depend on ``--provider``."""
    if args.output_root is None:
        args.output_root = DEFAULT_OUTPUT_ROOT_BY_PROVIDER[args.provider]
    args.output_root = Path(args.output_root).resolve()
    args.source_root = Path(args.source_root).resolve()
    args.save_debug = not args.no_debug


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    _resolve_provider_defaults(args)

    inputs: list[Path] = []
    for path in args.checkpoint:
        inputs.extend(_expand_input_path(Path(path).resolve()))
    if args.all_known:
        inputs.extend(_discover_known_inputs(args.source_root))

    unique_inputs: list[Path] = []
    seen: set[Path] = set()
    for path in inputs:
        if path in seen:
            continue
        seen.add(path)
        unique_inputs.append(path)

    if args.max_checkpoints is not None:
        unique_inputs = unique_inputs[: args.max_checkpoints]

    if not unique_inputs:
        parser.error("Pass --checkpoint PATH or --all-known.")

    missing = [str(path) for path in unique_inputs if not path.is_file()]
    if missing:
        raise SystemExit("Missing input file(s):\n  " + "\n  ".join(missing))

    configurator = _build_configurator(args)

    results: list[dict[str, Any]] = []
    for source_path in unique_inputs:
        result = evaluate_one(source_path, args, configurator)
        if result is not None:
            results.append(result)

    summary_name = f"{args.provider}_ragas_summary.json"
    summary_path = args.output_root / summary_name
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        summary_payload: dict[str, Any] = {
            "provider": args.provider,
            "ragas_max_workers": args.ragas_max_workers,
            "ragas_batch_size": args.ragas_batch_size,
            "results": results,
        }
        if args.provider == "aws":
            summary_payload.update({
                "model": args.aws_model,
                "embedding_model": args.aws_embedding_model,
                "region": args.aws_region,
                "profile": args.aws_profile,
                "max_tokens": args.aws_max_tokens,
            })
        elif args.provider == "nvidia":
            summary_payload.update({
                "model": args.nvidia_model,
                "embedding_model": args.nvidia_embedding_model,
                "base_url": args.nvidia_base_url,
                "max_tokens": args.nvidia_max_tokens,
                "reasoning_effort": nvidia_provider.resolve_reasoning_effort(args),
                "rate_limit_per_minute": args.nvidia_rate_limit_per_minute,
            })
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_payload, f, ensure_ascii=False, indent=2)
        print(f"\nSummary saved to: {summary_path}")

        if not args.no_aggregate:
            _run_aggregation_step(results, args)
    else:
        print(f"\nDry run completed. Inputs discovered: {len(unique_inputs)}")

    return 0


def _run_aggregation_step(results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Group results by comparison label and aggregate subset means per variant.

    Only checkpoints under ``.../comparisons/<label>/checkpoints/<variant>.json``
    are eligible — those are the multi-variant ablation runs where per-subset
    aggregation is meaningful. Single / RagBench / visual runs are skipped.
    """
    group_by_raw = (args.aggregate_group_by or "").strip()
    if not group_by_raw:
        return
    group_by_list = [g.strip() for g in group_by_raw.split(",") if g.strip()]
    invalid = [g for g in group_by_list if g not in SUPPORTED_GROUP_BY]
    if invalid:
        print(
            "  [warn] aggregation skipped: unsupported --aggregate-group-by "
            f"value(s): {', '.join(invalid)}. Valid: {', '.join(SUPPORTED_GROUP_BY)}"
        )
        return

    # group by (label_dir, dataset_path); each label_dir maps to one ablation run.
    groups: dict[Path, dict[str, Any]] = {}
    for res in results:
        ckpt = Path(res.get("checkpoint_path") or "")
        debug_path = res.get("debug_path")
        dataset_path = res.get("dataset_path")
        if not debug_path or not dataset_path or not ckpt:
            continue
        parts = ckpt.resolve().parts
        if "comparisons" not in parts:
            continue
        try:
            idx = parts.index("comparisons")
        except ValueError:
            continue
        if idx + 2 >= len(parts):
            continue
        label_dir = Path(*parts[: idx + 2])  # .../comparisons/<label>
        variant_name = Path(ckpt).stem
        entry = groups.setdefault(label_dir, {
            "dataset_path": dataset_path,
            "variants": [],
        })
        entry["variants"].append((variant_name, Path(debug_path)))

    if not groups:
        return

    print("\n" + "=" * 70)
    print(f"Per-subset aggregation (group_by={','.join(group_by_list)})")
    print("=" * 70)

    for label_dir, info in groups.items():
        # Mirror the comparison label under output_root: output_root/comparisons/<label>/aggregates/
        try:
            rel = label_dir.relative_to(args.source_root)
        except ValueError:
            rel = Path("comparisons") / label_dir.name
        out_dir = args.output_root / rel / "aggregates"
        print(f"\n  Comparison: {label_dir.name}")
        print(f"  Variants:   {len(info['variants'])}")
        print(f"  Output dir: {out_dir}")
        try:
            aggregate_comparison_run(
                variant_debug_paths=info["variants"],
                dataset_path=Path(info["dataset_path"]),
                out_dir=out_dir,
                group_by_list=group_by_list,
                etiquetas_es=args.aggregate_etiquetas_es,
                write_csv_too=True,
            )
        except FileNotFoundError as exc:
            print(f"  [warn] aggregation skipped for {label_dir.name}: {exc}")
        except Exception as exc:
            print(f"  [warn] aggregation failed for {label_dir.name}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
