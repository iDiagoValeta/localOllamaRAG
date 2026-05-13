"""
Re-evaluate stored RAG checkpoints with RAGAS using Amazon Bedrock.

This script does not regenerate RAG answers. It reads existing checkpoint JSON files
that contain model answers and retrieved contexts, rebuilds the RAGAS input payload,
and writes fresh RAGAS scores/debug artifacts under:

    research/evaluation/runs/ragas_aws_revaluation/

Example:
    python eval_ragas_aws_from_checkpoints.py --all-known --dry-run
    python eval_ragas_aws_from_checkpoints.py --checkpoint path/to/checkpoint.json --limit 5

Required:
    pip install langchain-aws boto3

AWS credentials via any of:
    - Environment: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
    - ~/.aws/credentials (aws configure)
    - --profile <name>

Bedrock model access must be enabled in the AWS console (Bedrock → Model access).

Candidate LLM judges:
    anthropic.claude-3-5-haiku-20241022-v1:0   (default) fast, cheap, best JSON following
    anthropic.claude-3-5-sonnet-20241022-v2:0  highest quality
    amazon.nova-lite-v1:0                      cheapest AWS option
    amazon.nova-pro-v1:0                       good quality, moderate cost
    meta.llama3-3-70b-instruct-v1:0            same model as NVIDIA (for cross-provider comparison)

Candidate embedding models:
    amazon.titan-embed-text-v2:0   (default) 100+ languages, 8K context, Catalan likely works
    cohere.embed-multilingual-v3   best multilingual on Bedrock, Catalan likely works
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "research" / "evaluation" / "run_eval.py").is_file():
            return candidate
    raise RuntimeError(
        "Could not find repository root from script location. "
        "Expected research/evaluation/run_eval.py in a parent directory."
    )


ROOT = _find_repo_root(Path(__file__).resolve().parent)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.evaluation import run_eval  # noqa: E402


DEFAULT_CHAT_MODEL = "anthropic.claude-3-5-haiku-20241022-v1:0"
DEFAULT_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
DEFAULT_REGION = "us-east-1"
DEFAULT_OUTPUT_ROOT = ROOT / "research" / "evaluation" / "runs" / "ragas_aws_revaluation"
DEFAULT_SOURCE_ROOT = ROOT / "research" / "evaluation" / "runs" / "ragas"
DEFAULT_MAX_TOKENS = 4096


def _load_json(path: Path) -> dict[str, Any]:
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            with path.open(encoding=encoding) as f:
                payload = json.load(f)
            break
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            last_error = exc
    else:
        raise UnicodeDecodeError(
            "utf-8",
            b"",
            0,
            1,
            f"Could not decode {path} as utf-8, utf-8-sig, or cp1252: {last_error}",
        )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _resolve_existing_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None

    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate.resolve()

    text = str(raw_path).replace("\\", "/")
    markers = [
        "/research/evaluation/datasets/",
        "/evaluation/datasets/",
    ]
    for marker in markers:
        if marker in text:
            suffix = text.split(marker, 1)[1]
            mapped = ROOT / "research" / "evaluation" / "datasets" / suffix
            if mapped.is_file():
                return mapped.resolve()

    if text.endswith("dataset_ragbench_text_10p_5q.json"):
        mapped = (
            ROOT
            / "research"
            / "evaluation"
            / "datasets"
            / "ragbench"
            / "prepared"
            / "dev_frozen"
            / "dataset_ragbench_text_10p_5q_dev10_frozen.json"
        )
        if mapped.is_file():
            return mapped.resolve()

    name = Path(text).name
    matches = sorted((ROOT / "research" / "evaluation" / "datasets").rglob(name))
    if matches:
        return matches[0].resolve()

    return None


def _coerce_contexts(raw_contexts: Any) -> list[str]:
    if isinstance(raw_contexts, list):
        return [str(ctx) for ctx in raw_contexts]
    if isinstance(raw_contexts, str):
        try:
            decoded = json.loads(raw_contexts)
        except json.JSONDecodeError:
            return [raw_contexts] if raw_contexts.strip() else []
        if isinstance(decoded, list):
            return [str(ctx) for ctx in decoded]
    return []


def _generation_from_visual_results(payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Visual inference JSON has no rows: {source_path}")

    questions: list[str] = []
    ground_truths: list[str] = []
    answers: list[str] = []
    contexts_list: list[list[str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        questions.append(str(row.get("question", "")))
        ground_truths.append(str(row.get("ground_truth", "")))
        answers.append(str(row.get("answer", "")))
        contexts_list.append(_coerce_contexts(row.get("contexts", [])))

    generation_meta = payload.get("generation", {})
    manifest = payload.get("manifest", {})
    if not isinstance(generation_meta, dict):
        generation_meta = {}
    if not isinstance(manifest, dict):
        manifest = {}

    dataset_path = _resolve_existing_path(
        generation_meta.get("dataset_path") or manifest.get("dataset_path")
    )
    checkpoint_path = _resolve_existing_path(generation_meta.get("checkpoint_path")) or source_path

    return {
        "dataset_path": str(dataset_path or ""),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "questions": questions,
        "ground_truths": ground_truths,
        "answers": answers,
        "contexts_list": contexts_list,
        "question_statuses": payload.get("question_statuses", []),
        "questions_count": len(questions),
        "indexed_fragments": int(generation_meta.get("indexed_fragments") or 0),
        "recomp_enabled": bool(generation_meta.get("recomp_enabled", True)),
        "pipeline_flags": generation_meta.get("pipeline_flags") or {},
        "eval_corpus": generation_meta.get("eval_corpus") or "ragbench",
        "docs_dir": generation_meta.get("docs_dir") or manifest.get("docs_dir"),
        "pipeline_seconds": float(generation_meta.get("pipeline_seconds") or 0.0),
        "tiene_ground_truth": any(bool(gt) for gt in ground_truths),
    }


def _generation_from_checkpoint(payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    answers = payload.get("answers")
    contexts_list = payload.get("contexts_list")
    if not isinstance(answers, list) or not isinstance(contexts_list, list):
        if isinstance(payload.get("rows"), list):
            return _generation_from_visual_results(payload, source_path)
        raise ValueError(f"Checkpoint does not contain answers/contexts_list: {source_path}")

    dataset_path = _resolve_existing_path(str(payload.get("dataset_path") or ""))
    if dataset_path is None:
        raise FileNotFoundError(
            f"Could not resolve dataset_path from {source_path}: {payload.get('dataset_path')!r}"
        )

    df = run_eval.normalizar_columnas(run_eval.cargar_dataset(str(dataset_path)))
    questions = [str(q) for q in df["question"].tolist()]
    ground_truths = [str(gt) for gt in df["ground_truth"].tolist()]
    count = len(questions)

    normalized_answers = [str(answer or "") for answer in answers[:count]]
    normalized_contexts = [_coerce_contexts(ctxs) for ctxs in contexts_list[:count]]
    while len(normalized_answers) < count:
        normalized_answers.append("")
    while len(normalized_contexts) < count:
        normalized_contexts.append([])

    return {
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(source_path.resolve()),
        "questions": questions,
        "ground_truths": ground_truths,
        "answers": normalized_answers,
        "contexts_list": normalized_contexts,
        "question_statuses": payload.get("question_statuses", []),
        "questions_count": count,
        "indexed_fragments": int(payload.get("indexed_fragments") or 0),
        "recomp_enabled": bool(payload.get("recomp_enabled", True)),
        "pipeline_flags": payload.get("pipeline_flags") or {},
        "eval_corpus": payload.get("eval_corpus") or "unknown",
        "docs_dir": payload.get("docs_dir"),
        "pipeline_seconds": float(payload.get("pipeline_seconds") or 0.0),
        "tiene_ground_truth": any(bool(gt) for gt in ground_truths),
    }


def _output_paths_for(source_path: Path, output_root: Path) -> tuple[Path, Path, Path]:
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
    candidates: set[Path] = set()
    for pattern in (
        "comparisons/*/checkpoints/*.json",
        "ragbench/**/checkpoint.json",
        "ragbench_visual/**/checkpoint.json",
    ):
        candidates.update(path.resolve() for path in source_root.glob(pattern) if path.is_file())

    return sorted(
        candidates,
        key=lambda path: str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
    )


def _expand_input_path(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(child.resolve() for child in path.glob("*.json") if child.is_file())
    return [path.resolve()]


def _metric_columns_in_csv(df: Any) -> list[str]:
    return [name for name in run_eval.METRIC_NAMES if name in df.columns]


def _failed_score_indexes(scores_csv: Path) -> list[int]:
    import pandas as pd

    df = pd.read_csv(scores_csv)
    metric_cols = _metric_columns_in_csv(df)
    if not metric_cols:
        raise ValueError(f"No RAGAS metric columns found in {scores_csv}")
    mask = df[metric_cols].isna().any(axis=1)
    return [int(idx) for idx in df.index[mask].tolist()]


def _subset_generation(generation: dict[str, Any], indexes: list[int]) -> dict[str, Any]:
    subset = dict(generation)
    for key in ("questions", "ground_truths", "answers", "contexts_list", "question_statuses"):
        value = generation.get(key)
        if isinstance(value, list):
            subset[key] = [value[i] for i in indexes if i < len(value)]
    subset["questions_count"] = len(indexes)
    subset["tiene_ground_truth"] = any(bool(gt) for gt in subset.get("ground_truths", []))
    return subset


def _merge_retry_scores(original_csv: Path, retry_csv: Path, retry_indexes: list[int]) -> dict[str, Any]:
    import pandas as pd

    original_df = pd.read_csv(original_csv)
    retry_df = pd.read_csv(retry_csv)
    metric_cols = _metric_columns_in_csv(original_df)
    recovered: list[dict[str, Any]] = []

    for retry_row_idx, original_row_idx in enumerate(retry_indexes):
        if retry_row_idx >= len(retry_df) or original_row_idx >= len(original_df):
            continue
        recovered_metrics = []
        for metric_name in metric_cols:
            original_value = original_df.at[original_row_idx, metric_name]
            retry_value = retry_df.at[retry_row_idx, metric_name]
            if pd.isna(original_value) and not pd.isna(retry_value):
                original_df.at[original_row_idx, metric_name] = retry_value
                recovered_metrics.append(metric_name)
        if recovered_metrics:
            recovered.append({"row": original_row_idx + 1, "metrics": recovered_metrics})

    original_df.to_csv(original_csv, index=False, encoding="utf-8")
    remaining_nan = int(original_df[metric_cols].isna().sum().sum()) if metric_cols else 0
    return {"recovered_rows": recovered, "remaining_nan_cells": remaining_nan}


def _build_aws_configurator(args: argparse.Namespace):
    def configurar_llm_aws(
        google_timeout: int | None = None,
        google_retries: int | None = None,
    ):
        try:
            from langchain_aws import BedrockEmbeddings, ChatBedrock
            from ragas.llms.base import LangchainLLMWrapper
        except ImportError as err:
            print(f"Error: {err}")
            print("Install with: pip install langchain-aws boto3")
            raise SystemExit(1) from err

        if args.profile:
            import boto3
            boto3.setup_default_session(
                region_name=args.region,
                profile_name=args.profile,
            )

        raw_eval_llm = ChatBedrock(
            model_id=args.model,
            region_name=args.region,
            model_kwargs={
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            },
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="LangchainLLMWrapper is deprecated.*",
                category=DeprecationWarning,
            )
            eval_llm = LangchainLLMWrapper(raw_eval_llm, bypass_n=True)

        eval_embeddings = None
        if args.embedding_model.lower() != "none":
            eval_embeddings = BedrockEmbeddings(
                model_id=args.embedding_model,
                region_name=args.region,
            )

        print(f"Evaluation LLM:        AWS Bedrock {args.model}")
        print(
            "Evaluation embeddings: "
            + ("disabled" if eval_embeddings is None else f"AWS Bedrock {args.embedding_model}")
        )
        print(f"Region: {args.region}" + (f"  profile: {args.profile}" if args.profile else ""))
        print(
            "RAGAS throughput config: "
            f"workers={args.ragas_max_workers}, "
            f"batch_size={args.ragas_batch_size or 'auto'}"
        )
        return eval_llm, eval_embeddings

    return configurar_llm_aws


def _apply_limit(generation: dict[str, Any], limit: int | None) -> dict[str, Any]:
    if not limit:
        return generation
    limited = dict(generation)
    for key in ("questions", "ground_truths", "answers", "contexts_list", "question_statuses"):
        value = limited.get(key)
        if isinstance(value, list):
            limited[key] = value[:limit]
    limited["questions_count"] = min(int(limited.get("questions_count") or 0), limit)
    limited["tiene_ground_truth"] = any(bool(gt) for gt in limited.get("ground_truths", []))
    return limited


def evaluate_one(source_path: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    source_path = source_path.resolve()
    payload = _load_json(source_path)
    generation = _generation_from_checkpoint(payload, source_path)
    generation = _apply_limit(generation, args.limit)

    run_dir, output_csv, debug_json = _output_paths_for(source_path, args.output_root)
    generation["output_path"] = str(output_csv.resolve())
    generation["debug_path"] = str(debug_json.resolve()) if args.save_debug else None

    retry_indexes: list[int] = []
    original_output_csv = output_csv
    original_debug_json = debug_json
    if args.retry_failed:
        if not output_csv.exists():
            raise FileNotFoundError(f"--retry-failed requires an existing scores CSV: {output_csv}")
        retry_indexes = _failed_score_indexes(output_csv)
        if not retry_indexes:
            print(f"[skip] no NaN metric cells found: {output_csv}")
            return None
        generation = _subset_generation(generation, retry_indexes)
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
        retry_rows = ", ".join(str(i + 1) for i in retry_indexes[:20])
        retry_suffix = "..." if len(retry_indexes) > 20 else ""
        print(f"Retry failed rows: {retry_rows}{retry_suffix}")
    print("=" * 80)

    if args.dry_run:
        return {
            "checkpoint_path": str(source_path),
            "output_path": str(output_csv),
            "questions_count": generation["questions_count"],
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    result = run_eval.evaluar_respuestas_con_ragas(
        generation=generation,
        save_debug=args.save_debug,
        ragas_timeout=args.ragas_timeout,
        ragas_max_retries=args.ragas_max_retries,
        ragas_max_wait=args.ragas_max_wait,
        ragas_max_workers=args.ragas_max_workers,
        ragas_batch_size=args.ragas_batch_size,
        ragas_metrics=args.metrics,
        raise_exceptions=args.raise_exceptions,
    )
    if args.retry_failed:
        merge_summary = _merge_retry_scores(original_output_csv, output_csv, retry_indexes)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RAGAS on stored checkpoints using Amazon Bedrock."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Checkpoint JSON or visual results JSON. Can be passed more than once.",
    )
    parser.add_argument(
        "--all-known",
        action="store_true",
        help="Evaluate all known checkpoint JSON files under research/evaluation/runs/ragas.",
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--profile", default=None, help="AWS credentials profile name.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--metrics", default=None, help="Comma-separated RAGAS metrics, or all.")
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
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-evaluate only rows with NaN metric cells in an existing scores.csv.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.source_root = args.source_root.resolve()
    args.save_debug = not args.no_debug

    inputs: list[Path] = []
    for path in args.checkpoint:
        inputs.extend(_expand_input_path(Path(path).resolve()))
    if args.all_known:
        inputs.extend(_discover_known_inputs(args.source_root))

    unique_inputs = []
    seen = set()
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

    run_eval.configurar_llm_evaluacion = _build_aws_configurator(args)

    results: list[dict[str, Any]] = []
    for source_path in unique_inputs:
        result = evaluate_one(source_path, args)
        if result is not None:
            results.append(result)

    summary_path = args.output_root / "aws_ragas_summary.json"
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": args.model,
                    "embedding_model": args.embedding_model,
                    "region": args.region,
                    "profile": args.profile,
                    "max_tokens": args.max_tokens,
                    "ragas_max_workers": args.ragas_max_workers,
                    "ragas_batch_size": args.ragas_batch_size,
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nSummary saved to: {summary_path}")
    else:
        print(f"\nDry run completed. Inputs discovered: {len(unique_inputs)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
