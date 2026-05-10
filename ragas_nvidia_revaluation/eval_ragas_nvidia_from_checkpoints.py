"""
Re-evaluate stored RAG checkpoints with RAGAS using NVIDIA's OpenAI-compatible API.

This script does not regenerate RAG answers. It reads existing checkpoint JSON files
that contain model answers and retrieved contexts, rebuilds the RAGAS input payload,
and writes fresh RAGAS scores/debug artifacts under:

    research/evaluation/runs/ragas_nvidia/

Example:
    python eval_ragas_nvidia_from_checkpoints.py --all-known --dry-run
    python eval_ragas_nvidia_from_checkpoints.py --checkpoint path/to/checkpoint.json --limit 5

Required environment:
    NVIDIA_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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

try:
    from langchain_core.embeddings import Embeddings
except ImportError:  # pragma: no cover - dependency is checked again before real execution.
    Embeddings = object  # type: ignore[misc,assignment]


DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_CHAT_MODEL = "openai/gpt-oss-120b"
DEFAULT_EMBEDDING_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
DEFAULT_OUTPUT_ROOT = ROOT / "research" / "evaluation" / "runs" / "ragas_nvidia_revaluation"
DEFAULT_SOURCE_ROOT = ROOT / "research" / "evaluation" / "runs" / "ragas"


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


class NvidiaEmbeddings(Embeddings):
    """NVIDIA embedding wrapper with required input_type for asymmetric models."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int,
        max_retries: int,
        rate_limiter: Any,
        query_input_type: str,
        document_input_type: str,
    ):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model
        self.rate_limiter = rate_limiter
        self.query_input_type = query_input_type
        self.document_input_type = document_input_type

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        self.rate_limiter.acquire(blocking=True)
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            extra_body={"input_type": input_type},
        )
        return [item.embedding for item in response.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, self.document_input_type)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], self.query_input_type)[0]


def _build_nvidia_configurator(args: argparse.Namespace):
    def configurar_llm_nvidia(
        google_timeout: int | None = None,
        google_retries: int | None = None,
    ):
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            print("NVIDIA_API_KEY not found in environment.")
            raise SystemExit(1)

        try:
            from langchain_core.rate_limiters import InMemoryRateLimiter
            from langchain_openai import ChatOpenAI
        except ImportError as err:
            print(f"Error: {err}")
            print("Install with: pip install langchain-openai")
            raise SystemExit(1) from err

        requests_per_second = max(args.rate_limit_per_minute, 1) / 60.0
        limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=args.rate_check_seconds,
            max_bucket_size=1,
        )

        eval_llm = ChatOpenAI(
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            max_retries=args.max_retries,
            max_completion_tokens=args.max_tokens,
            rate_limiter=limiter,
        )

        eval_embeddings = None
        if args.embedding_model.lower() != "none":
            eval_embeddings = NvidiaEmbeddings(
                api_key=api_key,
                base_url=args.base_url,
                model=args.embedding_model,
                timeout=args.timeout,
                max_retries=args.max_retries,
                rate_limiter=limiter,
                query_input_type=args.embedding_query_input_type,
                document_input_type=args.embedding_document_input_type,
            )

        print(f"Evaluation LLM: NVIDIA {args.model}")
        print(
            "Evaluation embeddings: "
            + ("disabled" if eval_embeddings is None else f"NVIDIA {args.embedding_model}")
        )
        print(f"NVIDIA base_url: {args.base_url}")
        print(f"Shared API rate limit: {args.rate_limit_per_minute} calls/minute")
        return eval_llm, eval_embeddings

    return configurar_llm_nvidia


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

    if output_csv.exists() and not args.overwrite:
        print(f"[skip] exists: {output_csv}")
        return None

    print("\n" + "=" * 80)
    print(f"Input:  {source_path}")
    print(f"Rows:   {generation['questions_count']}")
    print(f"Output: {output_csv}")
    if args.save_debug:
        print(f"Debug:  {debug_json}")
    print("=" * 80)

    if args.dry_run:
        return {
            "checkpoint_path": str(source_path),
            "output_path": str(output_csv),
            "questions_count": generation["questions_count"],
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_eval.evaluar_respuestas_con_ragas(
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RAGAS on stored checkpoints using NVIDIA's OpenAI-compatible API."
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
    parser.add_argument("--embedding-query-input-type", default="query")
    parser.add_argument("--embedding-document-input-type", default="passage")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--rate-limit-per-minute", type=int, default=40)
    parser.add_argument("--rate-check-seconds", type=float, default=0.25)
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

    run_eval.configurar_llm_evaluacion = _build_nvidia_configurator(args)

    results: list[dict[str, Any]] = []
    for source_path in unique_inputs:
        result = evaluate_one(source_path, args)
        if result is not None:
            results.append(result)

    summary_path = args.output_root / "nvidia_ragas_summary.json"
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": args.model,
                    "embedding_model": args.embedding_model,
                    "base_url": args.base_url,
                    "rate_limit_per_minute": args.rate_limit_per_minute,
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
