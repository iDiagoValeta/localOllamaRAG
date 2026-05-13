"""Benchmark multiple NVIDIA NIM LLM judges for RAGAS evaluation across es/ca/en.

Samples rows from 5 existing inference checkpoints (es, ca, ragbench_dev,
ragbench_test, ragbench_visual), filters truncated answers, runs RAGAS with
each candidate judge N times, and reports mean score / inter-run stability /
NaN rate per judge, metric, and language.

Candidate judges (from web research, May 2025):
  TIER 1:
    meta/llama-3.3-70b-instruct               - RAGAS-recommended, strong multilingual
    nvidia/llama-3.3-nemotron-super-49b-v1.5  - NVIDIA specialist, best instruction-following
    nvidia/llama-3.1-nemotron-ultra-253b-v1   - highest quality, slower
  TIER 2:
    mistralai/mistral-medium-3.5-128b         - current pipeline default
    mistralai/mistral-small-4-119b-2603       - speed/cost reference

Embedding candidates:
    nvidia/llama-3.2-nv-embedqa-1b-v2         - current default (26 langs, no CA officially)
    baai/bge-m3                               - 100+ langs; pass via --embedding-model if available

Usage:
    python judge_benchmark.py --dry-run
    python judge_benchmark.py --judges meta/llama-3.3-70b-instruct --stability-runs 1 --n-per-dataset 3
    python judge_benchmark.py --stability-runs 2 --n-per-dataset 10
    python judge_benchmark.py --reuse-benchmark runs/ragas_nvidia_judge_benchmark/sampled_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any

# ─── MODULE MAP ───────────────────────────────────────────────────────────────
# SECTION 1: REPO DETECTION & IMPORTS
# SECTION 2: CONFIGURATION
# SECTION 3: CHECKPOINT UTILITIES
# SECTION 4: TRUNCATION FILTER & SAMPLER
# SECTION 5: NVIDIA JUDGE BUILDER
# SECTION 6: EVALUATION LOOP
# SECTION 7: STATISTICS
# SECTION 8: REPORTING
# SECTION 9: CLI & MAIN
# ──────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────
# SECTION 1: REPO DETECTION & IMPORTS
# ─────────────────────────────────────────────

def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "research" / "evaluation" / "run_eval.py").is_file():
            return candidate
    raise RuntimeError(
        "Cannot find repository root. "
        "Expected research/evaluation/run_eval.py in a parent directory."
    )


ROOT = _find_repo_root(Path(__file__).resolve().parent)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.evaluation import run_eval  # noqa: E402

try:
    import pandas as pd
except ImportError:
    print("pandas required: pip install pandas")
    raise SystemExit(1)

try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    Embeddings = object  # type: ignore[misc,assignment]


# ─────────────────────────────────────────────
# SECTION 2: CONFIGURATION
# ─────────────────────────────────────────────

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_SOURCE_ROOT = ROOT / "research" / "evaluation" / "runs" / "ragas"
DEFAULT_OUTPUT_ROOT = ROOT / "research" / "evaluation" / "runs" / "ragas_nvidia_judge_benchmark"
DEFAULT_EMBEDDING_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
DEFAULT_STABILITY_RUNS = 2
DEFAULT_N_PER_DATASET = 10
DEFAULT_SEED = 42
DEFAULT_MIN_ANSWER_CHARS = 40
DEFAULT_MAX_TOKENS = 16384
DEFAULT_RATE_LIMIT = 40
DEFAULT_RATE_CHECK = 0.25
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3

JUDGE_CANDIDATES = [
    "meta/llama-3.3-70b-instruct",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "mistralai/mistral-medium-3.5-128b",
    "mistralai/mistral-small-4-119b-2603",
]

BENCHMARK_SOURCES = [
    (
        "es",
        "comparisons/todas_ablacion/checkpoints/baseline_all_on.json",
        "es",
    ),
    (
        "ca",
        "comparisons/todas_ablacion_ca_ca/checkpoints/baseline_all_on.json",
        "ca",
    ),
    (
        "ragbench_dev",
        "comparisons/ragbench_ablation_en_dev10_frozen/checkpoints/baseline_all_on.json",
        "en",
    ),
    (
        "ragbench_test",
        "ragbench/en_eval/dataset_ragbench_en_eval_text_25p_5q_eval/checkpoint.json",
        "en",
    ),
    (
        "ragbench_visual",
        "ragbench_visual/inference/image_table_25p_5q/checkpoint.json",
        "en",
    ),
]

_SENTENCE_ENDERS = frozenset(".!?:;»)")


# ─────────────────────────────────────────────
# SECTION 3: CHECKPOINT UTILITIES
# ─────────────────────────────────────────────

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
        raise ValueError(f"Could not decode {path}: {last_error}")
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _resolve_dataset_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_file():
        return candidate.resolve()
    text = str(raw).replace("\\", "/")
    for marker in ("/evaluation/datasets/", "/research/evaluation/datasets/"):
        if marker in text:
            suffix = text.split(marker, 1)[1]
            mapped = ROOT / "research" / "evaluation" / "datasets" / suffix
            if mapped.is_file():
                return mapped.resolve()
    if text.endswith("dataset_ragbench_text_10p_5q.json"):
        mapped = (
            ROOT / "research" / "evaluation" / "datasets"
            / "ragbench" / "prepared" / "dev_frozen"
            / "dataset_ragbench_text_10p_5q_dev10_frozen.json"
        )
        if mapped.is_file():
            return mapped.resolve()
    name = Path(text).name
    matches = sorted((ROOT / "research" / "evaluation" / "datasets").rglob(name))
    return matches[0].resolve() if matches else None


def _coerce_contexts(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(c) for c in raw]
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return [raw] if raw.strip() else []
        if isinstance(decoded, list):
            return [str(c) for c in decoded]
    return []


def _cargar_generacion(ckpt_path: Path) -> dict[str, Any]:
    payload = _load_json(ckpt_path)

    if isinstance(payload.get("rows"), list) and "answers" not in payload:
        rows = [r for r in payload["rows"] if isinstance(r, dict)]
        return {
            "questions": [str(r.get("question", "")) for r in rows],
            "answers": [str(r.get("answer", "")) for r in rows],
            "ground_truths": [str(r.get("ground_truth", "")) for r in rows],
            "contexts_list": [_coerce_contexts(r.get("contexts", [])) for r in rows],
        }

    dataset_path = _resolve_dataset_path(str(payload.get("dataset_path") or ""))
    if dataset_path is None:
        raise FileNotFoundError(
            f"Cannot resolve dataset_path in {ckpt_path}: {payload.get('dataset_path')!r}"
        )

    df = run_eval.normalizar_columnas(run_eval.cargar_dataset(str(dataset_path)))
    questions = [str(q) for q in df["question"].tolist()]
    ground_truths = [str(gt) for gt in df["ground_truth"].tolist()]
    n = len(questions)

    raw_answers = payload.get("answers") or []
    raw_contexts = payload.get("contexts_list") or []
    answers = [str(a or "") for a in raw_answers[:n]]
    contexts_list = [_coerce_contexts(c) for c in raw_contexts[:n]]
    while len(answers) < n:
        answers.append("")
    while len(contexts_list) < n:
        contexts_list.append([])

    return {
        "questions": questions,
        "answers": answers,
        "ground_truths": ground_truths,
        "contexts_list": contexts_list,
    }


# ─────────────────────────────────────────────
# SECTION 4: TRUNCATION FILTER & SAMPLER
# ─────────────────────────────────────────────

def _is_valid_answer(answer: str, min_chars: int) -> bool:
    text = answer.strip()
    return len(text) >= min_chars and bool(text) and text[-1] in _SENTENCE_ENDERS


def build_benchmark(
    source_root: Path,
    n_per_dataset: int,
    seed: int,
    min_chars: int,
) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    rng = random.Random(seed)

    for source_id, ckpt_rel, language in BENCHMARK_SOURCES:
        ckpt_path = source_root / ckpt_rel
        if not ckpt_path.is_file():
            print(f"  [warn] checkpoint not found, skipping: {ckpt_path}")
            continue

        gen = _cargar_generacion(ckpt_path)
        n_total = len(gen["questions"])

        valid: list[dict[str, Any]] = []
        for i in range(n_total):
            answer = gen["answers"][i]
            question = gen["questions"][i]
            contexts = gen["contexts_list"][i]
            ground_truth = gen["ground_truths"][i]
            if not _is_valid_answer(answer, min_chars):
                continue
            if not question.strip() or not contexts:
                continue
            valid.append({
                "source_id": source_id,
                "language": language,
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            })

        n_filtered = n_total - len(valid)
        sampled = rng.sample(valid, min(n_per_dataset, len(valid)))
        all_rows.extend(sampled)
        print(f"  {source_id:<18}  valid={len(valid)}/{n_total}  sampled={len(sampled)}  filtered_truncated={n_filtered}")

    return all_rows


# ─────────────────────────────────────────────
# SECTION 5: NVIDIA JUDGE BUILDER
# ─────────────────────────────────────────────

class NvidiaEmbeddings(Embeddings):
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


def _build_judge_configurator(
    model: str,
    base_url: str,
    embedding_model: str,
    max_tokens: int,
    rate_limit: int,
    rate_check: float,
    timeout: int,
    max_retries: int,
):
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
            from ragas.llms.base import LangchainLLMWrapper
        except ImportError as err:
            print(f"Missing dependency: {err}")
            print("pip install langchain-openai ragas")
            raise SystemExit(1) from err

        class NvidiaChatOpenAI(ChatOpenAI):
            def _get_request_payload(self, input_, *, stop=None, **kwargs):
                payload = super()._get_request_payload(input_, stop=stop, **kwargs)
                if "max_completion_tokens" in payload:
                    payload["max_tokens"] = payload.pop("max_completion_tokens")
                payload.pop("n", None)
                return payload

        limiter = InMemoryRateLimiter(
            requests_per_second=max(rate_limit, 1) / 60.0,
            check_every_n_seconds=rate_check,
            max_bucket_size=1,
        )

        raw_llm = NvidiaChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            top_p=1.0,
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,
            rate_limiter=limiter,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="LangchainLLMWrapper is deprecated.*",
                category=DeprecationWarning,
            )
            eval_llm = LangchainLLMWrapper(raw_llm, bypass_n=True)

        eval_embeddings = None
        if embedding_model.lower() != "none":
            eval_embeddings = NvidiaEmbeddings(
                api_key=api_key,
                base_url=base_url,
                model=embedding_model,
                timeout=timeout,
                max_retries=max_retries,
                rate_limiter=limiter,
                query_input_type="query",
                document_input_type="passage",
            )

        print(f"  Judge LLM:  {model}")
        print(f"  Embeddings: {embedding_model if eval_embeddings else 'disabled'}")
        return eval_llm, eval_embeddings

    return configurar_llm_nvidia


# ─────────────────────────────────────────────
# SECTION 6: EVALUATION LOOP
# ─────────────────────────────────────────────

def _judge_tag(model: str) -> str:
    return model.replace("/", "_").replace(".", "-")


def _annotate_metadata(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.read_csv(csv_path)
    for i in range(min(len(rows), len(df))):
        df.at[i, "source_id"] = rows[i]["source_id"]
        df.at[i, "language"] = rows[i]["language"]
    df.to_csv(csv_path, index=False, encoding="utf-8")


def run_benchmark(
    benchmark_rows: list[dict[str, Any]],
    judges: list[str],
    stability_runs: int,
    embedding_model: str,
    output_root: Path,
    base_url: str,
    max_tokens: int,
    rate_limit: int,
    rate_check: float,
    timeout: int,
    max_retries: int,
    ragas_workers: int,
    ragas_batch: int,
    ragas_timeout: int,
    ragas_max_retries: int,
    ragas_max_wait: int,
    overwrite: bool,
    dry_run: bool,
    save_debug: bool,
) -> list[dict[str, Any]]:
    questions = [r["question"] for r in benchmark_rows]
    answers = [r["answer"] for r in benchmark_rows]
    contexts_list = [r["contexts"] for r in benchmark_rows]
    ground_truths = [r["ground_truth"] for r in benchmark_rows]
    tiene_gt = any(bool(gt) for gt in ground_truths)

    all_results: list[dict[str, Any]] = []

    for judge_model in judges:
        judge_tag = _judge_tag(judge_model)
        for run_idx in range(stability_runs):
            output_dir = output_root / judge_tag / f"run_{run_idx}"
            output_csv = output_dir / "scores.csv"
            debug_json = output_dir / "debug.json"

            print(f"\n{'=' * 80}")
            print(f"Judge: {judge_model}  |  run {run_idx + 1}/{stability_runs}")
            print(f"{'=' * 80}")

            if output_csv.exists() and not overwrite:
                print(f"[skip] already exists: {output_csv}")
                df = pd.read_csv(output_csv)
                all_results.append({"judge": judge_model, "run": run_idx, "df": df})
                continue

            if dry_run:
                print(f"[dry-run] Would evaluate {len(questions)} rows with {judge_model}")
                continue

            output_dir.mkdir(parents=True, exist_ok=True)

            generation: dict[str, Any] = {
                "questions": questions,
                "ground_truths": ground_truths,
                "answers": answers,
                "contexts_list": contexts_list,
                "output_path": str(output_csv),
                "debug_path": str(debug_json) if save_debug else None,
                "questions_count": len(questions),
                "tiene_ground_truth": tiene_gt,
                "eval_corpus": "judge_benchmark",
                "checkpoint_path": str(output_root / "sampled_benchmark.json"),
                "indexed_fragments": 0,
                "recomp_enabled": True,
                "pipeline_flags": {},
                "docs_dir": None,
                "pipeline_seconds": 0.0,
            }

            run_eval.configurar_llm_evaluacion = _build_judge_configurator(
                model=judge_model,
                base_url=base_url,
                embedding_model=embedding_model,
                max_tokens=max_tokens,
                rate_limit=rate_limit,
                rate_check=rate_check,
                timeout=timeout,
                max_retries=max_retries,
            )

            run_eval.evaluar_respuestas_con_ragas(
                generation=generation,
                save_debug=save_debug,
                ragas_timeout=ragas_timeout,
                ragas_max_retries=ragas_max_retries,
                ragas_max_wait=ragas_max_wait,
                ragas_max_workers=ragas_workers,
                ragas_batch_size=ragas_batch,
            )

            if output_csv.exists():
                _annotate_metadata(output_csv, benchmark_rows)
                df = pd.read_csv(output_csv)
                all_results.append({"judge": judge_model, "run": run_idx, "df": df})

    return all_results


# ─────────────────────────────────────────────
# SECTION 7: STATISTICS
# ─────────────────────────────────────────────

def compute_statistics(
    all_results: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for run_info in all_results:
        judge = run_info["judge"]
        run_idx = run_info["run"]
        df = run_info["df"]
        for i, row in df.iterrows():
            lang = (
                row["language"]
                if "language" in df.columns and pd.notna(row.get("language"))
                else (benchmark_rows[int(i)]["language"] if int(i) < len(benchmark_rows) else "?")
            )
            src = (
                row["source_id"]
                if "source_id" in df.columns and pd.notna(row.get("source_id"))
                else (benchmark_rows[int(i)]["source_id"] if int(i) < len(benchmark_rows) else "?")
            )
            for metric in run_eval.METRIC_NAMES:
                if metric in df.columns:
                    records.append({
                        "judge": judge,
                        "run": run_idx,
                        "row": int(i),
                        "metric": metric,
                        "score": row[metric],
                        "source_id": src,
                        "language": lang,
                    })

    if not records:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    raw = pd.DataFrame(records)

    global_rows: list[dict[str, Any]] = []
    for (judge, metric), grp in raw.groupby(["judge", "metric"]):
        vals = grp["score"].dropna()
        global_rows.append({
            "judge": judge,
            "metric": metric,
            "mean": vals.mean() if len(vals) else float("nan"),
            "std": vals.std(ddof=0) if len(vals) > 1 else float("nan"),
            "nan_rate": grp["score"].isna().mean(),
            "n": len(grp),
        })
    global_stats = pd.DataFrame(global_rows)

    stability_rows: list[dict[str, Any]] = []
    for (judge, metric), grp in raw.groupby(["judge", "metric"]):
        per_row_std = grp.groupby("row")["score"].std(ddof=0)
        stability_rows.append({
            "judge": judge,
            "metric": metric,
            "mean_inter_run_std": per_row_std.mean(),
        })
    stability_stats = pd.DataFrame(stability_rows)

    lang_rows: list[dict[str, Any]] = []
    for (judge, language, metric), grp in raw.groupby(["judge", "language", "metric"]):
        vals = grp["score"].dropna()
        lang_rows.append({
            "judge": judge,
            "language": language,
            "metric": metric,
            "mean": vals.mean() if len(vals) else float("nan"),
            "nan_rate": grp["score"].isna().mean(),
        })
    lang_stats = pd.DataFrame(lang_rows)

    return global_stats, stability_stats, lang_stats


# ─────────────────────────────────────────────
# SECTION 8: REPORTING
# ─────────────────────────────────────────────

def _short_judge(model: str) -> str:
    parts = model.split("/")
    return parts[-1][:38] if len(parts) > 1 else model[:38]


def print_global_table(
    global_stats: pd.DataFrame,
    stability_stats: pd.DataFrame,
    judges: list[str],
) -> None:
    metrics = [m for m in run_eval.METRIC_NAMES if m in global_stats["metric"].values]
    if not metrics:
        print("No results to display.")
        return

    sep = "─"
    w = 14
    header = f"{'Judge':<40}"
    for m in metrics:
        header += f"  {m[:w]:>{w}}"
    header += f"  {'stab':>6}  {'nan%':>5}  {'overall':>8}"
    width = len(header)

    print(f"\n{'=' * width}")
    print("  JUDGE BENCHMARK — GLOBAL  (mean ± inter-run-std across all rows)")
    print(f"{'=' * width}")
    print(header)
    print(sep * width)

    stab_map: dict[tuple[str, str], float] = {}
    if not stability_stats.empty:
        for _, r in stability_stats.iterrows():
            stab_map[(r["judge"], r["metric"])] = r["mean_inter_run_std"]

    for judge_model in judges:
        subset = global_stats[global_stats["judge"] == judge_model]
        if subset.empty:
            continue
        row_str = f"{_short_judge(judge_model):<40}"
        means: list[float] = []
        nan_rates: list[float] = []
        stab_vals: list[float] = []
        for m in metrics:
            m_row = subset[subset["metric"] == m]
            if m_row.empty:
                row_str += f"  {'N/A':>{w}}"
                continue
            mean_val = float(m_row.iloc[0]["mean"])
            std_val = float(m_row.iloc[0]["std"])
            nan_val = float(m_row.iloc[0]["nan_rate"])
            stab = stab_map.get((judge_model, m), float("nan"))
            if not pd.isna(stab):
                stab_vals.append(stab)
            if pd.isna(mean_val):
                row_str += f"  {'N/A':>{w}}"
            else:
                entry = f"{mean_val:.3f}±{std_val:.3f}" if not pd.isna(std_val) else f"{mean_val:.3f}"
                row_str += f"  {entry:>{w}}"
                means.append(mean_val)
                nan_rates.append(nan_val)
        overall_nan = sum(nan_rates) / len(nan_rates) if nan_rates else float("nan")
        overall_mean = sum(means) / len(means) if means else float("nan")
        overall_stab = sum(stab_vals) / len(stab_vals) if stab_vals else float("nan")
        stab_str = f"{overall_stab:.3f}" if not pd.isna(overall_stab) else " N/A"
        row_str += f"  {stab_str:>6}  {overall_nan * 100:>4.1f}%  {overall_mean:>8.4f}"
        print(row_str)

    print("=" * width)
    print("  stab = mean inter-run std (lower = more stable between repeated evaluations)")


def print_language_table(lang_stats: pd.DataFrame, judges: list[str]) -> None:
    if lang_stats.empty:
        return
    languages = sorted(lang_stats["language"].dropna().unique())
    metrics = [m for m in run_eval.METRIC_NAMES if m in lang_stats["metric"].values]
    if not metrics or not languages:
        return

    print(f"\n{'=' * 80}")
    print("  BY LANGUAGE — mean across all metrics per judge")
    print(f"{'=' * 80}")
    lang_header = f"{'Judge':<40}"
    for lang in languages:
        lang_header += f"  {lang:>8}"
    print(lang_header)
    print("─" * 80)

    for judge_model in judges:
        row_str = f"{_short_judge(judge_model):<40}"
        for lang in languages:
            subset = lang_stats[(lang_stats["judge"] == judge_model) & (lang_stats["language"] == lang)]
            if subset.empty:
                row_str += f"  {'N/A':>8}"
            else:
                vals = [float(r["mean"]) for _, r in subset.iterrows() if not pd.isna(r.get("mean"))]
                overall = sum(vals) / len(vals) if vals else float("nan")
                row_str += f"  {overall:.4f}" if not pd.isna(overall) else f"  {'N/A':>8}"
        print(row_str)
    print("=" * 80)


def save_results(
    output_root: Path,
    benchmark_rows: list[dict[str, Any]],
    global_stats: pd.DataFrame,
    stability_stats: pd.DataFrame,
    lang_stats: pd.DataFrame,
    judges: list[str],
    embedding_model: str,
    stability_runs: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    benchmark_path = output_root / "sampled_benchmark.json"
    with benchmark_path.open("w", encoding="utf-8") as f:
        json.dump({"rows": benchmark_rows, "n": len(benchmark_rows)}, f, ensure_ascii=False, indent=2)
    print(f"\nBenchmark dataset:  {benchmark_path}")

    if not global_stats.empty:
        comparison_csv = output_root / "judge_comparison.csv"
        global_stats.to_csv(comparison_csv, index=False, encoding="utf-8")
        print(f"Global stats CSV:   {comparison_csv}")

        merged = global_stats.merge(stability_stats, on=["judge", "metric"], how="left") if not stability_stats.empty else global_stats.copy()

        comparison_json = output_root / "judge_comparison.json"
        with comparison_json.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "judges": judges,
                    "embedding_model": embedding_model,
                    "stability_runs": stability_runs,
                    "benchmark_n": len(benchmark_rows),
                    "global_stats": merged.to_dict(orient="records"),
                    "lang_stats": lang_stats.to_dict(orient="records") if not lang_stats.empty else [],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Full JSON results:  {comparison_json}")


# ─────────────────────────────────────────────
# SECTION 9: CLI & MAIN
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark NVIDIA NIM LLM judges for RAGAS evaluation (es/ca/en).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=JUDGE_CANDIDATES,
        metavar="MODEL",
        help="One or more judge model IDs.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--n-per-dataset", type=int, default=DEFAULT_N_PER_DATASET)
    parser.add_argument("--stability-runs", type=int, default=DEFAULT_STABILITY_RUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-answer-chars", type=int, default=DEFAULT_MIN_ANSWER_CHARS)
    parser.add_argument(
        "--reuse-benchmark",
        metavar="PATH",
        help="Load an existing sampled_benchmark.json to skip resampling.",
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--rate-limit-per-minute", type=int, default=DEFAULT_RATE_LIMIT)
    parser.add_argument("--rate-check-seconds", type=float, default=DEFAULT_RATE_CHECK)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--ragas-workers", type=int, default=3)
    parser.add_argument("--ragas-batch", type=int, default=3)
    parser.add_argument("--ragas-timeout", type=int, default=600)
    parser.add_argument("--ragas-max-retries", type=int, default=5)
    parser.add_argument("--ragas-max-wait", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show sampling preview without evaluating.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.output_root = Path(args.output_root).resolve()
    args.source_root = Path(args.source_root).resolve()

    print(f"\nSource root: {args.source_root}")
    print(f"Output root: {args.output_root}")
    print(f"Embedding:   {args.embedding_model}")
    print(f"Judges ({len(args.judges)}):")
    for j in args.judges:
        print(f"  - {j}")
    print(f"Stability:   {args.stability_runs} run(s) per judge")

    if args.reuse_benchmark:
        reuse_path = Path(args.reuse_benchmark).resolve()
        with reuse_path.open(encoding="utf-8") as f:
            data = json.load(f)
        benchmark_rows = data["rows"]
        print(f"\nReusing benchmark: {reuse_path}  ({len(benchmark_rows)} rows)")
    else:
        print("\nBuilding benchmark dataset...")
        benchmark_rows = build_benchmark(
            source_root=args.source_root,
            n_per_dataset=args.n_per_dataset,
            seed=args.seed,
            min_chars=args.min_answer_chars,
        )
        print(f"\nTotal benchmark rows: {len(benchmark_rows)}")

    if not benchmark_rows:
        print("Error: no valid rows found. Check --source-root.")
        return 1

    if args.dry_run:
        print("\nSample preview (first 5 rows):")
        for i, row in enumerate(benchmark_rows[:5]):
            print(f"\n  [{i}] [{row['source_id']}/{row['language']}]")
            print(f"       Q: {row['question'][:100]}")
            print(f"       A: {row['answer'][:100]}")
        if len(benchmark_rows) > 5:
            print(f"\n  ... ({len(benchmark_rows)} total rows)")
        print("\n[dry-run] Skipping evaluation.")
        return 0

    all_results = run_benchmark(
        benchmark_rows=benchmark_rows,
        judges=args.judges,
        stability_runs=args.stability_runs,
        embedding_model=args.embedding_model,
        output_root=args.output_root,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        rate_limit=args.rate_limit_per_minute,
        rate_check=args.rate_check_seconds,
        timeout=args.timeout,
        max_retries=args.max_retries,
        ragas_workers=args.ragas_workers,
        ragas_batch=args.ragas_batch,
        ragas_timeout=args.ragas_timeout,
        ragas_max_retries=args.ragas_max_retries,
        ragas_max_wait=args.ragas_max_wait,
        overwrite=args.overwrite,
        dry_run=False,
        save_debug=not args.no_debug,
    )

    if all_results:
        global_stats, stability_stats, lang_stats = compute_statistics(all_results, benchmark_rows)
        print_global_table(global_stats, stability_stats, args.judges)
        print_language_table(lang_stats, args.judges)
        save_results(
            output_root=args.output_root,
            benchmark_rows=benchmark_rows,
            global_stats=global_stats,
            stability_stats=stability_stats,
            lang_stats=lang_stats,
            judges=args.judges,
            embedding_model=args.embedding_model,
            stability_runs=args.stability_runs,
        )
    else:
        print("\nNo evaluation results collected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
