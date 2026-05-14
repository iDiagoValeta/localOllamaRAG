"""RAGAS execution + result rendering + debug JSON output.

# ─────────────────────────────────────────────
# MODULE MAP
# ─────────────────────────────────────────────
#
#  1. Metric constants
#  2. RAGAS lazy import + metric parsing
#  3. Console rendering + NaN summary + debug JSON
#  4. Trace extraction for justifications
#  5. evaluar_respuestas_con_ragas (accepts injected llm/embeddings)
#
# ─────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

import pandas as pd

from .providers.google import configurar_llm_evaluacion_google


# ─────────────────────────────────────────────
# SECTION 1: METRIC CONSTANTS
# ─────────────────────────────────────────────

METRIC_NAMES = [
    "answer_correctness",  # Factual Correctness
    "faithfulness",        # Faithfulness
    "answer_relevancy",    # Response Relevancy
    "context_precision",   # Context Precision
    "context_recall",      # Context Recall
]

METRIC_DISPLAY_NAMES = {
    "answer_correctness": "Factual Correctness",
    "faithfulness":       "Faithfulness",
    "answer_relevancy":   "Response Relevancy",
    "context_precision":  "Context Precision",
    "context_recall":     "Context Recall",
}

METRIC_DESCRIPTIONS = {
    "answer_correctness": "Factual precision vs ground truth (TP/FP/FN, F1)",
    "faithfulness":       "Factual consistency of the answer with the context",
    "answer_relevancy":   "Degree to which the answer addresses the question",
    "context_precision":  "Ranking precision of retrieved fragments",
    "context_recall":     "Coverage of required contexts",
}


# ─────────────────────────────────────────────
# SECTION 2: RAGAS LAZY IMPORT + METRIC PARSING
# ─────────────────────────────────────────────

def _importar_ragas_componentes() -> dict[str, Any]:
    """Import RAGAS lazily so generation-only flows do not require evaluator setup."""
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        )
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.run_config import RunConfig
    except ImportError as e:
        print("Install RAGAS and dependencies:")
        print("   pip install -r research/evaluation/requirements.txt")
        raise SystemExit(1) from e

    return {
        "evaluate": evaluate,
        "SingleTurnSample": SingleTurnSample,
        "EvaluationDataset": EvaluationDataset,
        "RunConfig": RunConfig,
        "metric_objects": {
            "answer_correctness": answer_correctness,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        },
    }


def parse_ragas_metric_names(
    metric_spec: str | None,
    tiene_ground_truth: bool,
) -> list[str]:
    """Resolve CLI metric names into the canonical RAGAS metric names."""
    default_names = [
        "answer_correctness",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ] if tiene_ground_truth else [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    if not metric_spec or metric_spec.strip().lower() == "all":
        return default_names

    aliases = {
        "answer_correctness": "answer_correctness",
        "factual_correctness": "answer_correctness",
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "response_relevancy": "answer_relevancy",
        "context_precision": "context_precision",
        "context_recall": "context_recall",
    }

    names: list[str] = []
    for raw_name in metric_spec.split(","):
        key = raw_name.strip().lower().replace("-", "_")
        if not key:
            continue
        if key not in aliases:
            valid = ", ".join(sorted(aliases))
            raise ValueError(f"Unknown RAGAS metric {raw_name!r}. Valid metrics: {valid}")
        name = aliases[key]
        if name == "answer_correctness" and not tiene_ground_truth:
            raise ValueError("answer_correctness requires ground truth in the dataset.")
        if name not in names:
            names.append(name)

    if not names:
        raise ValueError("No RAGAS metrics selected.")
    return names


def _es_google_resource_exhausted(exc: Exception) -> bool:
    """Detect Google GenAI quota/rate-limit failures through wrapped exceptions."""
    text = f"{type(exc).__name__}: {exc}"
    return "RESOURCE_EXHAUSTED" in text or "429" in text


# ─────────────────────────────────────────────
# SECTION 3: CONSOLE RENDERING + DEBUG JSON
# ─────────────────────────────────────────────

def imprimir_resultados(df_scores: pd.DataFrame, questions: list[str]) -> None:
    """Print detailed per-question scores and global averages to stdout."""
    metric_cols = [c for c in METRIC_NAMES if c in df_scores.columns]
    if not metric_cols:
        print("\nNo metric columns found in the results.")
        print(f"   Available columns: {list(df_scores.columns)}")
        return

    print("\n" + "=" * 70)
    print("  RAGAS RESULTS -- GLOBAL AVERAGES")
    print("=" * 70)

    medias = df_scores[metric_cols].mean(numeric_only=True).sort_values(ascending=False)
    for m, v in medias.items():
        desc = METRIC_DESCRIPTIONS.get(m, "")
        if pd.isna(v):
            print(f"  {m:25s}  {'N/A':>8s}   {desc}")
        else:
            print(f"  {m:25s}  {v:8.4f}   {desc}")

    media_global = medias.dropna().mean()
    if not pd.isna(media_global):
        print(f"\n  {'OVERALL MEAN SCORE':25s}  {media_global:8.4f}")

    print("\n" + "=" * 70)
    print("  PER-QUESTION DETAIL")
    print("=" * 70)

    for i, row in df_scores.iterrows():
        q = questions[i] if i < len(questions) else "?"
        q_short = q[:80] + "..." if len(q) > 80 else q
        scores_str = " | ".join(
            f"{c}: {row[c]:.3f}" if not pd.isna(row[c]) else f"{c}: N/A"
            for c in metric_cols
        )
        row_scores = [row[c] for c in metric_cols if not pd.isna(row[c])]
        media_q = sum(row_scores) / len(row_scores) if row_scores else float("nan")

        print(f"\n  [{i+1}] {q_short}")
        print(f"      {scores_str}")
        if not pd.isna(media_q):
            print(f"      Mean score: {media_q:.4f}")

    print("\n" + "=" * 70)


def resumen_nan_ragas(df: pd.DataFrame, metric_cols: list[str]) -> dict[str, Any]:
    """Build a compact summary of missing RAGAS metric cells."""
    rows: list[dict[str, Any]] = []
    by_metric: dict[str, int] = {}
    for metric_name in metric_cols:
        missing_count = int(df[metric_name].isna().sum())
        if missing_count:
            by_metric[metric_name] = missing_count

    if by_metric:
        for idx, row in df.iterrows():
            missing_metrics = [
                metric_name
                for metric_name in metric_cols
                if pd.isna(row.get(metric_name))
            ]
            if missing_metrics:
                rows.append({"index": int(idx) + 1, "missing_metrics": missing_metrics})

    return {
        "total_missing_cells": int(sum(by_metric.values())),
        "by_metric": by_metric,
        "rows": rows,
    }


# ─────────────────────────────────────────────
# SECTION 4: TRACE EXTRACTION
# ─────────────────────────────────────────────

def extraer_justificaciones_traces(traces: list, metric_cols: list) -> list[dict]:
    """Extract justification details from RAGAS evaluation traces."""
    justificaciones = []
    for trace in traces:
        justif = {}
        if not hasattr(trace, "__getitem__"):
            justificaciones.append(justif)
            continue
        for metric_name in metric_cols:
            if metric_name not in trace:
                continue
            metric_data = trace[metric_name]
            if isinstance(metric_data, dict):
                prompts = []
                for prompt_name, prompt_io in metric_data.items():
                    if isinstance(prompt_io, dict) and "output" in prompt_io:
                        out = prompt_io["output"]
                        if isinstance(out, dict):
                            prompts.append({"prompt": prompt_name, "output": out})
                        elif out is not None:
                            prompts.append({"prompt": prompt_name, "output": str(out)[:500]})
                if prompts:
                    justif[metric_name] = prompts
            elif metric_data is not None:
                justif[metric_name] = str(metric_data)[:500]
        justificaciones.append(justif)
    return justificaciones


def guardar_debug(
    result,
    questions: list,
    answers: list,
    ground_truths: list,
    contexts_list: list,
    debug_path: str,
) -> str:
    """Save a debug JSON file containing answers, contexts, scores, and justifications."""
    df = result.to_pandas()
    metric_cols = [c for c in METRIC_NAMES if c in df.columns]
    nan_summary = resumen_nan_ragas(df, metric_cols)

    traces = getattr(result, "traces", []) or []
    justificaciones = (
        extraer_justificaciones_traces(traces, metric_cols)
        if traces else [{}] * len(questions)
    )

    debug_entries = []
    for i in range(len(questions)):
        ctx_preview = []
        for ctx in (contexts_list[i] if i < len(contexts_list) else []):
            ctx_preview.append(ctx[:300] + "..." if len(ctx) > 300 else ctx)

        entry = {
            "index": i + 1,
            "question": questions[i],
            "model_answer": answers[i] if i < len(answers) else "",
            "ground_truth": ground_truths[i] if i < len(ground_truths) else "",
            "retrieved_contexts_preview": ctx_preview[:3],
            "contexts_count": len(contexts_list[i]) if i < len(contexts_list) else 0,
            "scores": {},
            "justifications": justificaciones[i] if i < len(justificaciones) else {},
        }
        for m in metric_cols:
            val = df.iloc[i][m] if i < len(df) else None
            entry["scores"][METRIC_DISPLAY_NAMES.get(m, m)] = (
                float(val) if val is not None and not pd.isna(val) else None
            )
        debug_entries.append(entry)

    debug_data = {
        "metrics_used": {
            METRIC_DISPLAY_NAMES.get(m, m): METRIC_DESCRIPTIONS.get(m, "")
            for m in metric_cols
        },
        "results": debug_entries,
        "global_averages": {
            METRIC_DISPLAY_NAMES.get(m, m): (
                float(df[m].mean()) if not pd.isna(df[m].mean()) else None
            )
            for m in metric_cols
        },
        "nan_summary": nan_summary,
    }

    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)
    return debug_path


# ─────────────────────────────────────────────
# SECTION 5: EVALUAR_RESPUESTAS_CON_RAGAS
# ─────────────────────────────────────────────

def evaluar_respuestas_con_ragas(
    generation: dict[str, Any],
    save_debug: bool = True,
    ragas_timeout: int = 90,
    ragas_max_retries: int = 5,
    ragas_max_wait: int = 60,
    ragas_max_workers: int = 1,
    ragas_batch_size: int | None = 5,
    ragas_metrics: str | None = None,
    google_timeout: int | None = None,
    google_retries: int | None = None,
    raise_exceptions: bool = False,
    llm_configurator: Callable | None = None,
) -> dict[str, Any]:
    """Evaluate previously generated RAG answers with RAGAS and save artifacts.

    Args:
        generation: Dict produced by ``generar_respuestas_rag`` or
            ``generation_from_checkpoint`` (questions, answers, contexts, etc.).
        llm_configurator: Optional callable ``(google_timeout, google_retries) ->
            (eval_llm, eval_embeddings)``. Defaults to the Gemini judge configurator.
            Providers (AWS, NVIDIA) pass their own configurator here.
    """
    ragas = _importar_ragas_componentes()
    configurator = llm_configurator or configurar_llm_evaluacion_google
    eval_llm, eval_embeddings = configurator(
        google_timeout=google_timeout,
        google_retries=google_retries,
    )

    questions = generation["questions"]
    ground_truths = generation["ground_truths"]
    answers = generation["answers"]
    contexts_list = generation["contexts_list"]

    print("\nBuilding EvaluationDataset for RAGAS...")
    samples = []
    for i in range(len(questions)):
        sample = ragas["SingleTurnSample"](
            user_input=questions[i],
            response=answers[i] if answers[i] else "",
            retrieved_contexts=contexts_list[i] if contexts_list[i] else [],
            reference=ground_truths[i] if ground_truths[i] else "",
        )
        samples.append(sample)
    eval_dataset = ragas["EvaluationDataset"](samples=samples)

    selected_metric_names = parse_ragas_metric_names(
        ragas_metrics,
        bool(generation.get("tiene_ground_truth")),
    )
    metrics = [ragas["metric_objects"][name] for name in selected_metric_names]

    print("\nRunning RAGAS evaluation (this may take a few minutes)...")
    print(
        "   RAGAS config: "
        f"timeout={ragas_timeout}s, retries={ragas_max_retries}, "
        f"max_wait={ragas_max_wait}s, workers={ragas_max_workers}, "
        f"batch_size={ragas_batch_size or 'auto'}, "
        f"metrics={','.join(selected_metric_names)}"
    )
    t_eval_start = time.time()

    eval_run_config = ragas["RunConfig"](
        timeout=ragas_timeout,
        max_retries=ragas_max_retries,
        max_wait=ragas_max_wait,
        max_workers=ragas_max_workers,
    )

    try:
        result = ragas["evaluate"](
            dataset=eval_dataset,
            metrics=metrics,
            llm=eval_llm,
            embeddings=eval_embeddings,
            run_config=eval_run_config,
            batch_size=ragas_batch_size,
            raise_exceptions=raise_exceptions,
        )
    except Exception as e:
        if _es_google_resource_exhausted(e):
            print("\nError: Google returned 429 RESOURCE_EXHAUSTED during RAGAS.")
            print("   The RAG checkpoint is valid; this failed in the evaluator, not in retrieval.")
            print("   Rerun without --raise-exceptions and the defaults already limit to")
            print("   workers=1 and batch_size=5, which should avoid the rate limit.")
            print("   If it persists, try: --ragas-max-workers 1 --ragas-batch-size 1 --ragas-max-wait 120")
            raise SystemExit(2) from e
        raise

    t_eval = time.time() - t_eval_start
    print(f"   Evaluation completed in {t_eval:.1f}s")

    df_scores = result.to_pandas()
    metric_cols = [c for c in METRIC_NAMES if c in df_scores.columns]
    nan_summary = resumen_nan_ragas(df_scores, metric_cols)
    imprimir_resultados(df_scores, questions)
    if nan_summary["total_missing_cells"]:
        print(
            "   RAGAS missing metric cells: "
            f"{nan_summary['total_missing_cells']} "
            f"across {len(nan_summary['rows'])} row(s)."
        )
        print(
            "   Missing by metric: "
            + ", ".join(
                f"{metric}={count}"
                for metric, count in nan_summary["by_metric"].items()
            )
        )

    output_path = generation["output_path"]
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    df_scores.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nResults saved to: {output_path}")

    debug_path = generation.get("debug_path")
    if save_debug and debug_path:
        os.makedirs(os.path.dirname(os.path.abspath(debug_path)), exist_ok=True)
        guardar_debug(
            result=result,
            questions=questions,
            answers=answers,
            ground_truths=ground_truths,
            contexts_list=contexts_list,
            debug_path=debug_path,
        )
        print(f"Debug saved to: {debug_path}")
    else:
        debug_path = None

    mean_scores = {}
    for metric_name in metric_cols:
        metric_value = df_scores[metric_name].mean(numeric_only=True)
        if not pd.isna(metric_value):
            mean_scores[metric_name] = float(metric_value)

    return {
        "dataset_path": generation["dataset_path"],
        "output_path": os.path.abspath(output_path),
        "debug_path": os.path.abspath(debug_path) if debug_path else None,
        "questions_count": len(questions),
        "indexed_fragments": generation.get("indexed_fragments", 0),
        "recomp_enabled": generation.get("recomp_enabled", True),
        "pipeline_flags": generation.get("pipeline_flags", {}),
        "eval_corpus": generation.get("eval_corpus", "unknown"),
        "docs_dir": generation.get("docs_dir"),
        "pipeline_seconds": generation.get("pipeline_seconds", 0.0),
        "evaluation_seconds": t_eval,
        "mean_scores": mean_scores,
        "nan_summary": nan_summary,
        "checkpoint_path": os.path.abspath(generation.get("checkpoint_path", "")),
    }
