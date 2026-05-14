"""Checkpoint I/O, validation, recovery, and re-evaluation helpers.

# ─────────────────────────────────────────────
# MODULE MAP
# ─────────────────────────────────────────────
#
#  1. Empty/status helpers
#  2. Checkpoint persistence and validation
#  3. Model-signature tracking
#  4. Re-evaluation helpers (load checkpoint into a generation payload)
#  5. Retry-only RAGAS support (failed rows merge)
#
# ─────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .datasets import (
    DATASETS_DIR,
    cargar_dataset,
    normalizar_columnas,
    single_run_dir,
)
from .pipeline_flags import normalizar_pipeline_flags


# ─────────────────────────────────────────────
# SECTION 1: EMPTY/STATUS HELPERS
# ─────────────────────────────────────────────

def respuesta_vacia(respuesta: Any) -> bool:
    """Return True when a generated answer is missing or blank."""
    return not isinstance(respuesta, str) or not respuesta.strip()


def indices_respuestas_vacias(answers: list[str], total: int) -> list[int]:
    """Find zero-based question indexes that still need a non-empty answer."""
    return [
        i
        for i in range(total)
        if i >= len(answers) or respuesta_vacia(answers[i])
    ]


def estado_pregunta_base(index: int, answer: Any = "") -> dict[str, Any]:
    """Build a checkpoint status entry for one evaluation question."""
    status = "ok" if not respuesta_vacia(answer) else "pending"
    return {
        "index": index,
        "question_number": index + 1,
        "status": status,
        "attempts": 0,
        "duration_seconds": 0.0,
        "reason": None,
        "error": None,
        "updated_at": None,
    }


def normalizar_estados_preguntas(
    raw_statuses: Any,
    answers: list[str],
    total: int,
) -> list[dict[str, Any]]:
    """Normalize checkpoint question statuses, supporting old checkpoints."""
    if isinstance(raw_statuses, list):
        statuses: list[dict[str, Any]] = []
        for i in range(total):
            raw = raw_statuses[i] if i < len(raw_statuses) and isinstance(raw_statuses[i], dict) else {}
            base = estado_pregunta_base(i, answers[i] if i < len(answers) else "")
            base.update({k: v for k, v in raw.items() if k in base})
            base["index"] = i
            base["question_number"] = i + 1
            if not respuesta_vacia(answers[i] if i < len(answers) else ""):
                base["status"] = "ok"
                base["reason"] = None
                base["error"] = None
            statuses.append(base)
        return statuses

    return [
        estado_pregunta_base(i, answers[i] if i < len(answers) else "")
        for i in range(total)
    ]


def indices_pendientes_generacion(
    answers: list[str],
    question_statuses: list[dict[str, Any]],
    total: int,
) -> list[int]:
    """Return indexes that need generation or retry."""
    pending = set(indices_respuestas_vacias(answers, total))
    for i, status in enumerate(question_statuses[:total]):
        if status.get("status") not in ("ok", "skipped"):
            pending.add(i)
    return sorted(pending)


def resumen_estados_fallidos(
    answers: list[str],
    question_statuses: list[dict[str, Any]],
    total: int,
) -> dict[str, list[int]]:
    """Group incomplete question numbers by diagnostic status/reason."""
    grouped: dict[str, list[int]] = {}
    for i in indices_respuestas_vacias(answers, total):
        status = question_statuses[i] if i < len(question_statuses) else {}
        key = str(status.get("reason") or status.get("status") or "empty_answer")
        grouped.setdefault(key, []).append(i + 1)
    return grouped


# ─────────────────────────────────────────────
# SECTION 2: CHECKPOINT PERSISTENCE AND VALIDATION
# ─────────────────────────────────────────────

def default_checkpoint_path(
    dataset_path: str, recomp_enabled: bool | None, suffix: str = ""
) -> str:
    """Return the checkpoint path used to resume generation question by question."""
    recomp_tag = "recomp_on" if recomp_enabled else "recomp_off"
    return os.path.join(single_run_dir(dataset_path, suffix), f"checkpoint_{recomp_tag}.json")


def cargar_checkpoint(checkpoint_path: str) -> dict[str, Any] | None:
    """Load a checkpoint file if it exists and is valid JSON."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
    with open(checkpoint_path, encoding="utf-8") as f:
        return json.load(f)


def guardar_checkpoint(checkpoint_path: str, payload: dict[str, Any]) -> None:
    """Persist the current question-by-question evaluation state."""
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def checkpoint_pipeline_flags_match(
    checkpoint: dict[str, Any],
    current_flags: dict[str, bool],
) -> bool:
    """Validate checkpoint compatibility with current runtime pipeline flags."""
    stored_flags = checkpoint.get("pipeline_flags")
    if stored_flags is not None:
        return normalizar_pipeline_flags(stored_flags) == normalizar_pipeline_flags(current_flags)
    # Backward compatibility for old checkpoints created before full flag tracking.
    return checkpoint.get("recomp_enabled") == current_flags.get("USAR_RECOMP_SYNTHESIS")


# ─────────────────────────────────────────────
# SECTION 3: MODEL-SIGNATURE TRACKING
# ─────────────────────────────────────────────

TRACKED_MODEL_FIELDS = ("modelo_rag", "modelo_chat", "modelo_embedding", "modelo_recomp")


def current_models_signature() -> dict[str, str]:
    """Snapshot of model names that influence answers and must invalidate checkpoints."""
    import rag.chat_pdfs as rag_runtime
    return {
        "modelo_rag": str(getattr(rag_runtime, "MODELO_RAG", "") or ""),
        "modelo_chat": str(getattr(rag_runtime, "MODELO_CHAT", "") or ""),
        "modelo_embedding": str(getattr(rag_runtime, "MODELO_EMBEDDING", "") or ""),
        "modelo_recomp": str(getattr(rag_runtime, "MODELO_RECOMP", "") or ""),
    }


def checkpoint_models_match(
    checkpoint: dict[str, Any],
    current_models: dict[str, str],
) -> tuple[bool, str | None]:
    """Validate checkpoint compatibility with the current generator / embedder."""
    has_any = any(checkpoint.get(k) is not None for k in TRACKED_MODEL_FIELDS)
    if not has_any:
        return True, (
            "checkpoint sin información de modelos (anterior a la validación por modelo); "
            "si cambiaste de modelo generador desde la última corrida, bórralo y relanza."
        )
    differences: list[str] = []
    for key in TRACKED_MODEL_FIELDS:
        stored = checkpoint.get(key)
        current = current_models.get(key, "")
        if stored is not None and str(stored) != current:
            differences.append(f"{key}: {stored!r} -> {current!r}")
    if differences:
        return False, "modelo(s) cambiado(s): " + "; ".join(differences)
    return True, None


def guardar_checkpoint_evaluacion(
    checkpoint_path: str,
    *,
    dataset_path: str,
    questions_count: int,
    recomp_enabled: bool,
    eval_corpus: str,
    output_path: str,
    debug_path: str,
    answers: list[str],
    contexts_list: list[list[str]],
    question_statuses: list[dict[str, Any]] | None = None,
    pipeline_flags: dict[str, bool] | None = None,
    docs_dir: str | None = None,
    ragbench_reranker_low_score_fallback: bool = False,
) -> None:
    """Save generation progress with a consistent payload."""
    payload = {
        "dataset_path": dataset_path,
        "questions_count": questions_count,
        "recomp_enabled": recomp_enabled,
        "pipeline_flags": normalizar_pipeline_flags(pipeline_flags),
        "eval_corpus": eval_corpus,
        "output_path": output_path,
        "debug_path": debug_path,
        "docs_dir": docs_dir,
        "ragbench_reranker_low_score_fallback": ragbench_reranker_low_score_fallback,
        "completed_questions": len([a for a in answers if not respuesta_vacia(a)]),
        "answers": answers,
        "contexts_list": contexts_list,
        "question_statuses": question_statuses or normalizar_estados_preguntas(
            None, answers, questions_count
        ),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    payload.update(current_models_signature())
    guardar_checkpoint(checkpoint_path, payload)


# ─────────────────────────────────────────────
# SECTION 4: RE-EVALUATION HELPERS (LOAD CHECKPOINT INTO GENERATION PAYLOAD)
# ─────────────────────────────────────────────

def coerce_contexts(raw_contexts: Any) -> list[str]:
    """Normalize stored contexts into a flat list of strings."""
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


def load_json_resilient(path: Path) -> dict[str, Any]:
    """Load JSON tolerating utf-8-sig and cp1252 (legacy Windows outputs)."""
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


def _resolve_dataset_path(raw_path: str | None) -> Path | None:
    """Best-effort dataset path resolution for checkpoints with stale absolute paths."""
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate.resolve()

    text = str(raw_path).replace("\\", "/")
    markers = ["/research/evaluation/datasets/", "/evaluation/datasets/"]
    for marker in markers:
        if marker in text:
            suffix = text.split(marker, 1)[1]
            mapped = Path(DATASETS_DIR) / suffix
            if mapped.is_file():
                return mapped.resolve()

    if text.endswith("dataset_ragbench_text_10p_5q.json"):
        mapped = (
            Path(DATASETS_DIR)
            / "ragbench"
            / "prepared"
            / "dev_frozen"
            / "dataset_ragbench_text_10p_5q_dev10_frozen.json"
        )
        if mapped.is_file():
            return mapped.resolve()

    name = Path(text).name
    matches = sorted(Path(DATASETS_DIR).rglob(name))
    if matches:
        return matches[0].resolve()
    return None


def _generation_from_visual_results(payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    """Build a generation payload from a visual-inference results.json."""
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
        contexts_list.append(coerce_contexts(row.get("contexts", [])))

    generation_meta = payload.get("generation", {})
    manifest = payload.get("manifest", {})
    if not isinstance(generation_meta, dict):
        generation_meta = {}
    if not isinstance(manifest, dict):
        manifest = {}

    dataset_path = _resolve_dataset_path(
        generation_meta.get("dataset_path") or manifest.get("dataset_path")
    )
    checkpoint_path = _resolve_dataset_path(generation_meta.get("checkpoint_path")) or source_path

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


def generation_from_checkpoint(payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    """Rebuild the generation payload that ``evaluar_respuestas_con_ragas`` consumes."""
    answers = payload.get("answers")
    contexts_list = payload.get("contexts_list")
    if not isinstance(answers, list) or not isinstance(contexts_list, list):
        if isinstance(payload.get("rows"), list):
            return _generation_from_visual_results(payload, source_path)
        raise ValueError(f"Checkpoint does not contain answers/contexts_list: {source_path}")

    dataset_path = _resolve_dataset_path(str(payload.get("dataset_path") or ""))
    if dataset_path is None:
        raise FileNotFoundError(
            f"Could not resolve dataset_path from {source_path}: {payload.get('dataset_path')!r}"
        )

    df = normalizar_columnas(cargar_dataset(str(dataset_path)))
    questions = [str(q) for q in df["question"].tolist()]
    ground_truths = [str(gt) for gt in df["ground_truth"].tolist()]
    count = len(questions)

    normalized_answers = [str(answer or "") for answer in answers[:count]]
    normalized_contexts = [coerce_contexts(ctxs) for ctxs in contexts_list[:count]]
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


# ─────────────────────────────────────────────
# SECTION 5: RETRY-ONLY RAGAS SUPPORT
# ─────────────────────────────────────────────

def metric_columns_in_csv(df: Any, metric_names: list[str]) -> list[str]:
    return [name for name in metric_names if name in df.columns]


def failed_score_indexes(scores_csv: Path, metric_names: list[str]) -> list[int]:
    import pandas as pd

    df = pd.read_csv(scores_csv)
    metric_cols = metric_columns_in_csv(df, metric_names)
    if not metric_cols:
        raise ValueError(f"No RAGAS metric columns found in {scores_csv}")
    mask = df[metric_cols].isna().any(axis=1)
    return [int(idx) for idx in df.index[mask].tolist()]


def subset_generation(generation: dict[str, Any], indexes: list[int]) -> dict[str, Any]:
    subset = dict(generation)
    for key in ("questions", "ground_truths", "answers", "contexts_list", "question_statuses"):
        value = generation.get(key)
        if isinstance(value, list):
            subset[key] = [value[i] for i in indexes if i < len(value)]
    subset["questions_count"] = len(indexes)
    subset["tiene_ground_truth"] = any(bool(gt) for gt in subset.get("ground_truths", []))
    return subset


def merge_retry_scores(
    original_csv: Path,
    retry_csv: Path,
    retry_indexes: list[int],
    metric_names: list[str],
) -> dict[str, Any]:
    import pandas as pd

    original_df = pd.read_csv(original_csv)
    retry_df = pd.read_csv(retry_csv)
    metric_cols = metric_columns_in_csv(original_df, metric_names)
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


def apply_limit(generation: dict[str, Any], limit: int | None) -> dict[str, Any]:
    """Trim a generation payload to the first ``limit`` rows."""
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
