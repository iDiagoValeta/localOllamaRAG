"""
RAGAS evaluation of the RAG pipeline.

Runs the full Teacher-RAG pipeline on a question dataset, then evaluates the
quality of the generated answers using RAGAS v0.2+ metrics.  Supported metrics
(per RAGAS documentation):

  - Context Recall: measures how well the retriever surfaces all relevant info.
  - Factual Correctness (answer_correctness): factual precision vs ground truth
    (TP / FP / FN).
  - Context Precision: evaluates the ranking of retrieved fragments.
  - Faithfulness: factual consistency of the answer with the retrieved context.
  - Response Relevancy (answer_relevancy): degree to which the answer addresses
    the original question.

Usage:
    python evaluation/run_eval.py single --corpus es|ca|en
    python evaluation/run_eval.py compare --corpus ca --label mi_ablacion
    python evaluation/run_eval.py ragbench --n-papers 10 --max-q 5

Default corpus: ``es`` (rag/docs/es + dataset_eval_es.json).
If you still pass the old path ``evaluation/dataset_eval_*.json``, it is
resolved automatically to ``evaluation/datasets/``.

Dependencies:
    - ragas
    - langchain-google-genai
    - pandas, chromadb
    - python-dotenv (optional)
"""


# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  CONFIGURATION
#  +-- 1. Environment setup    imports, .env loading, sys.path
#  +-- 2. Dataset loading      cargar_dataset, normalizar_columnas
#  +-- 3. Evaluation LLM       configurar_llm_evaluacion (Gemini + embeddings)
#
#  PIPELINE
#  +-- 4. Paths and checkpoints resolver_ruta_dataset, rutas, progreso por pregunta, reanudación
#  +-- 5. Result formatting    imprimir_resultados
#  +-- 6. Debug output         _extraer_justificaciones_traces, guardar_debug
#  +-- 7. Ragbench preparation  descargar_metadatos, seleccionar_papers, _ejecutar_ragbench
#
#  ENTRY
#  +-- 8. Main                 main()
#
# ─────────────────────────────────────────────

import os
import sys
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
from typing import Any, Literal
from pathlib import Path

# ─────────────────────────────────────────────
# SECTION 1: ENVIRONMENT SETUP
# ─────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(_env_path)
except ImportError:
    pass

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import pandas as pd
import chromadb

import rag.chat_pdfs as rag_runtime
from rag.chat_pdfs import (
    evaluar_pregunta_rag,
    indexar_documentos,
)

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download as _hf_hub_download
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

# ─────────────────────────────────────────────
# SECTION 2: DATASET LOADING
# ─────────────────────────────────────────────

def cargar_dataset(ruta: str) -> pd.DataFrame:
    """Load a dataset from a JSON, CSV, or Excel file into a DataFrame.

    Args:
        ruta: Path to the dataset file.

    Returns:
        A ``pd.DataFrame`` with the raw dataset contents.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = os.path.splitext(ruta)[1].lower()
    if ext == ".json":
        with open(ruta, encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(ruta)
    if ext == ".csv":
        return pd.read_csv(ruta, encoding="utf-8")
    raise ValueError(f"Unsupported format: {ext}")


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to a canonical ``question`` / ``ground_truth``
    schema, handling common Spanish and English aliases.

    Args:
        df: Raw dataset DataFrame.

    Returns:
        A new DataFrame with ``question`` and ``ground_truth`` columns.

    Raises:
        ValueError: If no question column can be found.
    """
    mapeo = {
        "pregunta": "question",
        "question": "question",
        "preguntas": "question",
        "ground_truth": "ground_truth",
        "respuesta_esperada": "ground_truth",
        "respuesta_referencia": "ground_truth",
        "reference": "ground_truth",
    }
    cols = {c.lower(): c for c in df.columns}
    out = {}
    for orig, target in mapeo.items():
        if orig in cols:
            out[target] = df[cols[orig]].tolist()
    if "question" not in out:
        raise ValueError("Dataset must have a 'question' or 'pregunta' column")
    if "ground_truth" not in out:
        out["ground_truth"] = [""] * len(out["question"])
    return pd.DataFrame(out)


# ─────────────────────────────────────────────
# SECTION 3: EVALUATION LLM AND EMBEDDINGS
# ─────────────────────────────────────────────

def _leer_env_int(nombre: str, default: int) -> int:
    """Read an integer environment variable with a safe fallback."""
    try:
        return int(os.getenv(nombre, str(default)))
    except (TypeError, ValueError):
        return default


def configurar_llm_evaluacion(
    google_timeout: int | None = None,
    google_retries: int | None = None,
):
    """Configure the Gemini LLM and embedding model used by RAGAS as the
    evaluation judge.

    Returns:
        A ``(eval_llm, eval_embeddings)`` tuple ready for ``ragas.evaluate()``.

    Raises:
        SystemExit: If the API key is missing or the Google GenAI package is
            not installed.
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        print("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment.")
        raise SystemExit(1)
    google_timeout = google_timeout or _leer_env_int("EVAL_GOOGLE_TIMEOUT", 45)
    google_retries = google_retries if google_retries is not None else _leer_env_int("EVAL_GOOGLE_RETRIES", 2)

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

        eval_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=gemini_key,
            temperature=0,
            request_timeout=google_timeout,
            retries=google_retries,
        )
        eval_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=gemini_key,
            request_options={"timeout": google_timeout},
        )
        print("Evaluation LLM: Gemini 2.5 Flash (langchain-google-genai)")
        print("Evaluation embeddings: Google gemini-embedding-001 (langchain-google-genai)")
        print(f"Google timeout/retries: {google_timeout}s / {google_retries}")
        return eval_llm, eval_embeddings
    except ImportError as err:
        print(f"Error: {err}")
        print("  Install with: pip install langchain-google-genai")
        raise SystemExit(1)

# ─────────────────────────────────────────────
# SECTION 4: PATHS AND CHECKPOINTS
# ─────────────────────────────────────────────

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(EVAL_DIR, "datasets")
LOCAL_DATASETS_DIR = os.path.join(DATASETS_DIR, "local")
RAGBENCH_DATASETS_DIR = os.path.join(DATASETS_DIR, "ragbench")
RAGBENCH_PREPARED_DIR = os.path.join(RAGBENCH_DATASETS_DIR, "prepared")
RUNS_DIR = os.path.join(EVAL_DIR, "runs")
RAGAS_RUNS_DIR = os.path.join(RUNS_DIR, "ragas")
SINGLE_RUNS_DIR = os.path.join(RAGAS_RUNS_DIR, "single")
COMPARISON_RUNS_DIR = os.path.join(RAGAS_RUNS_DIR, "comparisons")
RAGBENCH_RUNS_DIR = os.path.join(RAGAS_RUNS_DIR, "ragbench")
RAGBENCH_VISUAL_RAGAS_DIR = os.path.join(RAGAS_RUNS_DIR, "ragbench_visual")
TMP_DIR = os.path.join(EVAL_DIR, "tmp")

# Legacy constants remain available for explicit CLI overrides and old artifacts.
SCORES_DIR = os.path.join(RAGAS_RUNS_DIR, "scores")
DEBUG_DIR = os.path.join(RAGAS_RUNS_DIR, "debug")
CHECKPOINTS_DIR = os.path.join(RAGAS_RUNS_DIR, "checkpoints")
COMPARISON_SCORES_DIR = COMPARISON_RUNS_DIR
COMPARISON_DEBUG_DIR = COMPARISON_RUNS_DIR

BASELINE_PIPELINE_FLAGS = {
    "USAR_LLM_QUERY_DECOMPOSITION": True,
    "USAR_BUSQUEDA_HIBRIDA": True,
    "USAR_BUSQUEDA_EXHAUSTIVA": True,
    "USAR_RERANKER": True,
    "EXPANDIR_CONTEXTO": True,
    "USAR_OPTIMIZACION_CONTEXTO": True,
    "USAR_RECOMP_SYNTHESIS": True,
}

ABLATION_VARIANTS = [
    {
        "name": "baseline_all_on",
        "description": "All inference-time optional stages enabled.",
        "flags": BASELINE_PIPELINE_FLAGS,
    },
    {
        "name": "no_query_decomposition",
        "description": "Disable LLM query decomposition.",
        "flags": {**BASELINE_PIPELINE_FLAGS, "USAR_LLM_QUERY_DECOMPOSITION": False},
    },
    {
        "name": "no_lexical_search",
        "description": "Disable keyword/lexical Chroma search.",
        "flags": {**BASELINE_PIPELINE_FLAGS, "USAR_BUSQUEDA_HIBRIDA": False},
    },
    {
        "name": "no_exhaustive_search",
        "description": "Disable exhaustive full-collection text scan.",
        "flags": {**BASELINE_PIPELINE_FLAGS, "USAR_BUSQUEDA_EXHAUSTIVA": False},
    },
    {
        "name": "no_reranker",
        "description": "Disable Cross-Encoder reranking.",
        "flags": {**BASELINE_PIPELINE_FLAGS, "USAR_RERANKER": False},
    },
    {
        "name": "no_context_expansion",
        "description": "Disable adjacent-chunk context expansion.",
        "flags": {**BASELINE_PIPELINE_FLAGS, "EXPANDIR_CONTEXTO": False},
    },
    {
        "name": "no_context_optimization",
        "description": "Disable PDF artifact cleanup before generation.",
        "flags": {**BASELINE_PIPELINE_FLAGS, "USAR_OPTIMIZACION_CONTEXTO": False},
    },
    {
        "name": "no_recomp_synthesis",
        "description": "Disable RECOMP/LLM context synthesis.",
        "flags": {**BASELINE_PIPELINE_FLAGS, "USAR_RECOMP_SYNTHESIS": False},
    },
]

VARIANT_SUITES = {"ablation": ABLATION_VARIANTS}


def resolver_ruta_dataset(ruta: str) -> str:
    """Resolve a dataset path to an existing file.

    Bundled RAGAS JSON files live under ``evaluation/datasets/``. If the user
    passes a non-existent path but the basename matches ``dataset_eval_*.json``,
    the same file under ``evaluation/datasets/`` is tried (covers the legacy
    layout ``evaluation/dataset_eval_*.json`` and bare filenames from repo root).

    Args:
        ruta: Path as given by the user (relative or absolute).

    Returns:
        Absolute path to an existing file.

    Raises:
        FileNotFoundError: If no matching file exists.
    """
    expanded = os.path.abspath(os.path.expanduser(ruta))
    if os.path.isfile(expanded):
        return expanded
    name = os.path.basename(expanded)
    if name.startswith("dataset_eval_") and name.lower().endswith(".json"):
        alt = os.path.join(LOCAL_DATASETS_DIR, name)
        if os.path.isfile(alt):
            print(
                f"   Note: dataset path resolved to {alt} (input was {ruta!r}; "
                "use evaluation/datasets/… to silence this message)"
            )
            return alt
    raise FileNotFoundError(
        f"Dataset file not found: {expanded}. "
        f"Bundled datasets are under {os.path.join(EVAL_DIR, 'datasets')!r}, "
        f"e.g. {os.path.join(EVAL_DIR, 'datasets', 'dataset_eval_es.json')}"
    )


SUPPORTED_CORPORA = ("es", "ca", "en", "mix")


def _default_dataset_for_corpus(eval_corpus: str) -> str:
    """Default bundled JSON path for a local evaluation corpus."""
    if eval_corpus not in SUPPORTED_CORPORA:
        valid = ", ".join(SUPPORTED_CORPORA)
        raise ValueError(f"Unsupported corpus {eval_corpus!r}. Valid: {valid}")
    name = f"dataset_eval_{eval_corpus}.json"
    return os.path.join(LOCAL_DATASETS_DIR, name)


def _default_docs_dir_for_corpus(eval_corpus: str) -> str | None:
    """Default PDF folder for a local evaluation corpus.

    Returns the language-specific subfolder under ``rag/docs/``. ``es`` and
    ``mix`` return ``None`` so the RAG module default (``rag/docs/es``) is used.
    """
    _corpus_to_folder: dict[str, str | None] = {
        "es": None,
        "ca": os.path.join(_proj_root, "rag", "docs", "ca"),
        "en": os.path.join(_proj_root, "rag", "docs", "en"),
        "mix": None,
        "ragbench": os.path.join(_proj_root, "rag", "docs", "en"),
    }
    return _corpus_to_folder.get(eval_corpus)


def _artifact_suffix(eval_corpus: str) -> str:
    """Filename suffix for scores/debug/checkpoints (always includes language tag)."""
    _suffix_map: dict[str, str] = {
        "es": "_es",
        "ca": "_ca",
        "en": "_en",
        "mix": "_mix",
        "ragbench": "_en",
    }
    return _suffix_map.get(eval_corpus, f"_{eval_corpus}")


def _slugify(value: str) -> str:
    """Convert a free-form label into a filesystem-friendly slug."""
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "eval"


def _build_output_stem(dataset_path: str) -> str:
    """Create a stable stem for outputs derived from the dataset filename."""
    return _slugify(Path(dataset_path).stem)


def _single_run_dir(dataset_path: str, artifact_suffix: str = "") -> str:
    """Return the self-contained folder for one local RAGAS run."""
    return os.path.join(SINGLE_RUNS_DIR, f"{_build_output_stem(dataset_path)}{artifact_suffix}")


def _default_output_path(dataset_path: str, artifact_suffix: str = "") -> str:
    """Return the default CSV output path inside a self-contained single run."""
    return os.path.join(_single_run_dir(dataset_path, artifact_suffix), "scores.csv")


def _default_debug_path(dataset_path: str, artifact_suffix: str = "") -> str:
    """Return the default debug JSON path inside a self-contained single run."""
    return os.path.join(_single_run_dir(dataset_path, artifact_suffix), "debug.json")


def _default_checkpoint_path(
    dataset_path: str, recomp_enabled: bool | None, artifact_suffix: str = ""
) -> str:
    """Return the checkpoint path used to resume generation question by question."""
    recomp_tag = "recomp_on" if recomp_enabled else "recomp_off"
    return os.path.join(_single_run_dir(dataset_path, artifact_suffix), f"checkpoint_{recomp_tag}.json")


def _cargar_checkpoint(checkpoint_path: str) -> dict[str, Any] | None:
    """Load a checkpoint file if it exists and is valid JSON."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
    with open(checkpoint_path, encoding="utf-8") as f:
        return json.load(f)


def _guardar_checkpoint(checkpoint_path: str, payload: dict[str, Any]) -> None:
    """Persist the current question-by-question evaluation state."""
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _guardar_json(path: str, payload: dict[str, Any]) -> None:
    """Write a JSON payload ensuring the parent directory exists."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _build_run_slug(dataset_path: str, label: str | None, eval_corpus: str) -> str:
    """Build a stable folder slug for a comparison batch."""
    dataset_stem = Path(dataset_path).stem
    suffix = label.strip().replace(" ", "_") if label else f"{dataset_stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    if eval_corpus == "ca":
        suffix = f"{suffix}_ca"
    return suffix


def seleccionar_variantes(suite: str, variant_names: str | None = None) -> list[dict[str, Any]]:
    """Resolve requested variant names into concrete pipeline-flag specs."""
    available = {variant["name"]: variant for variant in VARIANT_SUITES[suite]}
    if not variant_names:
        return list(available.values())

    selected = []
    unknown = []
    for raw_name in variant_names.split(","):
        name = raw_name.strip()
        if not name:
            continue
        if name not in available:
            unknown.append(name)
        else:
            selected.append(available[name])

    if unknown:
        valid = ", ".join(available)
        raise ValueError(f"Unknown variant(s): {', '.join(unknown)}. Valid variants: {valid}")
    if not selected:
        raise ValueError("No variants selected.")
    return selected


def listar_variantes(suite: str = "ablation") -> None:
    """Print available variants for a suite."""
    print(f"Available variants for suite '{suite}':")
    for variant in VARIANT_SUITES[suite]:
        print(f"  {variant['name']}: {variant['description']}")


def _mean_score_deltas(results: list[dict[str, Any]], baseline_variant: str) -> dict[str, dict[str, float]]:
    """Compute per-variant mean-score deltas against the selected baseline."""
    by_variant = {entry["variant"]: entry for entry in results}
    baseline = by_variant.get(baseline_variant)
    if not baseline:
        return {}

    deltas: dict[str, dict[str, float]] = {}
    baseline_scores = baseline.get("mean_scores", {})
    for variant_name, entry in by_variant.items():
        if variant_name == baseline_variant:
            continue
        variant_deltas = {}
        for metric_name, variant_value in entry.get("mean_scores", {}).items():
            baseline_value = baseline_scores.get(metric_name)
            if baseline_value is None:
                continue
            variant_deltas[metric_name] = variant_value - baseline_value
        deltas[variant_name] = variant_deltas
    return deltas


def _respuesta_vacia(respuesta: Any) -> bool:
    """Return True when a generated answer is missing or blank."""
    return not isinstance(respuesta, str) or not respuesta.strip()


def _indices_respuestas_vacias(answers: list[str], total: int) -> list[int]:
    """Find zero-based question indexes that still need a non-empty answer."""
    return [
        i
        for i in range(total)
        if i >= len(answers) or _respuesta_vacia(answers[i])
    ]


def _estado_pregunta_base(index: int, answer: Any = "") -> dict[str, Any]:
    """Build a checkpoint status entry for one evaluation question."""
    status = "ok" if not _respuesta_vacia(answer) else "pending"
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


def _normalizar_estados_preguntas(
    raw_statuses: Any,
    answers: list[str],
    total: int,
) -> list[dict[str, Any]]:
    """Normalize checkpoint question statuses, supporting old checkpoints."""
    statuses: list[dict[str, Any]] = []
    if isinstance(raw_statuses, list):
        for i in range(total):
            raw = raw_statuses[i] if i < len(raw_statuses) and isinstance(raw_statuses[i], dict) else {}
            base = _estado_pregunta_base(i, answers[i] if i < len(answers) else "")
            base.update({k: v for k, v in raw.items() if k in base})
            base["index"] = i
            base["question_number"] = i + 1
            if not _respuesta_vacia(answers[i] if i < len(answers) else ""):
                base["status"] = "ok"
                base["reason"] = None
                base["error"] = None
            statuses.append(base)
        return statuses

    return [
        _estado_pregunta_base(i, answers[i] if i < len(answers) else "")
        for i in range(total)
    ]


def _indices_pendientes_generacion(
    answers: list[str],
    question_statuses: list[dict[str, Any]],
    total: int,
) -> list[int]:
    """Return indexes that need generation or retry."""
    pending = set(_indices_respuestas_vacias(answers, total))
    for i, status in enumerate(question_statuses[:total]):
        if status.get("status") not in ("ok", "skipped"):
            pending.add(i)
    return sorted(pending)


def _resumen_estados_fallidos(
    answers: list[str],
    question_statuses: list[dict[str, Any]],
    total: int,
) -> dict[str, list[int]]:
    """Group incomplete question numbers by diagnostic status/reason."""
    grouped: dict[str, list[int]] = {}
    for i in _indices_respuestas_vacias(answers, total):
        status = question_statuses[i] if i < len(question_statuses) else {}
        key = str(status.get("reason") or status.get("status") or "empty_answer")
        grouped.setdefault(key, []).append(i + 1)
    return grouped


def _es_dataset_ragbench(dataset_path: str, eval_corpus: str) -> bool:
    """Return True for RagBench runs, including prepared datasets run as corpus en."""
    if eval_corpus == "ragbench":
        return True
    path = Path(dataset_path)
    parts = {part.lower() for part in path.parts}
    return "ragbench_prepared" in parts or "ragbench" in path.stem.lower()


def _diagnosticar_fallo_generacion(
    pregunta: str,
    collection: Any,
    answer: str,
    contexts: list[str],
    timed_out: bool,
    error: Exception | None,
) -> str:
    """Classify why a question did not produce a usable answer."""
    if error is not None:
        return "excepcion"
    if timed_out:
        return "timeout"
    if not contexts:
        try:
            fragmentos_ranked, mejor_score, _ = rag_runtime.realizar_busqueda_hibrida(pregunta, collection)
            if not fragmentos_ranked:
                return "sin_contexto"
            fallback_ragbench = bool(
                getattr(rag_runtime, "EVAL_RAGBENCH_RERANKER_LOW_SCORE_FALLBACK", False)
            )
            if rag_runtime.USAR_RERANKER and mejor_score < rag_runtime.UMBRAL_RELEVANCIA and not fallback_ragbench:
                return "filtrada_por_reranker"
            if rag_runtime.USAR_RERANKER:
                fragmentos_filtrados = [
                    f for f in fragmentos_ranked
                    if f.get("score_reranker", f.get("score_final", 0)) >= rag_runtime.UMBRAL_SCORE_RERANKER
                ]
                if not fragmentos_filtrados and not fallback_ragbench:
                    return "filtrada_por_reranker"
        except Exception:
            return "sin_contexto"
        return "sin_contexto"
    if _respuesta_vacia(answer):
        return "respuesta_vacia"
    return "ok"


def _ejecutar_pregunta_con_timeout(
    pregunta: str,
    collection: Any,
    timeout_seconds: int,
) -> tuple[str, list[str], bool, Exception | None]:
    """Run one RAG question attempt with a best-effort wall-clock timeout."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(evaluar_pregunta_rag, pregunta, collection)
    try:
        answer, contexts = future.result(timeout=timeout_seconds)
        executor.shutdown(wait=True)
        return answer, contexts, False, None
    except _FuturesTimeout:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        return "", [], True, None
    except Exception as exc:
        executor.shutdown(wait=True)
        return "", [], False, exc


def _normalizar_pipeline_flags(flags: dict[str, Any] | None) -> dict[str, bool]:
    """Return a stable boolean pipeline-flags dict for checkpoints/manifests."""
    if not flags:
        return {}
    return {str(k): bool(v) for k, v in sorted(flags.items())}


def _checkpoint_pipeline_flags_match(
    checkpoint: dict[str, Any],
    current_flags: dict[str, bool],
) -> bool:
    """Validate checkpoint compatibility with current runtime pipeline flags."""
    stored_flags = checkpoint.get("pipeline_flags")
    if stored_flags is not None:
        return _normalizar_pipeline_flags(stored_flags) == _normalizar_pipeline_flags(current_flags)

    # Backward compatibility for old checkpoints created before full flag tracking.
    return checkpoint.get("recomp_enabled") == current_flags.get("USAR_RECOMP_SYNTHESIS")


_TRACKED_MODEL_FIELDS = ("modelo_rag", "modelo_chat", "modelo_embedding", "modelo_recomp")


def _current_models_signature() -> dict[str, str]:
    """Snapshot of model names that influence answers and must invalidate checkpoints."""
    return {
        "modelo_rag": str(getattr(rag_runtime, "MODELO_RAG", "") or ""),
        "modelo_chat": str(getattr(rag_runtime, "MODELO_CHAT", "") or ""),
        "modelo_embedding": str(getattr(rag_runtime, "MODELO_EMBEDDING", "") or ""),
        "modelo_recomp": str(getattr(rag_runtime, "MODELO_RECOMP", "") or ""),
    }


def _checkpoint_models_match(
    checkpoint: dict[str, Any],
    current_models: dict[str, str],
) -> tuple[bool, str | None]:
    """Validate checkpoint compatibility with the current generator / embedder.

    Returns ``(matches, warning_or_diff)``. Checkpoints written before model
    tracking is deployed have none of the ``_TRACKED_MODEL_FIELDS``; those are
    accepted for backward compatibility with a soft warning, so the user knows
    silent model drift is possible until the checkpoint is regenerated.
    """
    has_any = any(checkpoint.get(k) is not None for k in _TRACKED_MODEL_FIELDS)
    if not has_any:
        return True, (
            "checkpoint sin información de modelos (anterior a la validación por modelo); "
            "si cambiaste de modelo generador desde la última corrida, bórralo y relanza."
        )
    differences: list[str] = []
    for key in _TRACKED_MODEL_FIELDS:
        stored = checkpoint.get(key)
        current = current_models.get(key, "")
        if stored is not None and str(stored) != current:
            differences.append(f"{key}: {stored!r} -> {current!r}")
    if differences:
        return False, "modelo(s) cambiado(s): " + "; ".join(differences)
    return True, None


def _guardar_checkpoint_evaluacion(
    checkpoint_path: str,
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
        "pipeline_flags": _normalizar_pipeline_flags(pipeline_flags),
        "eval_corpus": eval_corpus,
        "output_path": output_path,
        "debug_path": debug_path,
        "docs_dir": docs_dir,
        "ragbench_reranker_low_score_fallback": ragbench_reranker_low_score_fallback,
        "completed_questions": len([a for a in answers if not _respuesta_vacia(a)]),
        "answers": answers,
        "contexts_list": contexts_list,
        "question_statuses": question_statuses or _normalizar_estados_preguntas(
            None, answers, questions_count
        ),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    payload.update(_current_models_signature())
    _guardar_checkpoint(checkpoint_path, payload)


def _parse_ragas_metric_names(
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
# SECTION 5: RESULT FORMATTING
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

def imprimir_resultados(df_scores: pd.DataFrame, questions: list[str]):
    """Print detailed per-question scores and global averages to stdout.

    Args:
        df_scores: DataFrame of RAGAS scores (one row per question).
        questions: Original question strings, aligned by index.
    """

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


# ─────────────────────────────────────────────
# SECTION 6: DEBUG OUTPUT
# ─────────────────────────────────────────────

def _extraer_justificaciones_traces(traces: list, metric_cols: list) -> list[dict]:
    """Extract justification details from RAGAS evaluation traces.

    The traces contain the LLM prompt outputs (TP/FP/FN counts, statement
    lists, etc.) that explain how each metric score was derived.

    Args:
        traces: List of trace objects from ``result.traces``.
        metric_cols: Metric column names to look for in each trace.

    Returns:
        A list of dictionaries (one per question) mapping metric names to
        their extracted justification payloads.
    """
    justificaciones = []
    for i, trace in enumerate(traces):
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
    """Save a debug JSON file containing model answers, retrieved contexts,
    per-question scores, and RAGAS justifications.

    Args:
        result: The RAGAS ``EvaluationResult`` object.
        questions: List of input questions.
        answers: List of model-generated answers.
        ground_truths: List of reference answers.
        contexts_list: List of lists of retrieved context strings.
        debug_path: Absolute path where the debug file will be written.

    Returns:
        Absolute path to the saved debug JSON file.
    """
    df = result.to_pandas()
    metric_cols = [c for c in METRIC_NAMES if c in df.columns]

    traces = getattr(result, "traces", []) or []
    justificaciones = _extraer_justificaciones_traces(traces, metric_cols) if traces else [{}] * len(questions)

    debug_entries = []
    for i in range(len(questions)):
        ctx_preview = []
        for j, ctx in enumerate(contexts_list[i] if i < len(contexts_list) else []):
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
            METRIC_DISPLAY_NAMES.get(m, m): float(df[m].mean())
            for m in metric_cols
        },
    }

    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)
    return debug_path


def _importar_ragas_componentes():
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
        print("   pip install -r evaluation/requirements.txt")
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


def _conectar_e_indexar(
    force_reindex: bool = False,
    solo_archivos: list[str] | None = None,
    add_missing_from_filter: bool = False,
) -> tuple[Any, int]:
    """Connect to the active Chroma collection and index documents if needed."""
    print(f"\nConnecting to ChromaDB: {rag_runtime.PATH_DB}")
    client = chromadb.PersistentClient(path=rag_runtime.PATH_DB)

    if force_reindex:
        print("   Force reindex requested. Rebuilding collection...")
        try:
            client.delete_collection(name=rag_runtime.COLLECTION_NAME)
        except Exception:
            pass
        collection = client.get_or_create_collection(name=rag_runtime.COLLECTION_NAME)
        total = indexar_documentos(rag_runtime.CARPETA_DOCS, collection, solo_archivos=solo_archivos)
        print(f"   Indexed {total} fragments.")
        return collection, total

    collection = client.get_or_create_collection(name=rag_runtime.COLLECTION_NAME)
    if collection.count() == 0:
        print("   Database empty. Indexing documents...")
        total = indexar_documentos(rag_runtime.CARPETA_DOCS, collection, solo_archivos=solo_archivos)
        print(f"   Indexed {total} fragments.")
    else:
        total = collection.count()
        print(f"   Fragments in collection: {total}")
        if solo_archivos and add_missing_from_filter:
            indexed_docs = set(rag_runtime.obtener_documentos_indexados(collection))
            missing_files = [f for f in solo_archivos if f not in indexed_docs]
            if missing_files:
                print(
                    "   Adding missing files to existing collection: "
                    + ", ".join(missing_files[:10])
                    + ("..." if len(missing_files) > 10 else "")
                )
                added = indexar_documentos(
                    rag_runtime.CARPETA_DOCS,
                    collection,
                    solo_archivos=missing_files,
                )
                total = collection.count()
                print(f"   Indexed {added} new fragments. Collection now has {total}.")
            else:
                print("   Collection already contains every file from the requested manifest.")
        elif solo_archivos:
            print("   Note: file filter only applies while indexing an empty/rebuilt collection.")
    return collection, total


def generar_respuestas_rag(
    dataset_path: str,
    output_path: str | None = None,
    debug_path: str | None = None,
    checkpoint_path: str | None = None,
    verbose: bool = False,
    force_reindex: bool = False,
    recomp_enabled: bool | None = None,
    pipeline_flags: dict[str, bool] | None = None,
    eval_corpus: str = "es",
    docs_dir: str | None = None,
    solo_archivos: list[str] | None = None,
    add_missing_from_filter: bool = False,
) -> dict[str, Any]:
    """Run only indexing/retrieval/generation and persist a reusable checkpoint."""
    previous_pipeline_flags = None
    docs_previous: tuple[str, str, str] | None = None
    previous_ragbench_reranker_fallback: bool | None = None
    flag_overrides = dict(pipeline_flags or {})
    if recomp_enabled is not None:
        flag_overrides["USAR_RECOMP_SYNTHESIS"] = bool(recomp_enabled)
    if flag_overrides:
        previous_pipeline_flags = rag_runtime.set_pipeline_flags(flag_overrides)

    resolved_docs_dir = docs_dir or _default_docs_dir_for_corpus(eval_corpus)
    if resolved_docs_dir is not None:
        docs_previous = rag_runtime.set_docs_folder_runtime(resolved_docs_dir)

    try:
        print(f"\nLoading dataset...")
        dataset_path = resolver_ruta_dataset(dataset_path)
        ragbench_reranker_fallback = _es_dataset_ragbench(dataset_path, eval_corpus)
        previous_ragbench_reranker_fallback = (
            rag_runtime.set_ragbench_reranker_low_score_fallback(ragbench_reranker_fallback)
        )
        sfx = _artifact_suffix(eval_corpus)
        resolved_output_path = os.path.abspath(output_path or _default_output_path(dataset_path, sfx))
        resolved_debug_path = os.path.abspath(debug_path or _default_debug_path(dataset_path, sfx))
        resolved_checkpoint_path = os.path.abspath(
            checkpoint_path
            or _default_checkpoint_path(dataset_path, rag_runtime.USAR_RECOMP_SYNTHESIS, sfx)
        )

        df = normalizar_columnas(cargar_dataset(dataset_path))
        questions = df["question"].tolist()
        ground_truths = df["ground_truth"].tolist()
        tiene_ground_truth = any(gt.strip() for gt in ground_truths)

        print(f"   Questions to evaluate: {len(questions)}")
        print(f"   Ground truth available: {'Yes' if tiene_ground_truth else 'No'}")
        print(f"   RECOMP synthesis: {'Enabled' if rag_runtime.USAR_RECOMP_SYNTHESIS else 'Disabled'}")
        current_pipeline_flags = rag_runtime.get_pipeline_flags()
        print(
            "   Pipeline flags: "
            + ", ".join(
                f"{name}={'on' if value else 'off'}"
                for name, value in current_pipeline_flags.items()
            )
        )
        print(f"   Eval corpus: {eval_corpus.upper()} -- PDFs: {rag_runtime.CARPETA_DOCS}")
        if ragbench_reranker_fallback:
            print(
                "   RagBench reranker fallback: enabled "
                "(low-scored reranker candidates are kept for generation)."
            )

        collection, total = _conectar_e_indexar(
            force_reindex=force_reindex,
            solo_archivos=solo_archivos,
            add_missing_from_filter=add_missing_from_filter,
        )

        print("\nRunning RAG pipeline for each question...")
        answers: list[str] = []
        contexts_list: list[list[str]] = []
        question_statuses: list[dict[str, Any]] = []
        checkpoint = _cargar_checkpoint(resolved_checkpoint_path)
        checkpoint_valid = False
        if checkpoint:
            models_match, models_note = _checkpoint_models_match(
                checkpoint, _current_models_signature()
            )
            structural_match = (
                checkpoint.get("dataset_path") == dataset_path
                and checkpoint.get("questions_count") == len(questions)
                and _checkpoint_pipeline_flags_match(checkpoint, current_pipeline_flags)
                and checkpoint.get("eval_corpus", "es") == eval_corpus
                and checkpoint.get("docs_dir", rag_runtime.CARPETA_DOCS) == rag_runtime.CARPETA_DOCS
            )
            if structural_match and models_match:
                checkpoint_valid = True
                answers = checkpoint.get("answers", [])
                contexts_list = checkpoint.get("contexts_list", [])
                question_statuses = _normalizar_estados_preguntas(
                    checkpoint.get("question_statuses"),
                    answers,
                    len(questions),
                )
                non_empty_answers = len([a for a in answers if not _respuesta_vacia(a)])
                print(
                    "   Resuming from checkpoint: "
                    f"{non_empty_answers}/{len(questions)} questions with non-empty answers "
                    f"({len(answers)}/{len(questions)} slots present)."
                )
                if models_note:
                    print(f"   [aviso] {models_note}")
            else:
                if structural_match and not models_match:
                    print(
                        f"   Existing checkpoint does not match this run ({models_note}). "
                        "Starting fresh progress."
                    )
                else:
                    print("   Existing checkpoint does not match this run. Starting fresh progress.")

        t_start = time.time()
        if len(answers) > len(questions):
            answers = answers[:len(questions)]
        if len(contexts_list) > len(questions):
            contexts_list = contexts_list[:len(questions)]

        while len(answers) < len(questions):
            answers.append("")
        while len(contexts_list) < len(questions):
            contexts_list.append([])
        question_statuses = _normalizar_estados_preguntas(
            question_statuses,
            answers,
            len(questions),
        )

        pending_answer_indexes = _indices_pendientes_generacion(
            answers,
            question_statuses,
            len(questions),
        )
        if pending_answer_indexes and checkpoint_valid:
            first_missing = ", ".join(str(i + 1) for i in pending_answer_indexes[:10])
            suffix = "..." if len(pending_answer_indexes) > 10 else ""
            print(
                "   Checkpoint contains empty answers. "
                f"Regenerating {len(pending_answer_indexes)} question(s): {first_missing}{suffix}"
            )

        _ollama_timeout = int(os.getenv("EVAL_OLLAMA_TIMEOUT", "300"))
        _max_attempts_per_question = max(1, int(os.getenv("EVAL_OLLAMA_ATTEMPTS", "2")))
        try:
            for i in pending_answer_indexes:
                q = questions[i]
                if verbose:
                    print(f"   [{i+1}/{len(questions)}] {q[:60]}...")
                answer, contexts = "", []
                timed_out = False
                last_error: Exception | None = None
                attempt_count = 0
                question_start = time.time()
                for attempt in range(_max_attempts_per_question):
                    attempt_count = attempt + 1
                    if attempt > 0:
                        print(
                            f"   [RETRY] Q{i+1} attempt {attempt + 1}/{_max_attempts_per_question} "
                            f"(previous: empty or timeout)."
                        )
                    answer, contexts, timed_out, last_error = _ejecutar_pregunta_con_timeout(
                        q,
                        collection,
                        _ollama_timeout,
                    )
                    if timed_out:
                        print(
                            f"   [TIMEOUT] Q{i+1} exceeded {_ollama_timeout}s "
                            "(attempt "
                            f"{attempt + 1}/{_max_attempts_per_question})."
                        )
                        break
                    if last_error is not None:
                        print(
                            f"   [ERROR] Q{i+1} failed on attempt "
                            f"{attempt + 1}/{_max_attempts_per_question}: {last_error}"
                        )
                    if not _respuesta_vacia(answer):
                        break

                reason = _diagnosticar_fallo_generacion(
                    q,
                    collection,
                    answer,
                    contexts,
                    timed_out,
                    last_error,
                )
                if _respuesta_vacia(answer) and _max_attempts_per_question > 1 and not timed_out:
                    print(
                        f"   [WARN] Q{i+1} still empty after {_max_attempts_per_question} attempts; "
                        f"reason={reason}. Rerun to retry only incomplete answers."
                    )
                answers[i] = answer
                contexts_list[i] = contexts
                question_statuses[i] = {
                    "index": i,
                    "question_number": i + 1,
                    "status": "ok" if not _respuesta_vacia(answer) else "failed",
                    "attempts": attempt_count,
                    "duration_seconds": round(time.time() - question_start, 3),
                    "reason": None if not _respuesta_vacia(answer) else reason,
                    "error": (
                        f"{type(last_error).__name__}: {last_error}"
                        if last_error is not None else None
                    ),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                _guardar_checkpoint_evaluacion(
                    resolved_checkpoint_path,
                    dataset_path=dataset_path,
                    questions_count=len(questions),
                    recomp_enabled=rag_runtime.USAR_RECOMP_SYNTHESIS,
                    eval_corpus=eval_corpus,
                    output_path=resolved_output_path,
                    debug_path=resolved_debug_path,
                    answers=answers,
                    contexts_list=contexts_list,
                    question_statuses=question_statuses,
                    pipeline_flags=current_pipeline_flags,
                    docs_dir=rag_runtime.CARPETA_DOCS,
                    ragbench_reranker_low_score_fallback=ragbench_reranker_fallback,
                )
        except ConnectionError as e:
            print(f"\nError: Could not connect to Ollama: {e}")
            print("   Make sure Ollama is running before launching the evaluation.")
            print("   Start Ollama with: ollama serve")
            raise SystemExit(1)

        t_rag = time.time() - t_start
        print(f"   Pipeline completed in {t_rag:.1f}s ({t_rag/len(questions):.1f}s/question)")

        empty_answer_indexes = _indices_respuestas_vacias(answers, len(questions))
        if empty_answer_indexes:
            listed = ", ".join(str(i + 1) for i in empty_answer_indexes[:20])
            suffix = "..." if len(empty_answer_indexes) > 20 else ""
            failed_summary = _resumen_estados_fallidos(
                answers,
                question_statuses,
                len(questions),
            )
            print(
                "\nError: RAGAS evaluation was not launched because "
                f"{len(empty_answer_indexes)} answer(s) are empty."
            )
            print(f"   Empty question indexes: {listed}{suffix}")
            for reason, indexes in sorted(failed_summary.items()):
                reason_listed = ", ".join(str(n) for n in indexes[:20])
                reason_suffix = "..." if len(indexes) > 20 else ""
                print(f"   {reason}: {reason_listed}{reason_suffix}")
            print(f"   Checkpoint: {resolved_checkpoint_path}")
            print("   Fix the RAG generation issue and rerun; only empty answers will be retried.")
            raise SystemExit(1)

        return {
            "dataset_path": os.path.abspath(dataset_path),
            "output_path": resolved_output_path,
            "debug_path": resolved_debug_path,
            "checkpoint_path": resolved_checkpoint_path,
            "questions": questions,
            "ground_truths": ground_truths,
            "answers": answers,
            "contexts_list": contexts_list,
            "question_statuses": question_statuses,
            "questions_count": len(questions),
            "indexed_fragments": total,
            "indexed_files_filter": solo_archivos,
            "recomp_enabled": rag_runtime.USAR_RECOMP_SYNTHESIS,
            "pipeline_flags": current_pipeline_flags,
            "eval_corpus": eval_corpus,
            "docs_dir": rag_runtime.CARPETA_DOCS,
            "ragbench_reranker_low_score_fallback": ragbench_reranker_fallback,
            "pipeline_seconds": t_rag,
            "tiene_ground_truth": tiene_ground_truth,
        }
    finally:
        if previous_ragbench_reranker_fallback is not None:
            rag_runtime.set_ragbench_reranker_low_score_fallback(previous_ragbench_reranker_fallback)
        if docs_previous is not None:
            rag_runtime.CARPETA_DOCS, rag_runtime.PATH_DB, rag_runtime.COLLECTION_NAME = (
                docs_previous
            )
        if previous_pipeline_flags is not None:
            rag_runtime.set_pipeline_flags(previous_pipeline_flags)


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
) -> dict[str, Any]:
    """Evaluate previously generated RAG answers with RAGAS and save artifacts."""
    ragas = _importar_ragas_componentes()
    eval_llm, eval_embeddings = configurar_llm_evaluacion(
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

    selected_metric_names = _parse_ragas_metric_names(
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
    imprimir_resultados(df_scores, questions)

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

    metric_cols = [c for c in METRIC_NAMES if c in df_scores.columns]
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
        "indexed_fragments": generation["indexed_fragments"],
        "recomp_enabled": generation["recomp_enabled"],
        "pipeline_flags": generation.get("pipeline_flags", {}),
        "eval_corpus": generation["eval_corpus"],
        "docs_dir": generation.get("docs_dir"),
        "pipeline_seconds": generation["pipeline_seconds"],
        "evaluation_seconds": t_eval,
        "mean_scores": mean_scores,
        "checkpoint_path": os.path.abspath(generation["checkpoint_path"]),
    }


def ejecutar_comparativa_pipeline(
    dataset_path: str,
    scores_root: str = COMPARISON_SCORES_DIR,
    debug_root: str = COMPARISON_DEBUG_DIR,
    verbose: bool = False,
    save_debug: bool = True,
    label: str | None = None,
    force_reindex: bool = False,
    eval_corpus: str = "es",
    docs_dir: str | None = None,
    ragas_timeout: int = 90,
    ragas_max_retries: int = 5,
    ragas_max_wait: int = 60,
    ragas_max_workers: int = 1,
    ragas_batch_size: int | None = 5,
    ragas_metrics: str | None = None,
    google_timeout: int | None = None,
    google_retries: int | None = None,
    raise_exceptions: bool = False,
    suite: str = "ablation",
    variant_names: str | None = None,
) -> dict[str, Any]:
    """Run all selected pipeline variants, then evaluate them with RAGAS."""
    variants = seleccionar_variantes(suite, variant_names)
    baseline_variant = (
        "baseline_all_on"
        if any(v["name"] == "baseline_all_on" for v in variants)
        else variants[0]["name"]
    )

    dataset_path = resolver_ruta_dataset(dataset_path)
    run_slug = _build_run_slug(dataset_path=dataset_path, label=label, eval_corpus=eval_corpus)
    if os.path.abspath(scores_root) == os.path.abspath(COMPARISON_RUNS_DIR):
        run_dir = os.path.join(COMPARISON_RUNS_DIR, run_slug)
        scores_dir = os.path.join(run_dir, "scores")
        debug_dir = os.path.join(run_dir, "debug")
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        summary_dir = run_dir
    else:
        scores_dir = os.path.join(scores_root, run_slug)
        debug_dir = os.path.join(debug_root, run_slug)
        checkpoints_dir = os.path.join(CHECKPOINTS_DIR, "comparison_runs", run_slug)
        summary_dir = debug_dir
    os.makedirs(scores_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    generations = []
    for index, variant in enumerate(variants):
        variant_name = variant["name"]
        should_reindex = force_reindex and index == 0
        print("\n" + "=" * 70)
        print(f"Launching RAG inference: {variant_name}")
        print(f"Variant: {variant['description']}")
        print("=" * 70)

        generation = generar_respuestas_rag(
            dataset_path=dataset_path,
            output_path=os.path.join(scores_dir, f"{variant_name}.csv"),
            debug_path=os.path.join(debug_dir, f"{variant_name}.json"),
            checkpoint_path=os.path.join(checkpoints_dir, f"{variant_name}.json"),
            verbose=verbose,
            force_reindex=should_reindex,
            pipeline_flags=variant["flags"],
            eval_corpus=eval_corpus,
            docs_dir=docs_dir,
        )
        generation["variant"] = variant_name
        generation["variant_description"] = variant["description"]
        generation["requested_pipeline_flags"] = dict(variant["flags"])
        generations.append(generation)

    print("\n" + "=" * 70)
    print("RAG inference finished for all variants. Starting RAGAS back-to-back.")
    print("=" * 70)

    results = []
    for generation in generations:
        variant_name = generation["variant"]
        print("\n" + "=" * 70)
        print(f"Launching RAGAS evaluation: {variant_name}")
        print("=" * 70)

        result = evaluar_respuestas_con_ragas(
            generation=generation,
            save_debug=save_debug,
            ragas_timeout=ragas_timeout,
            ragas_max_retries=ragas_max_retries,
            ragas_max_wait=ragas_max_wait,
            ragas_max_workers=ragas_max_workers,
            ragas_batch_size=ragas_batch_size,
            ragas_metrics=ragas_metrics,
            google_timeout=google_timeout,
            google_retries=google_retries,
            raise_exceptions=raise_exceptions,
        )
        result["variant"] = variant_name
        result["variant_description"] = generation["variant_description"]
        result["requested_pipeline_flags"] = generation["requested_pipeline_flags"]
        results.append(result)

    manifest = {
        "dataset_path": os.path.abspath(dataset_path),
        "eval_corpus": eval_corpus,
        "docs_dir": os.path.abspath(docs_dir) if docs_dir else _default_docs_dir_for_corpus(eval_corpus),
        "suite": suite,
        "baseline_variant": baseline_variant,
        "selected_variants": [variant["name"] for variant in variants],
        "scores_dir": os.path.abspath(scores_dir),
        "debug_dir": os.path.abspath(debug_dir),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": results,
        "mean_score_deltas_vs_baseline": _mean_score_deltas(results, baseline_variant),
    }
    _guardar_json(os.path.join(summary_dir, "comparison_summary.json"), manifest)
    return manifest


def ejecutar_evaluacion(
    dataset_path: str,
    output_path: str | None = None,
    debug_path: str | None = None,
    checkpoint_path: str | None = None,
    verbose: bool = False,
    save_debug: bool = True,
    force_reindex: bool = False,
    recomp_enabled: bool | None = None,
    pipeline_flags: dict[str, bool] | None = None,
    eval_corpus: str = "es",
    docs_dir: str | None = None,
    solo_archivos: list[str] | None = None,
    add_missing_from_filter: bool = False,
    ragas_timeout: int = 90,
    ragas_max_retries: int = 5,
    ragas_max_wait: int = 60,
    ragas_max_workers: int = 1,
    ragas_batch_size: int | None = 5,
    ragas_metrics: str | None = None,
    google_timeout: int | None = None,
    google_retries: int | None = None,
    raise_exceptions: bool = False,
) -> dict[str, Any]:
    """Run the full live RAGAS evaluation and persist its artifacts.

    Args:
        dataset_path: Path to the question dataset.
        output_path: CSV output path. Defaults to ``evaluation/runs/ragas/single/<tag>/scores.csv``.
        debug_path: Debug JSON output path. Defaults to ``evaluation/runs/ragas/single/<tag>/debug.json``.
        checkpoint_path: Progress JSON path used to resume question generation.
        verbose: Whether to print per-question progress.
        save_debug: Whether to persist the debug JSON.
        force_reindex: Whether to rebuild the ChromaDB collection before evaluation.
        recomp_enabled: Optional in-process override for ``USAR_RECOMP_SYNTHESIS``.
        eval_corpus: Corpus preset (``es``, ``ca``, ``en``, ``mix``). Resolves the PDF
            folder under ``rag/docs/<lang>/`` and adds ``_<lang>`` to output filenames.
        ragas_timeout: Per-call timeout used by RAGAS.
        ragas_max_retries: Max retries for failed RAGAS jobs.
        ragas_max_wait: Max backoff wait between RAGAS retries.
        ragas_max_workers: Concurrent RAGAS workers.
        ragas_batch_size: Optional RAGAS batch size.
        ragas_metrics: Comma-separated list of RAGAS metrics, or ``all``.
        google_timeout: Timeout passed to the Gemini LLM and embeddings clients.
        google_retries: Retries passed to the Gemini LLM client.
        raise_exceptions: Stop immediately on a RAGAS job failure.

    Returns:
        Summary dictionary with output paths, timings, and mean metrics.
    """
    generation = generar_respuestas_rag(
        dataset_path=dataset_path,
        output_path=output_path,
        debug_path=debug_path,
        checkpoint_path=checkpoint_path,
        verbose=verbose,
        force_reindex=force_reindex,
        recomp_enabled=recomp_enabled,
        pipeline_flags=pipeline_flags,
        eval_corpus=eval_corpus,
        docs_dir=docs_dir,
        solo_archivos=solo_archivos,
        add_missing_from_filter=add_missing_from_filter,
    )
    return evaluar_respuestas_con_ragas(
        generation=generation,
        save_debug=save_debug,
        ragas_timeout=ragas_timeout,
        ragas_max_retries=ragas_max_retries,
        ragas_max_wait=ragas_max_wait,
        ragas_max_workers=ragas_max_workers,
        ragas_batch_size=ragas_batch_size,
        ragas_metrics=ragas_metrics,
        google_timeout=google_timeout,
        google_retries=google_retries,
        raise_exceptions=raise_exceptions,
    )

# ─────────────────────────────────────────────
# SECTION 7: RAGBENCH PREPARATION
# ─────────────────────────────────────────────

HF_REPO = "vectara/open_ragbench"
HF_SUBDIR = "pdf/arxiv"
HF_METADATA_FILES = ("queries.json", "qrels.json", "answers.json", "pdf_urls.json")

ARXIV_DELAY_SECS = 5
ARXIV_TIMEOUT_SECS = 60
ARXIV_HEADERS = {
    "User-Agent": "MonkeyGrab-TFG-Eval/1.0 (academic research; Universitat Politecnica de Valencia)"
}

RAGBENCH_PDFS_DIR = os.path.join(_proj_root, "rag", "docs", "en")
_RAGBENCH_PREPARED_DIR = RAGBENCH_PREPARED_DIR
RAGBENCH_LEGACY_DEV_PDFS_DIR = RAGBENCH_PDFS_DIR
RAGBENCH_EVAL_PDFS_DIR = os.path.join(_proj_root, "rag", "docs", "en_ragbench_eval")
RAGBENCH_DEV_DOC_IDS_PATH = os.path.join(RAGBENCH_DATASETS_DIR, "ragbench_en_dev_doc_ids.json")
RAGBENCH_EVAL_PREPARED_DIR = os.path.join(_RAGBENCH_PREPARED_DIR, "en_eval")
RAGBENCH_DEV_FROZEN_PREPARED_DIR = os.path.join(_RAGBENCH_PREPARED_DIR, "dev_frozen")
RAGBENCH_EVAL_MANIFEST_PATH = os.path.join(RAGBENCH_EVAL_PREPARED_DIR, "ragbench_en_eval_manifest.json")

RAGBENCH_FINAL_PIPELINE_FLAGS = {
    **BASELINE_PIPELINE_FLAGS,
    "USAR_LLM_QUERY_DECOMPOSITION": False,
}


def descargar_metadatos() -> tuple[dict, dict, dict, dict]:
    """Download RagBench metadata through Hugging Face cache."""
    if not _HF_AVAILABLE:
        print("ERROR: huggingface_hub no instalado. Ejecuta: pip install huggingface-hub")
        raise SystemExit(1)

    print("Descargando metadatos de vectara/open_ragbench (cache tras la primera vez)...")
    loaded: dict = {}
    for fname in HF_METADATA_FILES:
        print(f"   {fname}...", end=" ", flush=True)
        local_path = _hf_hub_download(
            repo_id=HF_REPO,
            filename=f"{HF_SUBDIR}/{fname}",
            repo_type="dataset",
        )
        with open(local_path, encoding="utf-8") as f:
            loaded[fname] = json.load(f)
        print("OK")

    return (
        loaded["queries.json"],
        loaded["qrels.json"],
        loaded["answers.json"],
        loaded["pdf_urls.json"],
    )


def seleccionar_papers(
    queries: dict,
    qrels: dict,
    pdf_urls: dict,
    n_papers: int,
    source_filter: str | None,
    excluded_doc_ids: list[str] | None = None,
) -> list[str]:
    """Select top-N papers by eligible question count."""
    from collections import Counter
    excluded = set(excluded_doc_ids or [])
    paper_counts: Counter = Counter()
    for qid, qrel in qrels.items():
        doc_id = qrel.get("doc_id")
        if not doc_id or qid not in queries or doc_id not in pdf_urls:
            continue
        if doc_id in excluded:
            continue
        if source_filter and queries[qid].get("source") != source_filter:
            continue
        paper_counts[doc_id] += 1

    selected = [pid for pid, _ in paper_counts.most_common(n_papers)]
    src_label = f"source='{source_filter}'" if source_filter else "todos los tipos"
    print(f"\nPapers seleccionados ({src_label}, top-{n_papers} por numero de preguntas):")
    for pid in selected:
        print(f"   {pid}  ({paper_counts[pid]} preguntas elegibles)")
    return selected


def seleccionar_papers_objetivo(
    only_doc: str | None,
    queries: dict,
    qrels: dict,
    pdf_urls: dict,
    n_papers: int,
    source_filter: str | None,
    excluded_doc_ids: list[str] | None = None,
) -> list[str]:
    """Resolve either only_doc or the top-N paper selection."""
    if not only_doc:
        return seleccionar_papers(
            queries,
            qrels,
            pdf_urls,
            n_papers,
            source_filter,
            excluded_doc_ids=excluded_doc_ids,
        )

    doc_id = only_doc.strip()
    if doc_id not in pdf_urls:
        print(f"ERROR: doc_id '{doc_id}' no encontrado en pdf_urls del dataset.")
        raise SystemExit(1)
    if excluded_doc_ids and doc_id in set(excluded_doc_ids):
        print(f"ERROR: doc_id '{doc_id}' pertenece al dev split congelado y está excluido.")
        raise SystemExit(1)

    n_eligible = sum(
        1
        for qid, qrel in qrels.items()
        if qrel.get("doc_id") == doc_id
        and qid in queries
        and (not source_filter or queries[qid].get("source") == source_filter)
    )
    if n_eligible == 0:
        print(
            f"ERROR: no hay preguntas para '{doc_id}' con source='{source_filter}'. "
            "Prueba con --source all."
        )
        raise SystemExit(1)

    print(f"\n--only-doc: paper unico {doc_id} ({n_eligible} preguntas elegibles)")
    return [doc_id]


def construir_preguntas(
    queries: dict,
    qrels: dict,
    answers: dict,
    selected_papers: list[str],
    source_filter: str | None,
    max_per_paper: int,
) -> tuple[list[str], list[str], list[str]]:
    """Build aligned question, ground-truth and paper-id lists."""
    from collections import Counter
    selected_set = set(selected_papers)
    per_paper: dict[str, list[str]] = {p: [] for p in selected_papers}

    for qid, qrel in qrels.items():
        doc_id = qrel.get("doc_id")
        if doc_id not in selected_set or qid not in queries:
            continue
        if source_filter and queries[qid].get("source") != source_filter:
            continue
        per_paper[doc_id].append(qid)

    questions: list[str] = []
    ground_truths: list[str] = []
    paper_ids: list[str] = []

    for paper_id in selected_papers:
        chosen = per_paper[paper_id][:max_per_paper]
        if chosen:
            by_src = Counter(queries[q].get("source", "?") for q in chosen)
            print(f"   {paper_id}: {dict(by_src)}")
        for qid in chosen:
            questions.append(queries[qid]["query"])
            ground_truths.append(answers.get(qid, ""))
            paper_ids.append(paper_id)

    print(
        f"\nTotal preguntas: {len(questions)} de {len(selected_papers)} papers "
        f"(max {max_per_paper}/paper)"
    )
    return questions, ground_truths, paper_ids


def filtrar_por_pdfs_disponibles(
    questions: list[str],
    ground_truths: list[str],
    paper_ids: list[str],
    available_papers: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Keep only question rows whose PDF exists locally."""
    available_set = set(available_papers)
    filtered = [
        (q, gt, pid)
        for q, gt, pid in zip(questions, ground_truths, paper_ids)
        if pid in available_set
    ]
    if not filtered:
        return [], [], []
    out_q, out_gt, out_pid = map(list, zip(*filtered))
    return out_q, out_gt, out_pid


def descargar_pdfs(
    selected_papers: list[str],
    pdf_urls: dict,
    pdfs_dir: str,
    skip_existing: bool = True,
) -> list[str]:
    """Download selected PDFs from RagBench/arXiv."""
    if not _REQUESTS_AVAILABLE:
        print("ERROR: requests no instalado. Ejecuta: pip install requests")
        raise SystemExit(1)
    os.makedirs(pdfs_dir, exist_ok=True)
    successful: list[str] = []

    print(f"\nDescargando {len(selected_papers)} PDFs en {pdfs_dir}/")
    for i, paper_id in enumerate(selected_papers):
        out_path = os.path.join(pdfs_dir, f"{paper_id}.pdf")

        if skip_existing and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"   [{i + 1}/{len(selected_papers)}] {paper_id}  (en cache, omitido)")
            successful.append(paper_id)
            continue

        url = pdf_urls[paper_id]
        print(f"   [{i + 1}/{len(selected_papers)}] {paper_id} <- {url}")
        try:
            resp = _requests.get(url, headers=ARXIV_HEADERS, timeout=ARXIV_TIMEOUT_SECS)
            resp.raise_for_status()
        except Exception as e:
            print(f"      ERROR al descargar: {e}")
            continue

        content_type = resp.headers.get("Content-Type", "")
        if "application/pdf" not in content_type and not resp.content.startswith(b"%PDF"):
            print(f"      AVISO: Content-Type inesperado '{content_type}', omitido")
            continue

        with open(out_path, "wb") as fh:
            fh.write(resp.content)
        print(f"      {len(resp.content) / 1024:.0f} KB guardados")
        successful.append(paper_id)

        if i < len(selected_papers) - 1:
            time.sleep(ARXIV_DELAY_SECS)

    return successful


def obtener_pdfs_disponibles(selected_papers: list[str], pdfs_dir: str) -> list[str]:
    """Return selected papers whose local PDF exists and is non-empty."""
    return [
        pid
        for pid in selected_papers
        if os.path.exists(os.path.join(pdfs_dir, f"{pid}.pdf"))
        and os.path.getsize(os.path.join(pdfs_dir, f"{pid}.pdf")) > 0
    ]


def _safe_tag(value: str) -> str:
    """Convert an arbitrary string to a safe filesystem tag."""
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def cargar_doc_ids_dev_ragbench(path: str = RAGBENCH_DEV_DOC_IDS_PATH) -> list[str]:
    """Load the frozen RagBench EN dev-split doc_ids."""
    if not os.path.isfile(path):
        print(f"ERROR: no existe el listado de doc_ids dev: {path}")
        raise SystemExit(1)

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list) or not all(isinstance(x, str) and x.strip() for x in payload):
        print(f"ERROR: el fichero de doc_ids dev no tiene el formato esperado: {path}")
        raise SystemExit(1)

    seen: set[str] = set()
    ordered: list[str] = []
    for raw in payload:
        doc_id = raw.strip()
        if doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


def escribir_manifest_ragbench_eval(path: str, payload: dict[str, Any]) -> str:
    """Persist the prepared RagBench EN evaluation manifest."""
    _guardar_json(path, payload)
    return path


def cargar_manifest_ragbench_eval(path: str = RAGBENCH_EVAL_MANIFEST_PATH) -> dict[str, Any]:
    """Load the RagBench EN evaluation manifest."""
    if not os.path.isfile(path):
        print(f"ERROR: no existe el manifiesto RagBench EN: {path}")
        raise SystemExit(1)
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        print(f"ERROR: manifiesto RagBench EN inválido: {path}")
        raise SystemExit(1)
    return manifest


def _selected_pdf_filenames_from_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return stable PDF filenames from a RagBench EN manifest."""
    selected_papers = manifest.get("selected_papers") or []
    indexed_files = manifest.get("indexed_files") or []
    if indexed_files:
        return [str(name) for name in indexed_files]
    return [f"{paper_id}.pdf" for paper_id in selected_papers]


def escribir_dataset_preparado(
    questions: list[str],
    ground_truths: list[str],
    paper_ids: list[str],
    safe_tag: str,
    filename_prefix: str = "dataset_ragbench",
    output_dir: str | None = None,
) -> str:
    """Write a temporary JSON dataset consumable by ejecutar_evaluacion."""
    target_dir = output_dir or _RAGBENCH_PREPARED_DIR
    os.makedirs(target_dir, exist_ok=True)
    prepared_dataset = os.path.join(target_dir, f"{filename_prefix}_{safe_tag}.json")
    with open(prepared_dataset, "w", encoding="utf-8") as f:
        json.dump(
            [
                {"question": q, "ground_truth": gt, "paper_id": pid}
                for q, gt, pid in zip(questions, ground_truths, paper_ids)
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    return prepared_dataset


def preparar_ragbench_eval_en(
    source: str = "text",
    n_papers: int = 25,
    max_q: int = 5,
    skip_download: bool = False,
    docs_dir: str = RAGBENCH_EVAL_PDFS_DIR,
    manifest_path: str = RAGBENCH_EVAL_MANIFEST_PATH,
    excluded_doc_ids_path: str = RAGBENCH_DEV_DOC_IDS_PATH,
) -> dict[str, Any]:
    """Prepare the fixed English RagBench evaluation corpus and dataset."""
    if source == "all":
        print("ERROR: el flujo RagBench EN final requiere una fuente fija; usa source='text'.")
        raise SystemExit(1)
    if max_q < 1 or n_papers < 1:
        print("ERROR: n_papers y max_q deben ser >= 1.")
        raise SystemExit(1)

    source_filter = source
    excluded_doc_ids = cargar_doc_ids_dev_ragbench(excluded_doc_ids_path)

    print(f"\nPreparando RagBench EN final: source={source_filter}, n_papers={n_papers}, max_q={max_q}")
    print(f"Excluyendo dev split congelado: {len(excluded_doc_ids)} doc_ids")

    queries, qrels, answers_gt, pdf_urls = descargar_metadatos()
    selected_papers = seleccionar_papers_objetivo(
        only_doc=None,
        queries=queries,
        qrels=qrels,
        pdf_urls=pdf_urls,
        n_papers=n_papers,
        source_filter=source_filter,
        excluded_doc_ids=excluded_doc_ids,
    )

    print("\nSeleccionando preguntas:")
    questions, ground_truths, paper_ids = construir_preguntas(
        queries,
        qrels,
        answers_gt,
        selected_papers,
        source_filter,
        max_q,
    )
    if not questions:
        print("ERROR: No se seleccionaron preguntas con los filtros actuales.")
        raise SystemExit(1)

    if skip_download:
        print(f"\n--skip-download: usando PDFs existentes en {docs_dir}/")
        successful_papers = obtener_pdfs_disponibles(selected_papers, docs_dir)
    else:
        successful_papers = descargar_pdfs(selected_papers, pdf_urls, docs_dir)

    if len(successful_papers) < len(selected_papers):
        missing = sorted(set(selected_papers) - set(successful_papers))
        print(f"\nAVISO: {len(successful_papers)}/{len(selected_papers)} PDFs disponibles.")
        print(f"   Faltantes: {missing}")

    questions, ground_truths, paper_ids = filtrar_por_pdfs_disponibles(
        questions,
        ground_truths,
        paper_ids,
        successful_papers,
    )
    if not questions:
        print("ERROR: No quedan preguntas tras filtrar por PDFs disponibles.")
        print("   Ejecuta sin --skip-download para descargar los PDFs necesarios.")
        raise SystemExit(1)

    dataset_tag = f"{source_filter}_{len(successful_papers)}p_{max_q}q_eval"
    safe_tag = _safe_tag(dataset_tag)
    prepared_dataset = escribir_dataset_preparado(
        questions,
        ground_truths,
        paper_ids,
        safe_tag,
        filename_prefix="dataset_ragbench_en_eval",
        output_dir=RAGBENCH_EVAL_PREPARED_DIR,
    )
    indexed_files = [f"{paper_id}.pdf" for paper_id in successful_papers]
    manifest = {
        "manifest_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source_filter,
        "n_papers": len(successful_papers),
        "max_q": max_q,
        "docs_dir": os.path.abspath(docs_dir),
        "dataset_path": os.path.abspath(prepared_dataset),
        "selected_papers": successful_papers,
        "indexed_files": indexed_files,
        "excluded_doc_ids": excluded_doc_ids,
        "excluded_doc_ids_path": os.path.abspath(excluded_doc_ids_path),
    }
    escribir_manifest_ragbench_eval(manifest_path, manifest)
    print(f"\nDataset preparado en: {prepared_dataset}")
    print(f"Manifiesto escrito en: {manifest_path}")
    return manifest


def ejecutar_ragbench_eval_en(
    manifest_path: str = RAGBENCH_EVAL_MANIFEST_PATH,
    verbose: bool = False,
    save_debug: bool = True,
    force_reindex: bool = False,
    ragas_max_workers: int = 5,
    ragas_batch_size: int | None = 15,
) -> dict[str, Any]:
    """Run the fixed English RagBench evaluation from a prepared manifest."""
    manifest = cargar_manifest_ragbench_eval(manifest_path)
    dataset_path = os.path.abspath(manifest["dataset_path"])
    docs_dir = os.path.abspath(manifest["docs_dir"])
    indexed_files = _selected_pdf_filenames_from_manifest(manifest)
    if not indexed_files:
        print("ERROR: el manifiesto RagBench EN no contiene indexed_files.")
        raise SystemExit(1)

    safe_tag = _safe_tag(Path(dataset_path).stem)
    run_dir = os.path.join(RAGBENCH_RUNS_DIR, "en_eval", safe_tag)
    output_csv = os.path.join(run_dir, "scores.csv")
    output_debug = os.path.join(run_dir, "debug.json")
    checkpoint_path = os.path.join(run_dir, "checkpoint.json")

    print("\nEjecutando RagBench EN final con configuración fija:")
    print("   source=text, query_decomposition=off, resto de flags=on")
    print(f"   dataset: {dataset_path}")
    print(f"   docs_dir: {docs_dir}")
    print(f"   indexed files: {len(indexed_files)}")
    print(f"   RAGAS workers/batch: {ragas_max_workers}/{ragas_batch_size}")

    return ejecutar_evaluacion(
        dataset_path=dataset_path,
        output_path=output_csv,
        debug_path=None if not save_debug else output_debug,
        checkpoint_path=checkpoint_path,
        verbose=verbose,
        save_debug=save_debug,
        force_reindex=force_reindex,
        pipeline_flags=RAGBENCH_FINAL_PIPELINE_FLAGS,
        eval_corpus="ragbench",
        docs_dir=docs_dir,
        solo_archivos=indexed_files,
        add_missing_from_filter=True,
        ragas_timeout=600,
        ragas_max_retries=15,
        ragas_max_wait=120,
        ragas_max_workers=ragas_max_workers,
        ragas_batch_size=ragas_batch_size,
    )


def _ejecutar_ragbench(
    source: str = "text",
    only_doc: str | None = None,
    n_papers: int = 10,
    max_q: int = 5,
    skip_download: bool = False,
    force_reindex: bool = False,
    verbose: bool = False,
    save_debug: bool = True,
    recomp_enabled: bool | None = None,
    ragas_max_workers: int = 1,
    ragas_batch_size: int | None = 5,
    prepare_only: bool = False,
) -> None:
    """Prepare a RagBench subset and evaluate it with the unified runner."""
    if max_q < 1:
        print("ERROR: max_q debe ser >= 1.")
        raise SystemExit(1)
    if not only_doc and n_papers < 1:
        print("ERROR: n_papers debe ser >= 1.")
        raise SystemExit(1)

    source_filter = None if source == "all" else source
    print(f"\nFiltro de fuente: {source}")

    queries, qrels, answers_gt, pdf_urls = descargar_metadatos()
    selected_papers = seleccionar_papers_objetivo(
        only_doc=only_doc,
        queries=queries,
        qrels=qrels,
        pdf_urls=pdf_urls,
        n_papers=n_papers,
        source_filter=source_filter,
    )

    print("\nSeleccionando preguntas:")
    questions, ground_truths, paper_ids = construir_preguntas(
        queries, qrels, answers_gt, selected_papers, source_filter, max_q
    )
    if not questions:
        print("ERROR: No se seleccionaron preguntas con los filtros actuales.")
        raise SystemExit(1)

    if skip_download:
        print(f"\n--skip-download: usando PDFs existentes en {RAGBENCH_PDFS_DIR}/")
        successful_papers = obtener_pdfs_disponibles(selected_papers, RAGBENCH_PDFS_DIR)
    else:
        successful_papers = descargar_pdfs(selected_papers, pdf_urls, RAGBENCH_PDFS_DIR)

    if len(successful_papers) < len(selected_papers):
        missing = set(selected_papers) - set(successful_papers)
        print(f"\nAVISO: {len(successful_papers)}/{len(selected_papers)} PDFs disponibles.")
        print(f"   Faltantes: {missing}")

    questions, ground_truths, paper_ids = filtrar_por_pdfs_disponibles(
        questions, ground_truths, paper_ids, successful_papers
    )
    if not questions:
        print("ERROR: No quedan preguntas tras filtrar por PDFs disponibles.")
        print("   Ejecuta sin --skip-download para descargar los PDFs necesarios.")
        raise SystemExit(1)

    dataset_tag = only_doc.strip() if only_doc else f"{source}_{len(successful_papers)}p_{max_q}q"
    safe_tag = _safe_tag(dataset_tag)
    prepared_dataset = escribir_dataset_preparado(
        questions,
        ground_truths,
        paper_ids,
        safe_tag,
        output_dir=RAGBENCH_DEV_FROZEN_PREPARED_DIR,
    )
    solo_pdf = [f"{only_doc.strip()}.pdf"] if only_doc else None
    effective_recomp = rag_runtime.USAR_RECOMP_SYNTHESIS if recomp_enabled is None else bool(recomp_enabled)
    recomp_tag = "recomp_on" if effective_recomp else "recomp_off"
    run_dir = os.path.join(RAGBENCH_RUNS_DIR, "legacy", f"{safe_tag}_{recomp_tag}")
    output_csv = os.path.join(run_dir, "scores.csv")
    output_debug = os.path.join(run_dir, "debug.json")
    checkpoint_path = os.path.join(run_dir, "checkpoint.json")

    if prepare_only:
        print(f"\nDataset preparado en: {prepared_dataset}")
        print("--prepare-only: saltando generacion y RAGAS. Usa 'compare' con este dataset.")
        return

    print("\nDelegando generacion, checkpoints y RAGAS al runner unificado...")
    ejecutar_evaluacion(
        dataset_path=prepared_dataset,
        output_path=output_csv,
        debug_path=None if not save_debug else output_debug,
        checkpoint_path=checkpoint_path,
        verbose=verbose,
        save_debug=save_debug,
        force_reindex=force_reindex,
        recomp_enabled=recomp_enabled,
        eval_corpus="ragbench",
        docs_dir=RAGBENCH_PDFS_DIR,
        solo_archivos=solo_pdf,
        ragas_timeout=600,
        ragas_max_retries=15,
        ragas_max_wait=120,
        ragas_max_workers=ragas_max_workers,
        ragas_batch_size=ragas_batch_size,
    )


# ─────────────────────────────────────────────
# SECTION 8: MAIN
# ─────────────────────────────────────────────

def _add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default=None, help="Path to the dataset (JSON, CSV, or Excel).")
    parser.add_argument("--corpus", choices=SUPPORTED_CORPORA, default="es", help="Local corpus/dataset preset")
    parser.add_argument("--docs-dir", default=None, help="Override PDF folder for this run")
    parser.add_argument("--catalan", action="store_true", help="Legacy alias for --corpus ca")
    parser.add_argument("--verbose", action="store_true", help="Show per-question progress")
    parser.add_argument("--no-debug", action="store_true", help="Skip saving debug JSON")
    parser.add_argument("--ragas-timeout", type=int, default=90, help="Per-call RAGAS timeout in seconds")
    parser.add_argument("--ragas-max-retries", type=int, default=5, help="Maximum RAGAS retries per failed job")
    parser.add_argument("--ragas-max-wait", type=int, default=60, help="Maximum wait between RAGAS retries")
    parser.add_argument("--ragas-max-workers", type=int, default=1, help="Concurrent RAGAS workers")
    parser.add_argument("--ragas-batch-size", type=int, default=5, help="RAGAS batch size")
    parser.add_argument(
        "--ragas-metrics",
        default=None,
        help="Comma-separated RAGAS metrics or 'all'",
    )
    parser.add_argument("--google-timeout", type=int, default=None, help="Gemini timeout in seconds")
    parser.add_argument("--google-retries", type=int, default=None, help="Gemini retries")
    parser.add_argument("--raise-exceptions", action="store_true", help="Stop immediately on RAGAS failures")


def _resolve_corpus_arg(args: argparse.Namespace) -> str:
    if getattr(args, "catalan", False):
        return "ca"
    return args.corpus


def _run_single_from_args(args: argparse.Namespace) -> None:
    eval_corpus = _resolve_corpus_arg(args)
    dataset_arg = args.dataset if args.dataset is not None else _default_dataset_for_corpus(eval_corpus)
    recomp_arg = False if getattr(args, "no_recomp", False) else None

    ejecutar_evaluacion(
        dataset_path=dataset_arg,
        output_path=args.output,
        debug_path=args.debug_output,
        checkpoint_path=args.checkpoint,
        verbose=args.verbose,
        save_debug=not args.no_debug,
        force_reindex=args.force_reindex,
        recomp_enabled=recomp_arg,
        eval_corpus=eval_corpus,
        docs_dir=args.docs_dir,
        ragas_timeout=args.ragas_timeout,
        ragas_max_retries=args.ragas_max_retries,
        ragas_max_wait=args.ragas_max_wait,
        ragas_max_workers=args.ragas_max_workers,
        ragas_batch_size=args.ragas_batch_size,
        ragas_metrics=args.ragas_metrics,
        google_timeout=args.google_timeout,
        google_retries=args.google_retries,
        raise_exceptions=args.raise_exceptions,
    )


def _run_compare_from_args(args: argparse.Namespace) -> None:
    eval_corpus = _resolve_corpus_arg(args)
    dataset_arg = args.dataset if args.dataset is not None else _default_dataset_for_corpus(eval_corpus)

    manifest = ejecutar_comparativa_pipeline(
        dataset_path=dataset_arg,
        scores_root=args.scores_root,
        debug_root=args.debug_root,
        verbose=args.verbose,
        save_debug=not args.no_debug,
        label=args.label,
        force_reindex=args.reindex,
        eval_corpus=eval_corpus,
        docs_dir=args.docs_dir,
        suite=args.suite,
        variant_names=args.variants,
        ragas_timeout=args.ragas_timeout,
        ragas_max_retries=args.ragas_max_retries,
        ragas_max_wait=args.ragas_max_wait,
        ragas_max_workers=args.ragas_max_workers,
        ragas_batch_size=args.ragas_batch_size,
        ragas_metrics=args.ragas_metrics,
        google_timeout=args.google_timeout,
        google_retries=args.google_retries,
        raise_exceptions=args.raise_exceptions,
    )

    print("\nComparison finished.")
    print(f"Scores folder: {manifest['scores_dir']}")
    print(f"Debug folder:  {manifest['debug_dir']}")
    print("Summary file: " + os.path.join(manifest["debug_dir"], "comparison_summary.json"))


def _run_ragbench_prepare_from_args(args: argparse.Namespace) -> None:
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
    print(f"Dataset:   {manifest['dataset_path']}")
    print(f"Docs dir:  {manifest['docs_dir']}")
    print(f"Manifest:  {args.manifest}")


def _run_ragbench_eval_from_args(args: argparse.Namespace) -> None:
    result = ejecutar_ragbench_eval_en(
        manifest_path=args.manifest,
        verbose=args.verbose,
        save_debug=not args.no_debug,
        force_reindex=args.force_reindex,
        ragas_max_workers=args.ragas_max_workers,
        ragas_batch_size=args.ragas_batch_size,
    )
    print("\nRagBench EN evaluation finished.")
    print(f"CSV:   {result['output_path']}")
    if result.get("debug_path"):
        print(f"Debug: {result['debug_path']}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified RAG/RAGAS evaluation runner")
    subparsers = parser.add_subparsers(dest="command")

    single = subparsers.add_parser("single", help="Run one RAGAS evaluation")
    _add_common_eval_args(single)
    single.add_argument("--output", default=None, help="Output path for the results CSV")
    single.add_argument("--debug-output", default=None, help="Output path for ragas_debug.json")
    single.add_argument("--checkpoint", default=None, help="Path to the checkpoint JSON")
    single.add_argument("--force-reindex", action="store_true", help="Delete and rebuild the ChromaDB collection")
    single.add_argument("--no-recomp", action="store_true", help="Disable RECOMP synthesis for this run")
    single.set_defaults(func=_run_single_from_args)

    compare = subparsers.add_parser("compare", help="Run a pipeline ablation comparison")
    _add_common_eval_args(compare)
    compare.add_argument("--scores-root", default=COMPARISON_SCORES_DIR, help="Base directory for comparison CSV files")
    compare.add_argument("--debug-root", default=COMPARISON_DEBUG_DIR, help="Base directory for comparison debug files")
    compare.add_argument("--label", default=None, help="Folder suffix for this comparison batch")
    compare.add_argument("--suite", choices=sorted(VARIANT_SUITES), default="ablation", help="Variant suite to execute")
    compare.add_argument("--variants", default=None, help="Comma-separated variant names")
    compare.add_argument("--reindex", action="store_true", help="Rebuild the ChromaDB collection before the first variant")
    compare.set_defaults(func=_run_compare_from_args)

    list_variants = subparsers.add_parser("list-variants", help="List available comparison variants")
    list_variants.add_argument("--suite", choices=sorted(VARIANT_SUITES), default="ablation")
    list_variants.set_defaults(func=lambda args: listar_variantes(args.suite))

    ragbench = subparsers.add_parser("ragbench", help="Prepare and evaluate a RagBench subset")
    ragbench.add_argument("--source", default="text", choices=["text-image", "text-table", "text", "all"])
    ragbench.add_argument("--only-doc", default=None, help="Evaluate a single RagBench doc_id")
    ragbench.add_argument("--n-papers", type=int, default=10, help="Number of papers to select")
    ragbench.add_argument("--max-q", type=int, default=5, help="Maximum questions per paper")
    ragbench.add_argument("--skip-download", action="store_true", help="Use existing PDFs under rag/docs/en")
    ragbench.add_argument("--force-reindex", action="store_true", help="Rebuild the RagBench PDF collection")
    ragbench.add_argument("--verbose", action="store_true", help="Show per-question generation progress")
    ragbench.add_argument("--no-debug", action="store_true", help="Skip debug JSON")
    ragbench.add_argument("--ragas-max-workers", type=int, default=1, help="Concurrent RAGAS workers")
    ragbench.add_argument("--ragas-batch-size", type=int, default=5, help="RAGAS batch size")
    recomp_group = ragbench.add_mutually_exclusive_group()
    recomp_group.add_argument("--recomp", dest="recomp_enabled", action="store_true", help="Enable RECOMP synthesis")
    recomp_group.add_argument("--no-recomp", dest="recomp_enabled", action="store_false", help="Disable RECOMP synthesis")
    recomp_group.add_argument("--both-recomp", action="store_true", help="Run two passes: RECOMP enabled and disabled")
    ragbench.set_defaults(recomp_enabled=None)
    ragbench.add_argument("--prepare-only", action="store_true", help="Download PDFs and write dataset JSON; skip generation and RAGAS")
    ragbench.set_defaults(func=_run_ragbench_from_args)

    ragbench_prepare = subparsers.add_parser("ragbench-prepare", help="Prepare the fixed English RagBench evaluation corpus")
    ragbench_prepare.add_argument("--source", default="text", choices=["text"], help="Fixed RagBench source type")
    ragbench_prepare.add_argument("--n-papers", type=int, default=25, help="Number of evaluation papers to keep")
    ragbench_prepare.add_argument("--max-q", type=int, default=5, help="Maximum questions per paper")
    ragbench_prepare.add_argument("--skip-download", action="store_true", help="Use existing PDFs under the evaluation docs dir")
    ragbench_prepare.add_argument("--docs-dir", default=RAGBENCH_EVAL_PDFS_DIR, help="Target PDF folder for the final RagBench EN corpus")
    ragbench_prepare.add_argument("--manifest", default=RAGBENCH_EVAL_MANIFEST_PATH, help="Output path for the RagBench EN manifest")
    ragbench_prepare.add_argument("--excluded-doc-ids", default=RAGBENCH_DEV_DOC_IDS_PATH, help="JSON file with dev-split doc_ids to exclude")
    ragbench_prepare.set_defaults(func=_run_ragbench_prepare_from_args)

    ragbench_eval = subparsers.add_parser("ragbench-eval", help="Run the fixed English RagBench evaluation from its manifest")
    ragbench_eval.add_argument("--manifest", default=RAGBENCH_EVAL_MANIFEST_PATH, help="Input path for the RagBench EN manifest")
    ragbench_eval.add_argument("--force-reindex", action="store_true", help="Rebuild the evaluation Chroma collection from the manifest")
    ragbench_eval.add_argument("--verbose", action="store_true", help="Show per-question generation progress")
    ragbench_eval.add_argument("--no-debug", action="store_true", help="Skip debug JSON")
    ragbench_eval.add_argument("--ragas-max-workers", type=int, default=5, help="Concurrent RAGAS workers")
    ragbench_eval.add_argument("--ragas-batch-size", type=int, default=15, help="RAGAS batch size")
    ragbench_eval.set_defaults(func=_run_ragbench_eval_from_args)
    return parser


def _run_ragbench_from_args(args: argparse.Namespace) -> None:
    if args.both_recomp:
        common_kwargs = {
            "source": args.source,
            "only_doc": args.only_doc,
            "n_papers": args.n_papers,
            "max_q": args.max_q,
            "verbose": args.verbose,
            "save_debug": not args.no_debug,
            "ragas_max_workers": args.ragas_max_workers,
            "ragas_batch_size": args.ragas_batch_size,
        }
        _ejecutar_ragbench(
            **common_kwargs,
            skip_download=args.skip_download,
            force_reindex=args.force_reindex,
            recomp_enabled=True,
        )
        _ejecutar_ragbench(
            **common_kwargs,
            skip_download=True,
            force_reindex=False,
            recomp_enabled=False,
        )
        return

    _ejecutar_ragbench(
        source=args.source,
        only_doc=args.only_doc,
        n_papers=args.n_papers,
        max_q=args.max_q,
        skip_download=args.skip_download,
        force_reindex=args.force_reindex,
        verbose=args.verbose,
        save_debug=not args.no_debug,
        recomp_enabled=args.recomp_enabled,
        ragas_max_workers=args.ragas_max_workers,
        ragas_batch_size=args.ragas_batch_size,
        prepare_only=args.prepare_only,
    )


def main() -> None:
    """Unified entry point with backward-compatible single-run defaults."""
    legacy_commands = {"single", "compare", "list-variants", "ragbench", "ragbench-prepare", "ragbench-eval"}
    help_args = {"-h", "--help"}
    if len(sys.argv) == 1 or (
        len(sys.argv) > 1 and sys.argv[1] not in legacy_commands and sys.argv[1] not in help_args
    ):
        # Backward compatibility: ``python evaluation/run_eval.py --catalan`` still means single.
        sys.argv.insert(1, "single")

    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
