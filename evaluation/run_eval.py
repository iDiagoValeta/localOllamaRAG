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
    python evaluation/run_eval.py [--dataset PATH] [--output PATH] [--verbose] [--no-debug]
    python evaluation/run_eval.py --catalan   # rag/pdfs_ca + dataset_eval_ca.json + ``_ca`` outputs

Default dataset path: ``evaluation/datasets/dataset_eval_es.json`` (``dataset_eval_ca.json`` with ``--catalan``).
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
#
#  ENTRY
#  +-- 7. Main                 main()
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
    set_recomp_synthesis_enabled,
)

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
SCORES_DIR = os.path.join(EVAL_DIR, "scores")
DEBUG_DIR = os.path.join(EVAL_DIR, "debug")
CHECKPOINTS_DIR = os.path.join(DEBUG_DIR, "checkpoints")


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
        alt = os.path.join(EVAL_DIR, "datasets", name)
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


def _default_dataset_for_corpus(eval_corpus: Literal["es", "ca"]) -> str:
    """Default bundled JSON path for Spanish vs Catalan evaluation."""
    name = "dataset_eval_ca.json" if eval_corpus == "ca" else "dataset_eval_es.json"
    return os.path.join(EVAL_DIR, "datasets", name)


def _artifact_suffix(eval_corpus: Literal["es", "ca"]) -> str:
    """Filename suffix for scores/debug/checkpoints (Catalan corpus)."""
    return "_ca" if eval_corpus == "ca" else ""


def _slugify(value: str) -> str:
    """Convert a free-form label into a filesystem-friendly slug."""
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "eval"


def _build_output_stem(dataset_path: str) -> str:
    """Create a stable stem for outputs derived from the dataset filename."""
    return _slugify(Path(dataset_path).stem)


def _default_output_path(dataset_path: str, artifact_suffix: str = "") -> str:
    """Return the default CSV output path inside evaluation/scores."""
    return os.path.join(
        SCORES_DIR, f"ragas_scores_{_build_output_stem(dataset_path)}{artifact_suffix}.csv"
    )


def _default_debug_path(dataset_path: str, artifact_suffix: str = "") -> str:
    """Return the default debug JSON path inside evaluation/debug."""
    return os.path.join(
        DEBUG_DIR, f"ragas_debug_{_build_output_stem(dataset_path)}{artifact_suffix}.json"
    )


def _default_checkpoint_path(
    dataset_path: str, recomp_enabled: bool | None, artifact_suffix: str = ""
) -> str:
    """Return the checkpoint path used to resume generation question by question."""
    recomp_tag = "recomp_on" if recomp_enabled else "recomp_off"
    return os.path.join(
        CHECKPOINTS_DIR,
        f"ragas_progress_{_build_output_stem(dataset_path)}{artifact_suffix}_{recomp_tag}.json",
    )


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
) -> None:
    """Save generation progress with a consistent payload."""
    _guardar_checkpoint(
        checkpoint_path,
        {
            "dataset_path": dataset_path,
            "questions_count": questions_count,
            "recomp_enabled": recomp_enabled,
            "eval_corpus": eval_corpus,
            "output_path": output_path,
            "debug_path": debug_path,
            "completed_questions": len([a for a in answers if not _respuesta_vacia(a)]),
            "answers": answers,
            "contexts_list": contexts_list,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


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


def ejecutar_evaluacion(
    dataset_path: str,
    output_path: str | None = None,
    debug_path: str | None = None,
    checkpoint_path: str | None = None,
    verbose: bool = False,
    save_debug: bool = True,
    force_reindex: bool = False,
    recomp_enabled: bool | None = None,
    eval_corpus: Literal["es", "ca"] = "es",
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
        output_path: CSV output path. Defaults to ``evaluation/scores/``.
        debug_path: Debug JSON output path. Defaults to ``evaluation/debug/``.
        checkpoint_path: Progress JSON path used to resume question generation.
        verbose: Whether to print per-question progress.
        save_debug: Whether to persist the debug JSON.
        force_reindex: Whether to rebuild the ChromaDB collection before evaluation.
        recomp_enabled: Optional in-process override for ``USAR_RECOMP_SYNTHESIS``.
        eval_corpus: ``es`` uses ``rag/pdfs`` (default Chroma); ``ca`` uses ``rag/pdfs_ca``
            and adds ``_ca`` to default score/debug/checkpoint filenames.
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
    previous_recomp = None
    docs_previous: tuple[str, str, str] | None = None
    if recomp_enabled is not None:
        previous_recomp = set_recomp_synthesis_enabled(recomp_enabled)

    if eval_corpus == "ca":
        pdfs_ca = os.path.join(_proj_root, "rag", "pdfs_ca")
        docs_previous = rag_runtime.set_docs_folder_runtime(pdfs_ca)

    try:
        # ─────────────────────────────────────────────
        # 1. IMPORT RAGAS METRICS
        # ─────────────────────────────────────────────

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

        # ─────────────────────────────────────────────
        # 2. CONFIGURE EVALUATION LLM
        # ─────────────────────────────────────────────

        eval_llm, eval_embeddings = configurar_llm_evaluacion(
            google_timeout=google_timeout,
            google_retries=google_retries,
        )

        # ─────────────────────────────────────────────
        # 3. LOAD DATASET
        # ─────────────────────────────────────────────

        print(f"\nLoading dataset...")
        dataset_path = resolver_ruta_dataset(dataset_path)
        sfx = _artifact_suffix(eval_corpus)
        resolved_output_path = os.path.abspath(output_path or _default_output_path(dataset_path, sfx))
        resolved_debug_path = os.path.abspath(debug_path or _default_debug_path(dataset_path, sfx))
        resolved_checkpoint_path = os.path.abspath(
            checkpoint_path
            or _default_checkpoint_path(dataset_path, rag_runtime.USAR_RECOMP_SYNTHESIS, sfx)
        )

        df = cargar_dataset(dataset_path)
        df = normalizar_columnas(df)
        questions = df["question"].tolist()
        ground_truths = df["ground_truth"].tolist()

        tiene_ground_truth = any(gt.strip() for gt in ground_truths)

        print(f"   Questions to evaluate: {len(questions)}")
        print(f"   Ground truth available: {'Yes' if tiene_ground_truth else 'No'}")
        print(f"   RECOMP synthesis: {'Enabled' if rag_runtime.USAR_RECOMP_SYNTHESIS else 'Disabled'}")
        print(
            f"   Eval corpus: {eval_corpus.upper()} — PDFs: {rag_runtime.CARPETA_DOCS}"
        )

        # ─────────────────────────────────────────────
        # 4. CONNECT TO CHROMADB
        # ─────────────────────────────────────────────

        print(f"\nConnecting to ChromaDB: {rag_runtime.PATH_DB}")
        client = chromadb.PersistentClient(path=rag_runtime.PATH_DB)

        if force_reindex:
            print("   Force reindex requested. Rebuilding collection...")
            try:
                client.delete_collection(name=rag_runtime.COLLECTION_NAME)
            except Exception:
                pass
            collection = client.get_or_create_collection(name=rag_runtime.COLLECTION_NAME)
            total = indexar_documentos(rag_runtime.CARPETA_DOCS, collection)
            print(f"   Indexed {total} fragments.")
        else:
            collection = client.get_or_create_collection(name=rag_runtime.COLLECTION_NAME)
            if collection.count() == 0:
                print("   Database empty. Indexing documents...")
                total = indexar_documentos(rag_runtime.CARPETA_DOCS, collection)
                print(f"   Indexed {total} fragments.")
            else:
                print(f"   Fragments in collection: {collection.count()}")
                total = collection.count()

        # ─────────────────────────────────────────────
        # 5. RUN RAG PIPELINE PER QUESTION
        # ─────────────────────────────────────────────

        print("\nRunning RAG pipeline for each question...")
        answers: list[str] = []
        contexts_list: list[list[str]] = []
        checkpoint = _cargar_checkpoint(resolved_checkpoint_path)
        if checkpoint:
            if (
                checkpoint.get("dataset_path") == dataset_path
                and checkpoint.get("questions_count") == len(questions)
                and checkpoint.get("recomp_enabled") == rag_runtime.USAR_RECOMP_SYNTHESIS
                and checkpoint.get("eval_corpus", "es") == eval_corpus
            ):
                answers = checkpoint.get("answers", [])
                contexts_list = checkpoint.get("contexts_list", [])
                non_empty_answers = len([a for a in answers if not _respuesta_vacia(a)])
                print(
                    "   Resuming from checkpoint: "
                    f"{non_empty_answers}/{len(questions)} questions with non-empty answers "
                    f"({len(answers)}/{len(questions)} slots present)."
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

        pending_answer_indexes = _indices_respuestas_vacias(answers, len(questions))
        if pending_answer_indexes and checkpoint:
            first_missing = ", ".join(str(i + 1) for i in pending_answer_indexes[:10])
            suffix = "..." if len(pending_answer_indexes) > 10 else ""
            print(
                "   Checkpoint contains empty answers. "
                f"Regenerating {len(pending_answer_indexes)} question(s): {first_missing}{suffix}"
            )

        _ollama_timeout = int(os.getenv("EVAL_OLLAMA_TIMEOUT", "300"))
        try:
            for i in pending_answer_indexes:
                q = questions[i]
                if verbose:
                    print(f"   [{i+1}/{len(questions)}] {q[:60]}...")
                with ThreadPoolExecutor(max_workers=1) as _ex:
                    _fut = _ex.submit(evaluar_pregunta_rag, q, collection)
                    try:
                        answer, contexts = _fut.result(timeout=_ollama_timeout)
                    except _FuturesTimeout:
                        answer, contexts = "", []
                        print(
                            f"   [TIMEOUT] Q{i+1} exceeded {_ollama_timeout}s "
                            "— saved as empty, will retry on next run."
                        )
                answers[i] = answer
                contexts_list[i] = contexts
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
            print(
                "\nError: RAGAS evaluation was not launched because "
                f"{len(empty_answer_indexes)} answer(s) are empty."
            )
            print(f"   Empty question indexes: {listed}{suffix}")
            print(f"   Checkpoint: {resolved_checkpoint_path}")
            print("   Fix the RAG generation issue and rerun; only empty answers will be retried.")
            raise SystemExit(1)

        # ─────────────────────────────────────────────
        # 6. BUILD RAGAS EVALUATION DATASET
        # ─────────────────────────────────────────────

        print("\nBuilding EvaluationDataset for RAGAS...")
        samples = []
        for i in range(len(questions)):
            sample = SingleTurnSample(
                user_input=questions[i],
                response=answers[i] if answers[i] else "",
                retrieved_contexts=contexts_list[i] if contexts_list[i] else [],
                reference=ground_truths[i] if ground_truths[i] else "",
            )
            samples.append(sample)

        eval_dataset = EvaluationDataset(samples=samples)

        # ─────────────────────────────────────────────
        # 7. CONFIGURE METRICS
        # ─────────────────────────────────────────────

        metric_objects = {
            "answer_correctness": answer_correctness,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        selected_metric_names = _parse_ragas_metric_names(ragas_metrics, tiene_ground_truth)
        metrics = [metric_objects[name] for name in selected_metric_names]

        # ─────────────────────────────────────────────
        # 8. RUN RAGAS EVALUATION
        # ─────────────────────────────────────────────

        print("\nRunning RAGAS evaluation (this may take a few minutes)...")
        print(
            "   RAGAS config: "
            f"timeout={ragas_timeout}s, retries={ragas_max_retries}, "
            f"max_wait={ragas_max_wait}s, workers={ragas_max_workers}, "
            f"batch_size={ragas_batch_size or 'auto'}, "
            f"metrics={','.join(selected_metric_names)}"
        )
        t_eval_start = time.time()

        eval_run_config = RunConfig(
            timeout=ragas_timeout,
            max_retries=ragas_max_retries,
            max_wait=ragas_max_wait,
            max_workers=ragas_max_workers,
        )

        try:
            result = evaluate(
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

        # ─────────────────────────────────────────────
        # 9. DISPLAY RESULTS
        # ─────────────────────────────────────────────

        df_scores = result.to_pandas()
        imprimir_resultados(df_scores, questions)

        # ─────────────────────────────────────────────
        # 10. SAVE CSV
        # ─────────────────────────────────────────────

        output_dir = os.path.dirname(os.path.abspath(resolved_output_path))
        os.makedirs(output_dir, exist_ok=True)
        df_scores.to_csv(resolved_output_path, index=False, encoding="utf-8")
        print(f"\nResults saved to: {resolved_output_path}")

        # ─────────────────────────────────────────────
        # 11. SAVE DEBUG OUTPUT
        # ─────────────────────────────────────────────

        if save_debug:
            os.makedirs(os.path.dirname(os.path.abspath(resolved_debug_path)), exist_ok=True)
            guardar_debug(
                result=result,
                questions=questions,
                answers=answers,
                ground_truths=ground_truths,
                contexts_list=contexts_list,
                debug_path=resolved_debug_path,
            )
            print(f"Debug saved to: {resolved_debug_path}")
        else:
            resolved_debug_path = None

        metric_cols = [c for c in METRIC_NAMES if c in df_scores.columns]
        mean_scores = {}
        for metric_name in metric_cols:
            metric_value = df_scores[metric_name].mean(numeric_only=True)
            if not pd.isna(metric_value):
                mean_scores[metric_name] = float(metric_value)

        return {
            "dataset_path": os.path.abspath(dataset_path),
            "output_path": os.path.abspath(resolved_output_path),
            "debug_path": os.path.abspath(resolved_debug_path) if resolved_debug_path else None,
            "questions_count": len(questions),
            "indexed_fragments": total,
            "recomp_enabled": rag_runtime.USAR_RECOMP_SYNTHESIS,
            "eval_corpus": eval_corpus,
            "pipeline_seconds": t_rag,
            "evaluation_seconds": t_eval,
            "mean_scores": mean_scores,
            "checkpoint_path": os.path.abspath(resolved_checkpoint_path),
        }
    finally:
        if docs_previous is not None:
            rag_runtime.CARPETA_DOCS, rag_runtime.PATH_DB, rag_runtime.COLLECTION_NAME = (
                docs_previous
            )
        if previous_recomp is not None:
            set_recomp_synthesis_enabled(previous_recomp)


# ─────────────────────────────────────────────
# SECTION 7: MAIN
# ─────────────────────────────────────────────

def main():
    """Entry point: parse arguments, run the RAG pipeline on each question,
    evaluate with RAGAS, print results, and save outputs."""
    parser = argparse.ArgumentParser(
        description="Evaluate the Teacher RAG system with RAGAS v0.2+"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to the dataset (JSON, CSV, or Excel). Default: dataset_eval_es.json or dataset_eval_ca.json with --catalan",
    )
    parser.add_argument(
        "--catalan",
        action="store_true",
        help="Use rag/pdfs_ca for indexing and add _ca to default output/checkpoint file names; default dataset Catalan",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the results CSV",
    )
    parser.add_argument(
        "--debug-output",
        default=None,
        help="Output path for ragas_debug.json",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the checkpoint JSON used for question-by-question resume",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-question progress",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Skip saving the ragas_debug.json file",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Delete and rebuild the ChromaDB collection before evaluating",
    )
    parser.add_argument(
        "--no-recomp",
        action="store_true",
        help="Disable RECOMP synthesis for this run (overrides module default)",
    )
    parser.add_argument(
        "--ragas-timeout",
        type=int,
        default=90,
        help="Per-call RAGAS timeout in seconds. Default: 90",
    )
    parser.add_argument(
        "--ragas-max-retries",
        type=int,
        default=5,
        help="Maximum RAGAS retries per failed job. Default: 5",
    )
    parser.add_argument(
        "--ragas-max-wait",
        type=int,
        default=60,
        help="Maximum wait between RAGAS retries in seconds. Default: 60",
    )
    parser.add_argument(
        "--ragas-max-workers",
        type=int,
        default=1,
        help="Concurrent RAGAS workers. Lower this if Google rate limits. Default: 1",
    )
    parser.add_argument(
        "--ragas-batch-size",
        type=int,
        default=5,
        help="RAGAS batch size. Default: 5",
    )
    parser.add_argument(
        "--ragas-metrics",
        default=None,
        help=(
            "Comma-separated RAGAS metrics or 'all'. "
            "Valid: answer_correctness, faithfulness, answer_relevancy, "
            "context_precision, context_recall. Default: all available"
        ),
    )
    parser.add_argument(
        "--google-timeout",
        type=int,
        default=None,
        help="Timeout in seconds for Gemini LLM/embeddings calls. Default: 45 or EVAL_GOOGLE_TIMEOUT",
    )
    parser.add_argument(
        "--google-retries",
        type=int,
        default=None,
        help="Gemini LLM retries. Default: 2 or EVAL_GOOGLE_RETRIES",
    )
    parser.add_argument(
        "--raise-exceptions",
        action="store_true",
        help="Stop immediately when a RAGAS job fails instead of filling NaN scores",
    )
    args = parser.parse_args()

    eval_corpus: Literal["es", "ca"] = "ca" if args.catalan else "es"
    dataset_arg = args.dataset if args.dataset is not None else _default_dataset_for_corpus(eval_corpus)

    recomp_arg = False if args.no_recomp else None

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


if __name__ == "__main__":
    main()
