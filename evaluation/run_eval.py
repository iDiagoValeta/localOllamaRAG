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

def configurar_llm_evaluacion():
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
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

        eval_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_key,
            temperature=0,
        )
        eval_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=gemini_key,
        )
        print("Evaluation LLM: Gemini 2.0 Flash (langchain-google-genai)")
        print("Evaluation embeddings: Google gemini-embedding-001 (langchain-google-genai)")
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

        eval_llm, eval_embeddings = configurar_llm_evaluacion()

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
                print(
                    f"   Resuming from checkpoint: {len(answers)}/{len(questions)} questions already completed."
                )
            else:
                print("   Existing checkpoint does not match this run. Starting fresh progress.")

        t_start = time.time()

        try:
            for i, q in enumerate(questions[len(answers):], start=len(answers)):
                if verbose:
                    print(f"   [{i+1}/{len(questions)}] {q[:60]}...")
                answer, contexts = evaluar_pregunta_rag(q, collection)
                answers.append(answer)
                contexts_list.append(contexts)
                _guardar_checkpoint(
                    resolved_checkpoint_path,
                    {
                        "dataset_path": dataset_path,
                        "questions_count": len(questions),
                        "recomp_enabled": rag_runtime.USAR_RECOMP_SYNTHESIS,
                        "eval_corpus": eval_corpus,
                        "output_path": resolved_output_path,
                        "debug_path": resolved_debug_path,
                        "completed_questions": len(answers),
                        "answers": answers,
                        "contexts_list": contexts_list,
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
        except ConnectionError as e:
            print(f"\nError: Could not connect to Ollama: {e}")
            print("   Make sure Ollama is running before launching the evaluation.")
            print("   Start Ollama with: ollama serve")
            raise SystemExit(1)

        t_rag = time.time() - t_start
        print(f"   Pipeline completed in {t_rag:.1f}s ({t_rag/len(questions):.1f}s/question)")

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

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        if tiene_ground_truth:
            metrics.insert(0, answer_correctness)

        # ─────────────────────────────────────────────
        # 8. RUN RAGAS EVALUATION
        # ─────────────────────────────────────────────

        print("\nRunning RAGAS evaluation (this may take a few minutes)...")
        t_eval_start = time.time()

        eval_run_config = RunConfig(timeout=600, max_retries=15)

        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=eval_llm,
            embeddings=eval_embeddings,
            run_config=eval_run_config,
        )

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
    args = parser.parse_args()

    eval_corpus: Literal["es", "ca"] = "ca" if args.catalan else "es"
    dataset_arg = args.dataset if args.dataset is not None else _default_dataset_for_corpus(eval_corpus)

    ejecutar_evaluacion(
        dataset_path=dataset_arg,
        output_path=args.output,
        debug_path=args.debug_output,
        checkpoint_path=args.checkpoint,
        verbose=args.verbose,
        save_debug=not args.no_debug,
        force_reindex=args.force_reindex,
        eval_corpus=eval_corpus,
    )


if __name__ == "__main__":
    main()
