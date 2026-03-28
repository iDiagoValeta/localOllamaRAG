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
#  +-- 4. Result formatting    imprimir_resultados
#  +-- 5. Debug output         _extraer_justificaciones_traces, guardar_debug
#
#  ENTRY
#  +-- 6. Main                 main()
#
# ─────────────────────────────────────────────

import os
import sys
import json
import argparse
import time

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

from rag.chat_pdfs import (
    evaluar_pregunta_rag,
    PATH_DB,
    COLLECTION_NAME,
    CARPETA_DOCS,
    indexar_documentos,
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
# SECTION 4: RESULT FORMATTING
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
# SECTION 5: DEBUG OUTPUT
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
    eval_dir: str,
) -> str:
    """Save a debug JSON file containing model answers, retrieved contexts,
    per-question scores, and RAGAS justifications.

    Args:
        result: The RAGAS ``EvaluationResult`` object.
        questions: List of input questions.
        answers: List of model-generated answers.
        ground_truths: List of reference answers.
        contexts_list: List of lists of retrieved context strings.
        eval_dir: Directory where the debug file will be written.

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

    debug_path = os.path.join(eval_dir, "ragas_debug.json")
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)
    return debug_path


# ─────────────────────────────────────────────
# SECTION 6: MAIN
# ─────────────────────────────────────────────

def main():
    """Entry point: parse arguments, run the RAG pipeline on each question,
    evaluate with RAGAS, print results, and save outputs."""
    parser = argparse.ArgumentParser(
        description="Evaluate the Teacher RAG system with RAGAS v0.2+"
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(os.path.dirname(__file__), "dataset_eval.json"),
        help="Path to the dataset (JSON, CSV, or Excel)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the results CSV",
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
    args = parser.parse_args()

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
    df = cargar_dataset(args.dataset)
    df = normalizar_columnas(df)
    questions = df["question"].tolist()
    ground_truths = df["ground_truth"].tolist()

    tiene_ground_truth = any(gt.strip() for gt in ground_truths)

    print(f"   Questions to evaluate: {len(questions)}")
    print(f"   Ground truth available: {'Yes' if tiene_ground_truth else 'No'}")

    # ─────────────────────────────────────────────
    # 4. CONNECT TO CHROMADB
    # ─────────────────────────────────────────────

    print(f"\nConnecting to ChromaDB: {PATH_DB}")
    client = chromadb.PersistentClient(path=PATH_DB)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        print("   Database empty. Indexing documents...")
        total = indexar_documentos(CARPETA_DOCS, collection)
        print(f"   Indexed {total} fragments.")
    else:
        print(f"   Fragments in collection: {collection.count()}")

    # ─────────────────────────────────────────────
    # 5. RUN RAG PIPELINE PER QUESTION
    # ─────────────────────────────────────────────

    print("\nRunning RAG pipeline for each question...")
    answers = []
    contexts_list = []
    t_start = time.time()

    try:
        for i, q in enumerate(questions):
            if args.verbose:
                print(f"   [{i+1}/{len(questions)}] {q[:60]}...")
            answer, contexts = evaluar_pregunta_rag(q, collection)
            answers.append(answer)
            contexts_list.append(contexts)
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

    out_path = args.output
    if not out_path:
        out_path = os.path.join(
            os.path.dirname(__file__),
            "ragas_scores.csv",
        )
    df_scores.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nResults saved to: {out_path}")

    # ─────────────────────────────────────────────
    # 11. SAVE DEBUG OUTPUT
    # ─────────────────────────────────────────────

    if not args.no_debug:
        eval_dir = os.path.dirname(__file__)
        debug_path = guardar_debug(
            result=result,
            questions=questions,
            answers=answers,
            ground_truths=ground_truths,
            contexts_list=contexts_list,
            eval_dir=eval_dir,
        )
        print(f"Debug saved to: {debug_path}")


if __name__ == "__main__":
    main()
