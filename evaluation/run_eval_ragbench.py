"""
run_eval_ragbench.py -- Evaluación RAG con el dataset público vectara/open_ragbench.

Downloads a subset of arXiv PDFs from the open_ragbench HuggingFace dataset,
indexes them into a dedicated ChromaDB collection, and evaluates the MonkeyGrab
RAG pipeline end-to-end using RAGAS v0.2+ metrics:

    PDFs (arXiv) → ChromaDB (indexing) → Hybrid retrieval → Generation → RAGAS

This script complements run_eval.py (private TFG dataset) with a reproducible,
public benchmark that any reviewer can verify. Paper selection is automatic:
the top-N papers by question count are chosen from the dataset's text-only
query subset (no multimodal dependency required by default).

Usage:
    python evaluation/run_eval_ragbench.py
    python evaluation/run_eval_ragbench.py --n-papers 5 --max-q 3 --verbose
    python evaluation/run_eval_ragbench.py --skip-download --force-reindex
    python evaluation/run_eval_ragbench.py --all-sources   # include multimodal queries

Dependencies:
    - huggingface_hub
    - requests
    - ragas, langchain-google-genai
    - pandas, chromadb
    - python-dotenv (optional)
"""

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  CONFIGURATION
#  +-- 1. Imports & environment setup
#  +-- 2. Global constants
#
#  DATA ACQUISITION
#  +-- 3. Download dataset metadata (HuggingFace)
#  +-- 4. Select papers (top-N by question count)
#  +-- 5. Build question set
#  +-- 6. Download PDFs (arXiv)
#
#  EVALUATION PIPELINE
#  +-- 7. Index PDFs into ChromaDB
#  +-- 8. Configure RAGAS evaluation LLM
#  +-- 9. Run RAG pipeline per question
#  +-- 10. Run RAGAS evaluation
#
#  OUTPUT
#  +-- 11. Display results
#  +-- 12. Save CSV + debug JSON
#  +-- 13. main()
#
# ─────────────────────────────────────────────

import os
import sys
import json
import argparse
import time
from collections import Counter

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
import requests

from rag.chat_pdfs import (
    evaluar_pregunta_rag,
    indexar_documentos,
)

# ─────────────────────────────────────────────
# SECTION 2: GLOBAL CONSTANTS
# ─────────────────────────────────────────────

# --- 2.1 Dataset selection parameters ---
N_PAPERS = 10                 # number of arXiv papers to download and evaluate
MAX_QUESTIONS_PER_PAPER = 5   # cap per paper to keep evaluation balanced (~50 q total)
QUERY_SOURCE_FILTER = "text"  # "text" | "text-image" | "text-table" | None (all sources)

# --- 2.2 arXiv download settings ---
ARXIV_DELAY_SECS = 5          # seconds between downloads (be polite to arXiv)
ARXIV_TIMEOUT_SECS = 60       # per-request HTTP timeout
ARXIV_HEADERS = {
    "User-Agent": "MonkeyGrab-TFG-Eval/1.0 (academic research; Universitat Politecnica de Valencia)"
}

# --- 2.3 HuggingFace dataset ---
HF_REPO = "vectara/open_ragbench"
HF_SUBDIR = "pdf/arxiv"
HF_METADATA_FILES = ["queries.json", "qrels.json", "answers.json", "pdf_urls.json"]

# --- 2.4 Paths ---
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(EVAL_DIR)
RAGBENCH_PDFS_DIR = os.path.join(PROJ_ROOT, "rag", "ragbench_pdfs")
RAGBENCH_DB_PATH = os.path.join(PROJ_ROOT, "rag", "ragbench_vector_db")
RAGBENCH_COLLECTION = "ragbench_arxiv_eval"

# --- 2.5 Output files ---
OUTPUT_CSV = os.path.join(EVAL_DIR, "ragas_ragbench_scores.csv")
OUTPUT_DEBUG = os.path.join(EVAL_DIR, "ragas_ragbench_debug.json")

# --- 2.6 RAGAS metric labels ---
METRIC_NAMES = [
    "answer_correctness",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
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
# SECTION 3: DOWNLOAD DATASET METADATA
# ─────────────────────────────────────────────

def descargar_metadatos() -> tuple[dict, dict, dict, dict]:
    """Download the four metadata JSON files from vectara/open_ragbench on HuggingFace.

    Uses the HuggingFace Hub cache so subsequent runs avoid re-downloading.
    Files are stored in the default HF cache directory (~/.cache/huggingface/).

    Returns:
        Tuple of (queries, qrels, answers, pdf_urls) dictionaries.

    Raises:
        SystemExit: If huggingface_hub is not installed.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not found. Install with:")
        print("   pip install huggingface-hub")
        raise SystemExit(1)

    print("Downloading dataset metadata from HuggingFace (cached after first run)...")
    loaded = {}
    for fname in HF_METADATA_FILES:
        print(f"   {fname}...", end=" ", flush=True)
        local_path = hf_hub_download(
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


# ─────────────────────────────────────────────
# SECTION 4: SELECT PAPERS
# ─────────────────────────────────────────────

def seleccionar_papers(
    queries: dict,
    qrels: dict,
    pdf_urls: dict,
    n_papers: int,
    source_filter: str | None,
) -> list[str]:
    """Select the top-N papers by number of associated questions.

    Filters by query source type and ensures every selected paper has a
    downloadable PDF URL. Papers are ranked descending by question count
    so the evaluation covers the richest portion of the benchmark first.

    Args:
        queries: queries.json dict (UUID -> {query, type, source}).
        qrels: qrels.json dict (UUID -> {doc_id, section_id}).
        pdf_urls: pdf_urls.json dict (doc_id -> URL).
        n_papers: Number of papers to select.
        source_filter: If set, only count queries with this source value.
            Pass None to include all source types.

    Returns:
        Ordered list of doc_id strings (arXiv IDs with version suffix).
    """
    paper_counts: Counter = Counter()
    for qid, qrel in qrels.items():
        if qid not in queries:
            continue
        if qrel["doc_id"] not in pdf_urls:
            continue  # skip papers without a downloadable URL
        if source_filter and queries[qid].get("source") != source_filter:
            continue
        paper_counts[qrel["doc_id"]] += 1

    selected = [pid for pid, _ in paper_counts.most_common(n_papers)]

    src_label = f"source='{source_filter}'" if source_filter else "all sources"
    print(f"\nSelected {len(selected)} papers (top-{n_papers} by question count, {src_label}):")
    for pid in selected:
        print(f"   arxiv:{pid}  ({paper_counts[pid]} eligible questions)")

    return selected


# ─────────────────────────────────────────────
# SECTION 5: BUILD QUESTION SET
# ─────────────────────────────────────────────

def construir_preguntas(
    queries: dict,
    qrels: dict,
    answers: dict,
    selected_papers: list[str],
    source_filter: str | None,
    max_per_paper: int,
) -> tuple[list[str], list[str], list[str]]:
    """Build the evaluation question set from the selected papers.

    Collects up to max_per_paper questions per paper, filtered by source type,
    preserving the paper order defined in selected_papers.

    Args:
        queries: queries.json dict.
        qrels: qrels.json dict.
        answers: answers.json dict (UUID -> ground-truth string).
        selected_papers: Ordered list of paper IDs to include.
        source_filter: Query source filter (None = include all sources).
        max_per_paper: Maximum number of questions taken from each paper.

    Returns:
        Tuple of (questions, ground_truths, paper_ids) lists aligned by index.
    """
    selected_set = set(selected_papers)
    per_paper: dict[str, list[str]] = {p: [] for p in selected_papers}

    for qid, qrel in qrels.items():
        doc_id = qrel["doc_id"]
        if doc_id not in selected_set:
            continue
        if qid not in queries:
            continue
        if source_filter and queries[qid].get("source") != source_filter:
            continue
        per_paper[doc_id].append(qid)

    questions: list[str] = []
    ground_truths: list[str] = []
    paper_ids: list[str] = []

    for paper_id in selected_papers:
        for qid in per_paper[paper_id][:max_per_paper]:
            questions.append(queries[qid]["query"])
            ground_truths.append(answers.get(qid, ""))
            paper_ids.append(paper_id)

    print(f"\nQuestion set: {len(questions)} questions from {len(selected_papers)} papers "
          f"(max {max_per_paper}/paper)")
    return questions, ground_truths, paper_ids


# ─────────────────────────────────────────────
# SECTION 6: DOWNLOAD PDFs
# ─────────────────────────────────────────────

def descargar_pdfs(
    selected_papers: list[str],
    pdf_urls: dict,
    pdfs_dir: str,
    skip_existing: bool = True,
) -> list[str]:
    """Download PDFs from arXiv for the selected papers.

    Saves each PDF to pdfs_dir/{paper_id}.pdf. By default, skips papers
    whose PDF file already exists so re-runs avoid redundant downloads.
    Validates the response to detect arXiv rate-limit HTML pages.

    Args:
        selected_papers: List of arXiv paper IDs to download.
        pdf_urls: pdf_urls.json dict (paper_id -> URL).
        pdfs_dir: Local directory where PDFs will be saved.
        skip_existing: Skip already-downloaded files when True.

    Returns:
        List of paper IDs for which a valid PDF is available locally
        (downloaded now or pre-existing).
    """
    os.makedirs(pdfs_dir, exist_ok=True)
    successful: list[str] = []

    print(f"\nDownloading {len(selected_papers)} PDFs to {pdfs_dir}/")
    for i, paper_id in enumerate(selected_papers):
        out_path = os.path.join(pdfs_dir, f"{paper_id}.pdf")

        if skip_existing and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"   [{i+1}/{len(selected_papers)}] {paper_id}  (cached, skipping)")
            successful.append(paper_id)
            continue

        url = pdf_urls[paper_id]
        print(f"   [{i+1}/{len(selected_papers)}] {paper_id}  <- {url}")

        try:
            resp = requests.get(url, headers=ARXIV_HEADERS, timeout=ARXIV_TIMEOUT_SECS)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"      ERROR (download): {e}")
            continue

        # arXiv may serve an HTML rate-limit page instead of a PDF
        content_type = resp.headers.get("Content-Type", "")
        if "application/pdf" not in content_type and not resp.content.startswith(b"%PDF"):
            print(f"      WARNING: unexpected Content-Type '{content_type}' — skipping")
            continue

        with open(out_path, "wb") as fh:
            fh.write(resp.content)
        print(f"      {len(resp.content) / 1024:.0f} KB saved")
        successful.append(paper_id)

        # Be polite between requests (not after the last one)
        if i < len(selected_papers) - 1:
            time.sleep(ARXIV_DELAY_SECS)

    return successful


# ─────────────────────────────────────────────
# SECTION 7: CHROMADB INDEXING
# ─────────────────────────────────────────────

def conectar_e_indexar(pdfs_dir: str, force_reindex: bool) -> chromadb.Collection:
    """Connect to the dedicated ragbench ChromaDB and index PDFs if needed.

    Uses a separate database path (ragbench_vector_db) so the ragbench
    evaluation never contaminates the main production collection.

    Args:
        pdfs_dir: Directory containing the downloaded arXiv PDFs.
        force_reindex: If True, drop and rebuild the collection even when
            it already has data.

    Returns:
        The ChromaDB Collection ready for querying.
    """
    print(f"\nConnecting to ChromaDB (ragbench): {RAGBENCH_DB_PATH}")
    client = chromadb.PersistentClient(path=RAGBENCH_DB_PATH)
    collection = client.get_or_create_collection(name=RAGBENCH_COLLECTION)

    if force_reindex and collection.count() > 0:
        print("   --force-reindex: clearing existing collection...")
        client.delete_collection(RAGBENCH_COLLECTION)
        collection = client.get_or_create_collection(name=RAGBENCH_COLLECTION)

    if collection.count() == 0:
        print("   Collection empty. Indexing PDFs (this may take several minutes)...")
        total = indexar_documentos(pdfs_dir, collection)
        print(f"   Indexed {total} fragments.")
    else:
        print(f"   Collection has {collection.count()} fragments. "
              "Use --force-reindex to rebuild.")

    return collection


# ─────────────────────────────────────────────
# SECTION 8: RAGAS EVALUATION LLM
# ─────────────────────────────────────────────

def configurar_llm_evaluacion():
    """Configure the Gemini LLM and embeddings used by RAGAS as evaluation judge.

    Reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment. The same
    judge is used as in run_eval.py to ensure metric comparability.

    Returns:
        Tuple of (eval_llm, eval_embeddings).

    Raises:
        SystemExit: If the API key is missing or langchain-google-genai is absent.
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY not set in environment.")
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
        print("Evaluation LLM:        Gemini 2.0 Flash (langchain-google-genai)")
        print("Evaluation embeddings: gemini-embedding-001 (langchain-google-genai)")
        return eval_llm, eval_embeddings
    except ImportError as err:
        print(f"ERROR: {err}")
        print("  Install with: pip install langchain-google-genai")
        raise SystemExit(1)


# ─────────────────────────────────────────────
# SECTION 9: DISPLAY RESULTS
# ─────────────────────────────────────────────

def imprimir_resultados(df_scores: pd.DataFrame, questions: list[str]):
    """Print global RAGAS averages and per-question scores to stdout.

    Args:
        df_scores: DataFrame of RAGAS scores (one row per question).
        questions: Original question strings, aligned by index.
    """
    metric_cols = [c for c in METRIC_NAMES if c in df_scores.columns]
    if not metric_cols:
        print("\nNo metric columns found in results.")
        print(f"   Available columns: {list(df_scores.columns)}")
        return

    print("\n" + "=" * 70)
    print("  RAGAS RESULTS — vectara/open_ragbench — GLOBAL AVERAGES")
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
# SECTION 10: SAVE DEBUG JSON
# ─────────────────────────────────────────────

def _extraer_justificaciones_traces(traces: list, metric_cols: list) -> list[dict]:
    """Extract per-question LLM justification payloads from RAGAS traces.

    Args:
        traces: List of trace objects from result.traces.
        metric_cols: Metric column names to look for in each trace.

    Returns:
        List of dicts (one per question) mapping metric names to
        their extracted justification payloads.
    """
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
    questions: list[str],
    gen_answers: list[str],
    ground_truths: list[str],
    contexts_list: list[list[str]],
    paper_ids: list[str],
) -> str:
    """Save a debug JSON with model answers, contexts, scores, and justifications.

    Adds a paper_id field per entry (not present in run_eval.py) to allow
    grouping results by arXiv paper for analysis.

    Args:
        result: The RAGAS EvaluationResult object.
        questions: List of input questions.
        gen_answers: List of model-generated answers.
        ground_truths: List of reference answers.
        contexts_list: List of lists of retrieved context strings.
        paper_ids: arXiv paper ID associated with each question.

    Returns:
        Absolute path to the saved debug JSON file.
    """
    df = result.to_pandas()
    metric_cols = [c for c in METRIC_NAMES if c in df.columns]

    traces = getattr(result, "traces", []) or []
    justificaciones = (
        _extraer_justificaciones_traces(traces, metric_cols)
        if traces
        else [{}] * len(questions)
    )

    debug_entries = []
    for i in range(len(questions)):
        ctx_preview = [
            ctx[:300] + "..." if len(ctx) > 300 else ctx
            for ctx in (contexts_list[i] if i < len(contexts_list) else [])
        ]
        entry = {
            "index": i + 1,
            "paper_id": paper_ids[i] if i < len(paper_ids) else "",
            "question": questions[i],
            "model_answer": gen_answers[i] if i < len(gen_answers) else "",
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
        "dataset": HF_REPO,
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

    with open(OUTPUT_DEBUG, "w", encoding="utf-8") as fh:
        json.dump(debug_data, fh, ensure_ascii=False, indent=2)
    return OUTPUT_DEBUG


# ─────────────────────────────────────────────
# SECTION 11: MAIN
# ─────────────────────────────────────────────

def main():
    """Entry point: download ragbench subset, index, run RAG, evaluate with RAGAS."""
    parser = argparse.ArgumentParser(
        description="Evaluate MonkeyGrab RAG with vectara/open_ragbench (arXiv subset)"
    )
    parser.add_argument(
        "--n-papers", type=int, default=N_PAPERS,
        help=f"Number of arXiv papers to select and download (default: {N_PAPERS})"
    )
    parser.add_argument(
        "--max-q", type=int, default=MAX_QUESTIONS_PER_PAPER,
        help=f"Max questions per paper (default: {MAX_QUESTIONS_PER_PAPER})"
    )
    parser.add_argument(
        "--all-sources", action="store_true",
        help="Include multimodal queries (text-image, text-table). Default: text-only"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip PDF download phase and use existing files in rag/ragbench_pdfs/"
    )
    parser.add_argument(
        "--force-reindex", action="store_true",
        help="Drop and rebuild the ChromaDB collection even if it already has data"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each question as it is processed"
    )
    parser.add_argument(
        "--no-debug", action="store_true",
        help="Skip saving ragas_ragbench_debug.json"
    )
    args = parser.parse_args()

    source_filter = None if args.all_sources else QUERY_SOURCE_FILTER

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
        print("Install RAGAS: pip install -r evaluation/requirements.txt")
        raise SystemExit(1) from e

    # ─────────────────────────────────────────────
    # 2. CONFIGURE EVALUATION LLM
    # ─────────────────────────────────────────────

    eval_llm, eval_embeddings = configurar_llm_evaluacion()

    # ─────────────────────────────────────────────
    # 3. DOWNLOAD DATASET METADATA
    # ─────────────────────────────────────────────

    queries, qrels, answers_gt, pdf_urls = descargar_metadatos()

    # ─────────────────────────────────────────────
    # 4. SELECT PAPERS
    # ─────────────────────────────────────────────

    selected_papers = seleccionar_papers(
        queries, qrels, pdf_urls, args.n_papers, source_filter
    )

    # ─────────────────────────────────────────────
    # 5. BUILD QUESTION SET
    # ─────────────────────────────────────────────

    questions, ground_truths, paper_ids_per_q = construir_preguntas(
        queries, qrels, answers_gt, selected_papers, source_filter, args.max_q
    )

    # ─────────────────────────────────────────────
    # 6. DOWNLOAD PDFs
    # ─────────────────────────────────────────────

    if not args.skip_download:
        successful_papers = descargar_pdfs(selected_papers, pdf_urls, RAGBENCH_PDFS_DIR)

        if len(successful_papers) < len(selected_papers):
            print(f"\nWARNING: {len(successful_papers)}/{len(selected_papers)} PDFs available.")
            downloaded_set = set(successful_papers)
            filtered = [
                (q, gt, pid)
                for q, gt, pid in zip(questions, ground_truths, paper_ids_per_q)
                if pid in downloaded_set
            ]
            if filtered:
                questions, ground_truths, paper_ids_per_q = map(list, zip(*filtered))
            else:
                print("ERROR: No questions remain after filtering. Aborting.")
                raise SystemExit(1)
    else:
        print(f"\n--skip-download: using existing PDFs in {RAGBENCH_PDFS_DIR}/")

    # ─────────────────────────────────────────────
    # 7. INDEX INTO CHROMADB
    # ─────────────────────────────────────────────

    collection = conectar_e_indexar(RAGBENCH_PDFS_DIR, args.force_reindex)

    # ─────────────────────────────────────────────
    # 8. RUN RAG PIPELINE PER QUESTION
    # ─────────────────────────────────────────────

    print(f"\nRunning RAG pipeline for {len(questions)} questions...")
    gen_answers: list[str] = []
    contexts_list: list[list[str]] = []
    t_start = time.time()

    try:
        for i, q in enumerate(questions):
            if args.verbose:
                print(f"   [{i+1}/{len(questions)}] {q[:70]}...")
            answer, contexts = evaluar_pregunta_rag(q, collection)
            gen_answers.append(answer)
            contexts_list.append(contexts)
    except ConnectionError as e:
        print(f"\nERROR: Could not connect to Ollama: {e}")
        print("   Make sure Ollama is running: ollama serve")
        raise SystemExit(1)

    t_rag = time.time() - t_start
    print(f"   Completed in {t_rag:.1f}s ({t_rag / max(len(questions), 1):.1f}s/question)")

    # ─────────────────────────────────────────────
    # 9. BUILD RAGAS EVALUATION DATASET
    # ─────────────────────────────────────────────

    print("\nBuilding EvaluationDataset for RAGAS...")
    samples = [
        SingleTurnSample(
            user_input=questions[i],
            response=gen_answers[i] or "",
            retrieved_contexts=contexts_list[i] or [],
            reference=ground_truths[i] or "",
        )
        for i in range(len(questions))
    ]
    eval_dataset = EvaluationDataset(samples=samples)

    # ─────────────────────────────────────────────
    # 10. RUN RAGAS EVALUATION
    # ─────────────────────────────────────────────

    metrics = [
        answer_correctness,
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    print("\nRunning RAGAS evaluation (this may take several minutes)...")
    t_eval_start = time.time()

    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=RunConfig(timeout=600, max_retries=15),
    )

    t_eval = time.time() - t_eval_start
    print(f"   Evaluation completed in {t_eval:.1f}s")

    # ─────────────────────────────────────────────
    # 11. DISPLAY RESULTS
    # ─────────────────────────────────────────────

    df_scores = result.to_pandas()
    imprimir_resultados(df_scores, questions)

    # ─────────────────────────────────────────────
    # 12. SAVE CSV
    # ─────────────────────────────────────────────

    df_scores.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nResults CSV:   {OUTPUT_CSV}")

    # ─────────────────────────────────────────────
    # 13. SAVE DEBUG JSON
    # ─────────────────────────────────────────────

    if not args.no_debug:
        debug_path = guardar_debug(
            result=result,
            questions=questions,
            gen_answers=gen_answers,
            ground_truths=ground_truths,
            contexts_list=contexts_list,
            paper_ids=paper_ids_per_q,
        )
        print(f"Debug JSON:    {debug_path}")


if __name__ == "__main__":
    main()
