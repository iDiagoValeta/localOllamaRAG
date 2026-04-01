"""
run_eval_ragbench.py -- Evaluación RAG sobre el dataset público vectara/open_ragbench.

Descarga un subconjunto de PDFs arXiv del dataset de HuggingFace, los indexa
en una colección ChromaDB dedicada y evalúa el pipeline RAG end-to-end con RAGAS:

    PDFs (arXiv) → ChromaDB (indexado) → Recuperación híbrida → Generación → RAGAS

Por defecto solo se usan preguntas de tipo ``text-image`` (preguntas sobre imágenes
del paper). Usa ``--source all`` para incluir todos los tipos.

Uso desde la raíz del repositorio:
    python evaluation/run_eval_ragbench.py [opciones]

Invocaciones habituales:
    # Solo preguntas de imagen (por defecto), top-N papers:
    python evaluation/run_eval_ragbench.py

    # Un paper concreto, PDFs ya descargados, forzar re-indexado:
    python evaluation/run_eval_ragbench.py --only-doc 2403.20331v2 --skip-download --force-reindex

    # Todos los tipos de pregunta:
    python evaluation/run_eval_ragbench.py --source all

    # Verbose con pocos papers:
    python evaluation/run_eval_ragbench.py --n-papers 3 --max-q 5 --verbose

CLI flags:
    --source SOURCE     Tipo de fuente: text-image | text-table | text | all (default: text-image)
    --only-doc DOC_ID   Evalúa un solo paper (ej: 2403.20331v2). Sobreescribe --n-papers.
    --n-papers N        Número de papers a seleccionar (default: 10)
    --max-q N           Máximo de preguntas por paper (default: 5)
    --skip-download     No descarga PDFs; usa los que ya hay en rag/ragbench_pdfs/
    --force-reindex     Borra y reconstruye la colección ChromaDB
    --verbose           Imprime cada pregunta mientras se procesa
    --no-debug          No guarda ragas_ragbench_debug.json

Prerequisitos:
    - GOOGLE_API_KEY o GEMINI_API_KEY en .env (juez RAGAS: Gemini)
    - Ollama corriendo con los modelos de rag/chat_pdfs.py
    - pip install -r evaluation/requirements.txt

Salidas (en evaluation/):
    - ragas_ragbench_scores.csv
    - ragas_ragbench_debug.json (salvo --no-debug)

Dependencies:
    - huggingface_hub, requests
    - ragas, langchain-google-genai
    - pandas, chromadb
    - python-dotenv (opcional)
"""

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  CONFIGURATION
#  +-- 1. Imports & entorno      sys.path, .env, imports
#  +-- 2. Constantes globales    rutas, parámetros, nombres de métricas
#
#  SELECCIÓN DE DATOS
#  +-- 3. Metadatos HF           descargar_metadatos
#  +-- 4. Selección de papers    seleccionar_papers
#  +-- 5. Construcción preguntas construir_preguntas, filtrar_por_pdfs_disponibles
#
#  PIPELINE
#  +-- 6. Descarga PDFs          descargar_pdfs
#  +-- 7. Indexado ChromaDB      conectar_e_indexar
#  +-- 8. LLM evaluación         configurar_llm_evaluacion
#
#  SALIDA
#  +-- 9. Resultados             imprimir_resultados
#  +-- 10. Debug JSON            guardar_debug
#
#  ENTRY
#  +-- 11. main()
#
# ─────────────────────────────────────────────

import os
import sys
import json
import argparse
import time
from collections import Counter

# ─────────────────────────────────────────────
# SECTION 1: IMPORTS Y ENTORNO
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
# SECTION 2: CONSTANTES GLOBALES
# ─────────────────────────────────────────────

HF_REPO = "vectara/open_ragbench"
HF_SUBDIR = "pdf/arxiv"
HF_METADATA_FILES = ["queries.json", "qrels.json", "answers.json", "pdf_urls.json"]

# Tipos de fuente disponibles en RAGBench:
#   "text"        → preguntas sobre texto plano
#   "text-image"  → preguntas sobre figuras/imágenes del paper
#   "text-table"  → preguntas sobre tablas del paper
DEFAULT_SOURCE = "text"

N_PAPERS = 10
MAX_QUESTIONS_PER_PAPER = 5
ARXIV_DELAY_SECS = 5
ARXIV_TIMEOUT_SECS = 60
ARXIV_HEADERS = {
    "User-Agent": "MonkeyGrab-TFG-Eval/1.0 (academic research; Universitat Politecnica de Valencia)"
}

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(EVAL_DIR)
RAGBENCH_PDFS_DIR = os.path.join(PROJ_ROOT, "rag", "ragbench_pdfs")
RAGBENCH_DB_PATH = os.path.join(PROJ_ROOT, "rag", "ragbench_vector_db")
RAGBENCH_COLLECTION = "ragbench_arxiv_eval"
OUTPUT_CSV = os.path.join(EVAL_DIR, "ragas_ragbench_scores.csv")
OUTPUT_DEBUG = os.path.join(EVAL_DIR, "ragas_ragbench_debug.json")

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
# SECTION 3: METADATOS HUGGINGFACE
# ─────────────────────────────────────────────

def descargar_metadatos() -> tuple[dict, dict, dict, dict]:
    """Download the four metadata JSON files from vectara/open_ragbench on HuggingFace.

    Uses the HuggingFace Hub cache so subsequent runs avoid re-downloading.

    Returns:
        Tuple of (queries, qrels, answers, pdf_urls) dicts.

    Raises:
        SystemExit: If huggingface_hub is not installed.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub no instalado. Ejecuta: pip install huggingface-hub")
        raise SystemExit(1)

    print("Descargando metadatos del dataset desde HuggingFace (caché tras la primera vez)...")
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
# SECTION 4: SELECCIÓN DE PAPERS
# ─────────────────────────────────────────────

def seleccionar_papers(
    queries: dict,
    qrels: dict,
    pdf_urls: dict,
    n_papers: int,
    source_filter: str | None,
) -> list[str]:
    """Select the top-N papers by number of eligible questions.

    Papers without a downloadable PDF URL are excluded. Ranked descending
    by question count so the richest part of the benchmark is covered first.

    Args:
        queries: queries.json dict.
        qrels: qrels.json dict.
        pdf_urls: pdf_urls.json dict.
        n_papers: Number of papers to select.
        source_filter: If set, only count queries with this source value.
            Pass None to include all source types.

    Returns:
        Ordered list of doc_id strings.
    """
    paper_counts: Counter = Counter()
    for qid, qrel in qrels.items():
        doc_id = qrel.get("doc_id")
        if not doc_id or qid not in queries or doc_id not in pdf_urls:
            continue
        if source_filter and queries[qid].get("source") != source_filter:
            continue
        paper_counts[doc_id] += 1

    selected = [pid for pid, _ in paper_counts.most_common(n_papers)]
    src_label = f"source='{source_filter}'" if source_filter else "todos los tipos"
    print(f"\nPapers seleccionados ({src_label}, top-{n_papers} por nº de preguntas):")
    for pid in selected:
        print(f"   {pid}  ({paper_counts[pid]} preguntas elegibles)")
    return selected


def seleccionar_papers_objetivo(
    args: argparse.Namespace,
    queries: dict,
    qrels: dict,
    pdf_urls: dict,
    source_filter: str | None,
) -> list[str]:
    """Resolve the target paper list from CLI arguments.

    Supports single-document mode (--only-doc) and top-N mode.

    Args:
        args: Parsed CLI arguments.
        queries: queries.json dict.
        qrels: qrels.json dict.
        pdf_urls: pdf_urls.json dict.
        source_filter: Query source filter (None = all sources).

    Returns:
        Ordered list of paper IDs.

    Raises:
        SystemExit: If --only-doc is invalid or has no eligible questions.
    """
    if not args.only_doc:
        return seleccionar_papers(queries, qrels, pdf_urls, args.n_papers, source_filter)

    doc_id = args.only_doc.strip()
    if doc_id not in pdf_urls:
        print(f"ERROR: doc_id '{doc_id}' no encontrado en pdf_urls del dataset.")
        raise SystemExit(1)

    n_eligible = sum(
        1 for qid, qrel in qrels.items()
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

    print(f"\n--only-doc: paper único {doc_id} ({n_eligible} preguntas elegibles)")
    return [doc_id]


# ─────────────────────────────────────────────
# SECTION 5: CONSTRUCCIÓN DE PREGUNTAS
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

    Collects up to max_per_paper questions per paper, filtered by source type.

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

    print(f"\nTotal preguntas: {len(questions)} de {len(selected_papers)} papers "
          f"(máx {max_per_paper}/paper)")
    return questions, ground_truths, paper_ids


def filtrar_por_pdfs_disponibles(
    questions: list[str],
    ground_truths: list[str],
    paper_ids: list[str],
    available_papers: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Filter question triples to only include papers with a local PDF.

    Args:
        questions: Questions aligned by index.
        ground_truths: Ground truths aligned by index.
        paper_ids: Paper ID for each question.
        available_papers: Paper IDs that have a local PDF.

    Returns:
        Filtered (questions, ground_truths, paper_ids) tuple.
    """
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


# ─────────────────────────────────────────────
# SECTION 6: DESCARGA DE PDFs
# ─────────────────────────────────────────────

def descargar_pdfs(
    selected_papers: list[str],
    pdf_urls: dict,
    pdfs_dir: str,
    skip_existing: bool = True,
) -> list[str]:
    """Download PDFs from arXiv for the selected papers.

    Validates each response to detect arXiv rate-limit HTML pages.

    Args:
        selected_papers: List of arXiv paper IDs to download.
        pdf_urls: pdf_urls.json dict (paper_id -> URL).
        pdfs_dir: Local directory where PDFs will be saved.
        skip_existing: Skip already-downloaded files.

    Returns:
        List of paper IDs with a valid local PDF.
    """
    os.makedirs(pdfs_dir, exist_ok=True)
    successful: list[str] = []

    print(f"\nDescargando {len(selected_papers)} PDFs en {pdfs_dir}/")
    for i, paper_id in enumerate(selected_papers):
        out_path = os.path.join(pdfs_dir, f"{paper_id}.pdf")

        if skip_existing and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"   [{i+1}/{len(selected_papers)}] {paper_id}  (en caché, omitido)")
            successful.append(paper_id)
            continue

        url = pdf_urls[paper_id]
        print(f"   [{i+1}/{len(selected_papers)}] {paper_id}  <- {url}")

        try:
            resp = requests.get(url, headers=ARXIV_HEADERS, timeout=ARXIV_TIMEOUT_SECS)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"      ERROR al descargar: {e}")
            continue

        content_type = resp.headers.get("Content-Type", "")
        if "application/pdf" not in content_type and not resp.content.startswith(b"%PDF"):
            print(f"      AVISO: Content-Type inesperado '{content_type}' — omitido")
            continue

        with open(out_path, "wb") as fh:
            fh.write(resp.content)
        print(f"      {len(resp.content) / 1024:.0f} KB guardados")
        successful.append(paper_id)

        if i < len(selected_papers) - 1:
            time.sleep(ARXIV_DELAY_SECS)

    return successful


def obtener_pdfs_disponibles(selected_papers: list[str], pdfs_dir: str) -> list[str]:
    """Return paper IDs whose PDF already exists locally and is non-empty.

    Args:
        selected_papers: Candidate paper IDs.
        pdfs_dir: Directory where PDFs should exist.

    Returns:
        Paper IDs with a valid local PDF file.
    """
    return [
        pid for pid in selected_papers
        if os.path.exists(os.path.join(pdfs_dir, f"{pid}.pdf"))
        and os.path.getsize(os.path.join(pdfs_dir, f"{pid}.pdf")) > 0
    ]


# ─────────────────────────────────────────────
# SECTION 7: INDEXADO CHROMADB
# ─────────────────────────────────────────────

def conectar_e_indexar(
    pdfs_dir: str,
    force_reindex: bool,
    solo_archivos: list[str] | None = None,
) -> chromadb.Collection:
    """Connect to the dedicated ragbench ChromaDB and index PDFs if needed.

    Uses a separate DB path (ragbench_vector_db) so the ragbench evaluation
    never contaminates the production collection.

    Args:
        pdfs_dir: Directory containing the downloaded arXiv PDFs.
        force_reindex: If True, drop and rebuild the collection.
        solo_archivos: If set, only index these PDF filenames.

    Returns:
        The ChromaDB Collection ready for querying.
    """
    print(f"\nConectando a ChromaDB (ragbench): {RAGBENCH_DB_PATH}")
    client = chromadb.PersistentClient(path=RAGBENCH_DB_PATH)
    collection = client.get_or_create_collection(name=RAGBENCH_COLLECTION)

    if force_reindex and collection.count() > 0:
        print("   --force-reindex: eliminando colección existente...")
        client.delete_collection(RAGBENCH_COLLECTION)
        collection = client.get_or_create_collection(name=RAGBENCH_COLLECTION)

    if collection.count() == 0:
        print("   Colección vacía. Indexando PDFs (puede tardar varios minutos)...")
        if solo_archivos:
            print(f"   Solo indexando: {solo_archivos}")
        total = indexar_documentos(pdfs_dir, collection, solo_archivos=solo_archivos)
        print(f"   {total} fragmentos indexados.")
    else:
        print(f"   Colección con {collection.count()} fragmentos. "
              "Usa --force-reindex para reconstruir.")
        if solo_archivos:
            print(
                "   AVISO: --only-doc activo pero se reutiliza la colección existente. "
                "Usa --force-reindex para aislar ese documento."
            )

    return collection


# ─────────────────────────────────────────────
# SECTION 8: LLM DE EVALUACIÓN
# ─────────────────────────────────────────────

def configurar_llm_evaluacion():
    """Configure the Gemini LLM and embeddings used by RAGAS as evaluation judge.

    Reads GEMINI_API_KEY or GOOGLE_API_KEY from the environment.

    Returns:
        Tuple of (eval_llm, eval_embeddings).

    Raises:
        SystemExit: If the API key is missing or langchain-google-genai is absent.
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY o GOOGLE_API_KEY no encontrada en el entorno.")
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
        print("LLM evaluación:        Gemini 2.0 Flash (langchain-google-genai)")
        print("Embeddings evaluación: gemini-embedding-001 (langchain-google-genai)")
        return eval_llm, eval_embeddings
    except ImportError as err:
        print(f"ERROR: {err}")
        print("  Instala con: pip install langchain-google-genai")
        raise SystemExit(1)


# ─────────────────────────────────────────────
# SECTION 9: RESULTADOS
# ─────────────────────────────────────────────

def imprimir_resultados(df_scores: pd.DataFrame, questions: list[str]):
    """Print global RAGAS averages and per-question scores to stdout.

    Args:
        df_scores: DataFrame of RAGAS scores (one row per question).
        questions: Original question strings, aligned by index.
    """
    metric_cols = [c for c in METRIC_NAMES if c in df_scores.columns]
    if not metric_cols:
        print("\nNo se encontraron columnas de métricas en los resultados.")
        print(f"   Columnas disponibles: {list(df_scores.columns)}")
        return

    print("\n" + "=" * 70)
    print("  RAGAS RESULTS — vectara/open_ragbench — MEDIAS GLOBALES")
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
    print("  DETALLE POR PREGUNTA")
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
# SECTION 10: DEBUG JSON
# ─────────────────────────────────────────────

def _extraer_justificaciones_traces(traces: list, metric_cols: list) -> list[dict]:
    """Extract per-question LLM justification payloads from RAGAS traces.

    Args:
        traces: List of trace objects from result.traces.
        metric_cols: Metric column names to look for in each trace.

    Returns:
        List of dicts (one per question) mapping metric names to justifications.
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
        description="Evalúa MonkeyGrab RAG con vectara/open_ragbench (subconjunto arXiv)"
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        choices=["text-image", "text-table", "text", "all"],
        help=f"Tipo de fuente de preguntas (default: {DEFAULT_SOURCE}). "
             "Usa 'all' para incluir todos los tipos.",
    )
    parser.add_argument(
        "--only-doc",
        metavar="DOC_ID",
        default=None,
        help="Evalúa un solo paper por doc_id (ej: 2403.20331v2). "
             "Sobreescribe --n-papers; indexa solo DOC_ID.pdf al construir la colección.",
    )
    parser.add_argument(
        "--n-papers", type=int, default=N_PAPERS,
        help=f"Número de papers arXiv a seleccionar (default: {N_PAPERS})"
    )
    parser.add_argument(
        "--max-q", type=int, default=MAX_QUESTIONS_PER_PAPER,
        help=f"Máximo de preguntas por paper (default: {MAX_QUESTIONS_PER_PAPER})"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Omite la descarga de PDFs y usa los existentes en rag/ragbench_pdfs/"
    )
    parser.add_argument(
        "--force-reindex", action="store_true",
        help="Borra y reconstruye la colección ChromaDB aunque ya tenga datos"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Imprime cada pregunta mientras se procesa"
    )
    parser.add_argument(
        "--no-debug", action="store_true",
        help="No guarda ragas_ragbench_debug.json"
    )
    args = parser.parse_args()

    if args.max_q < 1:
        print("ERROR: --max-q debe ser >= 1.")
        raise SystemExit(1)
    if not args.only_doc and args.n_papers < 1:
        print("ERROR: --n-papers debe ser >= 1.")
        raise SystemExit(1)

    source_filter = None if args.source == "all" else args.source
    print(f"\nFiltro de fuente: {args.source}")

    import warnings
    try:
        from ragas import evaluate
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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
        print(f"ERROR importando RAGAS: {e}")
        print("  Instala con: pip install -r evaluation/requirements.txt")
        raise SystemExit(1) from e

    def _ensure_instance(m):
        return m() if isinstance(m, type) else m

    faithfulness       = _ensure_instance(faithfulness)
    answer_relevancy   = _ensure_instance(answer_relevancy)
    context_precision  = _ensure_instance(context_precision)
    context_recall     = _ensure_instance(context_recall)
    answer_correctness = _ensure_instance(answer_correctness)

    eval_llm, eval_embeddings = configurar_llm_evaluacion()

    queries, qrels, answers_gt, pdf_urls = descargar_metadatos()

    selected_papers = seleccionar_papers_objetivo(
        args=args,
        queries=queries,
        qrels=qrels,
        pdf_urls=pdf_urls,
        source_filter=source_filter,
    )

    print("\nSeleccionando preguntas:")
    questions, ground_truths, paper_ids_per_q = construir_preguntas(
        queries, qrels, answers_gt, selected_papers, source_filter, args.max_q
    )
    if not questions:
        print("ERROR: No se seleccionaron preguntas con los filtros actuales.")
        raise SystemExit(1)

    if not args.skip_download:
        successful_papers = descargar_pdfs(selected_papers, pdf_urls, RAGBENCH_PDFS_DIR)
    else:
        print(f"\n--skip-download: usando PDFs existentes en {RAGBENCH_PDFS_DIR}/")
        successful_papers = obtener_pdfs_disponibles(selected_papers, RAGBENCH_PDFS_DIR)

    if len(successful_papers) < len(selected_papers):
        missing = set(selected_papers) - set(successful_papers)
        print(f"\nAVISO: {len(successful_papers)}/{len(selected_papers)} PDFs disponibles.")
        print(f"   Faltantes: {missing}")

    questions, ground_truths, paper_ids_per_q = filtrar_por_pdfs_disponibles(
        questions, ground_truths, paper_ids_per_q, successful_papers
    )
    if not questions:
        print("ERROR: No quedan preguntas tras filtrar por PDFs disponibles.")
        print("   Ejecuta sin --skip-download para descargar los PDFs necesarios.")
        raise SystemExit(1)

    solo_pdf = [f"{args.only_doc.strip()}.pdf"] if args.only_doc else None
    collection = conectar_e_indexar(RAGBENCH_PDFS_DIR, args.force_reindex, solo_archivos=solo_pdf)

    print(f"\nEjecutando pipeline RAG para {len(questions)} preguntas...")
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
        print(f"\nERROR: No se pudo conectar a Ollama: {e}")
        print("   Asegúrate de que Ollama está corriendo: ollama serve")
        raise SystemExit(1)

    t_rag = time.time() - t_start
    print(f"   Completado en {t_rag:.1f}s ({t_rag / max(len(questions), 1):.1f}s/pregunta)")

    print("\nConstruyendo EvaluationDataset para RAGAS...")
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

    tiene_ground_truth = any(gt.strip() for gt in ground_truths)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    if tiene_ground_truth:
        metrics.insert(0, answer_correctness)

    print("\nEjecutando evaluación RAGAS (puede tardar varios minutos)...")
    t_eval_start = time.time()

    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=RunConfig(timeout=600, max_retries=15),
    )

    t_eval = time.time() - t_eval_start
    print(f"   Evaluación completada en {t_eval:.1f}s")

    df_scores = result.to_pandas()
    imprimir_resultados(df_scores, questions)

    df_scores.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nCSV resultados: {OUTPUT_CSV}")

    if not args.no_debug:
        debug_path = guardar_debug(
            result=result,
            questions=questions,
            gen_answers=gen_answers,
            ground_truths=ground_truths,
            contexts_list=contexts_list,
            paper_ids=paper_ids_per_q,
        )
        print(f"Debug JSON:     {debug_path}")


if __name__ == "__main__":
    main()
