"""RagBench (vectara/open_ragbench) dataset preparation: text and visual flows.

# ─────────────────────────────────────────────
# MODULE MAP
# ─────────────────────────────────────────────
#
#  1. Constants and HuggingFace metadata download
#  2. Paper selection (top-N, only-doc, visual filters)
#  3. Question construction + PDF availability filter
#  4. PDF download
#  5. Manifest and prepared-dataset I/O
#  6. preparar_ragbench_eval_en (text final corpus)
#  7. preparar_ragbench_visual (table/image corpus)
#  8. exportar_resultados_inferencia (visual run -> CSV/JSON)
#
# ─────────────────────────────────────────────
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .datasets import (
    PROJ_ROOT,
    RAGBENCH_DATASETS_DIR,
    RAGBENCH_DEV_FROZEN_DATASETS_DIR,
    RAGBENCH_EVAL_DATASETS_DIR,
    RAGBENCH_VISUAL_DATASETS_DIR,
    guardar_json,
    resolver_ruta_dataset,
    safe_tag,
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
# SECTION 1: CONSTANTS AND METADATA DOWNLOAD
# ─────────────────────────────────────────────

HF_REPO = "vectara/open_ragbench"
HF_SUBDIR = "pdf/arxiv"
HF_METADATA_FILES = ("queries.json", "qrels.json", "answers.json", "pdf_urls.json")

ARXIV_DELAY_SECS = 5
ARXIV_TIMEOUT_SECS = 60
ARXIV_HEADERS = {
    "User-Agent": "MonkeyGrab-TFG-Eval/1.0 (academic research; Universitat Politecnica de Valencia)"
}

RAGBENCH_DEV_PDFS_DIR = os.path.join(PROJ_ROOT, "rag", "docs", "en")
RAGBENCH_EVAL_PDFS_DIR = os.path.join(PROJ_ROOT, "rag", "docs", "en_ragbench_eval")
RAGBENCH_VISUAL_PDFS_DIR = os.path.join(PROJ_ROOT, "rag", "docs", "en_ragbench_visual")
RAGBENCH_DEV_DOC_IDS_PATH = os.path.join(
    RAGBENCH_DEV_FROZEN_DATASETS_DIR,
    "ragbench_en_dev_manifest_10p_5q_frozen.json",
)
RAGBENCH_EVAL_PREPARED_DIR = RAGBENCH_EVAL_DATASETS_DIR
RAGBENCH_DEV_FROZEN_PREPARED_DIR = RAGBENCH_DEV_FROZEN_DATASETS_DIR
RAGBENCH_VISUAL_PREPARED_DIR = RAGBENCH_VISUAL_DATASETS_DIR
RAGBENCH_EVAL_MANIFEST_PATH = os.path.join(
    RAGBENCH_EVAL_PREPARED_DIR,
    "ragbench_en_eval_manifest_40p.json",
)

DEFAULT_VISUAL_SOURCES = ("text-image", "text-table")
ALLOWED_VISUAL_SOURCES = set(DEFAULT_VISUAL_SOURCES)


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


# ─────────────────────────────────────────────
# SECTION 2: PAPER SELECTION
# ─────────────────────────────────────────────

def seleccionar_papers(
    queries: dict,
    qrels: dict,
    pdf_urls: dict,
    n_papers: int,
    source_filter: str | None,
    excluded_doc_ids: list[str] | None = None,
) -> list[str]:
    """Select top-N papers by eligible question count."""
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
            queries, qrels, pdf_urls, n_papers, source_filter,
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


def parse_visual_sources(raw: str | None) -> list[str]:
    """Parse and validate the requested RagBench visual source filters."""
    if not raw:
        return list(DEFAULT_VISUAL_SOURCES)
    sources = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(sources) - ALLOWED_VISUAL_SOURCES)
    if unknown:
        valid = ", ".join(sorted(ALLOWED_VISUAL_SOURCES))
        raise ValueError(f"Unsupported source(s): {', '.join(unknown)}. Valid: {valid}")
    if not sources:
        raise ValueError("At least one source must be selected.")
    return sources


def seleccionar_papers_visuales(
    queries: dict[str, Any],
    qrels: dict[str, Any],
    pdf_urls: dict[str, Any],
    sources: list[str],
    n_papers: int,
    excluded_doc_ids: list[str] | None = None,
    only_doc: str | None = None,
) -> list[str]:
    """Select RagBench papers with eligible table/image questions."""
    source_set = set(sources)
    excluded = set(excluded_doc_ids or [])

    if only_doc:
        doc_id = only_doc.strip()
        if doc_id not in pdf_urls:
            raise SystemExit(f"ERROR: doc_id '{doc_id}' no encontrado en pdf_urls.")
        if doc_id in excluded:
            raise SystemExit(f"ERROR: doc_id '{doc_id}' pertenece al dev split excluido.")
        eligible = [
            qid
            for qid, qrel in qrels.items()
            if qrel.get("doc_id") == doc_id
            and qid in queries
            and queries[qid].get("source") in source_set
        ]
        if not eligible:
            raise SystemExit(
                f"ERROR: no hay preguntas visuales para '{doc_id}' con sources={sources}."
            )
        print(f"\n--only-doc: {doc_id} ({len(eligible)} preguntas elegibles)")
        return [doc_id]

    counts: Counter[str] = Counter()
    for qid, qrel in qrels.items():
        doc_id = qrel.get("doc_id")
        if not doc_id or doc_id in excluded or doc_id not in pdf_urls or qid not in queries:
            continue
        if queries[qid].get("source") in source_set:
            counts[doc_id] += 1

    selected = [paper_id for paper_id, _ in counts.most_common(n_papers)]
    print(
        f"\nPapers seleccionados (sources={','.join(sources)}, "
        f"top-{n_papers} por preguntas elegibles):"
    )
    for paper_id in selected:
        print(f"   {paper_id}  ({counts[paper_id]} preguntas elegibles)")
    return selected


# ─────────────────────────────────────────────
# SECTION 3: QUESTION CONSTRUCTION
# ─────────────────────────────────────────────

def construir_preguntas(
    queries: dict,
    qrels: dict,
    answers: dict,
    selected_papers: list[str],
    source_filter: str | None,
    max_per_paper: int,
) -> tuple[list[str], list[str], list[str]]:
    """Build aligned question, ground-truth and paper-id lists."""
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


def construir_filas_visuales(
    queries: dict[str, Any],
    qrels: dict[str, Any],
    answers: dict[str, Any],
    selected_papers: list[str],
    sources: list[str],
    max_per_paper: int,
) -> list[dict[str, str]]:
    """Build dataset rows while preserving each question's RagBench source."""
    source_set = set(sources)
    selected_set = set(selected_papers)
    per_paper: dict[str, list[str]] = {paper_id: [] for paper_id in selected_papers}

    for qid, qrel in qrels.items():
        doc_id = qrel.get("doc_id")
        if doc_id not in selected_set or qid not in queries:
            continue
        if queries[qid].get("source") not in source_set:
            continue
        per_paper[doc_id].append(qid)

    rows: list[dict[str, str]] = []
    print("\nSeleccionando preguntas:")
    for paper_id in selected_papers:
        chosen = per_paper[paper_id][:max_per_paper]
        if chosen:
            by_source = Counter(str(queries[qid].get("source", "?")) for qid in chosen)
            print(f"   {paper_id}: {dict(by_source)}")
        for qid in chosen:
            rows.append({
                "question": str(queries[qid]["query"]),
                "ground_truth": str(answers.get(qid, "")),
                "paper_id": paper_id,
                "source_type": str(queries[qid].get("source", "")),
            })

    print(
        f"\nTotal preguntas: {len(rows)} de {len(selected_papers)} papers "
        f"(max {max_per_paper}/paper)"
    )
    return rows


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


def filtrar_filas_por_pdfs(rows: list[dict[str, str]], available_papers: list[str]) -> list[dict[str, str]]:
    """Keep only rows whose PDF was downloaded or exists locally."""
    available = set(available_papers)
    return [row for row in rows if row["paper_id"] in available]


# ─────────────────────────────────────────────
# SECTION 4: PDF DOWNLOAD
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# SECTION 5: MANIFEST AND PREPARED-DATASET I/O
# ─────────────────────────────────────────────

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
    dataset_path = manifest.get("dataset_path")
    if dataset_path:
        manifest["dataset_path"] = resolver_ruta_dataset(str(dataset_path))
    return manifest


def selected_pdf_filenames_from_manifest(manifest: dict[str, Any]) -> list[str]:
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
    tag: str,
    filename_prefix: str = "dataset_ragbench",
    output_dir: str | None = None,
) -> str:
    """Write a JSON dataset consumable by infer.py."""
    target_dir = output_dir or RAGBENCH_DATASETS_DIR
    os.makedirs(target_dir, exist_ok=True)
    prepared_dataset = os.path.join(target_dir, f"{filename_prefix}_{tag}.json")
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


def escribir_dataset_visual(rows: list[dict[str, str]], output_dir: Path, tag: str) -> Path:
    """Persist the visual-source dataset consumed by infer.py."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"dataset_ragbench_visual_{tag}.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ─────────────────────────────────────────────
# SECTION 6: PREPARAR_RAGBENCH_EVAL_EN
# ─────────────────────────────────────────────

def preparar_ragbench_eval_en(
    source: str = "text",
    n_papers: int = 40,
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
        queries, qrels, answers_gt, selected_papers, source_filter, max_q,
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
        questions, ground_truths, paper_ids, successful_papers,
    )
    if not questions:
        print("ERROR: No quedan preguntas tras filtrar por PDFs disponibles.")
        print("   Ejecuta sin --skip-download para descargar los PDFs necesarios.")
        raise SystemExit(1)

    dataset_tag = f"{source_filter}_{len(successful_papers)}p_{max_q}q_eval"
    tag = safe_tag(dataset_tag)
    prepared_dataset = escribir_dataset_preparado(
        questions, ground_truths, paper_ids, tag,
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
    guardar_json(manifest_path, manifest)
    print(f"\nDataset preparado en: {prepared_dataset}")
    print(f"Manifiesto escrito en: {manifest_path}")
    return manifest


# ─────────────────────────────────────────────
# SECTION 7: PREPARAR_RAGBENCH_VISUAL
# ─────────────────────────────────────────────

def preparar_ragbench_visual(
    sources: list[str],
    n_papers: int,
    max_q: int,
    skip_download: bool,
    docs_dir: Path,
    debug_dir: Path,
    only_doc: str | None = None,
    excluded_doc_ids_path: Path = Path(RAGBENCH_DEV_DOC_IDS_PATH),
) -> dict[str, Any]:
    """Prepare RagBench visual PDFs, dataset and manifest."""
    if max_q < 1:
        raise SystemExit("ERROR: max_q debe ser >= 1.")
    if not only_doc and n_papers < 1:
        raise SystemExit("ERROR: n_papers debe ser >= 1.")

    excluded_doc_ids = cargar_doc_ids_dev_ragbench(str(excluded_doc_ids_path))
    print(f"\nPreparando RagBench visual: sources={','.join(sources)}, max_q={max_q}")
    print(f"Excluyendo dev split congelado: {len(excluded_doc_ids)} doc_ids")

    queries, qrels, answers, pdf_urls = descargar_metadatos()
    selected_papers = seleccionar_papers_visuales(
        queries=queries,
        qrels=qrels,
        pdf_urls=pdf_urls,
        sources=sources,
        n_papers=n_papers,
        excluded_doc_ids=excluded_doc_ids,
        only_doc=only_doc,
    )
    if not selected_papers:
        raise SystemExit("ERROR: no se seleccionaron papers con preguntas visuales.")

    rows = construir_filas_visuales(
        queries=queries,
        qrels=qrels,
        answers=answers,
        selected_papers=selected_papers,
        sources=sources,
        max_per_paper=max_q,
    )
    if not rows:
        raise SystemExit("ERROR: no se seleccionaron preguntas con los filtros actuales.")

    if skip_download:
        print(f"\n--skip-download: usando PDFs existentes en {docs_dir}/")
        successful_papers = obtener_pdfs_disponibles(selected_papers, str(docs_dir))
    else:
        successful_papers = descargar_pdfs(selected_papers, pdf_urls, str(docs_dir))

    if len(successful_papers) < len(selected_papers):
        missing = sorted(set(selected_papers) - set(successful_papers))
        print(f"\nAVISO: {len(successful_papers)}/{len(selected_papers)} PDFs disponibles.")
        print(f"   Faltantes: {missing}")

    rows = filtrar_filas_por_pdfs(rows, successful_papers)
    if not rows:
        raise SystemExit(
            "ERROR: no quedan preguntas tras filtrar por PDFs disponibles. "
            "Ejecuta sin --skip-download para descargar los PDFs necesarios."
        )

    source_tag = "_".join(src.replace("text-", "") for src in sources)
    paper_tag = only_doc.strip() if only_doc else f"{len(successful_papers)}p"
    tag = safe_tag(f"{source_tag}_{paper_tag}_{max_q}q")
    dataset_path = escribir_dataset_visual(rows, debug_dir, tag)
    manifest_path = debug_dir / f"ragbench_visual_manifest_{tag}.json"
    indexed_files = [f"{paper_id}.pdf" for paper_id in successful_papers]
    manifest = {
        "manifest_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": sources,
        "n_papers": len(successful_papers),
        "max_q": max_q,
        "docs_dir": str(docs_dir.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "selected_papers": successful_papers,
        "indexed_files": indexed_files,
        "excluded_doc_ids": excluded_doc_ids,
        "excluded_doc_ids_path": str(excluded_doc_ids_path.resolve()),
    }
    guardar_json(str(manifest_path), manifest)
    print(f"\nDataset preparado en: {dataset_path}")
    print(f"Manifiesto escrito en: {manifest_path}")
    return manifest


# ─────────────────────────────────────────────
# SECTION 8: VISUAL RESULT EXPORT
# ─────────────────────────────────────────────

def _metadata_by_question(dataset_path: str) -> dict[str, dict[str, str]]:
    with open(dataset_path, encoding="utf-8") as f:
        rows = json.load(f)
    return {
        str(row.get("question", "")): {
            "source_type": str(row.get("source_type", "")),
            "paper_id": str(row.get("paper_id", "")),
        }
        for row in rows
        if isinstance(row, dict)
    }


def exportar_resultados_inferencia(
    generation: dict[str, Any],
    manifest: dict[str, Any],
    result_csv: Path,
    result_json: Path,
) -> None:
    """Write CSV and JSON inference artifacts without RAGAS metrics."""
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    result_json.parent.mkdir(parents=True, exist_ok=True)
    metadata_by_question = _metadata_by_question(generation["dataset_path"])

    rows = []
    for idx, question in enumerate(generation["questions"]):
        status = generation["question_statuses"][idx] if idx < len(generation["question_statuses"]) else {}
        metadata = metadata_by_question.get(question, {})
        rows.append({
            "question": question,
            "ground_truth": generation["ground_truths"][idx],
            "answer": generation["answers"][idx],
            "paper_id": metadata.get("paper_id", ""),
            "source_type": metadata.get("source_type", ""),
            "contexts": json.dumps(generation["contexts_list"][idx], ensure_ascii=False),
            "status": status.get("status", ""),
            "reason": status.get("reason", ""),
        })

    with result_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question", "ground_truth", "answer", "paper_id", "source_type",
                "contexts", "status", "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": manifest,
        "generation": {
            key: value
            for key, value in generation.items()
            if key not in {"questions", "ground_truths", "answers", "contexts_list", "question_statuses"}
        },
        "rows": rows,
        "question_statuses": generation["question_statuses"],
    }
    guardar_json(str(result_json), payload)
    print(f"\nResultados CSV:  {result_csv}")
    print(f"Resultados JSON: {result_json}")
