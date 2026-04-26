"""
RagBench visual-source inference without RAGAS.

Builds a RagBench subset from table/image questions, downloads the matching
PDFs into an isolated corpus folder, indexes that folder into its own ChromaDB
path, and runs RAG inference only. No RAGAS evaluation is launched.

Usage:
    python evaluation/run_ragbench_visual_inference.py --n-papers 25 --max-q 5
    python evaluation/run_ragbench_visual_inference.py --skip-download --force-reindex
"""

# ---------------------------------------------------------------------------
# MODULE MAP -- Section index
# ---------------------------------------------------------------------------
#
#  CONFIGURATION
#  +-- 1. Imports and constants
#
#  PREPARATION
#  +-- 2. Source parsing and RagBench row selection
#  +-- 3. Dataset and manifest writing
#
#  INFERENCE
#  +-- 4. RAG-only execution and exports
#
#  ENTRY
#  +-- 5. CLI parser and main
#
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluation.run_eval as run_eval


EVAL_DIR = ROOT / "evaluation"
DEFAULT_VISUAL_SOURCES = ("text-image", "text-table")
ALLOWED_VISUAL_SOURCES = set(DEFAULT_VISUAL_SOURCES)
RAGBENCH_VISUAL_PDFS_DIR = ROOT / "rag" / "docs" / "en_ragbench_visual"
VISUAL_DATASETS_DIR = EVAL_DIR / "datasets" / "ragbench" / "prepared" / "visual"
VISUAL_RUN_DIR = EVAL_DIR / "runs" / "inference" / "ragbench_visual"
VISUAL_RAGAS_RUN_DIR = EVAL_DIR / "runs" / "ragas" / "ragbench_visual"
VISUAL_PIPELINE_FLAGS = {
    **run_eval.RAGBENCH_FINAL_PIPELINE_FLAGS,
    "USAR_RERANKER": False,
}


def parse_sources(raw: str | None) -> list[str]:
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
            rows.append(
                {
                    "question": str(queries[qid]["query"]),
                    "ground_truth": str(answers.get(qid, "")),
                    "paper_id": paper_id,
                    "source_type": str(queries[qid].get("source", "")),
                }
            )

    print(
        f"\nTotal preguntas: {len(rows)} de {len(selected_papers)} papers "
        f"(max {max_per_paper}/paper)"
    )
    return rows


def filtrar_filas_por_pdfs(rows: list[dict[str, str]], available_papers: list[str]) -> list[dict[str, str]]:
    """Keep only rows whose PDF was downloaded or exists locally."""
    available = set(available_papers)
    return [row for row in rows if row["paper_id"] in available]


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def escribir_dataset_visual(rows: list[dict[str, str]], debug_dir: Path, tag: str) -> Path:
    """Persist the visual-source dataset consumed by run_eval.generar_respuestas_rag."""
    return _write_json(debug_dir / f"dataset_ragbench_visual_{tag}.json", rows)


def preparar_ragbench_visual(
    sources: list[str],
    n_papers: int,
    max_q: int,
    skip_download: bool,
    docs_dir: Path,
    debug_dir: Path,
    only_doc: str | None = None,
    excluded_doc_ids_path: Path = Path(run_eval.RAGBENCH_DEV_DOC_IDS_PATH),
) -> dict[str, Any]:
    """Prepare RagBench visual PDFs, dataset and manifest."""
    if max_q < 1:
        raise SystemExit("ERROR: max_q debe ser >= 1.")
    if not only_doc and n_papers < 1:
        raise SystemExit("ERROR: n_papers debe ser >= 1.")

    excluded_doc_ids = run_eval.cargar_doc_ids_dev_ragbench(str(excluded_doc_ids_path))
    print(f"\nPreparando RagBench visual: sources={','.join(sources)}, max_q={max_q}")
    print(f"Excluyendo dev split congelado: {len(excluded_doc_ids)} doc_ids")

    queries, qrels, answers, pdf_urls = run_eval.descargar_metadatos()
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
        successful_papers = run_eval.obtener_pdfs_disponibles(selected_papers, str(docs_dir))
    else:
        successful_papers = run_eval.descargar_pdfs(selected_papers, pdf_urls, str(docs_dir))

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
    tag = _safe_tag(f"{source_tag}_{paper_tag}_{max_q}q")
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
    _write_json(manifest_path, manifest)
    print(f"\nDataset preparado en: {dataset_path}")
    print(f"Manifiesto escrito en: {manifest_path}")
    return manifest


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
        rows.append(
            {
                "question": question,
                "ground_truth": generation["ground_truths"][idx],
                "answer": generation["answers"][idx],
                "paper_id": metadata.get("paper_id", ""),
                "source_type": metadata.get("source_type", ""),
                "contexts": json.dumps(generation["contexts_list"][idx], ensure_ascii=False),
                "status": status.get("status", ""),
                "reason": status.get("reason", ""),
            }
        )

    with result_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "ground_truth",
                "answer",
                "paper_id",
                "source_type",
                "contexts",
                "status",
                "reason",
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
            if key
            not in {
                "questions",
                "ground_truths",
                "answers",
                "contexts_list",
                "question_statuses",
            }
        },
        "rows": rows,
        "question_statuses": generation["question_statuses"],
    }
    _write_json(result_json, payload)
    print(f"\nResultados CSV:  {result_csv}")
    print(f"Resultados JSON: {result_json}")


def _load_visual_inference_generation(
    inference_json: Path,
    output_csv: Path,
    debug_json: Path | None,
) -> dict[str, Any]:
    """Rebuild run_eval's generation payload from visual inference artifacts."""
    payload = json.loads(inference_json.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"ERROR: no hay filas de inferencia en {inference_json}")

    questions: list[str] = []
    ground_truths: list[str] = []
    answers: list[str] = []
    contexts_list: list[list[str]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"ERROR: fila invalida en {inference_json}: {idx}")
        questions.append(str(row.get("question", "")))
        ground_truths.append(str(row.get("ground_truth", "")))
        answers.append(str(row.get("answer", "")))
        raw_contexts = row.get("contexts", "[]")
        if isinstance(raw_contexts, list):
            contexts = [str(ctx) for ctx in raw_contexts]
        else:
            try:
                decoded = json.loads(str(raw_contexts or "[]"))
            except json.JSONDecodeError:
                decoded = []
            contexts = [str(ctx) for ctx in decoded] if isinstance(decoded, list) else []
        contexts_list.append(contexts)

    generation_meta = payload.get("generation", {}) if isinstance(payload.get("generation"), dict) else {}
    manifest = payload.get("manifest", {}) if isinstance(payload.get("manifest"), dict) else {}
    dataset_path = generation_meta.get("dataset_path") or manifest.get("dataset_path") or ""
    checkpoint_path = generation_meta.get("checkpoint_path") or ""

    return {
        "dataset_path": str(Path(dataset_path).resolve()) if dataset_path else "",
        "output_path": str(output_csv.resolve()),
        "debug_path": str(debug_json.resolve()) if debug_json else None,
        "checkpoint_path": str(Path(checkpoint_path).resolve()) if checkpoint_path else str(inference_json.resolve()),
        "questions": questions,
        "ground_truths": ground_truths,
        "answers": answers,
        "contexts_list": contexts_list,
        "question_statuses": payload.get("question_statuses", []),
        "questions_count": len(questions),
        "indexed_fragments": int(generation_meta.get("indexed_fragments") or 0),
        "recomp_enabled": bool(generation_meta.get("recomp_enabled", True)),
        "pipeline_flags": generation_meta.get("pipeline_flags") or VISUAL_PIPELINE_FLAGS,
        "eval_corpus": generation_meta.get("eval_corpus") or "ragbench",
        "docs_dir": generation_meta.get("docs_dir") or manifest.get("docs_dir"),
        "pipeline_seconds": float(generation_meta.get("pipeline_seconds") or 0.0),
        "tiene_ground_truth": any(bool(gt) for gt in ground_truths),
    }


def evaluar_ragas_visual_desde_inferencia(
    inference_json: Path,
    scores_dir: Path,
    debug_dir: Path,
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
    """Run RAGAS over a completed visual inference JSON without regenerating answers."""
    if not inference_json.exists():
        raise SystemExit(f"ERROR: no existe el JSON de inferencia: {inference_json}")

    tag = inference_json.stem.replace("ragbench_visual_inference_", "")
    run_dir = scores_dir / tag
    output_csv = run_dir / "scores.csv"
    output_debug = run_dir / "debug.json"
    generation = _load_visual_inference_generation(
        inference_json=inference_json,
        output_csv=output_csv,
        debug_json=output_debug if save_debug else None,
    )

    print("\nEjecutando RAGAS sobre inferencia RagBench visual existente:")
    print(f"   inference_json: {inference_json}")
    print(f"   questions: {generation['questions_count']}")
    print(f"   output_csv: {output_csv}")
    if save_debug:
        print(f"   debug_json: {output_debug}")

    return run_eval.evaluar_respuestas_con_ragas(
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


def ejecutar_inferencia_visual(
    manifest: dict[str, Any],
    result_dir: Path,
    debug_dir: Path,
    verbose: bool = False,
    force_reindex: bool = False,
) -> dict[str, Any]:
    """Run RAG inference only and export non-RAGAS results."""
    dataset_path = str(Path(manifest["dataset_path"]).resolve())
    docs_dir = str(Path(manifest["docs_dir"]).resolve())
    indexed_files = [str(name) for name in manifest.get("indexed_files", [])]
    if not indexed_files:
        raise SystemExit("ERROR: el manifiesto no contiene indexed_files.")

    tag = _safe_tag(Path(dataset_path).stem.replace("dataset_ragbench_visual_", ""))
    run_dir = result_dir / tag
    result_csv = run_dir / "results.csv"
    result_json = run_dir / "results.json"
    checkpoint_path = run_dir / "checkpoint.json"

    print("\nEjecutando inferencia RagBench visual sin RAGAS:")
    print("   sources=" + ",".join(manifest.get("sources", [])))
    print("   query_decomposition=off, reranker=off, resto de flags=on")
    print(f"   dataset: {dataset_path}")
    print(f"   docs_dir: {docs_dir}")
    print(f"   indexed files: {len(indexed_files)}")
    print(f"   output dir: {run_dir}")

    generation = run_eval.generar_respuestas_rag(
        dataset_path=dataset_path,
        output_path=str(result_csv),
        debug_path=str(result_json),
        checkpoint_path=str(checkpoint_path),
        verbose=verbose,
        force_reindex=force_reindex,
        pipeline_flags=VISUAL_PIPELINE_FLAGS,
        eval_corpus="ragbench",
        docs_dir=docs_dir,
        solo_archivos=indexed_files,
        add_missing_from_filter=True,
    )
    exportar_resultados_inferencia(generation, manifest, result_csv, result_json)
    return {
        "dataset_path": dataset_path,
        "manifest": manifest,
        "output_csv": str(result_csv.resolve()),
        "output_json": str(result_json.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "docs_dir": docs_dir,
        "questions_count": generation["questions_count"],
        "indexed_fragments": generation["indexed_fragments"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and run RagBench table/image inference without RAGAS."
    )
    parser.add_argument("--sources", default=",".join(DEFAULT_VISUAL_SOURCES))
    parser.add_argument("--n-papers", type=int, default=25)
    parser.add_argument("--max-q", type=int, default=5)
    parser.add_argument("--only-doc", default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--docs-dir", default=str(RAGBENCH_VISUAL_PDFS_DIR))
    parser.add_argument("--debug-dir", default=str(VISUAL_DATASETS_DIR), help="Directory for prepared visual dataset and manifest.")
    parser.add_argument("--output-dir", default=str(VISUAL_RUN_DIR), help="Root directory for visual inference runs.")
    parser.add_argument("--ragas-only", action="store_true", help="Evaluate an existing visual inference JSON with RAGAS.")
    parser.add_argument(
        "--inference-json",
        default=None,
        help="Visual inference JSON to evaluate. Defaults to the tag derived from sources/n-papers/max-q.",
    )
    parser.add_argument("--ragas-scores-dir", default=str(VISUAL_RAGAS_RUN_DIR), help="Root directory for visual RAGAS runs.")
    parser.add_argument("--ragas-debug-dir", default=str(VISUAL_RAGAS_RUN_DIR), help="Deprecated; visual RAGAS debug is written next to scores.")
    parser.add_argument("--no-debug", action="store_true", help="Do not write RAGAS debug JSON.")
    parser.add_argument("--ragas-timeout", type=int, default=90)
    parser.add_argument("--ragas-max-retries", type=int, default=5)
    parser.add_argument("--ragas-max-wait", type=int, default=60)
    parser.add_argument("--ragas-max-workers", type=int, default=1)
    parser.add_argument("--ragas-batch-size", type=int, default=5)
    parser.add_argument("--ragas-metrics", default=None)
    parser.add_argument("--google-timeout", type=int, default=None)
    parser.add_argument("--google-retries", type=int, default=None)
    parser.add_argument("--raise-exceptions", action="store_true")
    parser.add_argument("--excluded-doc-ids", default=run_eval.RAGBENCH_DEV_DOC_IDS_PATH)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        sources = parse_sources(args.sources)
    except ValueError as exc:
        parser.error(str(exc))

    if args.ragas_only:
        if args.inference_json:
            inference_json = Path(args.inference_json)
        else:
            source_tag = "_".join(src.replace("text-", "") for src in sources)
            paper_tag = args.only_doc.strip() if args.only_doc else f"{args.n_papers}p"
            tag = _safe_tag(f"{source_tag}_{paper_tag}_{args.max_q}q")
            inference_json = Path(args.output_dir) / tag / "results.json"
        result = evaluar_ragas_visual_desde_inferencia(
            inference_json=inference_json,
            scores_dir=Path(args.ragas_scores_dir),
            debug_dir=Path(args.ragas_debug_dir),
            save_debug=not args.no_debug,
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
        print("\nRagBench visual RAGAS finished.")
        print(f"CSV:   {result['output_path']}")
        if result.get("debug_path"):
            print(f"Debug: {result['debug_path']}")
        return

    manifest = preparar_ragbench_visual(
        sources=sources,
        n_papers=args.n_papers,
        max_q=args.max_q,
        skip_download=args.skip_download,
        docs_dir=Path(args.docs_dir),
        debug_dir=Path(args.debug_dir),
        only_doc=args.only_doc,
        excluded_doc_ids_path=Path(args.excluded_doc_ids),
    )
    result = ejecutar_inferencia_visual(
        manifest=manifest,
        result_dir=Path(args.output_dir),
        debug_dir=Path(args.debug_dir),
        verbose=args.verbose,
        force_reindex=args.force_reindex,
    )
    print("\nRagBench visual inference finished.")
    print(f"CSV:        {result['output_csv']}")
    print(f"JSON:       {result['output_json']}")
    print(f"Checkpoint: {result['checkpoint_path']}")


if __name__ == "__main__":
    main()
