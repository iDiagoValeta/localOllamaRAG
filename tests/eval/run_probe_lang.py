"""run_probe_lang -- diagnostic language-axis probe for the search-set corpus gap.

Runs tests/eval/probe_cases_lang.jsonl through the real pipeline, isolated
from every store the product or run_eval.py already write to. See
docs/design/2026-07-28-loop-automejorable.md section 3 ("Sonda previa") for
why this exists: before authoring the ~55 cases the loop's search set needs
(same section, "Tamaño"), a small diagnostic batch decides -- per axis --
whether it is worth the effort. This script covers the language axis: the
es/ (Castilian) and ca/ (Valencian) document stores the product already
ships, which tests/eval/gold_cases.jsonl never exercises today.

Six documents, staged into their own FAISS collection (see PROBE_DOCS_DIR)
so this run can never write into `docs_es`/`docs_ca` (the product's live
stores), `dev_docs`/`blind_docs` (run_eval.py's gate stores), or any other
`rag/vector_db/*` collection.

This script does not gate anything and keeps no baseline: it prints a
pass/fail per case and a summary in the same shape run_eval.py's own report
uses, so results read the same way. The verdict per axis -- viable / viable
tras arreglo / inviable con esta fuente -- is a human call made by reading
that output, not something this script computes.

Usage:
    python tests/eval/run_probe_lang.py
    python tests/eval/run_probe_lang.py --models gemma4:e2b

Dependencies: identical to run_eval.py -- the full multimodal stack (FAISS,
MinerU, jina-clip-v2, sentence-transformers, BGE reranker) plus a running
local Ollama server and, in practice, a GPU.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# CONSTANTS

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]

# Mirrors run_eval.py's own sys.path bootstrap -- this script must be
# runnable as `python tests/eval/run_probe_lang.py` from the repo root with
# no PYTHONPATH set beforehand.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import run_eval  # noqa: E402  (tests/eval sibling module -- reused, never modified)

PROBE_CASES_FILE = EVAL_DIR / "probe_cases_lang.jsonl"

# Staged copies of the chosen PDFs, gitignored like run_eval.py's
# BLIND_DOCS_DIR. Its basename ("probe_docs_lang") is what derive_db_paths
# turns into the FAISS collection name and store path when passed below as
# both `carpeta` and `path_label` -- distinct from "es"/"ca" (the product's
# own stores), "dev_docs"/"blind_docs" (run_eval.py's gate stores), and
# everything else under rag/vector_db/. This is the isolation the design doc
# requires (section 3, "Sonda previa": "indexats en la seua propia col·lecció
# aïllada per a no alterar els stores contra els quals es mesura el
# producte").
PROBE_DOCS_DIR = EVAL_DIR / "probe_docs_lang"
PROBE_RESULTS_DIR = EVAL_DIR / "probe_runs_lang"

# Maps each probe case's "paper" slug to the product PDF it is read from.
# Copied into PROBE_DOCS_DIR under the slug's own filename rather than the
# product's accented one, mirroring run_eval.py's stage_blind_papers (which
# stages arXiv downloads under the gold case's "paper" slug instead of the
# cache's arxiv-id filename) -- for the same reason: indexing derives the
# stored `source` metadata from the filename it is given, and
# probe_cases_lang.jsonl's "paper" field is the slug, not the product's.
_SOURCE_PDFS: Dict[str, Path] = {
    "horchata-chufa": REPO_ROOT / "rag" / "docs" / "es" / "Horchata_de_chufa.pdf",
    "albufera": REPO_ROOT / "rag" / "docs" / "es" / "Parque_natural_de_la_Albufera.pdf",
    "cid-valencia": REPO_ROOT / "rag" / "docs" / "es" / "Rodrigo_Díaz_de_Vivar.pdf",
    "llotja-seda": REPO_ROOT / "rag" / "docs" / "ca" / "Llotja_de_la_Seda.pdf",
    "jaume-conqueridor": REPO_ROOT / "rag" / "docs" / "ca" / "Jaume_el_Conqueridor.pdf",
    "pilota-valenciana": REPO_ROOT / "rag" / "docs" / "ca" / "Pilota_valenciana.pdf",
}

DEFAULT_MODELS = ["gemma4:e2b"]


def load_probe_cases() -> List[Dict[str, Any]]:
    """Parse tests/eval/probe_cases_lang.jsonl into a list of case dicts."""
    cases = []
    for line in PROBE_CASES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def stage_probe_docs(cases: Sequence[Dict[str, Any]]) -> set:
    """Copy every PDF this batch of cases needs into PROBE_DOCS_DIR.

    Args:
        cases: Parsed probe_cases_lang.jsonl.

    Returns:
        The set of "<slug>.pdf" filenames now staged.

    Raises:
        run_eval.EvalSetupError: A case names a "paper" slug with no entry
            in _SOURCE_PDFS, or the mapped source PDF is missing.
    """
    required = {case["paper"] for case in cases}
    PROBE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    staged = set()
    for slug in sorted(required):
        source = _SOURCE_PDFS.get(slug)
        if source is None:
            raise run_eval.EvalSetupError(
                f"probe case references paper {slug!r} with no entry in "
                "run_probe_lang._SOURCE_PDFS"
            )
        if not source.exists():
            raise run_eval.EvalSetupError(f"probe source PDF missing: {source}")
        filename = f"{slug}.pdf"
        dest = PROBE_DOCS_DIR / filename
        if not dest.exists():
            shutil.copyfile(source, dest)
            print(f"[stage] {filename} <- {source.relative_to(REPO_ROOT)}")
        staged.add(filename)
    return staged


def run_probe(models: Sequence[str] = DEFAULT_MODELS, write_report: bool = True) -> Dict[str, Any]:
    """Run every probe case through the real pipeline, report-only (no gate).

    Mirrors run_eval.run_all_cases's two-phase shape (retrieve everything,
    release the GPU-resident retrieval models, then generate) at a much
    smaller scale -- six documents, one isolated collection, not run_eval.py's
    dev+blind corpus -- so it reuses run_eval.py's own per-case graders
    (run_retrieval_case, run_factual_case) instead of re-implementing them.
    grade.py itself is never touched: identical scoring, the same
    content-kind taxonomy, and the same record shape as tests/eval/runs/*.json,
    so a human reading this script's report reads it the same way.

    Args:
        models: Ollama generator models to evaluate.
        write_report: Write a dated JSON report under PROBE_RESULTS_DIR.

    Returns:
        A dict with "summary" (run_eval.build_summary()'s blocks) and
        "records" (every per-case result dict).

    Raises:
        run_eval.EvalSetupError: Ollama, a required model, or a source PDF
            is missing.
    """
    run_started = time.perf_counter()
    cases = load_probe_cases()
    required_models = set(models) | {run_eval.AUX_MODEL}
    run_eval.preflight_ollama(required_models)

    # Heavy import deferred past preflight, same reasoning as run_eval.evaluate().
    import rag.chat_pdfs as rag

    records: List[Dict[str, Any]] = []
    with run_eval._scoped_model_roles(
        rag,
        {"chat": run_eval.AUX_MODEL, "contextual": run_eval.AUX_MODEL, "recomp": run_eval.AUX_MODEL},
    ):
        required = stage_probe_docs(cases)
        retrieve, evidence, stack = run_eval.ensure_indexed(
            rag, PROBE_DOCS_DIR, required, "lang probe", path_label=PROBE_DOCS_DIR,
        )
        try:
            print(f"\n[run] {len(cases)} probe case(s) x up to {len(models)} model(s)\n", flush=True)
            pending: List[Dict[str, Any]] = []

            print("[phase 1/2] retrieval for every case", flush=True)
            for case in cases:
                t0 = time.perf_counter()
                try:
                    result = retrieve.run(case["question"])
                    retrieved = [run_eval._fragment_to_dict(f) for f in result.fragments]
                    selected, _metrics = evidence.select_evidence(result.fragments)
                    fragments = [run_eval._fragment_to_dict(f) for f in selected]
                except Exception as exc:  # noqa: BLE001 -- a bad case must not abort the whole probe
                    records.append({
                        "id": case["id"], "paper": case["paper"], "case_type": case["case_type"],
                        "lang": case["lang"], "model": None, "passed": False,
                        "infrastructure_error": True,
                        "reason": f"retrieval failed -- {type(exc).__name__}: {exc}",
                        "elapsed_seconds": round(time.perf_counter() - t0, 2),
                    })
                    print(f"  [ERROR] {case['id']} -- retrieval: {type(exc).__name__}: {exc}", flush=True)
                    continue
                elapsed = time.perf_counter() - t0

                if case["case_type"] in run_eval._RETRIEVAL_ONLY_CASE_TYPES:
                    record = run_eval.run_retrieval_case(case, retrieved, elapsed)
                    records.append(record)
                    status = "PASS" if record["passed"] else "FAIL"
                    print(f"  [{status}] {case['id']} ({elapsed:.1f}s) -- {record['reason']}", flush=True)
                else:
                    pending.append({"case": case, "fragments": fragments, "elapsed": elapsed})
                    print(f"  [retrieved] {case['id']} ({elapsed:.1f}s, {len(fragments)} fragments)", flush=True)

            run_eval._release_gpu_models(retrieve)

            print(f"\n[phase 2/2] generation for {len(pending)} case(s)", flush=True)
            with run_eval._generation_keep_alive(set(models) | {run_eval.AUX_MODEL}):
                for item in pending:
                    records.extend(
                        run_eval.run_factual_case(rag, item["case"], item["fragments"], models, item["elapsed"])
                    )
        finally:
            closer = getattr(stack.embedder, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass

    summary = run_eval.build_summary(records)
    elapsed_total = time.perf_counter() - run_started
    run_eval.print_summary(summary)
    print(f"total elapsed: {elapsed_total / 60:.1f} min")

    report_path: Optional[Path] = None
    if write_report:
        PROBE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = PROBE_RESULTS_DIR / f"{timestamp}_lang_probe.json"
        report_path.write_text(json.dumps({
            "run": {
                "timestamp": timestamp,
                "models": list(models),
                "aux_model": run_eval.AUX_MODEL,
                "num_cases": len(cases),
                "elapsed_seconds": round(elapsed_total, 1),
            },
            "summary": summary,
            "results": records,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nfull results: {report_path}")

    print(
        "\nThis is a diagnostic probe, not a gate: no baseline, no automated "
        "pass/fail verdict. Read the per-case results above (and the JSON "
        "report) and record a verdict per axis by hand -- viable / viable "
        "tras arreglo / inviable con esta fuente -- in "
        "docs/design/2026-07-28-loop-automejorable.md section 3."
    )
    return {
        "summary": summary,
        "records": records,
        "report_path": str(report_path) if report_path is not None else None,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help=f"Ollama generator models to evaluate (default: {DEFAULT_MODELS})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run_probe(models=args.models)
    except run_eval.EvalSetupError as exc:
        print(f"\nSETUP FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
