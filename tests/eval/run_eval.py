"""run_eval -- self-sufficient runner for the gold eval gate (real pipeline, no mocks).

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  +-- 1. CONSTANTS          paths, default models, ratchet margin
#  +-- 2. PREFLIGHT          Ollama reachability + required models present
#  +-- 3. CORPUS STAGING     blind-set download/staging + cache-aware indexing
#  +-- 4. CASE EXECUTION     one gold case -> graded record(s)
#  +-- 5. REPORTING          JSON artifact, console summary, baseline gate
#  +-- 6. CLI                main()
#
# ─────────────────────────────────────────────

This is the "gate completo" from docs/design/2026-07-26-monkeygrab-v2.md
section 7.2: a single command that turns tests/eval/gold_cases.jsonl into a
pass/fail verdict by running the *real* pipeline (Ollama generation and
embeddings, hybrid BM25+semantic retrieval, Cross-Encoder reranking) -- no
mocks, no LLM judge. It self-provisions everything it needs and fails fast
with an actionable message the moment a prerequisite is missing:

    1. Ollama must be reachable and every model this run needs must already
       be pulled (section 2).
    2. The blind-set arXiv papers are downloaded if missing, reusing
       fetch_papers.py (section 3).
    3. Both corpora (dev set under rag/docs/en/, blind set under
       tests/eval/blind_docs/) are indexed if they are not already, and
       re-verified afterwards -- a paper that still has no index after an
       indexing attempt aborts the run rather than silently skipping cases.

Usage:
    python tests/eval/run_eval.py
    python tests/eval/run_eval.py --models gemma4:e2b gemma4:e4b
    python tests/eval/run_eval.py --update-baseline

Dependencies: the full rag/requirements.txt stack (chromadb, ollama,
sentence-transformers, rank-bm25, pymupdf4llm) plus a running local Ollama
server and, in practice, a GPU -- the reranker and every model role run on
CPU otherwise, which is not a supported configuration for this gate.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests

# ─────────────────────────────────────────────
# SECTION 1: CONSTANTS
# ─────────────────────────────────────────────

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]

# Mirrors the sys.path bootstrap rag/chat_pdfs.py does for itself -- this
# script must be runnable as `python tests/eval/run_eval.py` from the repo
# root with no PYTHONPATH set beforehand, per the "one command, no manual
# steps" requirement.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import fetch_papers  # noqa: E402  (tests/eval sibling module)
import grade  # noqa: E402  (tests/eval sibling module)

GOLD_FILE = EVAL_DIR / "gold_cases.jsonl"
BASELINE_FILE = EVAL_DIR / "baseline_min_pass_rate.txt"
RESULTS_DIR = EVAL_DIR / "runs"

# Papers not already shipped in rag/docs/en/ are staged here under their gold
# case "paper" slug (fetch_papers.py caches them by arXiv id instead, which
# does not match the corpus filename convention indexar_documentos expects).
BLIND_DOCS_DIR = EVAL_DIR / "blind_docs"
DEV_DOCS_DIR = REPO_ROOT / "rag" / "docs" / "en"

OLLAMA_BASE_URL = "http://localhost:11434"

# Default RAG generator models: the two small ones the 8 GB GPU in this
# environment can hold comfortably. gemma4:e4b is supported via --models but
# not defaulted to -- its Q4_K_M weights alone (~9.6 GB) exceed local VRAM.
DEFAULT_MODELS = ["gemma4:e2b", "qwen3.5:0.8b"]

# Model used for every non-generator role during this run (query
# decomposition, contextual-retrieval enrichment, RECOMP synthesis, image
# OCR/captioning). Kept separate from --models so indexing cost does not
# multiply by the number of generator models under test.
AUX_MODEL = "gemma4:e2b"

# Safety margin subtracted from an observed pass rate before it is written as
# the new baseline floor (see _update_baseline) -- keeps normal run-to-run
# variance (model sampling, Ollama warmup) from flipping the gate red.
BASELINE_MARGIN = 0.05


class EvalSetupError(RuntimeError):
    """Raised when a prerequisite (Ollama, a model, an index) is missing.

    Caught in main() and printed as a clean actionable message -- this is a
    self-sufficiency failure, not a bug, so it does not need a traceback.
    """


# ─────────────────────────────────────────────
# SECTION 2: PREFLIGHT
# ─────────────────────────────────────────────


def _installed_ollama_models() -> List[str]:
    """Return installed Ollama model names, or raise EvalSetupError.

    Raises:
        EvalSetupError: Ollama is not reachable at OLLAMA_BASE_URL.
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise EvalSetupError(
            f"Ollama is not reachable at {OLLAMA_BASE_URL} ({exc}). "
            "Start it first, e.g. `ollama serve`."
        ) from exc
    return [m["name"] for m in response.json().get("models", []) if m.get("name")]


def preflight_ollama(required_models: Iterable[str]) -> None:
    """Fail fast, with an exact `ollama pull` list, if any model is missing.

    Args:
        required_models: Every Ollama model name this run will call.

    Raises:
        EvalSetupError: Ollama is unreachable, or one or more models are
            missing (message lists every missing model, not just the first).
    """
    installed = set(_installed_ollama_models())
    missing = sorted(set(required_models) - installed)
    if missing:
        pulls = "\n".join(f"  ollama pull {name}" for name in missing)
        raise EvalSetupError(
            f"{len(missing)} required Ollama model(s) not installed:\n{pulls}"
        )


# ─────────────────────────────────────────────
# SECTION 3: CORPUS STAGING
# ─────────────────────────────────────────────


def load_gold_cases() -> List[Dict[str, Any]]:
    """Parse tests/eval/gold_cases.jsonl into a list of case dicts."""
    cases = []
    for line in GOLD_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def _required_pdfs(cases: Sequence[Dict[str, Any]], source: str) -> Dict[str, str]:
    """Map ``{"<paper>.pdf": arxiv_id_or_""}`` for cases with the given source."""
    out: Dict[str, str] = {}
    for case in cases:
        if case["source"] == source:
            out[f"{case['paper']}.pdf"] = case.get("arxiv_id", "")
    return out


def stage_blind_papers(cases: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """Download (if needed) and stage blind-set PDFs under BLIND_DOCS_DIR.

    Reuses fetch_papers.py's idempotent, header-validated download so a
    truncated or rate-limited response is never mistaken for a cached paper.
    The downloaded file is copied (not moved, so the arXiv-id-named cache
    stays intact and reusable) to ``<paper-slug>.pdf`` because
    indexar_documentos derives the ChromaDB ``source`` metadata from the
    filename, and gold_cases.jsonl's ``paper`` field is the slug, not the id.

    Args:
        cases: Full parsed gold_cases.jsonl.

    Returns:
        ``{"<paper>.pdf": arxiv_id}`` for every blind-set paper, staged.
    """
    required = _required_pdfs(cases, "arxiv")
    if not required:
        return {}

    BLIND_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    arxiv_ids = sorted({arxiv_id for arxiv_id in required.values() if arxiv_id})
    print(f"[stage] fetching {len(arxiv_ids)} blind-set arXiv paper(s) if missing...")
    downloaded = fetch_papers.download_papers(arxiv_ids, fetch_papers.DEFAULT_CACHE_DIR)
    by_id = {path.stem: path for path in downloaded}

    for filename, arxiv_id in required.items():
        dest = BLIND_DOCS_DIR / filename
        if dest.exists():
            continue
        src = by_id.get(arxiv_id)
        if src is None:
            raise EvalSetupError(f"blind paper {filename!r} (arXiv {arxiv_id}) failed to download")
        shutil.copyfile(src, dest)
        print(f"[stage] {filename} <- papers_cache/{src.name}")

    return required


def ensure_indexed(rag, carpeta: Path, required_pdfs: Iterable[str], label: str):
    """Index whatever ``required_pdfs`` are missing from a corpus, then verify.

    Cache-aware: only files absent from the collection are indexed
    (``solo_archivos``), so a paper already indexed from a previous run is
    never reprocessed. Uses the real production entry points
    (``set_docs_folder_runtime`` + ``indexar_documentos``), the same ones the
    CLI's ``/reindex`` command uses.

    Args:
        rag: The imported rag.chat_pdfs module.
        carpeta: PDF directory to index.
        required_pdfs: Filenames (e.g. "resnet.pdf") that must end up indexed.
        label: Short name for log lines ("dev set", "blind set").

    Returns:
        The ChromaDB collection for this corpus.

    Raises:
        EvalSetupError: A required PDF is still missing from the index after
            an indexing attempt.
    """
    required = set(required_pdfs)
    rag.set_docs_folder_runtime(str(carpeta))
    try:
        import chromadb

        client = chromadb.PersistentClient(path=rag.PATH_DB)
        collection = client.get_or_create_collection(name=rag.COLLECTION_NAME)

        existing = set(rag.obtener_documentos_indexados(collection))
        missing = sorted(required - existing)
        if not missing:
            print(f"[index] {label}: cache hit, {len(required)} paper(s) already indexed")
        else:
            print(f"[index] {label}: indexing {len(missing)} missing paper(s): {missing}")
            t0 = time.perf_counter()
            rag.indexar_documentos(str(carpeta), collection, solo_archivos=missing, silent=True)
            print(f"[index] {label}: done in {time.perf_counter() - t0:.0f}s")

        still_missing = sorted(required - set(rag.obtener_documentos_indexados(collection)))
        if still_missing:
            raise EvalSetupError(
                f"{label}: {still_missing} still not indexed after an indexing attempt "
                "-- check the [index] log above for the underlying error"
            )
        return collection
    finally:
        rag.set_docs_folder_runtime(None)


def verify_all_papers_indexed(
    cases: Sequence[Dict[str, Any]], dev_collection, blind_collection, rag
) -> None:
    """Final pre-run check: every paper a case references has an index entry.

    Args:
        cases: Full parsed gold_cases.jsonl.
        dev_collection: Collection backing the dev-set corpus.
        blind_collection: Collection backing the blind-set corpus.
        rag: The imported rag.chat_pdfs module (for obtener_documentos_indexados).

    Raises:
        EvalSetupError: Names every (paper, source) still missing an index.
    """
    dev_docs = set(rag.obtener_documentos_indexados(dev_collection))
    blind_docs = set(rag.obtener_documentos_indexados(blind_collection)) if blind_collection else set()

    missing = []
    for case in cases:
        filename = f"{case['paper']}.pdf"
        indexed = dev_docs if case["source"] == "corpus" else blind_docs
        if filename not in indexed:
            missing.append(f"{case['paper']} ({case['source']})")
    if missing:
        raise EvalSetupError(f"papers referenced by gold cases but not indexed: {sorted(set(missing))}")


# ─────────────────────────────────────────────
# SECTION 4: CASE EXECUTION
# ─────────────────────────────────────────────


def _kind_from_format(fmt: Optional[str]) -> str:
    """Map an indexed chunk's ``format`` metadata to the grader's content kind.

    The pipeline tags chunks as ``markdown``/``plain_text`` (body text) or
    ``image`` (VLM-described figures); there is no ``table`` tag yet, so
    table_retrieval cases are expected to fail until table-aware extraction
    lands (documented in tests/eval/README.md, not a case bug).
    """
    return "image" if fmt == "image" else "text"


def _fragment_diag(fragments: Sequence[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    """Compact {source, page, format} list for a failure's diagnostic record."""
    out = []
    for f in fragments[:limit]:
        meta = f.get("metadata", {})
        out.append({"source": meta.get("source"), "page": meta.get("page"), "format": meta.get("format")})
    return out


def run_retrieval_case(case: Dict[str, Any], fragments: Sequence[Dict[str, Any]], elapsed: float) -> Dict[str, Any]:
    """Grade a figure_retrieval/table_retrieval case against retrieved content kinds."""
    hit_kinds = [_kind_from_format(f.get("metadata", {}).get("format")) for f in fragments]
    result = grade.grade_retrieval(hit_kinds, case)
    record = {
        "id": case["id"], "paper": case["paper"], "case_type": case["case_type"],
        "lang": case["lang"], "model": None, "passed": result["pass"],
        "reason": result["reason"], "elapsed_seconds": round(elapsed, 2),
    }
    if not result["pass"]:
        record["retrieved"] = _fragment_diag(fragments)
    return record


def run_factual_case(
    rag, case: Dict[str, Any], fragments: Sequence[Dict[str, Any]], models: Sequence[str], retrieval_elapsed: float
) -> List[Dict[str, Any]]:
    """Grade a factual_number/factual_concept case once per generator model.

    Retrieval is computed once by the caller and reused across models --
    ``preparar_fragmentos_para_generacion`` does not depend on the generator,
    only on retrieval + reranking, so re-running it per model would be pure
    waste (and, more importantly, would not exercise anything different).
    """
    if not fragments:
        return [{
            "id": case["id"], "paper": case["paper"], "case_type": case["case_type"],
            "lang": case["lang"], "model": model, "passed": False,
            "reason": "no fragments retrieved", "elapsed_seconds": round(retrieval_elapsed, 2),
        } for model in models]

    records = []
    for model in models:
        rag.set_model_roles_runtime({"rag": model})
        t0 = time.perf_counter()
        answer = rag.generar_respuesta_silenciosa(case["question"], list(fragments))
        gen_elapsed = time.perf_counter() - t0
        result = grade.grade_answer(answer, case)
        record = {
            "id": case["id"], "paper": case["paper"], "case_type": case["case_type"],
            "lang": case["lang"], "model": model, "passed": result["pass"],
            "reason": result["reason"],
            "elapsed_seconds": round(retrieval_elapsed + gen_elapsed, 2),
        }
        if not result["pass"]:
            record["answer"] = answer
            record["retrieved"] = _fragment_diag(fragments)
        records.append(record)
        status = "PASS" if result["pass"] else "FAIL"
        print(f"  [{status}] {case['id']} / {model} ({gen_elapsed:.1f}s) -- {result['reason']}", flush=True)
    return records


def run_all_cases(
    rag, cases: Sequence[Dict[str, Any]], dev_collection, blind_collection, models: Sequence[str]
) -> List[Dict[str, Any]]:
    """Run every gold case, grading retrieval-only cases once and factual cases per model."""
    records: List[Dict[str, Any]] = []
    for case in cases:
        collection = dev_collection if case["source"] == "corpus" else blind_collection
        t0 = time.perf_counter()
        fragmentos_ranked, _, _ = rag.realizar_busqueda_hibrida(case["question"], collection)
        fragmentos_finales, _ = rag.preparar_fragmentos_para_generacion(fragmentos_ranked, collection)
        retrieval_elapsed = time.perf_counter() - t0

        if case["case_type"] in ("figure_retrieval", "table_retrieval"):
            record = run_retrieval_case(case, fragmentos_finales, retrieval_elapsed)
            records.append(record)
            status = "PASS" if record["passed"] else "FAIL"
            print(f"  [{status}] {case['id']} ({retrieval_elapsed:.1f}s) -- {record['reason']}", flush=True)
        else:
            records.extend(run_factual_case(rag, case, fragmentos_finales, models, retrieval_elapsed))
    return records


# ─────────────────────────────────────────────
# SECTION 5: REPORTING
# ─────────────────────────────────────────────


def _bucket_stats(records: Sequence[Dict[str, Any]], key_fn) -> Dict[str, Dict[str, Any]]:
    """Group records by key_fn(record) and compute pass/total/rate per bucket."""
    buckets: Dict[str, Dict[str, int]] = {}
    for r in records:
        key = key_fn(r)
        b = buckets.setdefault(key, {"total": 0, "passed": 0})
        b["total"] += 1
        b["passed"] += int(r["passed"])
    return {
        key: {**b, "pass_rate": round(b["passed"] / b["total"], 4) if b["total"] else 0.0}
        for key, b in buckets.items()
    }


def build_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the overall / by-case-type / by-model / cross summary blocks."""
    total = len(records)
    passed = sum(1 for r in records if r["passed"])
    return {
        "overall": {
            "total": total, "passed": passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
        },
        "by_case_type": _bucket_stats(records, lambda r: r["case_type"]),
        "by_model": _bucket_stats(records, lambda r: r["model"] or "n/a (retrieval)"),
        "by_case_type_and_model": _bucket_stats(
            records, lambda r: f"{r['case_type']} / {r['model'] or 'n/a (retrieval)'}"
        ),
    }


def print_summary(summary: Dict[str, Any]) -> None:
    """Print a compact human-readable table -- the CI log's actual payoff."""
    o = summary["overall"]
    print(f"\n=== RESULT: {o['passed']}/{o['total']} passed ({o['pass_rate']:.1%}) ===\n")
    print("-- by case type --")
    for key, b in sorted(summary["by_case_type"].items()):
        print(f"  {key:<20} {b['passed']:>3}/{b['total']:<3} ({b['pass_rate']:.1%})")
    print("-- by model --")
    for key, b in sorted(summary["by_model"].items()):
        print(f"  {key:<20} {b['passed']:>3}/{b['total']:<3} ({b['pass_rate']:.1%})")


def _read_baseline() -> Optional[float]:
    """Read the current pass-rate floor, or None if no baseline exists yet."""
    if not BASELINE_FILE.exists():
        return None
    text = BASELINE_FILE.read_text(encoding="utf-8").strip()
    return float(text) if text else None


def _update_baseline(pass_rate: float) -> None:
    """Raise the baseline to ``pass_rate - BASELINE_MARGIN``, rounded down -- never lowers it."""
    candidate = math.floor(max(0.0, pass_rate - BASELINE_MARGIN) * 100) / 100
    current = _read_baseline()
    if current is not None and candidate <= current:
        print(f"[baseline] keeping {current:.2f} (candidate {candidate:.2f} is not higher)")
        return
    BASELINE_FILE.write_text(f"{candidate:.2f}\n", encoding="utf-8")
    verb = "seeded" if current is None else f"raised from {current:.2f}"
    print(f"[baseline] {verb} to {candidate:.2f} (observed {pass_rate:.4f} minus {BASELINE_MARGIN:.2f} margin)")


# ─────────────────────────────────────────────
# SECTION 6: CLI
# ─────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help=f"Ollama generator models to evaluate (default: {DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--update-baseline", action="store_true",
        help=(
            "After this run, raise tests/eval/baseline_min_pass_rate.txt to this "
            "run's pass rate minus a safety margin. Only ever raises it -- a lower "
            "candidate is ignored, never written."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_started = time.perf_counter()

    cases = load_gold_cases()
    required_models = set(args.models) | {AUX_MODEL, "embeddinggemma:latest"}

    try:
        preflight_ollama(required_models)

        # Heavy import (chromadb, sentence-transformers, rank-bm25...) only
        # after the cheap network/model preflight has already passed.
        import rag.chat_pdfs as rag

        rag.set_model_roles_runtime({
            "chat": AUX_MODEL, "contextual": AUX_MODEL, "recomp": AUX_MODEL, "ocr": AUX_MODEL,
        })

        blind_required = stage_blind_papers(cases)
        dev_required = set(_required_pdfs(cases, "corpus"))

        dev_collection = ensure_indexed(rag, DEV_DOCS_DIR, dev_required, "dev set")
        blind_collection = (
            ensure_indexed(rag, BLIND_DOCS_DIR, set(blind_required), "blind set")
            if blind_required else None
        )
        verify_all_papers_indexed(cases, dev_collection, blind_collection, rag)

        print(f"\n[run] {len(cases)} cases x up to {len(args.models)} model(s)\n")
        records = run_all_cases(rag, cases, dev_collection, blind_collection, args.models)
    except EvalSetupError as exc:
        print(f"\nSETUP FAILED: {exc}", file=sys.stderr)
        return 1

    summary = build_summary(records)
    elapsed_total = time.perf_counter() - run_started

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{timestamp}.json"
    out_path.write_text(json.dumps({
        "run": {
            "timestamp": timestamp, "models": args.models, "aux_model": AUX_MODEL,
            "num_cases": len(cases), "num_records": len(records),
            "elapsed_seconds": round(elapsed_total, 1),
        },
        "summary": summary,
        "results": records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print_summary(summary)
    print(f"\nfull results: {out_path}")
    print(f"total elapsed: {elapsed_total / 60:.1f} min")

    pass_rate = summary["overall"]["pass_rate"]
    threshold = _read_baseline()
    if threshold is None:
        print("\n[gate] no baseline yet (tests/eval/baseline_min_pass_rate.txt missing) -- not gating this run")
        gate_exit = 0
    elif pass_rate < threshold:
        print(f"\n[gate] FAILED: pass_rate {pass_rate:.4f} < baseline {threshold:.2f}")
        gate_exit = 1
    else:
        print(f"\n[gate] PASSED: pass_rate {pass_rate:.4f} >= baseline {threshold:.2f}")
        gate_exit = 0

    if args.update_baseline:
        _update_baseline(pass_rate)

    return gate_exit


if __name__ == "__main__":
    raise SystemExit(main())
