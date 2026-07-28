"""run_eval -- self-sufficient runner for the gold eval gate (real pipeline, no mocks).

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

# CONSTANTS

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]

# Mirrors the sys.path bootstrap rag/chat_pdfs.py does for itself -- this
# script must be runnable as `python tests/eval/run_eval.py` from the repo
# root with no PYTHONPATH set beforehand, per the "one command, no manual
# steps" requirement.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
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


# PREFLIGHT


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


# CORPUS STAGING


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
    """Index missing PDFs into the env-selected stack; return a ``Retrieve`` for it.

    Both stacks (default pymupdf+ollama+chroma and mineru+jina+faiss) go through
    ``IndexCorpus`` + ``build_stack`` here, so later case execution can call the
    same ``Retrieve`` use case without touching ``rag/engine/retrieval.py``.

    Args:
        rag: The imported rag.chat_pdfs module (runtime model/flag source).
        carpeta: PDF directory to index.
        required_pdfs: Filenames that must end up indexed.
        label: Short name for log lines ("dev set", "blind set").

    Returns:
        ``(Retrieve, Stack)`` for this corpus. Caller should ``close()`` the
        stack embedder when finished if it exposes that method.

    Raises:
        EvalSetupError: A required PDF is still missing after indexing.
    """
    from monkeygrab.adapters.chat.ollama_chat import OllamaChatModel
    from monkeygrab.adapters.extraction.mineru_extractor import MineruImageExtractor
    from monkeygrab.adapters.extraction.pymupdf_image_extractor import PymupdfImageExtractor
    from monkeygrab.adapters.lexical.bm25_index import Bm25LexicalIndex
    from monkeygrab.adapters.reranking.cross_encoder_reranker import CrossEncoderReranker
    from monkeygrab.application.answer import Answer
    from monkeygrab.application.index_corpus import IndexCorpus
    from monkeygrab.application.retrieve import Retrieve
    from monkeygrab.composition import build_stack
    from monkeygrab.config.stack import EXTRACTOR_MINERU, VECTOR_STORE_CHROMA

    required = set(required_pdfs)
    rag.set_docs_folder_runtime(str(carpeta))
    try:
        config = _eval_app_config(rag, carpeta)
        stack = build_stack(config)
        store = stack.vector_store
        existing = _sources_in_store(store)
        missing = sorted(required - existing)
        if not missing:
            print(f"[index] {label}: cache hit, {len(required)} paper(s) already indexed")
        else:
            print(f"[index] {label}: indexing {len(missing)} missing paper(s): {missing}")
            t0 = time.perf_counter()
            image_extractor = None
            ocr_model = None
            if config.flags.usar_embeddings_imagen:
                if config.stack.extractor == EXTRACTOR_MINERU:
                    image_extractor = MineruImageExtractor()
                else:
                    image_extractor = PymupdfImageExtractor(config.chunking)
                if not config.stack.is_multimodal:
                    ollama = config.models.ollama
                    ocr_model = OllamaChatModel(
                        config.models.ocr,
                        num_ctx=ollama.ocr_num_ctx,
                        keep_alive=ollama.keep_alive,
                        options={"temperature": 0.1, "num_predict": 2000},
                    )
            contextual_model = None
            if config.flags.usar_contextual_retrieval:
                ollama = config.models.ollama
                contextual_model = OllamaChatModel(
                    config.models.contextual,
                    num_ctx=ollama.contextual_num_ctx,
                    keep_alive=ollama.keep_alive,
                    options={"temperature": 0.1, "num_predict": 250},
                )
            indexer = IndexCorpus(
                stack.extractor,
                stack.embedder,
                store,
                config,
                contextual_model=contextual_model,
                image_extractor=image_extractor,
                ocr_chat_model=ocr_model,
            )
            for filename in missing:
                indexer.run(str(carpeta / filename), filename)
            print(f"[index] {label}: done in {time.perf_counter() - t0:.0f}s")

        still_missing = sorted(required - _sources_in_store(store))
        if still_missing:
            raise EvalSetupError(
                f"{label}: {still_missing} still not indexed after an indexing attempt "
                "-- check the [index] log above for the underlying error"
            )

        lexical = Bm25LexicalIndex(store, config.retrieval)
        # Reranking runs on whatever device the adapter detects, which is CUDA
        # here. It no longer has to share the card with the jina worker: the
        # runner retrieves every case first and only then generates, so the two
        # models are never resident at the same time.
        reranker = CrossEncoderReranker(config.models) if config.flags.usar_reranker else None
        # Query decomposition is off for EVERY stack, and that is a decision
        # about the comparison rather than about VRAM. It calls Ollama in the
        # middle of retrieval, which would put a generator beside the embedder
        # again; disabling it only for the multimodal stack would have made the
        # A/B differ in two variables instead of one, which is how a comparison
        # stops meaning anything. Both stacks lose the same stage.
        retrieve = Retrieve(
            stack.embedder,
            store,
            config,
            lexical_index=lexical,
            reranker=reranker,
            query_decomposer=None,
        )
        # The product does not generate straight from what retrieval returned:
        # it expands the top fragments with their neighbours and trims to the
        # character budget first. Building the same use case here is what keeps
        # this runner from grading answers written off evidence no user's query
        # would have produced.
        from rag.engine import wiring

        evidence = Answer(store, wiring.rag_chat_model(config), config)

        print(
            f"[index] {label}: stack={config.stack.slug} "
            f"chunks={store.count()} chroma={config.stack.vector_store == VECTOR_STORE_CHROMA}",
            flush=True,
        )
        return retrieve, evidence, stack
    finally:
        rag.set_docs_folder_runtime(None)


def verify_all_papers_indexed(
    cases: Sequence[Dict[str, Any]],
    dev_sources: set[str],
    blind_sources: set[str],
) -> None:
    """Final pre-run check: every paper a case references has an index entry."""
    missing = []
    for case in cases:
        filename = f"{case['paper']}.pdf"
        indexed = dev_sources if case["source"] == "corpus" else blind_sources
        if filename not in indexed:
            missing.append(f"{case['paper']} ({case['source']})")
    if missing:
        raise EvalSetupError(f"papers referenced by gold cases but not indexed: {sorted(set(missing))}")


# CASE EXECUTION


def _kind_from_format(fmt: Optional[str]) -> str:
    """Map an indexed chunk's ``format`` metadata to the grader's content kind."""
    if fmt == "image":
        return "image"
    if fmt == "table":
        return "table"
    return "text"


def _fragment_to_dict(fragment) -> Dict[str, Any]:
    """Convert a domain ``Fragment`` into the dict shape generation consumes.

    Delegates to the same conversion the CLI and the web app go through, so the
    fragments this runner feeds to generation are shaped exactly like the ones
    a real query produces. A second copy here would be a place for the gate to
    drift from the product without anything failing.
    """
    from rag.engine.wiring import fragment_to_dict

    return fragment_to_dict(fragment)


def _eval_app_config(rag, carpeta: Path):
    """AppConfig from env stack selectors, synced to live eval runtime roles."""
    from monkeygrab.config.app_config import AppConfig

    config = AppConfig.from_env().with_overrides(
        **{
            "paths.docs_folder": str(carpeta),
            "models.rag": rag.MODELO_RAG,
            "models.chat": rag.MODELO_CHAT,
            "models.embedding": rag.MODELO_EMBEDDING,
            "models.contextual": rag.MODELO_CONTEXTUAL,
            "models.recomp": rag.MODELO_RECOMP,
            "models.ocr": rag.MODELO_OCR,
            "flags.usar_contextual_retrieval": rag.USAR_CONTEXTUAL_RETRIEVAL,
            "flags.usar_embeddings_imagen": rag.USAR_EMBEDDINGS_IMAGEN,
            "flags.usar_llm_query_decomposition": rag.USAR_LLM_QUERY_DECOMPOSITION,
            "flags.usar_busqueda_hibrida": rag.USAR_BUSQUEDA_HIBRIDA,
            "flags.usar_reranker": rag.USAR_RERANKER,
            "flags.expandir_contexto": rag.EXPANDIR_CONTEXTO,
            "flags.usar_optimizacion_contexto": rag.USAR_OPTIMIZACION_CONTEXTO,
            "flags.usar_recomp_synthesis": rag.USAR_RECOMP_SYNTHESIS,
        }
    )
    # Contextual retrieval stays off for the multimodal stack, and this is the
    # one asymmetry the A/B cannot remove cheaply: it is a property of the index,
    # not of the query, so equalising it would mean rebuilding an index — with an
    # LLM call per chunk on one side, or discarding the existing production index
    # on the other. It is reported as a known limitation instead of hidden, and
    # it handicaps the multimodal stack, so a win there is a conservative result.
    #
    # The reranking pool is NOT shrunk any more. It was, to survive a CPU
    # CrossEncoder, and that silently made the two stacks differ in which
    # candidates ever reached the top-k — a second variable in a comparison meant
    # to have one.
    if config.stack.is_multimodal:
        config = config.with_overrides(**{"flags.usar_contextual_retrieval": False})
    return config


def _sources_in_store(store) -> set[str]:
    return {f.metadata.source for f in store.get_page(None, 0)}


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
        try:
            answer = rag.generar_respuesta_silenciosa(case["question"], list(fragments))
        except Exception as exc:
            # A dead or overloaded Ollama must not be reported as a quality
            # regression: the run is inconclusive, which is a different verdict
            # from "the system got worse". Marked as an infrastructure error so
            # the gate can refuse to compare against the baseline at all.
            gen_elapsed = time.perf_counter() - t0
            records.append({
                "id": case["id"], "paper": case["paper"], "case_type": case["case_type"],
                "lang": case["lang"], "model": model, "passed": False,
                "infrastructure_error": True,
                "reason": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(retrieval_elapsed + gen_elapsed, 2),
            })
            print(f"  [ERROR] {case['id']} / {model} -- {type(exc).__name__}: {exc}", flush=True)
            continue
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


def _release_gpu_models(*retrievers) -> None:
    """Free every retrieval model holding GPU memory, before generation starts.

    Both matter, and missing either one is fatal in the same way. The embedder
    owns a worker process; the reranker holds its weights resident. On an 8 GB
    card either is enough to stop Ollama from loading a generator, and Ollama
    does not fail fast when memory is short — it blocks until the read times
    out, so the symptom is a 15-minute hang rather than an error.

    Adapters without either hook (the Ollama embedder) need nothing.
    """
    for retrieve in retrievers:
        for attribute, method in (("_embedder", "close"), ("_reranker", "release")):
            target = getattr(retrieve, attribute, None)
            hook = getattr(target, method, None)
            if callable(hook):
                try:
                    hook()
                except Exception as exc:  # noqa: BLE001 - shutdown must not mask results
                    print(f"[phase] {attribute}.{method}() failed, continuing: {exc}", flush=True)


def run_all_cases(
    rag,
    cases: Sequence[Dict[str, Any]],
    retrieve_dev,
    retrieve_blind,
    evidence_dev,
    evidence_blind,
    models: Sequence[str],
) -> List[Dict[str, Any]]:
    """Run every gold case via ``Retrieve``, retrieving everything before
    generating anything.

    The pipeline uses its models in sequence -- embed, rerank, generate -- so
    they never actually need to be resident together. Interleaving per case
    forced exactly that, and on an 8 GB card the previous shape either collided
    into HTTP 500s or paid a 28 s embedder cold start for every factual case.
    Splitting the run into phases lets each model own the GPU and starts the
    embedder once, which is the difference between a three-hour run and a
    twenty-minute one.

    The cost is memory: every case's retrieved fragments are held until the
    generation phase. At 51 cases and a handful of fragments each, that is
    kilobytes.
    """
    records: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []

    print("[phase 1/2] retrieval for every case (embedder + reranker on GPU)", flush=True)
    for case in cases:
        is_dev = case["source"] == "corpus"
        retrieve = retrieve_dev if is_dev else retrieve_blind
        evidence = evidence_dev if is_dev else evidence_blind
        t0 = time.perf_counter()
        try:
            result = retrieve.run(case["question"])
            retrieved = [_fragment_to_dict(f) for f in result.fragments]
            selected, _metrics = evidence.select_evidence(result.fragments)
            fragments = [_fragment_to_dict(f) for f in selected]
        except Exception as exc:
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

        # Retrieval-only cases are already decided; grading them now keeps the
        # generation phase to the cases that actually need a model.
        #
        # They are graded on what retrieval RETURNED, not on the expanded
        # evidence: neighbour expansion pulls in chunks by position, so a
        # figure sitting next to a retrieved paragraph would otherwise count as
        # "retrieved" and quietly inflate the figure and table scores -- the
        # two the multimodal stack exists to move.
        if case["case_type"] in ("figure_retrieval", "table_retrieval"):
            record = run_retrieval_case(case, retrieved, elapsed)
            records.append(record)
            status = "PASS" if record["passed"] else "FAIL"
            print(f"  [{status}] {case['id']} ({elapsed:.1f}s) -- {record['reason']}", flush=True)
        else:
            # Factual cases are only graded in phase 2, but they still need a
            # heartbeat here: without one, a run whose first cases are all
            # factual prints nothing for minutes and a hung Ollama is
            # indistinguishable from normal progress.
            pending.append({"case": case, "fragments": fragments, "elapsed": elapsed})
            print(
                f"  [retrieved] {case['id']} ({elapsed:.1f}s, {len(fragments)} fragments)",
                flush=True,
            )

    _release_gpu_models(retrieve_dev, retrieve_blind)

    print(f"\n[phase 2/2] generation for {len(pending)} case(s) (Ollama alone on GPU)", flush=True)
    for item in pending:
        records.extend(
            run_factual_case(rag, item["case"], item["fragments"], models, item["elapsed"])
        )
    return records


# REPORTING


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


# CLI


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
    from monkeygrab.config.stack import EMBEDDER_OLLAMA, stack_from_env

    stack_cfg = stack_from_env()
    required_models = set(args.models) | {AUX_MODEL}
    if stack_cfg.embedder == EMBEDDER_OLLAMA:
        required_models.add("embeddinggemma:latest")

    stacks_to_close = []
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

        retrieve_dev, evidence_dev, stack_dev = ensure_indexed(
            rag, DEV_DOCS_DIR, dev_required, "dev set"
        )
        stacks_to_close.append(stack_dev)
        retrieve_blind = None
        evidence_blind = None
        stack_blind = None
        if blind_required:
            retrieve_blind, evidence_blind, stack_blind = ensure_indexed(
                rag, BLIND_DOCS_DIR, set(blind_required), "blind set"
            )
            stacks_to_close.append(stack_blind)

        verify_all_papers_indexed(
            cases,
            _sources_in_store(stack_dev.vector_store),
            _sources_in_store(stack_blind.vector_store) if stack_blind else set(),
        )

        print(
            f"\n[run] stack={stack_cfg.slug} {len(cases)} cases x "
            f"up to {len(args.models)} model(s)\n",
            flush=True,
        )
        records = run_all_cases(
            rag, cases, retrieve_dev, retrieve_blind, evidence_dev, evidence_blind, args.models
        )
    except EvalSetupError as exc:
        print(f"\nSETUP FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        for stack in stacks_to_close:
            closer = getattr(stack.embedder, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass

    summary = build_summary(records)
    elapsed_total = time.perf_counter() - run_started

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{timestamp}_{stack_cfg.slug}.json"
    out_path.write_text(json.dumps({
        "run": {
            "timestamp": timestamp,
            "stack": stack_cfg.slug,
            "models": args.models,
            "aux_model": AUX_MODEL,
            "num_cases": len(cases),
            "num_records": len(records),
            "elapsed_seconds": round(elapsed_total, 1),
            "retrieval_path": "monkeygrab.application.retrieve.Retrieve",
        },
        "summary": summary,
        "results": records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print_summary(summary)
    print(f"\nfull results: {out_path}")
    print(f"total elapsed: {elapsed_total / 60:.1f} min")

    pass_rate = summary["overall"]["pass_rate"]
    threshold = _read_baseline()

    broken = [r for r in records if r.get("infrastructure_error")]
    if broken:
        print(f"\n[gate] INCONCLUSIVE: {len(broken)} case(s) could not be evaluated")
        for record in broken[:5]:
            print(f"        {record['id']}: {record['reason']}")
        if len(broken) > 5:
            print(f"        ... and {len(broken) - 5} more")
        print("        Not compared against the baseline: this run measured nothing.")
        return 1

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
