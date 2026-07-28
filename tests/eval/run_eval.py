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

Dependencies: the full multimodal stack (FAISS, MinerU, jina-clip-v2,
sentence-transformers and Ollama) plus a running local Ollama
server and, in practice, a GPU -- the reranker and every model role run on
CPU otherwise, which is not a supported configuration for this gate.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

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

import grade  # noqa: E402  (tests/eval sibling module)

GOLD_FILE = EVAL_DIR / "gold_cases.jsonl"
BASELINE_FILE = EVAL_DIR / "baseline_min_pass_rate.txt"
RESULTS_DIR = EVAL_DIR / "runs"

# Papers not already shipped in rag/docs/en/ are staged here under their gold
# case "paper" slug (fetch_papers.py caches them by arXiv id instead, which
# does not match the corpus filename convention indexar_documentos expects).
BLIND_DOCS_DIR = EVAL_DIR / "blind_docs"
DEV_DOCS_DIR = REPO_ROOT / "rag" / "docs" / "en"

# Fed into derive_db_paths in place of DEV_DOCS_DIR when building the dev-set
# config (see _eval_app_config's path_label) -- never read from disk, only its
# basename matters. Deriving the FAISS collection from DEV_DOCS_DIR directly
# would give this run the same "docs_en" collection the product's own store
# uses, and that store has a second writer: the web UI's indexing path writes
# chunks into it under its own flags and never records this gate's fingerprint.
# Reusing it here would silently measure a mixture of two pipelines, and the
# first eval run after a user adds a document there would call store.clear()
# on a store that still has product data in it. The blind set never needed
# this because BLIND_DOCS_DIR's basename already differs from "en".
EVAL_DEV_LABEL = EVAL_DIR / "dev_docs"

OLLAMA_BASE_URL = "http://localhost:11434"

# The baseline was calibrated with this model. Other models can be measured
# explicitly, but must not be mixed into the same ratchet.
DEFAULT_MODELS = ["gemma4:e2b"]

# Model used for the auxiliary Ollama roles during this run: query
# decomposition, contextual-retrieval enrichment and RECOMP synthesis. Kept
# separate from --models so indexing cost does not multiply by the number of
# generator models under test.
AUX_MODEL = "gemma4:e2b"

# Safety margin subtracted from an observed pass rate before it is written as
# the new baseline floor (see _update_baseline) -- keeps normal run-to-run
# variance (model sampling, Ollama warmup) from flipping the gate red.
BASELINE_MARGIN = 0.05

# Case types decided entirely by retrieval: they skip generation in
# run_all_cases (a pass says a fragment of the right kind was surfaced, not
# that any answer used it) and are reported in their own build_summary bucket,
# because blending them into "answer" would read as answer quality that isn't
# there. One definition drives both, so a third retrieval-only type can't be
# added to the dispatch without also changing how it's reported.
_RETRIEVAL_ONLY_CASE_TYPES = ("figure_retrieval", "table_retrieval")


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
    # Deferred so importing this module -- which test_index_reuse.py and
    # test_summary_split.py do, to reach pure helpers like should_rebuild and
    # build_summary -- never requires requests. The fast CI gate installs
    # only pytest, precisely to prove those helpers need nothing else.
    import requests

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
    indexing derives the stored ``source`` metadata from the
    filename, and gold_cases.jsonl's ``paper`` field is the slug, not the id.

    Args:
        cases: Full parsed gold_cases.jsonl.

    Returns:
        ``{"<paper>.pdf": arxiv_id}`` for every blind-set paper, staged.
    """
    # Deferred for the same reason as the `requests` import in
    # _installed_ollama_models: fetch_papers.py itself imports requests at
    # module level, so importing it eagerly here would defeat the
    # dependency-free fast CI gate just as much as importing requests
    # directly would.
    import fetch_papers

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


def should_rebuild(stored: Optional[str], expected: str) -> bool:
    """Whether a store must be discarded and rebuilt before it can be measured.

    Args:
        stored: The fingerprint the store recorded, or ``None`` if it recorded
            none.
        expected: The fingerprint of the configuration this run will use.

    Returns:
        True unless the store provably matches. An unknown recipe rebuilds:
        reusing chunks whose provenance cannot be established would report one
        number for a mixture of two pipelines.
    """
    return stored != expected


def ensure_indexed(
    rag,
    carpeta: Path,
    required_pdfs: Iterable[str],
    label: str,
    path_label: Optional[Path] = None,
):
    """Index missing PDFs into the multimodal stack; return its use cases.

    Args:
        rag: The imported rag.chat_pdfs module (runtime model/flag source).
        carpeta: PDF directory to index.
        required_pdfs: Filenames that must end up indexed.
        label: Short name for log lines ("dev set", "blind set").
        path_label: Forwarded to _eval_app_config to isolate the FAISS
            collection from carpeta's own basename (see EVAL_DEV_LABEL).

    Returns:
        ``(Retrieve, Stack)`` for this corpus. Caller should ``close()`` the
        stack embedder when finished if it exposes that method.

    Raises:
        EvalSetupError: A required PDF is still missing after indexing.
    """
    from monkeygrab.adapters.chat.ollama_chat import OllamaChatModel
    from monkeygrab.adapters.extraction.mineru_extractor import MineruImageExtractor
    from monkeygrab.adapters.lexical.bm25_index import Bm25LexicalIndex
    from monkeygrab.adapters.reranking.cross_encoder_reranker import CrossEncoderReranker
    from monkeygrab.application.answer import Answer
    from monkeygrab.application.index_corpus import IndexCorpus
    from monkeygrab.application.index_fingerprint import compute_index_fingerprint
    from monkeygrab.application.retrieve import Retrieve
    from monkeygrab.composition import build_stack

    required = set(required_pdfs)
    rag.set_docs_folder_runtime(str(carpeta))
    try:
        config = _eval_app_config(rag, carpeta, path_label=path_label)
        stack = build_stack(config)
        store = stack.vector_store
        expected_fingerprint = compute_index_fingerprint(config)
        stored_fingerprint = store.read_fingerprint()
        existing = set()
        if should_rebuild(stored_fingerprint, expected_fingerprint):
            if store.count():
                print(
                    f"[index] {label}: recipe changed "
                    f"({stored_fingerprint or 'unrecorded'} -> {expected_fingerprint}), "
                    "discarding and rebuilding",
                    flush=True,
                )
                store.clear()
        else:
            existing = _sources_in_store(store)
        missing = sorted(required - existing)
        if not missing:
            print(f"[index] {label}: cache hit, {len(required)} paper(s) already indexed")
        else:
            print(f"[index] {label}: indexing {len(missing)} missing paper(s): {missing}")
            t0 = time.perf_counter()
            image_extractor = None
            if config.flags.usar_embeddings_imagen:
                image_extractor = MineruImageExtractor()
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

        # Written only after every required paper is confirmed present, so an
        # aborted indexing run never leaves a fingerprint claiming a complete
        # index. A half-built store keeps its old (or absent) value and gets
        # rebuilt on the next run.
        store.write_fingerprint(expected_fingerprint)

        lexical = Bm25LexicalIndex(store, config.retrieval)
        # Reranking runs on whatever device the adapter detects, which is CUDA
        # here. It no longer has to share the card with the jina worker: the
        # runner retrieves every case first and only then generates, so the two
        # models are never resident at the same time.
        reranker = CrossEncoderReranker() if config.flags.usar_reranker else None
        # Query decomposition stays off so Ollama never competes with the
        # multimodal embedder for the same 8 GB of VRAM.
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
            f"[index] {label}: stack=mineru-jina_clip-faiss "
            f"chunks={store.count()}",
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


def _eval_app_config(rag, carpeta: Path, path_label: Optional[Path] = None):
    """Build the fixed multimodal evaluation configuration.

    Args:
        rag: The imported rag.chat_pdfs module (runtime model/flag source).
        carpeta: PDF directory this run indexes from.
        path_label: When given, derive the FAISS store location/collection
            name from this path's basename instead of carpeta's, without
            changing which folder is actually indexed. See EVAL_DEV_LABEL for
            why the dev set needs this and the blind set does not.
    """
    from monkeygrab.config.app_config import AppConfig
    from monkeygrab.config.paths import derive_db_paths

    config = AppConfig.from_env().with_overrides(
        **{
            "paths.docs_folder": str(carpeta),
            "models.rag": rag.MODELO_RAG,
            "models.chat": rag.MODELO_CHAT,
            "models.contextual": rag.MODELO_CONTEXTUAL,
            "models.recomp": rag.MODELO_RECOMP,
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
    config = config.with_overrides(**{"flags.usar_contextual_retrieval": False})
    if path_label is not None:
        path_db, collection_name = derive_db_paths(str(path_label), config.paths.data_dir)
        config = config.with_overrides(
            **{"paths.path_db": path_db, "paths.collection_name": collection_name}
        )
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


# Measured in issue #27 (2026-07-29): with the product default KEEP_ALIVE=0,
# Ollama discards the model's weights after every call, so each factual case
# in phase 2 paid a ~200s cold load -- a fixed per-call cost, not variance.
# That reload buys nothing here specifically: run_all_cases already calls
# _release_gpu_models() before phase 2 starts, so Ollama owns the GPU alone
# from that point and there is nothing else waiting on the VRAM it would
# free. 120s comfortably survives the sub-second gap between one case's
# generation calls and the next (grading plus a dict lookup, no GPU work),
# without pinning the model in memory once the harness itself exits.
_EVAL_GENERATION_KEEP_ALIVE_SECONDS = "120"


@contextlib.contextmanager
def _generation_keep_alive():
    """Scope OLLAMA_KEEP_ALIVE to this process, for phase 2's generation calls only.

    generar_respuesta_silenciosa (rag/engine/generation.py) builds a fresh
    AppConfig on every call via wiring.app_config_from_runtime(), which reads
    OLLAMA_KEEP_ALIVE straight from the environment -- the config this module
    builds for retrieval (_eval_app_config) is a separate object that never
    reaches that call, so the environment is the only lever available here.

    Deliberately scoped to the phase 2 block (set after _release_gpu_models,
    restored once generation finishes) rather than the whole run: phase 1
    never touches Ollama, so it cannot benefit, and leaving it set past
    main() would let a non-zero keep-alive leak into anything else importing
    rag.chat_pdfs afterwards. Does not touch the product default -- see
    issue #27: keep_alive=0 exists because the embedder and reranker
    contend for the same 8 GB (issue #25), and raising it globally would
    make that contention worse.
    """
    previous = os.environ.get("OLLAMA_KEEP_ALIVE")
    os.environ["OLLAMA_KEEP_ALIVE"] = _EVAL_GENERATION_KEEP_ALIVE_SECONDS
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OLLAMA_KEEP_ALIVE", None)
        else:
            os.environ["OLLAMA_KEEP_ALIVE"] = previous


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
        if case["case_type"] in _RETRIEVAL_ONLY_CASE_TYPES:
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
    with _generation_keep_alive():
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


def _rate(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pass/total/rate over ``records``; an empty bucket rates 0.0, not an error."""
    total = len(records)
    passed = sum(1 for r in records if r["passed"])
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
    }


def build_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the overall / by-case-type / by-model / cross summary blocks."""
    retrieval_only = [r for r in records if r["case_type"] in _RETRIEVAL_ONLY_CASE_TYPES]
    answered = [r for r in records if r["case_type"] not in _RETRIEVAL_ONLY_CASE_TYPES]
    return {
        "overall": _rate(records),
        "retrieval_only": _rate(retrieval_only),
        "answer": _rate(answered),
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
    r, a = summary["retrieval_only"], summary["answer"]
    print(f"  retrieval only  {r['passed']:>3}/{r['total']:<3} ({r['pass_rate']:.1%})")
    print(f"  answered        {a['passed']:>3}/{a['total']:<3} ({a['pass_rate']:.1%})\n")
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
    required_models = set(args.models) | {AUX_MODEL}
    stack_slug = "mineru-jina_clip-faiss"

    stacks_to_close = []
    try:
        preflight_ollama(required_models)

        # Heavy import (FAISS, sentence-transformers, rank-bm25...) only
        # after the cheap network/model preflight has already passed.
        import rag.chat_pdfs as rag

        rag.set_model_roles_runtime({
            "chat": AUX_MODEL, "contextual": AUX_MODEL, "recomp": AUX_MODEL,
        })

        blind_required = stage_blind_papers(cases)
        dev_required = set(_required_pdfs(cases, "corpus"))

        retrieve_dev, evidence_dev, stack_dev = ensure_indexed(
            rag, DEV_DOCS_DIR, dev_required, "dev set", path_label=EVAL_DEV_LABEL
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
            f"\n[run] stack={stack_slug} {len(cases)} cases x "
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
    out_path = RESULTS_DIR / f"{timestamp}_{stack_slug}.json"
    out_path.write_text(json.dumps({
        "run": {
            "timestamp": timestamp,
            "stack": stack_slug,
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
