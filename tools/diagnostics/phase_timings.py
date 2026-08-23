#!/usr/bin/env python3
"""Per-phase wall-clock probe for the query pipeline (issue #12).

Issue #12 ("does Rust buy anything measurable here?") cannot be answered from
the evaluation gate's per-case totals: the gate records one elapsed number
per case, so the time *inside* native/library calls (Ollama, torch, FAISS,
the isolated jina-clip worker) is not separated from the time *between* them
(the Python orchestration a rewrite could actually move). This probe makes
that split by wrapping every port in a timing recorder -- the hexagonal
boundary means no product code changes are needed to measure it.

For each sampled gold case it reports wall time per phase:

- retrieval: query decomposition (Ollama), embedding (isolated worker),
  vector search (FAISS), BM25 (rank_bm25), reranking (torch cross-encoder)
- generation: RECOMP synthesis (Ollama) and the answer stream (Ollama),
  with Ollama's own reported in-server duration for comparison

Everything left over is Python glue: orchestration, dict/JSON plumbing and
HTTP overhead -- the only part a language rewrite could shrink.

Usage:
    python tools/diagnostics/phase_timings.py                # 6 factual + 2 retrieval-only cases
    python tools/diagnostics/phase_timings.py --factual 3 --retrieval-only 1

Dependencies:
    - The full product stack (rag/requirements.txt), a running Ollama with
      the configured role models, and a CUDA GPU.
    - The dev-set index already built under rag/vector_db/ (run the product
      or the eval gate once first). The probe never indexes anything.

WARNING: one GPU tenant only. Do not run this alongside the evaluation gate
or the harness loop -- it contends for the card and pollutes their latency
measurements, and its own numbers would be equally worthless.
"""

import argparse
import functools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

GOLD_CASES = ROOT / "tests" / "eval" / "gold_cases.jsonl"

# Port methods the probe times, per phase bucket. Anything a wrapped port
# exposes that is not listed here passes through untouched.
_TIMED_METHODS = {
    "decompose": ("generate",),              # query decomposer (ChatModel)
    "embed": ("embed",),                     # jina-clip worker (Embedder)
    "vector_search": ("query", "get_by_ids"),  # FAISS store (VectorStore)
    "bm25": ("search",),                     # rank_bm25 (LexicalIndex)
    "rerank": ("rerank",),                   # cross-encoder (Reranker)
    "recomp": ("generate",),                 # RECOMP synthesis (ChatModel)
}

_BUCKETS = ("decompose", "embed", "vector_search", "bm25", "rerank", "recomp", "generate")


class _Timer:
    """Cumulative seconds and call counts per phase bucket."""

    def __init__(self):
        self.seconds = {bucket: 0.0 for bucket in _BUCKETS}
        self.calls = {bucket: 0 for bucket in _BUCKETS}

    def record(self, bucket: str, elapsed: float) -> None:
        self.seconds[bucket] += elapsed
        self.calls[bucket] += 1

    def snapshot(self):
        return dict(self.seconds), dict(self.calls)


def _timed(inner, timer: _Timer, bucket: str):
    """Wrap a port so every call to the bucket's methods is timed."""
    if inner is None:
        return None
    methods = _TIMED_METHODS[bucket]

    class _Wrapper:
        def __getattr__(self, name):
            attr = getattr(inner, name)
            if name not in methods or not callable(attr):
                return attr

            @functools.wraps(attr)
            def call(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return attr(*args, **kwargs)
                finally:
                    timer.record(bucket, time.perf_counter() - started)

            return call

    return _Wrapper()


def _timed_stream(inner, timer: _Timer):
    """Wrap the generator ChatModel; stream() must time the full iteration."""

    class _Wrapper:
        def stream(self, prompt, *, system=None):
            started = time.perf_counter()
            try:
                yield from inner.stream(prompt, system=system)
            finally:
                timer.record("generate", time.perf_counter() - started)

        def __getattr__(self, name):
            return getattr(inner, name)

    return _Wrapper()


def _load_cases(n_factual: int, n_retrieval_only: int):
    """Sample gold cases from the dev corpus: factual cases from both ends of
    the question-length range (long ones exercise LLM query decomposition,
    short ones the keyword-variant path), plus retrieval-only cases."""
    factual, retrieval_only = [], []
    with open(GOLD_CASES, encoding="utf-8") as fh:
        for line in fh:
            case = json.loads(line)
            if case.get("source") != "corpus":
                continue
            if case["case_type"] in ("factual_number", "factual_concept"):
                factual.append(case)
            else:
                retrieval_only.append(case)
    factual.sort(key=lambda c: len(c["question"]))
    n_short = max(1, n_factual // 2) if n_factual > 1 else n_factual
    n_long = n_factual - n_short
    picked = factual[:n_short] + (factual[len(factual) - n_long:] if n_long else [])
    return picked + retrieval_only[:n_retrieval_only]


def _fmt(seconds: float) -> str:
    return f"{seconds:7.2f}s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--factual", type=int, default=6, help="answered cases to sample")
    parser.add_argument("--retrieval-only", type=int, default=2, help="retrieval-only cases to sample")
    parser.add_argument(
        "--eval-dev-set",
        action="store_true",
        help="measure the evaluation gate's dev-set store (tests/eval/run_eval.py's "
        "EVAL_DEV_LABEL collection) instead of the product's live store",
    )
    args = parser.parse_args()

    import rag.chat_pdfs as rag
    from monkeygrab.application.answer import Answer
    from monkeygrab.application.retrieve import Retrieve
    from rag.engine import wiring

    config = wiring.app_config_from_runtime()
    if args.eval_dev_set:
        from monkeygrab.config.paths import derive_db_paths

        path_db, collection_name = derive_db_paths(
            str(ROOT / "tests" / "eval" / "dev_docs"), config.paths.data_dir
        )
        config = config.with_overrides(
            **{"paths.path_db": path_db, "paths.collection_name": collection_name}
        )
    store = wiring.vector_store(config)
    if store.count() == 0:
        print(
            f"The active store ({config.paths.path_db}) is empty. Run the CLI or the "
            "eval gate once to build the index; this probe measures queries, not indexing."
        )
        return 1

    timer = _Timer()
    timed_store = _timed(store, timer, "vector_search")
    retrieve = Retrieve(
        _timed(wiring.embedder(config), timer, "embed"),
        timed_store,
        config,
        lexical_index=_timed(wiring.lexical_index(store, config), timer, "bm25"),
        reranker=_timed(wiring.reranker(config), timer, "rerank")
        if config.flags.usar_reranker
        else None,
        query_decomposer=_timed(wiring.query_decomposer(config), timer, "decompose")
        if config.flags.usar_llm_query_decomposition
        else None,
    )
    answer = Answer(
        timed_store,
        _timed_stream(wiring.rag_chat_model(config), timer),
        config,
        recomp_chat_model=_timed(wiring.recomp_chat_model(config), timer, "recomp")
        if config.flags.usar_recomp_synthesis
        else None,
        system_prompt=rag.SYSTEM_PROMPT_RAG,
    )

    cases = _load_cases(args.factual, args.retrieval_only)
    print(
        f"store: {store.count()} chunks | {len(cases)} cases | "
        "config from env > settings.json > defaults"
    )
    print(f"{'case':38} {'retriev':>8} {'gen':>8} {'ollama':>8} {'total':>8}")

    per_case = []
    for case in cases:
        before_seconds, before_calls = timer.snapshot()
        started = time.perf_counter()
        result = retrieve.run(case["question"])
        t_retrieval = time.perf_counter() - started
        t_gen, ollama_s = 0.0, 0.0
        if case["case_type"] in ("factual_number", "factual_concept"):
            started = time.perf_counter()
            answer_result = answer.run(case["question"], result.fragments)
            t_gen = time.perf_counter() - started
            gen_stats = answer_result.metrics.get("generation") or {}
            ollama_s = (gen_stats.get("total_duration") or 0) / 1e9
        after_seconds, after_calls = timer.snapshot()
        per_case.append((case, before_seconds, after_seconds, before_calls, after_calls,
                         t_retrieval, t_gen, ollama_s))
        print(
            f"{case['id'][:38]:38} {_fmt(t_retrieval)} {_fmt(t_gen)} {_fmt(ollama_s)}"
            f" {_fmt(t_retrieval + t_gen)}"
        )

    phase_totals = {bucket: 0.0 for bucket in _BUCKETS}
    phase_calls = {bucket: 0 for bucket in _BUCKETS}
    for _, before_s, after_s, before_c, after_c, *_ in per_case:
        for bucket in _BUCKETS:
            phase_totals[bucket] += after_s[bucket] - before_s[bucket]
            phase_calls[bucket] += after_c[bucket] - before_c[bucket]

    wall = sum(t_r + t_g for *_, t_r, t_g, _ in per_case)
    accounted = sum(phase_totals.values())
    glue = max(0.0, wall - accounted)
    ollama_reported = sum(o for *_, o in per_case)

    print("\nPer-phase totals across the sample (wall clock):")
    for bucket in _BUCKETS:
        print(f"  {bucket:14} {_fmt(phase_totals[bucket])}  ({phase_calls[bucket]} calls)")
    print(f"\n  total wall     {_fmt(wall)}")
    print(
        f"  Ollama self-reported in-server generation: {_fmt(ollama_reported)} "
        f"of the {_fmt(phase_totals['generate'])} generate bucket"
    )
    if wall:
        print(
            f"  Python glue (wall not inside any port call): {_fmt(glue)}"
            f"  ({100.0 * glue / wall:.1f}% of wall)"
        )
    print("\nIssue #12 reads this as: the glue share is what a rewrite could move;")
    print("everything else is already native (Ollama, torch, FAISS, rank_bm25, the jina worker).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
