# `src/monkeygrab/` — core

Hexagonal core of MonkeyGrab: the retrieval/generation logic as swappable
ports and adapters, independent of FAISS, Ollama, or any specific model.
Full rationale and phased rollout: [`docs/design/2026-07-26-monkeygrab-v2.md`](../../docs/design/2026-07-26-monkeygrab-v2.md).

## Layers

```
domain/        entities, zero infrastructure imports    (Chunk, Fragment, ExtractedPage, ...)
ports/         Protocols the application layer depends on (PdfExtractor, Embedder, VectorStore, ...)
application/   use cases: IndexCorpus, Retrieve, Answer, Study, + pure helpers (rrf_fusion,
               text_chunking, context_assembly, keywords)
config/        AppConfig — immutable, built once via from_env(), changed via with_overrides() (never mutated)
adapters/      port implementations (MinerU, embeddinggemma-2 over llama-server, FAISS, Ollama, BM25, CrossEncoder)
```

Dependency rule: `application` → `domain`/`ports`/`config`; `ports` → `domain`;
`domain`/`config` → nothing internal. `adapters` implement `ports` but are
never imported *by* `domain`/`ports`/`config`/`application` — only wired in
by whatever composes the app. Enforced by AST-parsing every import in
[`tests/unit/test_architecture_boundaries.py`](../../tests/unit/test_architecture_boundaries.py),
not by convention.

> [!IMPORTANT]
> **Hard-fail policy.** Every adapter raises on failure instead of degrading —
> no silent CUDA→CPU→disabled reranker and no hidden extraction fallback.
> A caller that wants a fallback chain builds it explicitly from two ports; an
> adapter never hides a second strategy inside itself. See each port's
> docstring under `ports/` for its specific failure contract.

## Current wiring (read before assuming more than this)

**All three stages run through the core.** `rag/engine/indexing.py` runs
`IndexCorpus` with MinerU, embeddinggemma-2 and FAISS.
`rag/engine/retrieval.py` runs `Retrieve` and
`rag/engine/generation.py` runs `Answer`. `tests/eval/run_eval.py` constructs
the same use cases, so the evaluation gate and the shipped product cannot
measure different behaviour.

`application/index_fingerprint.py`'s recipe digest was, until issue #36, read
only by `run_eval.py`. `rag/engine/indexing.py` now writes it after a full
folder index (never after an incremental add, which cannot vouch for the rest
of an existing store) and exposes `index_fingerprint_mismatch()`; the
web app calls it to warn when a stored index no longer matches the active
config, and leave reindexing to the user rather than triggering it — an hour
of MinerU + embedding must never start silently. A store with no recorded
fingerprint (every index built before this feature existed) reads as unknown,
not stale. The wiring adds the installed MinerU version to the recipe when it
can read it (issue #426); without MinerU both the write and the check leave it
out, so a query-only machine does not read every store as stale.

Each entry point under `rag/engine/` is wiring plus conversion between the
domain entities the core speaks and the dicts the interfaces consume. None of
them holds pipeline logic.

`Answer` exposes its three steps separately — `select_evidence`,
`build_user_message`, `stream` — as well as a `run` that composes them. The
split exists because a caller streaming to a web client sends the cited
sources before the first token arrives, which it cannot do if preparing the
prompt and generating from it are one call.

`Study` is the other caller of the generator, and it produces structure where
`Answer` produces prose: a summary, an outline, a quiz. All three take
fragments the caller already retrieved and differ only in schema, prompt, and
how strict the parse is. That last part is the design decision worth knowing
before touching it — an outline truncates, a summary drops unusable sections,
and a quiz refuses anything it cannot grade safely, because a wrong answer key
is the only one of the three that misleads without looking wrong.

`rag/engine/wiring.py` is the only bridge between the mutable runtime globals
in `rag/chat_pdfs.py` and the immutable `AppConfig` this package takes. It
also caches the two components too expensive to rebuild per query: the
Cross-Encoder weights and the tokenized BM25 corpus.

The fixed multimodal stack is built by `composition.py` from that `AppConfig`
(`build_extractor`, `build_image_extractor`, `build_vector_store`,
`build_embedder`, `build_stack`). `wiring.py` derives the config and builds the
rest: a chat model per role, and the reranker and BM25 index it caches between
queries.

## Adapters whose dependencies collide with the product's

A dependency that does not fit the product's pinned stack runs outside it
rather than downgrading the product environment. MinerU runs as an external
CLI from its own isolated interpreter (`.venv-mineru`), never a product Python
dependency. embeddinggemma-2 runs in an external `llama-server` the adapter
reaches over HTTP (`adapters/embedding/llama_server_embedder.py`). Both
adapters stay pure stdlib, so their unit tests never need either runtime.

## Adding a new adapter (e.g. to compare a retrieval technology)

1. Pick the port it implements (`ports/<name>.py`) and read its docstring —
   it states the exact method contract and the failure policy.
2. Write the adapter under `adapters/<category>/<name>.py`. Ports are
   `Protocol`s, not base classes: no inheritance needed, just match the
   method signatures.
3. Raise on failure; do not add a fallback inside the adapter.
4. Add a unit test under `tests/unit/adapters/` that doubles the
   infrastructure the adapter calls (see the existing adapter tests for the
   pattern) — `tests/unit/adapters/test_adapters_do_not_import_rag.py` is the
   one boundary check specific to this directory.
5. Register it in `composition.py` only when it replaces the current
   implementation; the product exposes one composition, not runtime selectors.
