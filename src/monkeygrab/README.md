# `src/monkeygrab/` — core

Hexagonal core of MonkeyGrab: the retrieval/generation logic as swappable
ports and adapters, independent of Chroma, Ollama, or any specific model.
Full rationale and phased rollout: [`docs/design/2026-07-26-monkeygrab-v2.md`](../../docs/design/2026-07-26-monkeygrab-v2.md).

## Layers

```
domain/        entities, zero infrastructure imports    (Chunk, Fragment, ExtractedPage, ...)
ports/         Protocols the application layer depends on (PdfExtractor, Embedder, VectorStore, ...)
application/   use cases: IndexCorpus, Retrieve, Answer, + pure helpers (rrf_fusion, text_chunking, context_assembly)
config/        AppConfig — immutable, built once via from_env(), changed via with_overrides() (never mutated)
adapters/      port implementations (pymupdf, Chroma, Ollama, BM25, CrossEncoder today)
```

Dependency rule: `application` → `domain`/`ports`/`config`; `ports` → `domain`;
`domain`/`config` → nothing internal. `adapters` implement `ports` but are
never imported *by* `domain`/`ports`/`config`/`application` — only wired in
by whatever composes the app. Enforced by AST-parsing every import in
[`tests/unit/test_architecture_boundaries.py`](../../tests/unit/test_architecture_boundaries.py),
not by convention.

> [!IMPORTANT]
> **Hard-fail policy.** Every adapter raises on failure instead of degrading —
> no silent CUDA→CPU→disabled reranker, no silent pymupdf4llm→pypdf fallback.
> A caller that wants a fallback chain builds it explicitly from two ports; an
> adapter never hides a second strategy inside itself. See each port's
> docstring under `ports/` for its specific failure contract.

## Current wiring (read before assuming more than this)

`rag/engine/*` still owns the pipeline entry points the CLI and web UI call.
It imports specific pure functions out of `application/` (`rrf_fusion`,
`text_chunking`, `context_assembly`, plus two private helpers from
`retrieve.py`/`answer.py`) and calls them in place of the code that used to
live inline — each substitution is checked byte-for-byte against the
original in `tests/unit/application/*_equivalence.py`.

The full use-case classes (`IndexCorpus`, `Retrieve`, `Answer`) exist and are
independently unit-tested, but nothing in the CLI or web layer constructs or
calls them yet — that wiring is future work. Don't document it as done.

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
5. There is no runtime adapter-selection mechanism yet (no `PDF_EXTRACTOR`-style
   switch in `AppConfig`) — construct the adapter and pass it into the
   relevant use case directly until that wiring exists.
