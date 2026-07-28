# `src/monkeygrab/` — core

Hexagonal core of MonkeyGrab: the retrieval/generation logic as swappable
ports and adapters, independent of Chroma, Ollama, or any specific model.
Full rationale and phased rollout: [`docs/design/2026-07-26-monkeygrab-v2.md`](../../docs/design/2026-07-26-monkeygrab-v2.md).

## Layers

```
domain/        entities, zero infrastructure imports    (Chunk, Fragment, ExtractedPage, ...)
ports/         Protocols the application layer depends on (PdfExtractor, Embedder, VectorStore, ...)
application/   use cases: IndexCorpus, Retrieve, Answer, + pure helpers (rrf_fusion, text_chunking,
               context_assembly, keywords)
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

**All three stages run through the core.** `rag/engine/indexing.py` runs
`IndexCorpus` with backends from `composition.build_stack` (env:
`PDF_EXTRACTOR`, `VECTOR_STORE`, `EMBEDDER`; the default is still pymupdf +
Ollama + Chroma). `rag/engine/retrieval.py` runs `Retrieve` and
`rag/engine/generation.py` runs `Answer`. `tests/eval/run_eval.py` constructs
the same use cases, so the evaluation gate and the shipped product cannot
measure different behaviour.

Each entry point under `rag/engine/` is wiring plus conversion between the
domain entities the core speaks and the dicts the interfaces consume. None of
them holds pipeline logic.

`Answer` exposes its three steps separately — `select_evidence`,
`build_user_message`, `stream` — as well as a `run` that composes them. The
split exists because a caller streaming to a web client sends the cited
sources before the first token arrives, which it cannot do if preparing the
prompt and generating from it are one call.

`rag/engine/wiring.py` is the only bridge between the mutable runtime globals
in `rag/chat_pdfs.py` and the immutable `AppConfig` this package takes. It
also caches the two components too expensive to rebuild per query: the
Cross-Encoder weights and the tokenized BM25 corpus.

## Adapters whose dependencies collide with the product's

`adapters/embedding/jina_clip_embedder.py` is the first adapter whose
library doesn't fit the product's pinned stack at all: jina-clip-v2's remote
code only loads under an older transformers than the one the product runs.
Rather than downgrade the product environment for one model, the pattern is
to run the dependency in its own isolated interpreter (here, `.venv-mineru`)
as a persistent subprocess, speaking line-JSON over stdin/stdout
(`jina_clip_worker.py`, never imported by product code). MinerU follows the
same isolation idea as an external CLI (never a product Python dependency).
The jina adapter itself stays pure stdlib either way, so its unit tests never
need the isolated environment.

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
5. Register it in `composition.py` (`build_extractor` / `build_vector_store` /
   `build_embedder`) behind a `StackConfig` choice, and document the env var
   in `config/stack.py`.
