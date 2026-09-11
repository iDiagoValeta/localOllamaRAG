# `rag/` — interfaces

> [!TIP]
> Looking to **install and run** MonkeyGrab? See the [root README](../README.md).

`rag/` holds the Flask + React web app, the pywebview desktop
wrapper, and the entry points they call into the pipeline.

[`chat_pdfs.py`](chat_pdfs.py) is the public facade: configuration,
prompts, and re-exports of everything the web app, the tests and the
evaluation runner import. Treat its exported names as a contract; renaming one
breaks callers at once.

```
chat_pdfs.py    facade: runtime configuration, prompts, re-exports
web/            Flask backend, React frontend, desktop (pywebview) entry point
engine/         pipeline entry points
docs/           corpus PDFs, one folder per language store
```

## How `engine/` relates to the core

`engine/` is wiring, not logic. Indexing, retrieval and generation build the
required ports and run `IndexCorpus`, `Retrieve` and `Answer` from
[`src/monkeygrab/`](../src/monkeygrab/README.md).
[`engine/wiring.py`](engine/wiring.py) is the single bridge between this
package's mutable runtime configuration and the immutable `AppConfig` the core
expects. MinerU, Jina CLIP and FAISS are built by the fixed composition root.

The consequence worth knowing: the web app, desktop wrapper and evaluation
gate execute the same indexing, retrieval and generation implementations.

`wiring.py` also owns the jina-clip worker's lifetime, since it holds the one
instance the whole process shares. A worker that dies between calls (the OOM
killer picks it first, being the largest resident process after the model
server) is replaced on the next request rather than making every later query
fail until a restart. The adapter itself still never respawns its own worker,
so a genuine crash loop stays visible as repeated failures; `wiring` is the
caller its docstring points at. `release_embedder()` frees that worker after
an indexing run that failed, where it would otherwise sit on ~1.7 GiB that
the next attempt needs.

The Cross-Encoder reranker is cached the same way, and `release_reranker()`
drops its GPU weights on demand -- called from `/api/settings` when the
reranker flag is turned off, since nothing will use it again until it is
turned back on. It is deliberately not called on the per-query retrieval
path itself: `wiring.rag_chat_model()`'s docstring has the measurement that
decision rests on.

[`engine/settings.py`](engine/settings.py) owns the other half of that
agreement: the model roles, active store and pipeline flags the web control
panel saves are read at startup so the session reopens under the user's
choices. The environment outranks the file, the file outranks the defaults
in `chat_pdfs.py`.

- **Hexagonal core, layers, how to add an adapter:** [`src/monkeygrab/README.md`](../src/monkeygrab/README.md)
- **Design rationale and phased rollout:** [`docs/design/2026-07-26-monkeygrab-v2.md`](../docs/design/2026-07-26-monkeygrab-v2.md)
- **Pipeline behavior as currently observed:** [`tests/characterization/`](../tests/characterization/)
