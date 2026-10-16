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
  sources.py    document/page references for an answer (shared by web and headless)
headless/       Flask service for other applications: health.py (probe), service.py (index/status/answer over the engine entry points), app.py (routes)
docs/           corpus PDFs, one folder per language store
```

## How `engine/` relates to the core

`engine/` is wiring, not logic. Indexing, retrieval and generation build the
required ports and run `IndexCorpus`, `Retrieve` and `Answer` from
[`src/monkeygrab/`](../src/monkeygrab/README.md).
[`engine/wiring.py`](engine/wiring.py) is the single bridge between this
package's mutable runtime configuration and the immutable `AppConfig` the core
expects. MinerU, embeddinggemma-2 and FAISS are built by the composition root.

The consequence worth knowing: the web app, desktop wrapper and evaluation
gate execute the same indexing, retrieval and generation implementations.

The Cross-Encoder reranker is cached, and `release_reranker()`
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

That agreement is web-only. `headless/` never loads `settings.json` and
mutates no runtime global, so a headless process runs under environment plus
`chat_pdfs.py` defaults alone — and the `USAR_*` pipeline flags are bound
from the environment for that process, since the web-UI toggles are what
headless ignores. Operator notes (auth, store pinning, 8 GB levers, what
`/health`'s commit means to Daimon) live in the [root README](../README.md)'s
headless section, not here.

- **Hexagonal core, layers, how to add an adapter:** [`src/monkeygrab/README.md`](../src/monkeygrab/README.md)
- **Design rationale and phased rollout:** [`docs/design/2026-07-26-monkeygrab-v2.md`](../docs/design/2026-07-26-monkeygrab-v2.md)
- **Pipeline behavior as currently observed:** [`tests/characterization/`](../tests/characterization/)
