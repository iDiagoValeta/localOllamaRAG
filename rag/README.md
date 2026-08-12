# `rag/` — interfaces

> [!TIP]
> Looking to **install and run** MonkeyGrab? See the [root README](../README.md).

`rag/` holds the terminal CLI, the Flask + React web app, the pywebview desktop
wrapper, and the entry points they call into the pipeline.

[`chat_pdfs.py`](chat_pdfs.py) is the public facade: configuration,
prompts, and re-exports of everything the CLI, the web app, the tests and the
evaluation runner import. Treat its exported names as a contract; renaming one
breaks callers in three places at once.

```
chat_pdfs.py    facade: runtime configuration, prompts, re-exports
cli/            interactive terminal app and its i18n strings
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

`wiring.py` also owns freeing what it cached: `release_gpu_models()` closes
the jina-clip worker and drops the reranker's weights, and
`engine/generation.py` calls it right before every RAG generation call, so
retrieval's GPU tenants are gone before Ollama is asked to load the
generator. `/chat` mode never reaches this — it calls Ollama directly — so
nothing needs to guard the call by mode.

The consequence worth knowing: the CLI, web app, desktop wrapper and evaluation
gate execute the same indexing, retrieval and generation implementations.

- **Hexagonal core, layers, how to add an adapter:** [`src/monkeygrab/README.md`](../src/monkeygrab/README.md)
- **Design rationale and phased rollout:** [`docs/design/2026-07-26-monkeygrab-v2.md`](../docs/design/2026-07-26-monkeygrab-v2.md)
- **Pipeline behavior as currently observed:** [`tests/characterization/`](../tests/characterization/)
