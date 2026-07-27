# `rag/` — interfaces

> [!TIP]
> Looking to **install and run** MonkeyGrab? See the [root README](../README.md).

`rag/` holds the two interfaces — the terminal CLI and the Flask + React web
app — and the entry points they call into the pipeline.

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

`engine/` is wiring, not logic. Indexing and retrieval build port adapters and
run the corresponding use case in
[`src/monkeygrab/`](../src/monkeygrab/README.md); generation still owns its own
implementation. [`engine/wiring.py`](engine/wiring.py) is the single bridge
between this package's mutable runtime configuration and the immutable
`AppConfig` the core expects.

The consequence worth knowing: retrieval behaves identically in the CLI, the
web app and the evaluation gate, because all three run the same use case.
Generation is shared by construction, since every caller goes through this
facade.

- **Hexagonal core, layers, how to add an adapter:** [`src/monkeygrab/README.md`](../src/monkeygrab/README.md)
- **Design rationale and phased rollout:** [`docs/design/2026-07-26-monkeygrab-v2.md`](../docs/design/2026-07-26-monkeygrab-v2.md)
- **Pipeline behavior as currently observed:** [`tests/characterization/`](../tests/characterization/)
