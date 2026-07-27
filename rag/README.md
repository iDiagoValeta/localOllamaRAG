# `rag/` — interfaces

> [!TIP]
> Looking to **install and run** MonkeyGrab? See the [root README](../README.md).

Today `rag/` is the CLI, the web UI, and the pipeline entry points they call.
[`rag/chat_pdfs.py`](chat_pdfs.py) is the public facade: configuration,
prompts, and re-exports from `rag/engine/*`, which every caller (CLI, web,
tests) is expected to keep importing — see `rag/chat_pdfs.py`'s own
module-map docstring for the full symbol list.

`rag/engine/*` increasingly delegates pure logic to
[`src/monkeygrab/application/`](../src/monkeygrab/README.md) (RRF fusion,
chunking, context assembly) while still owning the orchestration and all
infrastructure calls (Chroma, Ollama). That is the real architecture
reference now — signatures and parameters live in the code, not duplicated
here.

- **Hexagonal core, layers, how to add an adapter:** [`src/monkeygrab/README.md`](../src/monkeygrab/README.md)
- **Design rationale and phased rollout:** [`docs/design/2026-07-26-monkeygrab-v2.md`](../docs/design/2026-07-26-monkeygrab-v2.md)
- **Pipeline behavior as currently observed:** [`tests/characterization/`](../tests/characterization/)
