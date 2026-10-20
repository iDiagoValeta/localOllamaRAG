# MonkeyGrab — Desktop app packaging

Builds **`MonkeyGrab.exe`**, a native-window desktop wrapper around the Flask
backend + React UI + RAG engine. The window is provided by
[pywebview](https://pywebview.flowrl.com/) (Edge WebView2 on Windows); the whole
Python stack is frozen with [PyInstaller](https://pyinstaller.org/) so the target
machine does not need the product's Python environment.

> [!IMPORTANT]
> **Ollama is not bundled.** A local RAG runs the LLM on the machine, and the
> models are multi-GB and hardware-specific — they cannot live inside the `.exe`.
> Every machine that runs MonkeyGrab needs **Ollama installed** with the desired
> models pulled (`ollama pull …`). The app starts Ollama automatically at launch if
> it is installed but not running, and lets you assign models per role from the
> **Models** tab.
>
> **The embedder is not bundled either.** Text and images are embedded by
> `google/embeddinggemma-2` served by a local `llama-server` (see
> [Target prerequisites](#target-prerequisites)). The app never starts that
> process; at launch it probes `<MONKEYGRAB_EMBEDDER_URL>/health` and logs a
> warning when the server is down, so the missing prerequisite is visible before
> the first indexing run or question.

## Build

```bash
pip install pyinstaller pywebview          # build-time deps (once)
python packaging/build_exe.py              # builds UI + icon + freezes the app
```

Reuse an existing React build with `--skip-frontend`. Output lands in
`dist/MonkeyGrab/MonkeyGrab.exe` (a one-folder bundle — distribute the whole
`MonkeyGrab/` folder, or zip it / wrap it in an installer such as Inno Setup).

Before anything is built, the script runs `python tools/fetch_corpus.py --check`
and aborts if any PDF the manifest requires is missing: the bundle ships the
built-in corpora, so packaging a fresh clone without fetching them would produce
an app that cannot answer. Run `python tools/fetch_corpus.py` first (see
[`AGENTS.md`](../AGENTS.md) section 8).

Direct PyInstaller invocation (equivalent):

```bash
pyinstaller packaging/MonkeyGrab.spec --noconfirm --clean --distpath dist
```

## What ships where

| Content | Location |
|---|---|
| App code, ML libs, Python runtime, built-in corpora (`es`/`ca`/`en`), React UI | inside the bundle (`MonkeyGrab/_internal/`) |
| The three FAISS language stores, chat history and debug dumps | per-user data dir `MONKEYGRAB_DATA_DIR` (default `%LOCALAPPDATA%\MonkeyGrab`) |
| MinerU extraction cache | `<MONKEYGRAB_DATA_DIR>\mineru_cache` (default `%LOCALAPPDATA%\MonkeyGrab\mineru_cache`) |
| Isolated MinerU runtime | external `.venv-mineru/` beside `MonkeyGrab.exe` |

Routing writable data to `MONKEYGRAB_DATA_DIR` keeps the install read-only-safe;
in source/dev runs the variable is unset and everything stays repo-relative.
The desktop also pins `MINERU_CACHE_DIR` under that directory (never overriding
an explicit export), because MinerU's own fallback would otherwise write
`.mineru_cache` inside the read-only install folder. The three document stores
are seeded on first start from the bundled corpora into
`<MONKEYGRAB_DATA_DIR>\docs` (via `MONKEYGRAB_DOCS_ROOT`), and an existing
store is never overwritten or merged, so uploads and deletions survive upgrades.

## Target prerequisites

- **Ollama** running locally with the desired models.
- **The `llama-server` embedder**: the `llama-server` binary from
  [llama.cpp](https://github.com/ggml-org/llama.cpp) plus the GGUF conversion of
  [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2),
  running with embeddings enabled on the default port:

  ```bash
  llama-server -m embeddinggemma-2.gguf --embedding --port 8091
  ```

  `MONKEYGRAB_EMBEDDER_URL` defaults to `http://127.0.0.1:8091`; point it at
  another host/port if the server listens elsewhere. The server is a
  prerequisite the app checks but never starts, and `MONKEYGRAB_EMBEDDER_DIM`
  selects the Matryoshka width (512 by default) that the index was built with.
- **The isolated MinerU runtime** at
  `MonkeyGrab/.venv-mineru/Scripts/python.exe`. The current PyInstaller recipe
  does not create or bundle it; prepare it separately with the MinerU
  dependencies before distributing the folder.
- **Edge WebView2 runtime** — preinstalled on Windows 11; on older Windows install
  the [Evergreen runtime](https://developer.microsoft.com/microsoft-edge/webview2/).

## Notes

- The bundle is **several GB** because of `torch` (CUDA build) + `transformers`.
  A smaller, GPU-agnostic build is possible by freezing from an env
  with the CPU build of `torch`.
- First use of the cross-encoder reranker downloads its model from Hugging Face
  (one-time, needs internet) into the user cache.
- A desktop bundle without `.venv-mineru` starts, but indexing and retrieval
  fail visibly because the fixed multimodal stack has no fallback.
