# MonkeyGrab — Desktop app packaging

Builds **`MonkeyGrab.exe`**, a native-window desktop wrapper around the Flask
backend + React UI + RAG engine. The window is provided by
[pywebview](https://pywebview.flowrl.com/) (Edge WebView2 on Windows); the whole
Python stack is frozen with [PyInstaller](https://pyinstaller.org/) so the target
machine needs **no Python**.

> [!IMPORTANT]
> **Ollama is not bundled.** A local RAG runs the LLM on the machine, and the
> models are multi-GB and hardware-specific — they cannot live inside the `.exe`.
> Every machine that runs MonkeyGrab needs **Ollama installed** with the desired
> models pulled (`ollama pull …`). The app detects Ollama, offers to start it, and
> lets you assign models per role from the **Models** tab.

## Build

```bash
pip install pyinstaller pywebview          # build-time deps (once)
python packaging/build_exe.py              # builds UI + icon + freezes the app
```

Reuse an existing React build with `--skip-frontend`. Output lands in
`dist/MonkeyGrab/MonkeyGrab.exe` (a one-folder bundle — distribute the whole
`MonkeyGrab/` folder, or zip it / wrap it in an installer such as Inno Setup).

Direct PyInstaller invocation (equivalent):

```bash
pyinstaller packaging/MonkeyGrab.spec --noconfirm --clean --distpath dist
```

## What ships where

| Content | Location |
|---|---|
| App code, ML libs, Python runtime, built-in corpora (`es`/`ca`/`en`), React UI | inside the bundle (`MonkeyGrab/_internal/`) |
| Vector DBs, user-created stores, chat history, debug dumps | per-user data dir `MONKEYGRAB_DATA_DIR` (default `%LOCALAPPDATA%\MonkeyGrab`) |

Routing writable data to `MONKEYGRAB_DATA_DIR` keeps the install read-only-safe;
in source/dev runs the variable is unset and everything stays repo-relative.

## Target prerequisites

- **Ollama** running locally with the desired models.
- **Edge WebView2 runtime** — preinstalled on Windows 11; on older Windows install
  the [Evergreen runtime](https://developer.microsoft.com/microsoft-edge/webview2/).

## Notes

- The bundle is **several GB** because of `torch` (CUDA build) + `transformers` +
  `chromadb`. A smaller, GPU-agnostic build is possible by freezing from an env
  with the CPU build of `torch`.
- First use of the cross-encoder reranker downloads its model from Hugging Face
  (one-time, needs internet) into the user cache.
