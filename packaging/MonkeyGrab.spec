# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the MonkeyGrab desktop app.

Builds a one-folder bundle (dist/MonkeyGrab/MonkeyGrab.exe) that wraps the Flask
backend + React UI + RAG engine in a native window (pywebview). Heavy ML deps
(torch, transformers, sentence-transformers, FAISS, onnxruntime) are collected
wholesale to maximize the chance of a working freeze; the bundle is large by
design. Ollama is NOT bundled -- it is an external prerequisite on each machine.

Build:
    pyinstaller packaging/MonkeyGrab.spec --noconfirm
"""

import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

PROJ = os.path.dirname(os.path.abspath(SPECPATH))  # noqa: F821 (SPECPATH injected by PyInstaller)

datas = []
binaries = []
hiddenimports = []

# --- Application resources: built React UI + built-in corpora ----------------
_dist = os.path.join(PROJ, "rag", "web", "frontend", "dist")
if os.path.isdir(_dist):
    datas.append((_dist, os.path.join("rag", "web", "frontend", "dist")))
for _corpus in ("es", "ca", "en"):
    _src = os.path.join(PROJ, "rag", "docs", _corpus)
    if os.path.isdir(_src):
        datas.append((_src, os.path.join("rag", "docs", _corpus)))

# Our own packages so dynamically referenced modules survive. `monkeygrab` lives
# under src/, which is not on the default search path, so the layout root has to
# be announced before collecting it -- rag.engine delegates its logic there, and
# without this the frozen app raises ModuleNotFoundError on first import while
# working perfectly from a source checkout.
_SRC = os.path.join(PROJ, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
hiddenimports += collect_submodules("rag")
hiddenimports += collect_submodules("monkeygrab")

# jina_clip_worker.py is never imported by the frozen app -- JinaClipEmbedder
# launches it as a SCRIPT with the external .venv-mineru interpreter (see
# src/monkeygrab/adapters/embedding/jina_clip_embedder.py). collect_submodules
# above only bundles it as compiled bytecode inside the PYZ archive, which an
# external interpreter cannot run as `python jina_clip_worker.py`. Ship it as
# a literal data file at the same path it already has relative to its package
# in the source tree (monkeygrab/adapters/embedding/), so JinaClipEmbedder's
# existing Path(__file__).with_name("jina_clip_worker.py") lookup finds it
# under sys._MEIPASS at runtime, exactly as it finds it beside the source
# file in dev -- verified with a standalone PyInstaller onedir build that
# __file__ for an archived module resolves to a real path mirroring the
# source tree under sys._MEIPASS. No frozen/dev branch needed in the adapter,
# same shape as composition._isolated_python() (#26).
_worker_script = os.path.join(_SRC, "monkeygrab", "adapters", "embedding", "jina_clip_worker.py")
if os.path.isfile(_worker_script):
    datas.append((_worker_script, os.path.join("monkeygrab", "adapters", "embedding")))

# --- Heavy third-party packages: collect data + binaries + submodules --------
_collect = [
    "faiss", "sentence_transformers", "transformers", "tokenizers",
    "torch", "onnxruntime", "huggingface_hub", "safetensors",
    "PIL", "rank_bm25", "ollama",
    "flask", "flask_cors", "werkzeug", "jinja2",
    "datasets", "accelerate",   # eagerly imported by sentence_transformers 5.x
    "webview",
]
for _pkg in _collect:
    try:
        d, b, h = collect_all(_pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        # A package that is not importable in this env is simply skipped.
        pass

# Package metadata (importlib.metadata.version checks fail at runtime otherwise;
# The HF/torch stack introspects its own version.
for _pkg in (
    "faiss-cpu", "transformers", "torch", "sentence_transformers", "tokenizers",
    "huggingface_hub", "safetensors", "tqdm", "numpy", "onnxruntime",
    "pydantic", "regex", "pyyaml", "filelock", "ollama", "flask", "rank_bm25",
    "Pillow",
):
    try:
        datas += copy_metadata(_pkg, recursive=True)
    except Exception:
        pass

# pywebview (Windows) talks to the OS WebView2 through pythonnet/winforms.
hiddenimports += [
    "clr",
    "webview.platforms.winforms",
    "webview.platforms.edgechromium",
]
for _pkg in ("pythonnet", "clr_loader"):
    try:
        d, b, h = collect_all(_pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# Trim large unused stacks to keep build time and size sane.
excludes = [
    "matplotlib", "tkinter", "IPython", "notebook", "jupyter",
    "pytest", "PyQt5", "PyQt6", "PySide2", "PySide6",
    "tensorflow", "jax",
]

block_cipher = None

a = Analysis(
    [os.path.join(PROJ, "rag", "web", "desktop.py")],
    pathex=[PROJ, _SRC],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

_icon = os.path.join(PROJ, "packaging", "MonkeyGrab.ico")
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MonkeyGrab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=_icon if os.path.isfile(_icon) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MonkeyGrab",
)
