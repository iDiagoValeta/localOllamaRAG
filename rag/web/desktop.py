"""MonkeyGrab -- Desktop application entry point.

Wraps the Flask web app (rag.web.app) in a native OS window via pywebview and
runs the server in a background thread. This is the module frozen by PyInstaller
into ``MonkeyGrab.exe``; it is also runnable directly during development:

    python rag/web/desktop.py                 # native window
    MONKEYGRAB_NO_WINDOW=1 python rag/web/desktop.py   # headless server only

Behavior:
    * Writable data (vector DBs, user stores, history, debug dumps) is routed to
      a per-user directory through ``MONKEYGRAB_DATA_DIR`` so a read-only install
      never tries to write inside the application bundle. A source run without
      the variable keeps the engine's repo-relative default (``rag/``), and the
      boot log and port file follow whichever root the engine resolves.
    * A loading window appears immediately; the heavy engine import and the
      server start happen off the GUI thread, then the window navigates to the UI.
    * Any boot failure is written to ``<data_dir>/monkeygrab_boot.log`` (and shown
      in the window) so packaged builds are diagnosable without a console.
    * Closing the window shuts the server down and exits the process.
    * If Ollama is installed but not running, it is started automatically in the
      background at boot (no-op when already running or not installed).
    * The ``llama-server`` that serves embeddinggemma-2 is probed at boot. When
      it is down, a clear warning is printed and appended to the boot log; the
      window still opens, because only indexing and retrieval need it.

Environment:
    * ``MONKEYGRAB_NO_WINDOW=1`` -- run the server only (no GUI); useful for
      testing the frozen bundle and for headless/server use.
    * ``MONKEYGRAB_PORT`` -- bind a fixed port instead of an ephemeral one.

Prerequisite (on every machine): Ollama installed and the desired models pulled,
plus the embeddinggemma-2 ``llama-server`` documented in
``packaging/README.md``. The embedder adapter never starts that server itself.

Dependencies:
    - pywebview (native window), werkzeug/flask (server), rag.web.app (backend).
"""


import os
import shutil
import sys
import socket
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request


# IMPORTS AND PATH SETUP

# Make ``rag.web.app`` importable when run as a loose script from the repo root.
# When frozen by PyInstaller the package is already on the import path.
_here = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
if not getattr(sys, "frozen", False) and _project_root not in sys.path:
    sys.path.insert(0, _project_root)


APP_NAME = "MonkeyGrab"
HOST = "127.0.0.1"

# A dead llama-server must not hold the window back: loopback answers fast, and
# a boot warning that arrives seconds late is not a boot warning.
_EMBEDDER_HEALTH_TIMEOUT_SECONDS = 3.0


class _NullStream:
    """Minimal stand-in for a missing std stream (windowed builds have none)."""

    encoding = "utf-8"

    def write(self, *_a):
        return 0

    def read(self, *_a):
        return ""

    def readline(self, *_a):
        return ""

    def flush(self):
        pass

    def isatty(self):
        return False

    def close(self):
        pass


def _ensure_std_streams() -> None:
    """Replace any ``None`` std stream with a no-op stream.

    A windowed PyInstaller build runs without a console, so ``sys.stdin`` /
    ``sys.stdout`` / ``sys.stderr`` are ``None``. Downstream code (the CLI display
    backend probe, logging) calls ``.isatty()`` / ``.write()`` on them; provide
    safe stand-ins before importing the engine so those calls never crash.
    """
    for _name in ("stdin", "stdout", "stderr"):
        if getattr(sys, _name, None) is None:
            setattr(sys, _name, _NullStream())

LOADING_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>MonkeyGrab</title>
<style>
  html,body{height:100%;margin:0;background:#0b0a10;color:#e4e4e7;
    font-family:system-ui,Segoe UI,sans-serif;display:flex;align-items:center;
    justify-content:center}
  .box{text-align:center}
  .ring{width:46px;height:46px;margin:0 auto 18px;border-radius:50%;
    border:3px solid rgba(240,164,114,.25);border-top-color:#F0A472;
    animation:spin 1s linear infinite}
  h1{font-size:20px;font-weight:800;letter-spacing:-.01em;margin:0 0 6px}
  p{font-size:13px;color:#a1a1aa;margin:0}
  @keyframes spin{to{transform:rotate(360deg)}}
</style></head>
<body><div class="box"><div class="ring"></div>
<h1>MonkeyGrab</h1><p>Iniciando el motor RAG local…</p></div></body></html>
"""


# DATA DIRECTORY

# Mirrors rag/engine/settings.STORE_IDS. Importing that module would pull the
# engine (and its ML stack) in at module scope, which this entry point and its
# dependency-free unit tests deliberately avoid.
_STORE_IDS = ("en", "es", "ca")


def _default_data_dir() -> str:
    """Per-user writable data directory for vector DBs, stores, history and debug."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, APP_NAME)


# rag/chat_pdfs.py derives its data root from its own package directory. A source
# run without MONKEYGRAB_DATA_DIR has to resolve to that same folder, not to a
# per-user path the engine never writes to (issue #400).
_RAG_DIR = os.path.dirname(os.path.dirname(_here))


def _data_dir() -> str:
    """Resolve the active data directory exactly as ``rag/chat_pdfs.py`` does.

    The boot log and the server port file must live where the engine keeps its
    vector DBs, history and debug dumps: a diagnostic filed under a different
    root is a diagnostic nobody finds. The env override wins in every mode, a
    frozen build falls back to the per-user directory, and a source run keeps
    the engine's repo-relative default (``rag/``).

    Returns:
        Absolute path to the writable data root.
    """
    explicit = os.environ.get("MONKEYGRAB_DATA_DIR")
    if explicit:
        return os.path.abspath(explicit)
    if getattr(sys, "frozen", False):
        return os.path.abspath(_default_data_dir())
    return _RAG_DIR


def _ensure_docs_root(data_dir: str) -> None:
    """Pin the docs root under ``data_dir`` and seed it from the bundle once.

    The bundled ``rag/docs`` sits inside the install folder (Program Files),
    where upload, reindex and document deletion all fail with PermissionError.
    ``setdefault`` keeps an explicit export winning, the same precedence
    ``MONKEYGRAB_DATA_DIR`` follows.

    A store is copied only when its destination folder is entirely absent.
    Re-seeding must never overwrite an edited PDF nor merge back one the user
    deleted on purpose, so an existing folder is left untouched.
    """
    docs_root = os.path.join(data_dir, "docs")
    os.environ.setdefault("MONKEYGRAB_DOCS_ROOT", docs_root)
    source_root = os.path.join(_RAG_DIR, "docs")
    os.makedirs(docs_root, exist_ok=True)
    for store in _STORE_IDS:
        source = os.path.join(source_root, store)
        destination = os.path.join(docs_root, store)
        if os.path.isdir(source) and not os.path.exists(destination):
            shutil.copytree(source, destination)
    manifest_source = os.path.join(source_root, "corpus_manifest.json")
    manifest_destination = os.path.join(docs_root, "corpus_manifest.json")
    if os.path.isfile(manifest_source) and not os.path.exists(manifest_destination):
        shutil.copy2(manifest_source, manifest_destination)


def _ensure_data_dir() -> str:
    """Pin ``MONKEYGRAB_DATA_DIR`` (defaulting to a per-user path) and create it.

    Also pins ``MINERU_CACHE_DIR`` under that directory unless the user already
    exported one: MinerU's own fallback is ``<repo_root>/.mineru_cache``, which
    a packaged install cannot write (issue #343). A frozen build additionally
    pins ``MONKEYGRAB_DOCS_ROOT`` and seeds it from the bundled corpora (#344).

    Must run before importing the engine, which reads the variable at import time.
    A user-provided ``MONKEYGRAB_DATA_DIR`` is respected as-is; source runs without
    the variable keep repo-relative behavior.
    """
    data_dir = _data_dir()
    if not os.environ.get("MONKEYGRAB_DATA_DIR"):
        if not getattr(sys, "frozen", False):
            # Nothing to pin: the engine derives this same repo-relative path.
            return ""
        os.environ["MONKEYGRAB_DATA_DIR"] = data_dir
    os.makedirs(data_dir, exist_ok=True)
    # setdefault, not assignment: an explicit export wins, the same precedence
    # MONKEYGRAB_DATA_DIR itself follows. The packager runs with the bundle in
    # Program Files, where MinerU's repo-relative fallback is read-only.
    os.environ.setdefault("MINERU_CACHE_DIR", os.path.join(data_dir, "mineru_cache"))
    if getattr(sys, "frozen", False):
        _ensure_docs_root(data_dir)
    return data_dir


# SERVER LIFECYCLE

def _resolve_port() -> int:
    """Return ``MONKEYGRAB_PORT`` if set/valid, otherwise a free ephemeral port."""
    raw = os.environ.get("MONKEYGRAB_PORT")
    if raw and raw.isdigit():
        return int(raw)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def _wait_until_ready(port: int, timeout: float = 30.0) -> bool:
    """Poll the loopback port until it accepts connections (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((HOST, port)) == 0:
                return True
        time.sleep(0.15)
    return False


def _start_server(port: int):
    """Import the backend and start a threaded werkzeug server. Returns the server."""
    from werkzeug.serving import make_server
    from rag.web.app import app

    server = make_server(HOST, port, app, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True, name="monkeygrab-server").start()
    return server


def _log_boot_error(exc: BaseException) -> str:
    """Write a boot failure (with traceback) to the data dir so frozen runs are diagnosable."""
    try:
        path = os.path.join(_data_dir(), "monkeygrab_boot.log")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("MonkeyGrab failed to start.\n\n")
            f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        return path
    except Exception:
        return ""


def _log_boot_warning(message: str) -> str:
    """Append a non-fatal startup warning to the boot log; return its path.

    A windowed build has no console, so stderr alone would lose the message.
    Appending keeps a warning visible even when it precedes a later fatal error.
    """
    try:
        path = os.path.join(_data_dir(), "monkeygrab_boot.log")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"WARNING: {message}\n")
        return path
    except Exception:
        return ""


# EMBEDDER HEALTH

def _embedder_warning_message(health_url: str, reason: str) -> str:
    """One line naming the failing endpoint, why, and how to start llama-server."""
    port = urllib.parse.urlsplit(health_url).port or 8091
    return (
        f"Embedder (llama-server) is not reachable at {health_url}: {reason}. "
        "Start it with `llama-server -m <embeddinggemma-2.gguf> --embedding "
        f"--port {port} -b 2048 -ub 2048`, or point MONKEYGRAB_EMBEDDER_URL at a running server. "
        "MonkeyGrab will open, but indexing and retrieval fail until the "
        "embedder answers."
    )


def _embedder_warning() -> str:
    """Probe the llama-server embedder and return a warning when it is down.

    The embedding adapter reaches embeddinggemma-2 over HTTP and never starts
    the server itself, so a clean machine fails at the first indexing or
    question with a message far from the missing prerequisite (issue #342).

    Returns:
        An empty string when ``GET <base_url>/health`` answers 200, otherwise a
        one-line explanation naming the endpoint and how to start it.

    Raises:
        Nothing for an unreachable or failing server: a probe failure is the
        signal, and a malformed configuration surfaces to the caller.
    """
    # Deferred so importing this module stays standard-library only, which is
    # what lets its unit tests run in the dependency-free fast gate.
    from monkeygrab.config import AppConfig

    # The same AppConfig the embedding adapter resolves, so the check can never
    # name a different server from the one the pipeline will call.
    base_url = AppConfig.from_env().models.embedding.base_url.rstrip("/")
    health_url = f"{base_url}/health"
    try:
        with urllib.request.urlopen(  # nosec B310 -- local endpoint from app config
            health_url, timeout=_EMBEDDER_HEALTH_TIMEOUT_SECONDS
        ) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        return _embedder_warning_message(health_url, f"HTTP {exc.code}")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return _embedder_warning_message(health_url, str(exc))
    if status != 200:
        return _embedder_warning_message(health_url, f"HTTP {status}")
    return ""


def _warn_if_embedder_unreachable() -> None:
    """Print and persist the embedder warning; no-op when the server answers."""
    warning = _embedder_warning()
    if not warning:
        return
    print(f"MonkeyGrab: {warning}", file=sys.stderr, flush=True)
    _log_boot_warning(warning)


# WINDOW BOOT

def _boot(window, port: int) -> None:
    """Import the backend, start the server, then navigate the window to the UI.

    Runs on a worker thread (started by ``webview.start``) so the loading window
    stays responsive while the engine and its heavy ML dependencies import.
    """
    try:
        globals()["_SERVER"] = _start_server(port)
        # Bring up Ollama in the background if it is installed but not running, so
        # the UI is usable without the user starting the server by hand.
        try:
            from rag.web.app import start_ollama_if_needed
            start_ollama_if_needed()
        except Exception:
            pass
        # The embedder is not autostarted (its process is the user's, per the
        # design doc), so say once when it is missing. Guarded like the Ollama
        # block above: a failed probe must never abort the window.
        try:
            _warn_if_embedder_unreachable()
        except Exception:
            pass
        _wait_until_ready(port)
        window.load_url(f"http://{HOST}:{port}/")
    except Exception as e:  # surface the failure in the window + a log file
        log = _log_boot_error(e)
        msg = str(e).replace("<", "&lt;").replace(">", "&gt;")
        window.load_html(
            "<body style='background:#0b0a10;color:#fca5a5;font-family:system-ui;"
            "padding:40px'><h2>No se pudo iniciar MonkeyGrab</h2>"
            f"<pre style='white-space:pre-wrap;color:#e4e4e7'>{msg}</pre>"
            f"<p style='color:#71717a'>Detalle: {log}</p></body>"
        )


# HEADLESS MODE

def _run_headless(port: int) -> None:
    """Run the server with no GUI (testing / server use). Blocks until interrupted."""
    try:
        server = _start_server(port)
    except Exception as e:
        log = _log_boot_error(e)
        print(f"MonkeyGrab failed to start: {e}\nDetail: {log}", file=sys.stderr)
        traceback.print_exc()
        raise
    # Record the bound port so external tooling can find a server on an ephemeral port.
    try:
        with open(os.path.join(_data_dir(), "monkeygrab_server.port"), "w", encoding="utf-8") as f:
            f.write(str(port))
    except Exception:
        pass
    print(f"MonkeyGrab server on http://{HOST}:{port}/  (Ctrl+C to stop)", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.shutdown()


# ENTRY POINT

def main() -> None:
    """Launch the desktop window (or a headless server) and run until closed."""
    # Every pipeline stage (reranker included) runs on the GPU by default; the
    # reranker falls back to CPU on its own when the GPU load fails. Set
    # RERANKER_DEVICE=cpu in the environment to force CPU.
    _ensure_std_streams()
    _ensure_data_dir()
    port = _resolve_port()

    if os.environ.get("MONKEYGRAB_NO_WINDOW"):
        _run_headless(port)
        return

    import webview

    window = webview.create_window(
        APP_NAME,
        html=LOADING_HTML,
        width=1280,
        height=820,
        min_size=(940, 640),
        background_color="#0b0a10",
    )
    webview.start(_boot, (window, port))

    server = globals().get("_SERVER")
    if server is not None:
        try:
            server.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
