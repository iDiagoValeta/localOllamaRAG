"""MonkeyGrab -- Desktop application entry point.

Wraps the Flask web app (rag.web.app) in a native OS window via pywebview and
runs the server in a background thread. This is the module frozen by PyInstaller
into ``MonkeyGrab.exe``; it is also runnable directly during development:

    python rag/web/desktop.py                 # native window
    MONKEYGRAB_NO_WINDOW=1 python rag/web/desktop.py   # headless server only

Behavior:
    * Writable data (vector DBs, user stores, history, debug dumps) is routed to
      a per-user directory through ``MONKEYGRAB_DATA_DIR`` so a read-only install
      never tries to write inside the application bundle.
    * A loading window appears immediately; the heavy engine import and the
      server start happen off the GUI thread, then the window navigates to the UI.
    * Any boot failure is written to ``<data_dir>/monkeygrab_boot.log`` (and shown
      in the window) so packaged builds are diagnosable without a console.
    * Closing the window shuts the server down and exits the process.
    * If Ollama is installed but not running, it is started automatically in the
      background at boot (no-op when already running or not installed).

Environment:
    * ``MONKEYGRAB_NO_WINDOW=1`` -- run the server only (no GUI); useful for
      testing the frozen bundle and for headless/server use.
    * ``MONKEYGRAB_PORT`` -- bind a fixed port instead of an ephemeral one.

Prerequisite (on every machine): Ollama installed and the desired models pulled.

Dependencies:
    - pywebview (native window), werkzeug/flask (server), rag.web.app (backend).
"""


import os
import sys
import socket
import threading
import time
import traceback


# IMPORTS AND PATH SETUP

# Make ``rag.web.app`` importable when run as a loose script from the repo root.
# When frozen by PyInstaller the package is already on the import path.
_here = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
if not getattr(sys, "frozen", False) and _project_root not in sys.path:
    sys.path.insert(0, _project_root)


APP_NAME = "MonkeyGrab"
HOST = "127.0.0.1"


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

def _default_data_dir() -> str:
    """Per-user writable data directory for vector DBs, stores, history and debug."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, APP_NAME)


def _data_dir() -> str:
    """Resolve the active data directory (env override or per-user default)."""
    return os.environ.get("MONKEYGRAB_DATA_DIR") or _default_data_dir()


def _ensure_data_dir() -> str:
    """Pin ``MONKEYGRAB_DATA_DIR`` (defaulting to a per-user path) and create it.

    Must run before importing the engine, which reads the variable at import time.
    A user-provided ``MONKEYGRAB_DATA_DIR`` is respected as-is; source runs without
    the variable keep repo-relative behavior.
    """
    data_dir = os.environ.get("MONKEYGRAB_DATA_DIR")
    if not data_dir:
        if getattr(sys, "frozen", False):
            data_dir = _default_data_dir()
            os.environ["MONKEYGRAB_DATA_DIR"] = data_dir
        else:
            return ""
    os.makedirs(data_dir, exist_ok=True)
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
