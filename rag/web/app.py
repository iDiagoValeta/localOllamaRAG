"""MonkeyGrab -- Local web interface.

Flask server for the full RAG pipeline. Provides the same functionality
as the CLI: CHAT mode (free conversation) and RAG mode (document query).

Serves the React interface (build in rag/web/frontend/dist/) and the API at /api/*.

Usage (from project root):
    python rag/web/app.py

Open http://127.0.0.1:5000 in your browser.

Dependencies:
    - flask, flask-cors, werkzeug
    - rag.chat_pdfs (project internal module)
"""

import gc
import os
import sys
import json
import time
import shutil
import getpass
import subprocess
import threading
from urllib.parse import urlsplit
from typing import Generator, Optional

import requests
from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory
from flask_cors import CORS


# IMPORTS AND FLASK SETUP

# sys.path must include project root so `rag.chat_pdfs` resolves from rag/web/
_here = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rag.chat_pdfs as rag_engine
from rag.engine import wiring
from monkeygrab.config.env import read_env_docs_root
from monkeygrab.config.models import EMBEDDING_DISPLAY_NAME

_web_dir = os.path.dirname(os.path.abspath(__file__))
_react_dist = os.path.join(_web_dir, "frontend", "dist")

app = Flask(
    __name__,
    static_folder=os.path.join(_react_dist, "assets")
    if os.path.isdir(os.path.join(_react_dist, "assets"))
    else None,
)
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# CORS for local development (Vite on :3000 -> Flask on :5000).
_cors_origins = os.getenv(
    "MONKEYGRAB_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)
CORS(
    app,
    resources={r"/api/*": {"origins": [o.strip() for o in _cors_origins.split(",") if o.strip()]}},
)

# Host and Origin guard (issue #346). The API has no authentication, so what
# keeps a web page from driving it is the browser: DNS rebinding makes a
# hostile page's requests arrive with its own hostname in Host, and a
# cross-site form POST (multipart, no CORS preflight) always carries a foreign
# Origin. Loopback names are allowed by default; a server bound to another
# address names it in MONKEYGRAB_ALLOWED_HOSTS (comma separated hostnames,
# no port) or MONKEYGRAB_HOST.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _allowed_hostnames() -> frozenset:
    extra = os.getenv("MONKEYGRAB_ALLOWED_HOSTS", "")
    names = {h.strip().lower() for h in extra.split(",") if h.strip()}
    bound = os.getenv("MONKEYGRAB_HOST", "").strip().lower()
    if bound and bound != "0.0.0.0":  # nosec B104 - compared, not bound: a wildcard is no hostname
        names.add(bound)
    return _LOOPBACK_HOSTS | names


def _hostname_of(netloc: str) -> str:
    """Return the lowercase hostname of a ``host[:port]`` string, brackets kept for IPv6."""
    netloc = netloc.strip().lower()
    if netloc.startswith("["):
        return netloc.split("]", 1)[0] + "]"
    return netloc.rsplit(":", 1)[0] if netloc.count(":") == 1 else netloc


@app.before_request
def _reject_foreign_host_or_origin():
    allowed = _allowed_hostnames()
    if _hostname_of(request.host) not in allowed:
        return jsonify({"ok": False, "error": "host_not_allowed"}), 403
    origin = request.headers.get("Origin")
    if origin and request.method not in _SAFE_METHODS:
        # "null" (sandboxed iframes, file://) has no host and is rejected too.
        if _hostname_of(urlsplit(origin).netloc) not in allowed:
            return jsonify({"ok": False, "error": "origin_not_allowed"}), 403
    return None


_state = {
    "mode": "chat",
    "historial_chat": [],
    # Bumped by /api/clear. A streaming turn only persists itself when this
    # value still matches the one it started with (issue #394).
    "history_generation": 0,
    "collection": None,
    "indexing": False,
    "indexing_error": None,
    "indexing_failed": False,  # True if the last attempt failed permanently
    "indexing_done_empty": False,  # True if indexing completed with 0 chunks (no PDFs)
    "indexing_progress": None,  # {"file": str, "file_index": int, "total_files": int}
    "indexing_failed_files": [],  # names from the last run that did not land (issue #323)
}
_indexing_lock = threading.RLock()

_CORPUS_PRESET_IDS = frozenset(rag_engine.STORE_IDS)

# Vector-store layout: three fixed stores, one per language, each bound to a
# versioned docs folder under rag/docs/. There are no user-created stores and no
# hide/restore concept — the three always exist (possibly empty). English is the
# default. Their vector DBs live under DATA_DIR/vector_db/ (user-writable).
# The packaged app pins this to a writable per-user copy before importing the
# backend; a source run keeps the repo-relative bundle-versioned folder (#344).
_DOCS_ROOT = read_env_docs_root(os.path.join(_project_root, "rag", "docs"))
_BUILTIN_STORES = {
    "en": "English",
    "es": "Castellano",
    "ca": "Valencià",
}

# Ollama runtime control. Taken from the engine rather than re-read from the
# environment so the endpoint the control panel probes is the one the pipeline
# generates against; resolving it twice is how the two used to disagree.
OLLAMA_BASE_URL = rag_engine.OLLAMA_BASE_URL
_model_caps_cache: dict = {}  # digest -> capabilities list


def _infer_corpus_preset() -> str | None:
    """Return ``es``/``ca``/``en`` when ``CARPETA_DOCS`` basename matches a preset tree."""
    base = os.path.basename(os.path.abspath(rag_engine.CARPETA_DOCS))
    return base if base in _CORPUS_PRESET_IDS else None


def _init_history_payload(historial: list) -> list[dict]:
    """Copy the persisted chat turns the UI can paint.

    Entries in ``historial_chat.json`` are not guaranteed to be chat turns, and
    the frontend only knows ``user``/``assistant``; anything else is dropped
    rather than sent to a bubble that cannot render it.

    Args:
        historial: Raw entries loaded from the persisted history.

    Returns:
        A list of fresh ``{"role", "content"}`` dicts.
    """
    return [
        {"role": entry["role"], "content": entry["content"]}
        for entry in historial
        if isinstance(entry, dict)
        and entry.get("role") in ("user", "assistant")
        and isinstance(entry.get("content"), str)
    ]


def _init_paths_payload() -> dict:
    """Paths and local identity exposed to the web UI (corpus selection + greeting)."""
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    return {
        "docs_folder": os.path.abspath(rag_engine.CARPETA_DOCS),
        "corpus_preset": _infer_corpus_preset(),
        "active_store": os.path.basename(os.path.abspath(rag_engine.CARPETA_DOCS)),
        "user": user,
    }


def _safe_pdf_basename(filename: str) -> str:
    """Validate a PDF filename while preserving Unicode characters.

    Shared by the routes that receive a name from the client (DELETE/GET by
    name) and by the upload routes. ``secure_filename`` cannot be used for the
    latter: it transliterates or drops every non-ASCII character, so names like
    ``报告.pdf`` collapse into an unusable stem.
    """
    normalized = (filename or "").replace("\\", "/")
    basename = os.path.basename(normalized)
    if basename != normalized or basename in ("", ".", ".."):
        return ""
    if not basename.lower().endswith(".pdf"):
        return ""
    return basename


# COLLECTION MANAGEMENT


def _invalidate_collection_if_deleted():
    """Clear state to start fresh if the DB folder was deleted."""
    store_dir = os.path.join(rag_engine.PATH_DB, rag_engine.COLLECTION_NAME)
    if _state["collection"] is not None and not os.path.exists(store_dir):
        _state["collection"] = None
        rag_engine.reset_vector_store_cache()
        _state["indexing_failed"] = False
        _state["indexing_error"] = None
        _state["indexing_done_empty"] = False
        _state["indexing_failed_files"] = []
        gc.collect()


def _get_collection():
    """Obtain or create the FAISS store for the active corpus."""
    with _indexing_lock:
        _invalidate_collection_if_deleted()
        if _state["collection"] is None:
            _state["collection"] = rag_engine.obtener_vector_store()
        return _state["collection"]


def _run_indexing_bg():
    """Run document indexing in the background in silent mode (no Rich display)."""
    try:
        coll = _get_collection()

        def _on_progress(info):
            _state["indexing_progress"] = info

        failed_files = []

        def _on_file_failed(filename, _error):
            failed_files.append(filename)

        total_chunks = rag_engine.indexar_documentos(
            rag_engine.CARPETA_DOCS,
            coll,
            silent=True,
            progress_callback=_on_progress,
            on_file_failed=_on_file_failed,
        )
        _state["indexing_error"] = None
        _state["indexing_failed"] = False
        # A partial run is not a failed run: the store holds the files that did
        # land and retrieval keeps working. But the user has to see which ones
        # did not, or a lost document looks like a clean index (issue #323).
        _state["indexing_failed_files"] = failed_files
        if total_chunks == 0:
            _state["indexing_done_empty"] = True
        else:
            _state["indexing_done_empty"] = False
    except Exception as e:
        _state["indexing_error"] = str(e)
        _state["indexing_failed"] = True
        _state["indexing_failed_files"] = []
        # The worker that just failed a whole re-index is holding ~1.7 GiB and
        # nothing is going to use it again. Left alive it makes the *next*
        # attempt fail on memory this failure is holding, blaming a PID with
        # no visible connection to anything the user did (issue #191). Safe
        # here specifically: the store this run would have filled is empty, so
        # no retrieval can be mid-flight against it.
        rag_engine.release_embedder()
    finally:
        _state["indexing"] = False
        _state["indexing_progress"] = None


def _ensure_indexed():
    """Start background indexing if the collection is empty. Non-blocking.

    Does not restart if the previous attempt failed (use api_reindex to force).
    Does not restart if indexing already completed with no PDFs (indexing_done_empty).
    """
    if _state["indexing_failed"] or _state["indexing_done_empty"]:
        return
    coll = _get_collection()
    if coll.count() == 0 and not _state["indexing"]:
        with _indexing_lock:
            if (
                coll.count() == 0
                and not _state["indexing"]
                and not _state["indexing_failed"]
                and not _state["indexing_done_empty"]
            ):
                _state["indexing"] = True
                _state["indexing_error"] = None
                _state["indexing_progress"] = None
                threading.Thread(target=_run_indexing_bg, daemon=True).start()


def _try_begin_indexing() -> bool:
    """Claim the single index-mutation slot while holding ``_indexing_lock``.

    Checking ``_state["indexing"]`` and marking it are one critical section:
    two requests that checked separately could both start a re-index and leave
    a half-written store (issue #394).

    Returns:
        True when the caller now owns the slot and must release it with
        :func:`_release_indexing` or hand it to a background run with
        :func:`_hand_over_indexing`; False when a run is already in progress.
    """
    with _indexing_lock:
        if _state["indexing"]:
            return False
        _state["indexing"] = True
        _state["indexing_error"] = None
        _state["indexing_progress"] = None
        return True


def _release_indexing() -> None:
    """Release a slot taken by :func:`_try_begin_indexing`."""
    with _indexing_lock:
        _state["indexing"] = False
        _state["indexing_progress"] = None


def _hand_over_indexing() -> None:
    """Release the slot and let the background run claim it.

    The release and ``_ensure_indexed``'s own claim happen under the same
    lock, so no other mutating request can slip in between (issue #394).
    """
    with _indexing_lock:
        _state["indexing"] = False
        _ensure_indexed()


def _reset_db():
    """Clear the active FAISS store."""
    coll = _get_collection()
    coll.clear()
    _state["indexing_done_empty"] = False


# STREAMING HELPERS


def _chat_stream(pregunta: str) -> Generator[str, None, None]:
    """Generate tokens from chat mode via streaming.

    Goes through the ``chat`` role's ``ChatModel``, so
    ``MONKEYGRAB_CHAT_BACKEND=openai`` applies here exactly as it does to the
    other roles instead of always talking to Ollama (issue #371). Earlier
    turns travel as ``history`` messages rather than being flattened into the
    prompt, which is what keeps the conversation multi-turn. The adapter is
    built with the configured request timeout and generation deadline, so a
    wedged server raises instead of blocking the Flask worker forever --
    api_chat's generate() already turns any exception from this generator
    into an SSE `error` event.

    Args:
        pregunta: User question text.

    Yields:
        Token strings as they are produced by the model.
    """
    config = wiring.app_config_from_runtime()
    ollama = config.models.ollama
    model = wiring.chat_model_for_role(
        config,
        "chat",
        # Capped like the RAG role (#254): an unbounded answer holds the
        # model's only slot for as long as it keeps writing (issue #371).
        options={"temperature": 0.7, "top_p": 0.9, "num_predict": 4096},
        num_ctx=8192,
        keep_alive=ollama.keep_alive,
        generation_deadline=ollama.generation_deadline,
    )
    mensajes_recientes = _state["historial_chat"][-(rag_engine.MAX_HISTORIAL_MENSAJES) :]

    for chunk in model.stream(
        pregunta,
        system=rag_engine.SYSTEM_PROMPT_CHAT,
        history=mensajes_recientes,
    ):
        if chunk.text:
            yield chunk.text


from rag.engine.sources import format_sources

# Kept as an alias: tests/test_web_routes.py imports this name from here.
_format_sources = format_sources


def _sse_event(event: str, payload: dict) -> str:
    """Serialize one Server-Sent Event with a named event type."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _collection_document_details(coll) -> list:
    """Return per-document metadata without changing the legacy documents list."""
    try:
        fragments = coll.get_page(None, 0)
    except Exception:
        return []

    details = {}
    for fragment in fragments:
        meta = fragment.metadata
        doc = meta.source
        if not doc:
            continue
        info = details.setdefault(
            doc,
            {"name": doc, "fragments": 0, "pages": set(), "formats": set()},
        )
        info["fragments"] += 1
        info["pages"].add(meta.page)
        if meta.format:
            info["formats"].add(meta.format)

    normalized = []
    for doc in sorted(details):
        info = details[doc]
        normalized.append(
            {
                "name": info["name"],
                "fragments": info["fragments"],
                "pages": len(info["pages"]),
                "formats": sorted(info["formats"]),
            }
        )
    return normalized


def _ollama_version() -> Optional[str]:
    """Return the running Ollama version string, or ``None`` if unreachable."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/version", timeout=2)
        if r.ok:
            return r.json().get("version")
    except requests.RequestException:
        return None
    return None


def _ollama_capabilities(name: str) -> list | None:
    """Capability list (``embedding``/``vision``/...) for a model, ``None`` if unknown.

    ``None`` rather than ``[]`` on failure, so the caller can tell "Ollama said
    none" from "Ollama did not answer" and not cache the second (issue #459).
    """
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/show", json={"model": name}, timeout=5)
        if r.ok:
            return r.json().get("capabilities") or []
    except requests.RequestException:
        pass
    return None


def _ollama_models() -> list:
    """List installed Ollama models with size, family and capabilities.

    Capabilities come from ``/api/show`` and are cached by digest so repeated
    panel refreshes stay cheap.
    """
    r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    r.raise_for_status()
    models = []
    for m in r.json().get("models", []):
        name = m.get("name")
        if not name:
            continue
        digest = m.get("digest") or name
        details = m.get("details") or {}
        caps = _model_caps_cache.get(digest)
        if caps is None:
            caps = _ollama_capabilities(name)
            if caps is not None:
                _model_caps_cache[digest] = caps
            caps = caps or []
        models.append(
            {
                "name": name,
                "size": m.get("size"),
                "family": details.get("family"),
                "parameter_size": details.get("parameter_size"),
                "capabilities": caps,
                "embedding": "embedding" in caps,
                "vision": "vision" in caps,
            }
        )
    models.sort(key=lambda x: x["name"].lower())
    return models


def _start_ollama_process() -> Optional[str]:
    """Launch ``ollama serve`` detached and poll until ready (~12s). Returns version or None."""
    exe = shutil.which("ollama")
    if not exe:
        return None
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )
    subprocess.Popen(
        [exe, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    for _ in range(24):
        time.sleep(0.5)
        version = _ollama_version()
        if version:
            return version
    return None


def start_ollama_if_needed() -> None:
    """Start Ollama in the background if it is installed but not already running.

    Non-blocking: launches ``ollama serve`` on a daemon thread so app startup is
    never delayed by the Ollama boot. A no-op when Ollama is already reachable or
    not installed. Called at desktop/web launch so the UI works without the user
    starting the server by hand.
    """
    if _ollama_version() is not None:
        return
    if shutil.which("ollama") is None:
        return
    threading.Thread(target=_start_ollama_process, daemon=True, name="ollama-autostart").start()


def _store_pdf_count(folder: str) -> int:
    """Count PDF files directly inside a store folder."""
    try:
        return sum(1 for f in os.listdir(folder) if f.lower().endswith(".pdf"))
    except OSError:
        return 0


def _store_entry(name: str, label: str, folder: str) -> dict:
    """Build the JSON descriptor for one vector store."""
    path_db, collection_name = rag_engine._derivar_paths_db(folder)
    active = os.path.abspath(folder) == os.path.abspath(rag_engine.CARPETA_DOCS)
    # Opening a store creates its directory empty, so a store selected once,
    # or one whose indexing failed, used to read as indexed (issue #451).
    # Stored rows are what count, as the headless decided in #320; the file
    # is read rather than the store opened, which would create the directory.
    meta = os.path.join(path_db, collection_name, "meta.jsonl")
    indexed = os.path.isfile(meta) and os.path.getsize(meta) > 0
    entry = {
        "name": name,
        "label": label,
        "docs_folder": os.path.abspath(folder),
        "pdf_count": _store_pdf_count(folder),
        "indexed": indexed,
        "active": active,
        "fragments": None,
    }
    if active:
        try:
            entry["fragments"] = _get_collection().count()
        except Exception:
            pass
    return entry


def _all_stores() -> list:
    """List the three fixed language stores (English, Castellano, Valencià)."""
    stores = []
    for sid, label in _BUILTIN_STORES.items():
        folder = os.path.join(_DOCS_ROOT, sid)
        if os.path.isdir(folder):
            stores.append(_store_entry(sid, label, folder))
    return stores


def _active_store_name() -> str:
    """Basename of the active docs folder (the store identifier)."""
    return os.path.basename(os.path.abspath(rag_engine.CARPETA_DOCS))


def _switch_store_folder(folder: str) -> None:
    """Point the engine at a docs folder and reset the collection state.

    The caller owns the indexing slot (``_try_begin_indexing``): this helper
    must not open the window another mutating request could slip through
    (issue #394).
    """
    with _indexing_lock:
        rag_engine.set_docs_folder_runtime(folder)
        _state["collection"] = None
        rag_engine.reset_vector_store_cache()
        _state["indexing_failed"] = False
        _state["indexing_error"] = None
        _state["indexing_done_empty"] = False
        _state["indexing_progress"] = None
        _state["indexing_failed_files"] = []
    gc.collect()


def _pipeline_error_payload(exc: Exception) -> dict:
    """Return a stable error code plus a human fallback for a pipeline failure.

    The web UI translates on ``error`` (issue #379), so the code is the
    contract and ``message`` is only what an API client without i18n shows.

    Embedding runs on llama-server, a different process from Ollama. Checking
    it first is what keeps an embedder outage from being reported as "Ollama
    is down" -- the adapter names its server in the error, and the UI then
    points the user at the process that actually failed.

    Args:
        exc: Exception raised by retrieval, embedding or generation.

    Returns:
        JSON-ready payload with ``ok``, ``error`` and ``message``.
    """
    text = str(exc).lower()
    if "llama-server" in text:
        return {
            "ok": False,
            "error": "embedder_unavailable",
            "message": "No se pudo conectar con el embedder (llama-server). Comprueba que está en ejecución.",
        }
    if isinstance(exc, ConnectionError) or "ollama" in text or "connect" in text:
        return {
            "ok": False,
            "error": "model_unavailable",
            "message": "No se pudo conectar con Ollama. Comprueba que está en ejecución (pestaña Modelos).",
        }
    return {
        "ok": False,
        "error": "generation_error",
        "message": f"Error al procesar la consulta: {exc}",
    }


# API ROUTES


@app.route("/")
def index():
    """Serve the main page -- React build."""
    react_index = os.path.join(_react_dist, "index.html")
    if os.path.isfile(react_index):
        return send_from_directory(_react_dist, "index.html")
    return (
        "<h1>MonkeyGrab</h1><p>Build React no encontrado. Ejecuta: <code>cd rag/web/frontend && pnpm install && pnpm run build</code></p>",
        503,
        {"Content-Type": "text/html; charset=utf-8"},
    )


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    """Serve static files from the React build (JS/CSS bundles).

    Args:
        filename: Relative path within the assets directory.
    """
    assets_dir = os.path.join(_react_dist, "assets")
    return send_from_directory(assets_dir, filename)


@app.route("/logo-dark.png")
@app.route("/logo-light.png")
@app.route("/logo.png")
@app.route("/logo.jpg")
def serve_logo():
    """Serve the requested MonkeyGrab logo (Vite copies ``public/`` into ``dist/``
    root). The UI asks for ``logo.png`` and greys it in CSS, so one file serves
    both themes; the per-theme routes stay for older clients and resolve to it
    through the fallback below."""
    name = os.path.basename(request.path)  # route is a fixed literal — safe
    path = os.path.join(_react_dist, name)
    if os.path.isfile(path):
        return send_from_directory(_react_dist, name)
    for fallback in ("logo-dark.png", "logo-light.png", "logo.jpg", "logo.png"):
        if os.path.isfile(os.path.join(_react_dist, fallback)):
            return send_from_directory(_react_dist, fallback)
    return "", 404


def _api_init_logic():
    """Core logic for api_init.

    Returns:
        Tuple of (response_dict, status_code).
    """
    _invalidate_collection_if_deleted()

    # If permanently failed, report error without restarting the loop
    if _state["indexing_failed"]:
        return {
            "ok": False,
            "code": "indexing_failed",
            "error": _state["indexing_error"] or "La indexación falló",
        }, 500

    _ensure_indexed()

    if _state["indexing"]:
        resp = {
            "ok": False,
            "indexing": True,
            "code": "indexing_in_progress",
            "error": "Indexando documentos, por favor espera...",
        }
        if _state["indexing_progress"]:
            resp["progress"] = _state["indexing_progress"]
        resp.update(_init_paths_payload())
        return resp, 202

    coll = _get_collection()
    if coll.count() == 0:
        if _state["indexing_done_empty"]:
            docs = []
            _state["historial_chat"] = rag_engine.cargar_historial()
            return {
                "ok": True,
                "mode": _state["mode"],
                "total_fragments": 0,
                "documents": docs,
                "document_details": [],
                "history": _init_history_payload(_state["historial_chat"]),
                "history_count": len(_state["historial_chat"]),
                "failed_files": list(_state["indexing_failed_files"]),
                **_init_paths_payload(),
            }, 200
        return {
            "ok": False,
            "indexing": True,
            "code": "indexing_in_progress",
            "error": "Iniciando indexación de documentos...",
            **_init_paths_payload(),
        }, 202

    docs = rag_engine.obtener_documentos_indexados(coll)
    _state["historial_chat"] = rag_engine.cargar_historial()

    return {
        "ok": True,
        "mode": _state["mode"],
        "total_fragments": coll.count(),
        "documents": docs,
        "document_details": _collection_document_details(coll),
        "history": _init_history_payload(_state["historial_chat"]),
        "history_count": len(_state["historial_chat"]),
        # Detection only -- a mismatch is never auto-reindexed here (issue #36):
        # a settings change must not silently trigger a MinerU + embedding pass.
        # Reindexing stays the explicit /api/reindex action.
        "fingerprint_stale": rag_engine.index_fingerprint_mismatch(coll),
        # Files from the last run that did not land. Empty means the index is
        # a complete index of the folder; non-empty is a warning the caller
        # must not mistake for success (issue #323).
        "failed_files": list(_state["indexing_failed_files"]),
        **_init_paths_payload(),
    }, 200


@app.route("/api/init", methods=["GET"])
def api_init():
    """Initialize the system and return its status."""
    try:
        resp, status = _api_init_logic()
        return jsonify(resp), status
    except Exception:
        # Corrupt or deleted DB: invalidate and retry once
        _state["collection"] = None
        rag_engine.reset_vector_store_cache()
        _state["indexing_failed"] = False
        _state["indexing_error"] = None
        _state["indexing_done_empty"] = False
        _state["indexing_failed_files"] = []
        gc.collect()
        try:
            resp, status = _api_init_logic()
            return jsonify(resp), status
        except Exception as e2:
            return jsonify({"ok": False, "code": "init_failed", "error": str(e2)}), 500


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Process a message in chat mode. Supports SSE streaming."""
    data = request.get_json() or {}
    pregunta = (data.get("message") or "").strip()
    stream = data.get("stream", True)

    if not pregunta:
        return jsonify({"ok": False, "code": "empty_message", "error": "Mensaje vacío"}), 400

    if stream:

        def generate():
            generation = _state["history_generation"]
            full = ""
            try:
                for token in _chat_stream(pregunta):
                    full += token
                    yield _sse_event("token", {"token": token})
                # A /api/clear while the answer streamed means this turn was
                # deleted on purpose; persisting it would resurrect it.
                if generation == _state["history_generation"]:
                    _state["historial_chat"].append({"role": "user", "content": pregunta})
                    _state["historial_chat"].append({"role": "assistant", "content": full})
                    rag_engine.guardar_historial(_state["historial_chat"])
                yield _sse_event("done", {"done": True})
            except Exception as e:
                yield _sse_event("error", _pipeline_error_payload(e))

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return jsonify({"ok": False, "code": "stream_required", "error": "Usar stream=true"}), 400


@app.route("/api/rag", methods=["POST"])
def api_rag():
    """Process a RAG query. Supports SSE streaming."""
    data = request.get_json() or {}
    pregunta = (data.get("message") or "").strip()
    stream = data.get("stream", True)

    if len(pregunta) < rag_engine.MIN_LONGITUD_PREGUNTA_RAG:
        return jsonify(
            {
                "ok": False,
                "error": "question_too_short",
                "message": "Pregunta demasiado corta. Formula una pregunta concreta.",
            }
        ), 400

    coll = _get_collection()
    try:
        fragmentos_ranked, _, metricas = rag_engine.realizar_busqueda_hibrida(pregunta, coll)

        if not fragmentos_ranked:
            return jsonify(
                {
                    "ok": False,
                    "error": "no_results",
                    "message": "No se encontró información relevante en los documentos.",
                }
            ), 200

        fragmentos_finales, metricas_contexto = rag_engine.preparar_fragmentos_para_generacion(
            fragmentos_ranked,
            coll,
        )
        metricas = {**metricas, "fase_contexto": metricas_contexto}
        if not fragmentos_finales:
            return jsonify(
                {
                    "ok": False,
                    "error": "no_results",
                    "message": "No se encontró información relevante.",
                }
            ), 200

        sources = _format_sources(fragmentos_finales)

        mensaje_usuario = rag_engine._preparar_mensaje_usuario_rag(pregunta, fragmentos_finales)
    except Exception as e:
        # Retrieval / context prep failed (commonly Ollama down for embeddings).
        return jsonify(_pipeline_error_payload(e)), 200

    if stream:

        def generate():
            raw = ""
            # Sources go first: they are known before generation starts, and
            # a failure mid-answer would otherwise lose them with the 'done'
            # event that used to be their only carrier (issue #376). 'done'
            # still repeats them for clients that read only that event.
            yield _sse_event("sources", {"sources": sources})
            try:
                for token in rag_engine.generar_tokens_respuesta(mensaje_usuario):
                    raw += token
                    yield _sse_event("token", {"token": token})
                try:
                    rag_engine.guardar_debug_rag(
                        pregunta,
                        mensaje_usuario,
                        raw,
                        fragmentos_finales,
                        metricas=metricas,
                    )
                except Exception:
                    pass
                yield _sse_event("done", {"done": True, "sources": sources})
            except Exception as e:
                if raw:
                    try:
                        rag_engine.guardar_debug_rag(
                            pregunta,
                            mensaje_usuario,
                            raw,
                            fragmentos_finales,
                            metricas=metricas,
                        )
                    except Exception:
                        pass
                yield _sse_event("error", _pipeline_error_payload(e))

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        respuesta = rag_engine._generar_respuesta_stream(mensaje_usuario)
    except Exception as e:
        # Same JSON contract as a retrieval failure above (#379); unwrapped,
        # Flask answered 500 with an HTML page (issue #460).
        return jsonify(_pipeline_error_payload(e)), 200
    rag_engine.guardar_debug_rag(
        pregunta,
        mensaje_usuario,
        respuesta,
        fragmentos_finales,
        metricas=metricas,
    )
    return jsonify(
        {
            "ok": True,
            "response": respuesta,
            "sources": sources,
        }
    )


@app.route("/api/mode", methods=["POST"])
def api_mode():
    """Switch the active mode (chat/rag)."""
    data = request.get_json() or {}
    mode = (data.get("mode") or "chat").lower()
    if mode not in ("chat", "rag"):
        return jsonify({"ok": False, "code": "invalid_mode", "error": "Modo inválido"}), 400
    _state["mode"] = mode
    return jsonify({"ok": True, "mode": mode})


@app.route("/api/clear", methods=["POST"])
def api_clear():
    """Clear the chat history."""
    rag_engine.limpiar_historial(_state["historial_chat"])
    _state["historial_chat"] = []
    _state["history_generation"] += 1
    return jsonify({"ok": True})


@app.route("/api/docs/<path:filename>", methods=["DELETE"])
def api_delete_doc(filename):
    """Delete a document from FAISS and remove the PDF file from disk.

    Args:
        filename: Name of the PDF file to delete.
    """
    filename = _safe_pdf_basename(filename)
    if not filename:
        return jsonify({"ok": False, "code": "invalid_filename", "error": "Nombre de archivo inválido"}), 400
    # The background indexer adds chunks outside _indexing_lock, so the lock
    # alone let a delete land mid-run and the indexer re-add the document it
    # removed (issue #453). Claim the turn like every other index mutation.
    if not _try_begin_indexing():
        return jsonify({"ok": False, "error": "indexing_in_progress"}), 409
    try:
        # Resolve the folder and the store once, under the same lock: a store
        # switch in between used to delete the chunks from one store and the
        # PDF from another (issue #394).
        with _indexing_lock:
            filepath = os.path.join(os.path.abspath(rag_engine.CARPETA_DOCS), filename)
            try:
                coll = _get_collection()
                coll.delete_source(filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                docs = rag_engine.obtener_documentos_indexados(coll)
                return jsonify({"ok": True, "documents": docs, "total_fragments": coll.count()})
            except Exception as e:
                return jsonify({"ok": False, "code": "delete_failed", "error": str(e)}), 500
    finally:
        _release_indexing()


@app.route("/api/pdf/<path:filename>", methods=["GET"])
def api_serve_pdf(filename):
    """Serve a PDF file for in-browser viewing.

    Args:
        filename: Name of the PDF file to serve.
    """
    filename = _safe_pdf_basename(filename)
    if not filename:
        return jsonify({"ok": False, "code": "invalid_filename", "error": "Archivo inválido"}), 400
    docs_folder = os.path.abspath(rag_engine.CARPETA_DOCS)
    return send_from_directory(docs_folder, filename, mimetype="application/pdf")


# The three Study artifacts share one route because they share one shape:
# pick a document, load its chunks, ask the generator for structure. Three
# routes would have been three copies of the document lookup and three places
# for the error contract to drift apart.
_STUDY_KINDS = ("summary", "outline", "quiz")

# Allowlist for the Study language: the three store display names plus
# "Català" as an alias of "Valencià" (same ca store, both names in use).
# The use case interpolates this into the prompt ("in {language}"), so free
# text here is prompt injection by design -- reject before any engine work.
_STUDY_LANGUAGES = ("English", "Castellano", "Valencià", "Català")


@app.route("/api/study", methods=["POST"])
def api_study():
    """Build a summary, an outline or a quiz over one indexed document.

    Body: ``{"kind": "summary"|"outline"|"quiz", "document": str,
    "language": str|None, "question_count": int|None}``.

    ``language`` follows an allowlist (``None`` plus the store display
    names): it is interpolated into the prompt, so anything else is a 400
    before any engine work.

    A malformed generator reply comes back as 422 with ``kind: "malformed"``,
    separate from a 500: the model ignored the format, which is actionable
    (another model may not) and is not the server failing. The quiz raises
    rather than returning questions whose answer key could not be verified,
    so an empty panel here means "not safe to grade", never "nothing found".

    Validation failures carry a stable ``error`` code plus a human
    ``message`` (issue #379): the UI translates the code, so these never
    reach the user as English text inside a Spanish or Valencian template.
    """
    data = request.get_json(silent=True) or {}
    kind = str(data.get("kind") or "summary").strip().lower()
    if kind not in _STUDY_KINDS:
        return jsonify(
            {
                "ok": False,
                "error": "invalid_kind",
                "message": f"kind must be one of {', '.join(_STUDY_KINDS)}",
            }
        ), 400

    documento = str(data.get("document") or "").strip()
    if not documento:
        return jsonify(
            {
                "ok": False,
                "error": "document_required",
                "message": "document is required",
            }
        ), 400

    idioma = data.get("language") or None
    if idioma is not None:
        idioma = str(idioma).strip() or None
    if idioma is not None and idioma not in _STUDY_LANGUAGES:
        return jsonify(
            {
                "ok": False,
                "error": "invalid_language",
                "message": f"language must be one of {', '.join(_STUDY_LANGUAGES)}",
            }
        ), 400
    coll = _get_collection()
    if documento not in rag_engine.obtener_documentos_indexados(coll):
        return jsonify(
            {
                "ok": False,
                "error": "document_not_indexed",
                "message": f"{documento} is not indexed",
            }
        ), 404

    fragmentos = rag_engine.fragmentos_de_documento(documento)
    if not fragmentos:
        return jsonify(
            {
                "ok": False,
                "error": "document_without_fragments",
                "message": f"{documento} has no indexed fragments",
            }
        ), 409

    try:
        if kind == "summary":
            artifact = rag_engine.resumir_fragmentos(fragmentos, idioma=idioma)
        elif kind == "outline":
            artifact = rag_engine.esquema_de_fragmentos(fragmentos, idioma=idioma)
        else:
            count = data.get("question_count")
            kwargs = {"idioma": idioma}
            if count is not None:
                kwargs["num_preguntas"] = int(count)
            artifact = rag_engine.cuestionario_de_fragmentos(fragmentos, **kwargs)
    except (
        rag_engine.MalformedSummaryError,
        rag_engine.MalformedOutlineError,
        rag_engine.MalformedQuizError,
    ) as exc:
        return jsonify({"ok": False, "kind": "malformed", "code": "study_malformed", "error": str(exc)[:400]}), 422
    except ValueError as exc:
        return jsonify({"ok": False, "code": "study_invalid_request", "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "code": "study_failed", "error": str(exc)[:400]}), 500

    return jsonify({"ok": True, "kind": kind, "artifact": artifact})


@app.route("/api/reindex", methods=["POST"])
def api_reindex():
    """Re-index everything with the current pipeline settings.

    Accepts optional PDF files to add before re-indexing.
    """
    if not _try_begin_indexing():
        # Same answer as before #394: a reindex already running is the
        # outcome the caller asked for, not a client error.
        return jsonify(
            {
                "ok": True,
                "indexing": True,
                "message": "Ya hay una indexación en curso.",
                "progress": _state["indexing_progress"],
            }
        ), 202

    try:
        files = request.files.getlist("file") or []
        uploads = []
        for f in files:
            # Validate every name before writing anything: an invalid one must
            # not leave the earlier files of the same request on disk.
            filename = _safe_pdf_basename(f.filename) if f else ""
            if not filename:
                _release_indexing()
                return jsonify({"ok": False, "code": "invalid_filename", "error": "Nombre de archivo inválido"}), 400
            uploads.append((f, filename))

        folder = os.path.abspath(rag_engine.CARPETA_DOCS)
        for f, filename in uploads:
            os.makedirs(folder, exist_ok=True)
            f.save(os.path.join(folder, filename))

        _reset_db()
        # Reset failure and empty state to allow new indexing
        _state["indexing_failed"] = False
        _state["indexing_error"] = None
        _state["indexing_done_empty"] = False
    except Exception as e:
        _release_indexing()
        return jsonify({"ok": False, "code": "reindex_failed", "error": str(e)}), 500

    _hand_over_indexing()
    return jsonify(
        {
            "ok": True,
            "indexing": True,
            "message": "Re-indexación iniciada.",
            "total_fragments": 0,
            "documents": [],
            "progress": _state["indexing_progress"],
        }
    ), 202


_SETTINGS_MAP = {
    "contextualRetrieval": "USAR_CONTEXTUAL_RETRIEVAL",
    "queryDecomposition": "USAR_LLM_QUERY_DECOMPOSITION",
    "hybridSearch": "USAR_BUSQUEDA_HIBRIDA",
    "imageIndexing": "USAR_EMBEDDINGS_IMAGEN",
    "imageDescription": "USAR_DESCRIPCION_IMAGEN",
    "reranker": "USAR_RERANKER",
    "expandContext": "EXPANDIR_CONTEXTO",
    "optimizeContext": "USAR_OPTIMIZACION_CONTEXTO",
    "recompSynthesis": "USAR_RECOMP_SYNTHESIS",
}


@app.route("/api/settings", methods=["GET"])
def api_settings_get():
    """Return the current state of the pipeline flags."""
    current = {}
    for fe_key, engine_var in _SETTINGS_MAP.items():
        current[fe_key] = getattr(rag_engine, engine_var, False)
    return jsonify({"ok": True, "settings": current})


@app.route("/api/settings", methods=["POST"])
def api_settings_post():
    """Update the RAG pipeline flags at runtime."""
    data = request.get_json() or {}
    # Validate before mutating: a rejected request must leave the session on
    # the configuration it had. Only a JSON bool counts -- bool("false") is
    # True, so coercing would turn the flag on while reporting success
    # (issue #357).
    for fe_key in _SETTINGS_MAP:
        if fe_key in data and not isinstance(data[fe_key], bool):
            return jsonify({"ok": False, "error": "invalid_flag_value", "key": fe_key}), 400
    updated = {}
    for fe_key, engine_var in _SETTINGS_MAP.items():
        if fe_key in data:
            val = data[fe_key]
            setattr(rag_engine, engine_var, val)
            updated[fe_key] = val
            # Turned off: no query will call reranker() again until this is
            # flipped back on, so its resident CUDA weights are now held for
            # nothing (issue #239) -- give the VRAM back.
            if engine_var == "USAR_RERANKER" and not val:
                rag_engine.release_reranker()
    rag_engine.guardar_ajustes_persistidos()
    # contextualRetrieval, imageIndexing and imageDescription are index-time
    # flags (part of index_recipe): flipping any of them is the most likely
    # way a store goes stale mid-session, with no restart and no store switch
    # to otherwise trigger a fresh check -- see /api/init's own
    # fingerprint_stale field.
    return jsonify(
        {
            "ok": True,
            "settings": updated,
            "fingerprint_stale": rag_engine.index_fingerprint_mismatch(_get_collection()),
        }
    )


@app.route("/api/ollama", methods=["GET"])
def api_ollama_status():
    """Report whether the local Ollama server is reachable."""
    version = _ollama_version()
    return jsonify(
        {"ok": True, "running": version is not None, "version": version, "host": OLLAMA_BASE_URL}
    )


@app.route("/api/ollama/start", methods=["POST"])
def api_ollama_start():
    """Start ``ollama serve`` if it is not already running."""
    version = _ollama_version()
    if version is not None:
        return jsonify({"ok": True, "running": True, "version": version})
    if shutil.which("ollama") is None:
        return jsonify({"ok": False, "running": False, "error": "ollama_not_found"}), 404
    try:
        version = _start_ollama_process()
    except Exception as e:
        return jsonify({"ok": False, "running": False, "code": "ollama_start_failed", "error": str(e)}), 500
    return jsonify({"ok": version is not None, "running": version is not None, "version": version})


@app.route("/api/ollama/models", methods=["GET"])
def api_ollama_models():
    """List installed Ollama models for the role selectors."""
    if _ollama_version() is None:
        return jsonify({"ok": False, "error": "ollama_not_running", "models": []}), 200
    try:
        return jsonify({"ok": True, "models": _ollama_models()})
    except Exception as e:
        return jsonify({"ok": False, "code": "ollama_models_failed", "error": str(e), "models": []}), 500


@app.route("/api/models", methods=["GET"])
def api_models_get():
    """Return the current model assigned to each pipeline role."""
    return jsonify({"ok": True, "roles": rag_engine.get_model_roles()})


@app.route("/api/models", methods=["POST"])
def api_models_post():
    """Reassign Ollama generation roles at runtime."""
    if _state["indexing"]:
        return jsonify({"ok": False, "error": "indexing_in_progress"}), 409
    data = request.get_json() or {}
    overrides = {
        k: v
        for k, v in data.items()
        if k in rag_engine.MODEL_ROLE_VARS and isinstance(v, str) and v.strip()
    }
    if not overrides:
        return jsonify({"ok": False, "error": "no_valid_roles"}), 400

    try:
        roles = rag_engine.set_model_roles_runtime(overrides)
    except ValueError as e:
        return jsonify({"ok": False, "code": "invalid_model_roles", "error": str(e)}), 400

    rag_engine.guardar_ajustes_persistidos()
    # The contextual role only enters index_recipe while contextual retrieval
    # is on (the default) -- reassigning it mid-session is just as likely to
    # make the active store stale as toggling the flag itself; see
    # api_settings_post's comment.
    return jsonify(
        {
            "ok": True,
            "roles": roles,
            "fingerprint_stale": rag_engine.index_fingerprint_mismatch(_get_collection()),
        }
    )


@app.route("/api/stores", methods=["GET"])
def api_stores_list():
    """List the three fixed language stores (English, Castellano, Valencià)."""
    return jsonify(
        {
            "ok": True,
            "stores": _all_stores(),
            "active": _active_store_name(),
            "embedding": EMBEDDING_DISPLAY_NAME,
        }
    )


@app.route("/api/stores/select", methods=["POST"])
def api_stores_select():
    """Switch the active vector store (one of the three fixed language stores)."""
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    folder = rag_engine.resolver_carpeta_store(name)
    if not folder:
        return jsonify({"ok": False, "error": "unknown_store"}), 404
    if not _try_begin_indexing():
        return jsonify({"ok": False, "error": "indexing_in_progress"}), 409
    try:
        with _indexing_lock:
            _switch_store_folder(folder)
            rag_engine.guardar_ajustes_persistidos()
        coll = _get_collection()
        total = coll.count()
    except Exception:
        _release_indexing()
        raise
    if total == 0:
        _hand_over_indexing()
    else:
        _release_indexing()
    docs = rag_engine.obtener_documentos_indexados(coll) if total > 0 else []
    return jsonify(
        {
            "ok": True,
            "active": name,
            "indexing": _state["indexing"],
            "total_fragments": total,
            "documents": docs,
            "document_details": _collection_document_details(coll) if total > 0 else [],
            "fingerprint_stale": rag_engine.index_fingerprint_mismatch(coll)
            if total > 0
            else False,
            "stores": _all_stores(),
            **_init_paths_payload(),
        }
    )


# Apply any persisted UI choices (model roles / store / flags) before serving.
# The reader and the file live in rag/engine/settings.py because the CLI applies
# the same choices at startup: two interfaces over one index must not run under
# two configurations (issue #36 follow-up).
rag_engine.cargar_ajustes_persistidos()


# ENTRY POINT


def main():
    port = int(os.getenv("MONKEYGRAB_PORT", "5000"))
    host = os.getenv("MONKEYGRAB_HOST", "127.0.0.1")
    has_react = os.path.isfile(os.path.join(_react_dist, "index.html"))
    start_ollama_if_needed()
    print(f"\n  MonkeyGrab Web — http://{host}:{port}")
    if has_react:
        print(f"  Frontend React: {_react_dist}")
    else:
        print(f"  ⚠  Build React no encontrado en {_react_dist}")
        print("     Ejecuta: cd rag/web/frontend && pnpm install && pnpm run build")
        print("     (Usando template legacy como fallback)")
    print()
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
