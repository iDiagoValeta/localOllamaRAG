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
from collections import Counter
from typing import Generator, Optional
from werkzeug.utils import secure_filename

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

_web_dir = os.path.dirname(os.path.abspath(__file__))
_react_dist = os.path.join(_web_dir, "frontend", "dist")

app = Flask(
    __name__,
    static_folder=os.path.join(_react_dist, "assets") if os.path.isdir(os.path.join(_react_dist, "assets")) else None,
)
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# CORS for local development (Vite on :3000 -> Flask on :5000).
_cors_origins = os.getenv(
    "MONKEYGRAB_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)
CORS(app, resources={r"/api/*": {"origins": [o.strip() for o in _cors_origins.split(",") if o.strip()]}})

_state = {
    "mode": "chat",
    "historial_chat": [],
    "collection": None,
    "indexing": False,
    "indexing_error": None,
    "indexing_failed": False,       # True if the last attempt failed permanently
    "indexing_done_empty": False,   # True if indexing completed with 0 chunks (no PDFs)
    "indexing_progress": None,      # {"file": str, "file_index": int, "total_files": int}
}
_indexing_lock = threading.RLock()

_CORPUS_PRESET_IDS = frozenset({"es", "ca", "en"})

# Vector-store layout: three fixed stores, one per language, each bound to a
# versioned docs folder under rag/docs/. There are no user-created stores and no
# hide/restore concept — the three always exist (possibly empty). English is the
# default. Their vector DBs live under DATA_DIR/vector_db/ (user-writable).
_DOCS_ROOT = os.path.join(_project_root, "rag", "docs")
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

# Persisted UI choices (model roles, active store, pipeline flags) so the desktop
# app remembers them across restarts. Lives in the writable data dir.
_SETTINGS_FILE = os.path.join(rag_engine.DATA_DIR, "settings.json")


def _infer_corpus_preset() -> str | None:
    """Return ``es``/``ca``/``en`` when ``CARPETA_DOCS`` basename matches a preset tree."""
    base = os.path.basename(os.path.abspath(rag_engine.CARPETA_DOCS))
    return base if base in _CORPUS_PRESET_IDS else None


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
    """Validate a routed PDF filename while preserving Unicode characters."""
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

        total_chunks = rag_engine.indexar_documentos(
            rag_engine.CARPETA_DOCS,
            coll,
            silent=True,
            progress_callback=_on_progress,
        )
        _state["indexing_error"] = None
        _state["indexing_failed"] = False
        if total_chunks == 0:
            _state["indexing_done_empty"] = True
        else:
            _state["indexing_done_empty"] = False
    except Exception as e:
        _state["indexing_error"] = str(e)
        _state["indexing_failed"] = True
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
            if coll.count() == 0 and not _state["indexing"] and not _state["indexing_failed"] and not _state["indexing_done_empty"]:
                _state["indexing"] = True
                _state["indexing_error"] = None
                _state["indexing_progress"] = None
                threading.Thread(target=_run_indexing_bg, daemon=True).start()


def _reset_db():
    """Clear the active FAISS store."""
    coll = _get_collection()
    coll.clear()
    _state["indexing_done_empty"] = False


# STREAMING HELPERS

def _chat_stream(pregunta: str) -> Generator[str, None, None]:
    """Generate tokens from chat mode via streaming.

    Args:
        pregunta: User question text.

    Yields:
        Token strings as they are produced by the model.
    """
    from monkeygrab.adapters.chat.ollama_chat import ollama_client_for

    messages = [{"role": "system", "content": rag_engine.SYSTEM_PROMPT_CHAT}]
    mensajes_recientes = _state["historial_chat"][-(rag_engine.MAX_HISTORIAL_MENSAJES):]
    messages.extend(mensajes_recientes)
    messages.append({"role": "user", "content": pregunta})

    stream = ollama_client_for(OLLAMA_BASE_URL).chat(
        model=rag_engine.MODELO_CHAT,
        messages=messages,
        stream=True,
        think=False,
        keep_alive=rag_engine.OLLAMA_KEEP_ALIVE,
        options={"temperature": 0.7, "top_p": 0.9, "num_ctx": 8192},
    )

    for chunk in stream:
        content = chunk.get("message", {}).get("content", "") or chunk.get("content", "")
        if content:
            yield content


def _format_sources(fragments: list) -> list:
    """Format source references for the JSON response.

    Args:
        fragments: List of fragment dicts with metadata (source, page).
            Fragments must be ordered by descending relevance score so that
            the first occurrence of each document is its highest-scoring fragment.

    Returns:
        List of dicts with 'document', 'pages', and 'best_page' keys, sorted
        by document name. 'best_page' is the page of the highest-scoring fragment
        for that document, intended as the default scroll target in the viewer.
    """
    sources_map = {}
    for frag in fragments:
        meta = frag.get("metadata", {})
        doc = meta.get("source", "?")
        page = meta.get("page", 0)
        page_num = page + 1 if isinstance(page, int) else page
        score = frag.get("score_reranker", frag.get("score_final", 0.0))
        if doc not in sources_map:
            sources_map[doc] = {"pages": set(), "best_page": page_num, "best_score": score}
        else:
            if score > sources_map[doc]["best_score"]:
                sources_map[doc]["best_score"] = score
                sources_map[doc]["best_page"] = page_num
        sources_map[doc]["pages"].add(page_num)

    return [
        {"document": doc, "pages": sorted(info["pages"]), "best_page": info["best_page"]}
        for doc, info in sorted(sources_map.items())
    ]


def _sse_event(event: str, payload: dict) -> str:
    """Serialize one Server-Sent Event with a named event type."""
    return (
        f"event: {event}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    )


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
        normalized.append({
            "name": info["name"],
            "fragments": info["fragments"],
            "pages": len(info["pages"]),
            "formats": sorted(info["formats"]),
        })
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


def _ollama_capabilities(name: str) -> list:
    """Best-effort capability list (``embedding``/``vision``/...) for a model."""
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/show", json={"model": name}, timeout=5)
        if r.ok:
            return r.json().get("capabilities") or []
    except requests.RequestException:
        pass
    return []


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
            _model_caps_cache[digest] = caps
        models.append({
            "name": name,
            "size": m.get("size"),
            "family": details.get("family"),
            "parameter_size": details.get("parameter_size"),
            "capabilities": caps,
            "embedding": "embedding" in caps,
            "vision": "vision" in caps,
        })
    models.sort(key=lambda x: x["name"].lower())
    return models


def _start_ollama_process() -> Optional[str]:
    """Launch ``ollama serve`` detached and poll until ready (~12s). Returns version or None."""
    exe = shutil.which("ollama")
    if not exe:
        return None
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
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
    path_db, _ = rag_engine._derivar_paths_db(folder)
    active = os.path.abspath(folder) == os.path.abspath(rag_engine.CARPETA_DOCS)
    indexed = os.path.isdir(path_db) and bool(os.listdir(path_db))
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


def _resolve_store_folder(name: str) -> Optional[str]:
    """Return the absolute docs folder for a store name, or ``None`` if unknown."""
    if name in _BUILTIN_STORES:
        folder = os.path.join(_DOCS_ROOT, name)
        return folder if os.path.isdir(folder) else None
    return None


def _active_store_name() -> str:
    """Basename of the active docs folder (the store identifier)."""
    return os.path.basename(os.path.abspath(rag_engine.CARPETA_DOCS))


def _switch_store_folder(folder: str) -> None:
    """Point the engine at a docs folder and reset all collection/indexing state."""
    with _indexing_lock:
        rag_engine.set_docs_folder_runtime(folder)
        _state["collection"] = None
        rag_engine.reset_vector_store_cache()
        _state["indexing"] = False
        _state["indexing_failed"] = False
        _state["indexing_error"] = None
        _state["indexing_done_empty"] = False
        _state["indexing_progress"] = None
    gc.collect()


def _save_persisted_settings() -> None:
    """Persist model roles, active store and pipeline flags to the data dir.

    Best-effort: lets the desktop app reopen with the user's last choices instead
    of the startup defaults.
    """
    try:
        data = {
            "roles": rag_engine.get_model_roles(),
            "active_store": _active_store_name(),
            "flags": {var: bool(getattr(rag_engine, var, False)) for var in _SETTINGS_MAP.values()},
        }
        os.makedirs(os.path.dirname(_SETTINGS_FILE), exist_ok=True)
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_persisted_settings() -> None:
    """Apply persisted settings at startup: model roles, then store, then flags.

    Roles are applied before the store. Missing or invalid entries are skipped;
    the engine defaults stand when nothing is saved.
    """
    try:
        # utf-8-sig tolerates a BOM if the file was hand-edited with a Windows tool.
        with open(_SETTINGS_FILE, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return

    roles = data.get("roles") or {}
    valid = {
        k: v for k, v in roles.items()
        if k in rag_engine.MODEL_ROLE_VARS and isinstance(v, str) and v.strip()
    }
    if valid:
        try:
            rag_engine.set_model_roles_runtime(valid)
        except Exception:
            pass

    store = data.get("active_store")
    if store and store != _active_store_name():
        folder = _resolve_store_folder(store)
        if folder:
            try:
                rag_engine.set_docs_folder_runtime(folder)
            except Exception:
                pass

    for var, val in (data.get("flags") or {}).items():
        if var in _SETTINGS_MAP.values():
            setattr(rag_engine, var, bool(val))


def _ollama_error_payload(exc: Exception) -> dict:
    """Friendly JSON error for an Ollama/generation failure (avoids a raw 500 page)."""
    text = str(exc).lower()
    if isinstance(exc, ConnectionError) or "ollama" in text or "connect" in text:
        return {
            "ok": False,
            "error": "ollama_unavailable",
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
    root). The theme switch requests ``logo-dark.png`` / ``logo-light.png``."""
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
        return {"ok": False, "error": _state["indexing_error"] or "La indexación falló"}, 500

    _ensure_indexed()

    if _state["indexing"]:
        resp = {
            "ok": False,
            "indexing": True,
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
                "history_count": len(_state["historial_chat"]),
                **_init_paths_payload(),
            }, 200
        return {
            "ok": False,
            "indexing": True,
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
        "history_count": len(_state["historial_chat"]),
        # Detection only -- a mismatch is never auto-reindexed here (issue #36):
        # a settings change must not silently trigger a MinerU + jina-clip pass.
        # Reindexing stays the explicit /api/reindex action.
        "fingerprint_stale": rag_engine.index_fingerprint_mismatch(coll),
        **_init_paths_payload(),
    }, 200


@app.route("/api/corpus", methods=["POST"])
def api_corpus():
    """Switch the PDF corpus folder and its derived FAISS path."""
    if _state["indexing"]:
        return jsonify({"ok": False, "error": "indexing_in_progress"}), 409
    data = request.get_json() or {}
    preset = (data.get("preset") or "").strip().lower()
    if preset not in _CORPUS_PRESET_IDS:
        return jsonify({"ok": False, "error": "invalid_preset"}), 400
    abs_path = os.path.join(_project_root, "rag", "docs", preset)
    if not os.path.isdir(abs_path):
        return jsonify({"ok": False, "error": "missing_docs_folder", "path": abs_path}), 400
    with _indexing_lock:
        rag_engine.set_docs_folder_runtime(abs_path)
        _state["collection"] = None
        rag_engine.reset_vector_store_cache()
        _state["indexing"] = False
        _state["indexing_failed"] = False
        _state["indexing_error"] = None
        _state["indexing_done_empty"] = False
        _state["indexing_progress"] = None
    gc.collect()
    coll = _get_collection()
    total_fragments = coll.count()
    if total_fragments == 0:
        _ensure_indexed()
    docs = rag_engine.obtener_documentos_indexados(coll) if total_fragments > 0 else []
    return jsonify({
        "ok": True,
        "preset": preset,
        "indexing": _state["indexing"],
        "total_fragments": total_fragments,
        "documents": docs,
        "document_details": _collection_document_details(coll) if total_fragments > 0 else [],
        **_init_paths_payload(),
    })


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
        gc.collect()
        try:
            resp, status = _api_init_logic()
            return jsonify(resp), status
        except Exception as e2:
            return jsonify({"ok": False, "error": str(e2)}), 500


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Process a message in chat mode. Supports SSE streaming."""
    data = request.get_json() or {}
    pregunta = (data.get("message") or "").strip()
    stream = data.get("stream", True)

    if not pregunta:
        return jsonify({"ok": False, "error": "Mensaje vacío"}), 400

    if stream:
        def generate():
            full = ""
            try:
                for token in _chat_stream(pregunta):
                    full += token
                    yield _sse_event("token", {"token": token})
                _state["historial_chat"].append({"role": "user", "content": pregunta})
                _state["historial_chat"].append({"role": "assistant", "content": full})
                rag_engine.guardar_historial(_state["historial_chat"])
                yield _sse_event("done", {"done": True})
            except Exception as e:
                yield _sse_event("error", {"error": str(e)})

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return jsonify({"ok": False, "error": "Usar stream=true"}), 400


@app.route("/api/rag", methods=["POST"])
def api_rag():
    """Process a RAG query. Supports SSE streaming."""
    data = request.get_json() or {}
    pregunta = (data.get("message") or "").strip()
    stream = data.get("stream", True)

    if len(pregunta) < rag_engine.MIN_LONGITUD_PREGUNTA_RAG:
        return jsonify({
            "ok": False,
            "error": "question_too_short",
            "message": "Pregunta demasiado corta. Formula una pregunta concreta.",
        }), 400

    coll = _get_collection()
    try:
        fragmentos_ranked, _, metricas = rag_engine.realizar_busqueda_hibrida(
            pregunta, coll
        )

        if not fragmentos_ranked:
            return jsonify({
                "ok": False,
                "error": "no_results",
                "message": "No se encontró información relevante en los documentos.",
            }), 200

        fragmentos_finales, metricas_contexto = rag_engine.preparar_fragmentos_para_generacion(
            fragmentos_ranked,
            coll,
        )
        metricas = {**metricas, "fase_contexto": metricas_contexto}
        if not fragmentos_finales:
            return jsonify({
                "ok": False,
                "error": "no_results",
                "message": "No se encontró información relevante.",
            }), 200

        sources = _format_sources(fragmentos_finales)

        mensaje_usuario = rag_engine._preparar_mensaje_usuario_rag(
            pregunta, fragmentos_finales
        )
    except Exception as e:
        # Retrieval / context prep failed (commonly Ollama down for embeddings).
        return jsonify(_ollama_error_payload(e)), 200

    if stream:
        def generate():
            raw = ""
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
                yield _sse_event("error", {"error": str(e)})

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    respuesta = rag_engine._generar_respuesta_stream(mensaje_usuario)
    rag_engine.guardar_debug_rag(
        pregunta,
        mensaje_usuario,
        respuesta,
        fragmentos_finales,
        metricas=metricas,
    )
    return jsonify({
        "ok": True,
        "response": respuesta,
        "sources": sources,
    })


@app.route("/api/mode", methods=["POST"])
def api_mode():
    """Switch the active mode (chat/rag)."""
    data = request.get_json() or {}
    mode = (data.get("mode") or "chat").lower()
    if mode not in ("chat", "rag"):
        return jsonify({"ok": False, "error": "Modo inválido"}), 400
    _state["mode"] = mode
    return jsonify({"ok": True, "mode": mode})


@app.route("/api/clear", methods=["POST"])
def api_clear():
    """Clear the chat history."""
    rag_engine.limpiar_historial(_state["historial_chat"])
    _state["historial_chat"] = []
    return jsonify({"ok": True})


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Return database statistics."""
    coll = _get_collection()
    docs = rag_engine.obtener_documentos_indexados(coll)
    return jsonify({
        "ok": True,
        "total_fragments": coll.count(),
        "documents": docs,
        "document_details": _collection_document_details(coll),
        "indexing": _state["indexing"],
    })


@app.route("/api/docs", methods=["GET"])
def api_docs():
    """List indexed documents."""
    coll = _get_collection()
    docs = rag_engine.obtener_documentos_indexados(coll)
    return jsonify({
        "ok": True,
        "documents": docs,
        "document_details": _collection_document_details(coll),
        "total_fragments": coll.count(),
        "indexing": _state["indexing"],
    })


@app.route("/api/docs/<path:filename>", methods=["DELETE"])
def api_delete_doc(filename):
    """Delete a document from FAISS and remove the PDF file from disk.

    Args:
        filename: Name of the PDF file to delete.
    """
    filename = _safe_pdf_basename(filename)
    if not filename:
        return jsonify({"ok": False, "error": "Nombre de archivo inválido"}), 400
    filepath = os.path.join(rag_engine.CARPETA_DOCS, filename)
    try:
        coll = _get_collection()
        coll.delete_source(filename)
        if os.path.isfile(filepath):
            os.remove(filepath)
        docs = rag_engine.obtener_documentos_indexados(coll)
        return jsonify({"ok": True, "documents": docs, "total_fragments": coll.count()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/pdf/<path:filename>", methods=["GET"])
def api_serve_pdf(filename):
    """Serve a PDF file for in-browser viewing.

    Args:
        filename: Name of the PDF file to serve.
    """
    filename = _safe_pdf_basename(filename)
    if not filename:
        return jsonify({"ok": False, "error": "Archivo inválido"}), 400
    docs_folder = os.path.abspath(rag_engine.CARPETA_DOCS)
    return send_from_directory(docs_folder, filename, mimetype="application/pdf")


@app.route("/api/topics", methods=["GET"])
def api_topics():
    """Return a summary of indexed content/topics."""
    coll = _get_collection()
    docs = rag_engine.obtener_documentos_indexados(coll)
    docs_data = []

    for doc_name in docs:
        doc_info = {"name": doc_name}
        try:
            fragments = [
                fragment
                for fragment in coll.get_page(None, 0)
                if fragment.metadata.source == doc_name
            ]
            if fragments:
                pages = {fragment.metadata.page for fragment in fragments}
                doc_info["pages"] = len(pages)
                doc_info["fragments"] = len(fragments)
                texto = " ".join(fragment.doc for fragment in fragments[:20])
                palabras = texto.split()
                significativas = [
                    p.strip('.,;:()[]{}"\'-\'').lower()
                    for p in palabras
                    if len(p) > 5
                    and p.strip('.,;:()[]{}"\'-\'').lower()
                    not in rag_engine.STOPWORDS
                ]
                frecuencias = Counter(significativas)
                top = [w for w, _ in frecuencias.most_common(10)]
                doc_info["terms"] = ", ".join(top) if top else None
        except Exception:
            pass
        docs_data.append(doc_info)

    return jsonify({"ok": True, "topics": docs_data})


@app.route("/api/reindex", methods=["POST"])
def api_reindex():
    """Re-index everything with the current pipeline settings.

    Accepts optional PDF files to add before re-indexing.
    """
    try:
        if _state["indexing"]:
            return jsonify({
                "ok": True,
                "indexing": True,
                "message": "Ya hay una indexación en curso.",
                "progress": _state["indexing_progress"],
            }), 202

        files = request.files.getlist("file") or []
        for f in files:
            if f and f.filename and f.filename.lower().endswith(".pdf"):
                filename = secure_filename(f.filename)
                dest = os.path.join(rag_engine.CARPETA_DOCS, filename)
                os.makedirs(rag_engine.CARPETA_DOCS, exist_ok=True)
                f.save(dest)

        _reset_db()
        # Reset failure and empty state to allow new indexing
        _state["indexing_failed"] = False
        _state["indexing_error"] = None
        _state["indexing_done_empty"] = False
        _ensure_indexed()
        return jsonify({
            "ok": True,
            "indexing": True,
            "message": "Re-indexación iniciada.",
            "total_fragments": 0,
            "documents": [],
            "progress": _state["indexing_progress"],
        }), 202
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload PDF(s) and index them.

    When add_only=1: add without full re-indexing, using current settings.
    """
    add_only = request.args.get("add_only", "").lower() in ("1", "true", "yes")
    files = request.files.getlist("file") or (
        request.files.get("file") and [request.files["file"]] or []
    )
    if not files:
        return jsonify({"ok": False, "error": "No se envió ningún archivo"}), 400

    saved = []
    for f in files:
        if not f or not f.filename or not f.filename.lower().endswith(".pdf"):
            continue
        filename = secure_filename(f.filename)
        dest = os.path.join(rag_engine.CARPETA_DOCS, filename)
        os.makedirs(rag_engine.CARPETA_DOCS, exist_ok=True)
        f.save(dest)
        saved.append(filename)

    if not saved:
        return jsonify({"ok": False, "error": "Ningún PDF válido"}), 400

    try:
        coll = _get_collection()
        if add_only:
            rag_engine.indexar_documentos(
                rag_engine.CARPETA_DOCS, coll, solo_archivos=saved
            )
            total = coll.count()
        else:
            _reset_db()
            _state["indexing_failed"] = False
            _state["indexing_error"] = None
            _state["indexing_done_empty"] = False
            _ensure_indexed()
            return jsonify({
                "ok": True,
                "indexing": True,
                "message": "PDF guardado. Re-indexación iniciada.",
                "files": saved,
                "total_fragments": 0,
                "documents": [],
                "progress": _state["indexing_progress"],
            }), 202
        docs = rag_engine.obtener_documentos_indexados(coll)
        return jsonify({
            "ok": True,
            "filename": saved[0] if len(saved) == 1 else None,
            "files": saved,
            "total_fragments": total,
            "documents": docs,
            "document_details": _collection_document_details(coll),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


_SETTINGS_MAP = {
    "contextualRetrieval": "USAR_CONTEXTUAL_RETRIEVAL",
    "queryDecomposition": "USAR_LLM_QUERY_DECOMPOSITION",
    "hybridSearch": "USAR_BUSQUEDA_HIBRIDA",
    "imageIndexing": "USAR_EMBEDDINGS_IMAGEN",
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
    updated = {}
    for fe_key, engine_var in _SETTINGS_MAP.items():
        if fe_key in data:
            val = bool(data[fe_key])
            # The reranker is only activated if the dependency is available
            if engine_var == "USAR_RERANKER" and val and not rag_engine.RERANKER_AVAILABLE:
                updated[fe_key] = False
                continue
            setattr(rag_engine, engine_var, val)
            updated[fe_key] = val
    _save_persisted_settings()
    # contextualRetrieval and imageIndexing are index-time flags (part of
    # index_recipe): flipping either one is the most likely way a store goes
    # stale mid-session, with no restart and no store switch to otherwise
    # trigger a fresh check -- see /api/init's own fingerprint_stale field.
    return jsonify({
        "ok": True,
        "settings": updated,
        "fingerprint_stale": rag_engine.index_fingerprint_mismatch(_get_collection()),
    })


@app.route("/api/ollama", methods=["GET"])
def api_ollama_status():
    """Report whether the local Ollama server is reachable."""
    version = _ollama_version()
    return jsonify({"ok": True, "running": version is not None, "version": version, "host": OLLAMA_BASE_URL})


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
        return jsonify({"ok": False, "running": False, "error": str(e)}), 500
    return jsonify({"ok": version is not None, "running": version is not None, "version": version})


@app.route("/api/ollama/models", methods=["GET"])
def api_ollama_models():
    """List installed Ollama models for the role selectors."""
    if _ollama_version() is None:
        return jsonify({"ok": False, "error": "ollama_not_running", "models": []}), 200
    try:
        return jsonify({"ok": True, "models": _ollama_models()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "models": []}), 500


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
        k: v for k, v in data.items()
        if k in rag_engine.MODEL_ROLE_VARS and isinstance(v, str) and v.strip()
    }
    if not overrides:
        return jsonify({"ok": False, "error": "no_valid_roles"}), 400

    try:
        roles = rag_engine.set_model_roles_runtime(overrides)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    _save_persisted_settings()
    # The contextual role only enters index_recipe while contextual retrieval
    # is on (the default) -- reassigning it mid-session is just as likely to
    # make the active store stale as toggling the flag itself; see
    # api_settings_post's comment.
    return jsonify({
        "ok": True,
        "roles": roles,
        "fingerprint_stale": rag_engine.index_fingerprint_mismatch(_get_collection()),
    })


@app.route("/api/stores", methods=["GET"])
def api_stores_list():
    """List the three fixed language stores (English, Castellano, Valencià)."""
    return jsonify({
        "ok": True,
        "stores": _all_stores(),
        "active": _active_store_name(),
        "embedding": "jinaai/jina-clip-v2",
    })


@app.route("/api/stores/select", methods=["POST"])
def api_stores_select():
    """Switch the active vector store (one of the three fixed language stores)."""
    if _state["indexing"]:
        return jsonify({"ok": False, "error": "indexing_in_progress"}), 409
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    folder = _resolve_store_folder(name)
    if not folder:
        return jsonify({"ok": False, "error": "unknown_store"}), 404
    _switch_store_folder(folder)
    _save_persisted_settings()
    coll = _get_collection()
    total = coll.count()
    if total == 0:
        _ensure_indexed()
    docs = rag_engine.obtener_documentos_indexados(coll) if total > 0 else []
    return jsonify({
        "ok": True,
        "active": name,
        "indexing": _state["indexing"],
        "total_fragments": total,
        "documents": docs,
        "document_details": _collection_document_details(coll) if total > 0 else [],
        "fingerprint_stale": rag_engine.index_fingerprint_mismatch(coll) if total > 0 else False,
        "stores": _all_stores(),
        **_init_paths_payload(),
    })


# Apply any persisted UI choices (model roles / store / flags) before serving.
_load_persisted_settings()


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
