"""
MonkeyGrab -- Local web interface.

Flask server for the full RAG pipeline. Provides the same functionality
as the CLI: CHAT mode (free conversation) and RAG mode (document query).

Serves the React interface (build in web/zip/dist/) and the API at /api/*.

Usage (from project root):
    python web/app.py
    # or: python -m web.app

Open http://127.0.0.1:5000 in your browser.

Dependencies:
    - flask, flask-cors, chromadb, werkzeug
    - rag.chat_pdfs (project internal module)
"""

import gc
import io
import os
import sys
import shutil
import json
import time
import threading
import contextlib
from collections import Counter
from typing import Generator
from werkzeug.utils import secure_filename

import chromadb
from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory
from flask_cors import CORS


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Ensure project root is in sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rag.chat_pdfs as rag_engine

# React build directory (web/zip/dist/)
_web_dir = os.path.dirname(os.path.abspath(__file__))
_react_dist = os.path.join(_web_dir, "zip", "dist")

app = Flask(
    __name__,
    static_folder=os.path.join(_react_dist, "assets") if os.path.isdir(os.path.join(_react_dist, "assets")) else None,
)
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# CORS for development (Vite on :3000 -> Flask on :5000)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Global state (simple session for local use)
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
_indexing_lock = threading.Lock()


# ─────────────────────────────────────────────
# COLLECTION MANAGEMENT
# ─────────────────────────────────────────────

def _invalidate_collection_if_deleted():
    """Clear state to start fresh if the DB folder was deleted."""
    path_db = rag_engine.PATH_DB
    if not os.path.exists(os.path.dirname(path_db)):
        _state["collection"] = None
        _state["indexing_failed"] = False
        _state["indexing_error"] = None
        _state["indexing_done_empty"] = False
        gc.collect()


def _get_collection():
    """Obtain or create the ChromaDB collection, invalidating if the folder was deleted.

    Returns:
        The ChromaDB collection object.
    """
    _invalidate_collection_if_deleted()
    if _state["collection"] is None:
        os.makedirs(os.path.dirname(rag_engine.PATH_DB), exist_ok=True)
        client = chromadb.PersistentClient(path=rag_engine.PATH_DB)
        _state["collection"] = client.get_or_create_collection(
            name=rag_engine.COLLECTION_NAME
        )
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
    """Release the ChromaDB collection and delete it via the API (without touching the filesystem).

    Avoids WinError 32 on Windows during re-indexing by using delete_collection
    instead of rmtree.
    """
    _state["collection"] = None
    _state["indexing_done_empty"] = False
    gc.collect()
    last_error = None
    for attempt in range(5):
        time.sleep(0.5 * (attempt + 1))  # 0.5s, 1s, 1.5s, 2s, 2.5s
        try:
            client = chromadb.PersistentClient(path=rag_engine.PATH_DB)
            client.delete_collection(name=rag_engine.COLLECTION_NAME)
            return
        except ValueError:
            return  # Collection did not exist
        except Exception as e:
            last_error = e
            if attempt >= 4:
                raise last_error


# ─────────────────────────────────────────────
# STREAMING HELPERS
# ─────────────────────────────────────────────

def _chat_stream(pregunta: str) -> Generator[str, None, None]:
    """Generate tokens from chat mode via streaming.

    Args:
        pregunta: User question text.

    Yields:
        Token strings as they are produced by the model.
    """
    import ollama

    messages = [{"role": "system", "content": rag_engine.SYSTEM_PROMPT_CHAT}]
    mensajes_recientes = _state["historial_chat"][-(rag_engine.MAX_HISTORIAL_MENSAJES):]
    messages.extend(mensajes_recientes)
    messages.append({"role": "user", "content": pregunta})

    stream = ollama.chat(
        model=rag_engine.MODELO_CHAT,
        messages=messages,
        stream=True,
        options={"temperature": 0.7, "top_p": 0.9, "num_ctx": 8192},
    )

    for chunk in stream:
        content = chunk.get("message", {}).get("content", "") or chunk.get("content", "")
        if content:
            yield content


def _rag_stream(mensaje_usuario: str) -> Generator[str, None, None]:
    """Generate tokens from RAG mode via streaming.

    Args:
        mensaje_usuario: Full user message including context.

    Yields:
        Token strings as they are produced by the model.
    """
    import ollama

    # System prompt is baked into the RAG model Modelfile (not sent via API).
    stream = ollama.chat(
        model=rag_engine.MODELO_RAG,
        messages=[{"role": "user", "content": mensaje_usuario}],
        stream=True,
        options={
            "temperature": 0.15,
            "top_p": 0.85,
            "repeat_penalty": 1.15,
            "num_ctx": 8192,
        },
    )

    for chunk in stream:
        content = chunk.get("message", {}).get("content", "") or chunk.get("content", "")
        if content:
            yield content


def _format_sources(fragments: list) -> list:
    """Format source references for the JSON response.

    Args:
        fragments: List of fragment dicts with metadata (source, page).

    Returns:
        List of dicts with 'document' and 'pages' keys, sorted by document name.
    """
    sources_map = {}
    for frag in fragments:
        meta = frag.get("metadata", {})
        doc = meta.get("source", "?")
        page = meta.get("page", 0)
        page_num = page + 1 if isinstance(page, int) else page
        if doc not in sources_map:
            sources_map[doc] = set()
        sources_map[doc].add(page_num)

    return [
        {"document": doc, "pages": sorted(pages)}
        for doc, pages in sorted(sources_map.items())
    ]


# ─────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────


@app.route("/")
def index():
    """Serve the main page -- React build."""
    react_index = os.path.join(_react_dist, "index.html")
    if os.path.isfile(react_index):
        return send_from_directory(_react_dist, "index.html")
    return (
        "<h1>MonkeyGrab</h1><p>Build React no encontrado. Ejecuta: <code>cd web/zip && npm install && npm run build</code></p>",
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


@app.route("/logo.png")
@app.route("/logo.jpg")
def serve_logo():
    """Serve the MonkeyGrab logo."""
    for name in ("logo.jpg", "logo.png"):
        path = os.path.join(_react_dist, name)
        if os.path.isfile(path):
            return send_from_directory(_react_dist, name)
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
                "history_count": len(_state["historial_chat"]),
            }, 200
        return {
            "ok": False,
            "indexing": True,
            "error": "Iniciando indexación de documentos...",
        }, 202

    docs = rag_engine.obtener_documentos_indexados(coll)
    _state["historial_chat"] = rag_engine.cargar_historial()

    return {
        "ok": True,
        "mode": _state["mode"],
        "total_fragments": coll.count(),
        "documents": docs,
        "history_count": len(_state["historial_chat"]),
    }, 200


@app.route("/api/init", methods=["GET"])
def api_init():
    """Initialize the system and return its status."""
    try:
        resp, status = _api_init_logic()
        return jsonify(resp), status
    except Exception as e:
        # Corrupt or deleted DB: invalidate and retry once
        _state["collection"] = None
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
            for token in _chat_stream(pregunta):
                full += token
                yield f"data: {json.dumps({'token': token})}\n\n"
            _state["historial_chat"].append({"role": "user", "content": pregunta})
            _state["historial_chat"].append({"role": "assistant", "content": full})
            rag_engine.guardar_historial(_state["historial_chat"])
            yield f"data: {json.dumps({'done': True})}\n\n"

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
    fragmentos_ranked, mejor_score, metricas = rag_engine.realizar_busqueda_hibrida(
        pregunta, coll
    )

    if not fragmentos_ranked:
        return jsonify({
            "ok": False,
            "error": "no_results",
            "message": "No se encontró información relevante en los documentos.",
        }), 200

    if mejor_score < rag_engine.UMBRAL_RELEVANCIA:
        return jsonify({
            "ok": False,
            "error": "out_of_scope",
            "message": "Pregunta fuera del ámbito de los documentos indexados.",
        }), 200

    if rag_engine.USAR_RERANKER:
        fragmentos_filtrados = [
            f
            for f in fragmentos_ranked
            if f.get("score_reranker", f.get("score_final", 0))
            >= rag_engine.UMBRAL_SCORE_RERANKER
        ]
        if not fragmentos_filtrados:
            return jsonify({
                "ok": False,
                "error": "no_results",
                "message": "No se encontró información relevante.",
            }), 200
        fragmentos_ranked = fragmentos_filtrados

    fragmentos_finales = fragmentos_ranked[: rag_engine.TOP_K_FINAL]
    ids_usados = {f["id"] for f in fragmentos_finales}

    if (
        rag_engine.EXPANDIR_CONTEXTO
        and fragmentos_finales
        and "chunk" in fragmentos_finales[0]["metadata"]
    ):
        for frag in fragmentos_finales[: rag_engine.N_TOP_PARA_EXPANSION]:
            ids_vecinos = rag_engine.expandir_con_chunks_adyacentes(
                frag["id"], frag["metadata"], n_vecinos=1
            )
            if ids_vecinos:
                try:
                    vecinos = coll.get(
                        ids=ids_vecinos, include=["documents", "metadatas"]
                    )
                    for v_doc, v_meta in zip(
                        vecinos["documents"], vecinos["metadatas"]
                    ):
                        v_id = f"{v_meta['source']}_pag{v_meta['page']}_chunk{v_meta.get('chunk', 0)}"
                        if v_id not in ids_usados:
                            fragmentos_finales.append({
                                "doc": v_doc,
                                "metadata": v_meta,
                                "distancia": float("inf"),
                                "score_final": 0.0,
                                "id": v_id,
                            })
                            ids_usados.add(v_id)
                except Exception:
                    pass

    contexto_total = sum(len(f["doc"]) for f in fragmentos_finales)
    if contexto_total > rag_engine.MAX_CONTEXTO_CHARS:
        fragmentos_truncados = []
        chars_acum = 0
        for f in fragmentos_finales:
            if chars_acum + len(f["doc"]) > rag_engine.MAX_CONTEXTO_CHARS:
                break
            fragmentos_truncados.append(f)
            chars_acum += len(f["doc"])
        fragmentos_finales = fragmentos_truncados

    sources = _format_sources(fragmentos_finales)

    # Build context and user message (shared by streaming and non-streaming)
    if rag_engine.USAR_RECOMP_SYNTHESIS:
        contexto_str = rag_engine.sintetizar_contexto_recomp(
            fragmentos_finales, query_usuario=pregunta
        )
    else:
        contexto_str = rag_engine.construir_contexto_para_modelo(fragmentos_finales)

    mensaje_usuario = f"{pregunta}\n\n<context>{contexto_str}</context>"

    if stream:
        def generate():
            raw = ""
            for token in _rag_stream(mensaje_usuario):
                raw += token
            rag_engine.guardar_debug_rag(
                pregunta,
                mensaje_usuario,
                raw,
                fragmentos_finales,
                metricas=metricas,
            )
            words = raw.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    respuesta = rag_engine.generar_respuesta_silenciosa(pregunta, fragmentos_finales)
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
    })


@app.route("/api/docs", methods=["GET"])
def api_docs():
    """List indexed documents."""
    coll = _get_collection()
    docs = rag_engine.obtener_documentos_indexados(coll)
    return jsonify({"ok": True, "documents": docs})


@app.route("/api/docs/<path:filename>", methods=["DELETE"])
def api_delete_doc(filename):
    """Delete a document: remove its chunks from ChromaDB and the PDF file from disk.

    Args:
        filename: Name of the PDF file to delete.
    """
    if not filename or not filename.lower().endswith(".pdf"):
        return jsonify({"ok": False, "error": "Nombre de archivo inválido"}), 400
    filename = secure_filename(os.path.basename(filename))
    filepath = os.path.join(rag_engine.CARPETA_DOCS, filename)
    try:
        coll = _get_collection()
        coll.delete(where={"source": filename})
        if os.path.isfile(filepath):
            os.remove(filepath)
        docs = rag_engine.obtener_documentos_indexados(coll)
        return jsonify({"ok": True, "documents": docs, "total_fragments": coll.count()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/topics", methods=["GET"])
def api_topics():
    """Return a summary of indexed content/topics."""
    coll = _get_collection()
    docs = rag_engine.obtener_documentos_indexados(coll)
    docs_data = []

    for doc_name in docs:
        doc_info = {"name": doc_name}
        try:
            all_data = coll.get(
                where={"source": doc_name},
                include=["documents", "metadatas"],
                limit=100,
            )
            if all_data["documents"]:
                pages = {meta["page"] for meta in all_data["metadatas"]}
                doc_info["pages"] = len(pages)
                doc_info["fragments"] = len(all_data["documents"])
                texto = " ".join(all_data["documents"][:20])
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
        return jsonify({"ok": True, "total_fragments": 0, "indexing": True, "documents": []})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# Pipeline settings mapping (frontend name -> engine global variable)
_SETTINGS_MAP = {
    "contextualRetrieval": "USAR_CONTEXTUAL_RETRIEVAL",
    "queryDecomposition": "USAR_LLM_QUERY_DECOMPOSITION",
    "hybridSearch": "USAR_BUSQUEDA_HIBRIDA",
    "exhaustiveSearch": "USAR_BUSQUEDA_EXHAUSTIVA",
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
    return jsonify({"ok": True, "settings": updated})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload PDF(s) and index them.

    When add_only=1: add without full re-indexing (uses current settings).
    """
    add_only = request.args.get("add_only", "").lower() in ("1", "true", "yes")
    files = request.files.getlist("file") or (request.files.get("file") and [request.files["file"]] or [])
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
            coll = _get_collection()
            total = coll.count()
        docs = rag_engine.obtener_documentos_indexados(coll)
        return jsonify({
            "ok": True,
            "filename": saved[0] if len(saved) == 1 else None,
            "files": saved,
            "total_fragments": total,
            "documents": docs,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    port = int(os.getenv("MONKEYGRAB_PORT", "5000"))
    host = os.getenv("MONKEYGRAB_HOST", "127.0.0.1")
    has_react = os.path.isfile(os.path.join(_react_dist, "index.html"))
    print(f"\n  MonkeyGrab Web — http://{host}:{port}")
    if has_react:
        print(f"  Frontend React: {_react_dist}")
    else:
        print(f"  ⚠  Build React no encontrado en {_react_dist}")
        print(f"     Ejecuta: cd web/zip && npm install && npm run build")
        print(f"     (Usando template legacy como fallback)")
    print()
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
