"""Flask surface of the headless service. Routes only; the work is in ``service``.

Bound to 127.0.0.1 by ``__main__``. With ``MONKEYGRAB_HEADLESS_TOKEN`` set,
every request must carry ``Authorization: Bearer <token>``. Without a token
the service runs without auth on loopback; exposing the port beyond loopback
without a token is unsupported.
"""
import hmac
import json
import logging
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

from flask import Flask, Response, abort, jsonify, request, stream_with_context

# Outside pytest (pytest.ini sets pythonpath = . src), nothing else puts src/
# on sys.path before a monkeygrab import runs; rag.chat_pdfs does that as a
# module-level side effect, so it must be the first project import here.
import rag.chat_pdfs  # noqa: F401
from monkeygrab.config.app_config import AppConfig
from rag.engine import wiring
from rag.headless import health as health_module
from rag.headless import service as service_module
from rag.headless.service import QuestionTooShort, StoreBusy, StoreConflict, StorePaths, StoreRegistry


def _sse_event(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _paths_from(values: Dict[str, Any]) -> Tuple[str, str]:
    """Both paths are mandatory: the service owns no default folder."""
    missing = [name for name in ("docs_folder", "data_dir") if not values.get(name)]
    if missing:
        abort(400, description=f"missing {', '.join(missing)}")
    return str(values["docs_folder"]), str(values["data_dir"])


def create_app(
    *,
    service: Any = service_module,
    health: Callable[[AppConfig], health_module.HealthReport] = health_module.cached_probe,
    token: Optional[str] = None,
    registry: Optional[StoreRegistry] = None,
) -> Flask:
    """Build the Flask app.

    Args:
        service: Module-like object with ``index_store``, ``status_store``
            and ``answer_stream`` (the real module by default; a double in tests).
        health: The probe to run on ``/health``. Cached with a short TTL by
            default (``health_module.cached_probe``) so Daimon's polling never
            pays a torch/CUDA subprocess per request; a double in tests.
        token: Bearer token to require, or ``None`` for no auth. Without a
            token the app serves loopback only; exposing the port beyond
            loopback without a token is unsupported.
        registry: Store-id registry; one per process by default.
    """
    app = Flask(__name__)
    if token is None:
        # Non-breaking notice: Daimon already injects a token, so a set token
        # stays silent; only the no-auth loopback case warns.
        logging.warning("running without auth on loopback; exposing beyond loopback unsupported")
    stores = registry or StoreRegistry()

    @app.before_request
    def _require_token():
        if token is None:
            return None
        supplied = request.headers.get("Authorization", "")
        # bytes, not str: compare_digest rejects non-ASCII str operands with a
        # TypeError, which before_request would otherwise turn into an HTML 500.
        if not hmac.compare_digest(supplied.encode("utf-8"), f"Bearer {token}".encode("utf-8")):
            abort(401)
        return None

    @app.errorhandler(400)
    @app.errorhandler(401)
    def _http_error(error):
        return jsonify({"ok": False, "error": "bad_request" if error.code == 400 else "unauthorized",
                        "message": getattr(error, "description", str(error))}), error.code

    def _resolve(store_id: str, values: Dict[str, Any]) -> StorePaths:
        docs_folder, data_dir = _paths_from(values)
        try:
            return stores.resolve(store_id, docs_folder, data_dir)
        except StoreConflict as exc:
            abort(409, description=str(exc))

    @app.errorhandler(409)
    def _conflict(error):
        return jsonify({"ok": False, "error": "store_conflict", "message": error.description}), 409

    @app.get("/health")
    def get_health():
        report = health(wiring.app_config_from_runtime())
        return jsonify(report.to_dict()), (200 if report.ok else 503)

    @app.post("/stores/<store_id>/index")
    def post_index(store_id: str):
        paths = _resolve(store_id, request.get_json(silent=True) or {})
        try:
            with stores.indexing(store_id):
                result = service.index_store(paths)
        except StoreBusy as exc:
            return jsonify({"ok": False, "error": "index_in_progress", "message": str(exc)}), 409
        except Exception as exc:
            logging.exception("indexing failed for store %s", store_id)
            return jsonify({"ok": False, "error": "index_failed", "message": str(exc)}), 502
        return jsonify({"store_id": store_id, **result})

    @app.get("/stores/<store_id>/status")
    def get_status(store_id: str):
        paths = _resolve(store_id, request.args.to_dict())
        try:
            result = service.status_store(paths)
        except Exception as exc:
            logging.exception("status failed for store %s", store_id)
            return jsonify({"ok": False, "error": "status_failed", "message": str(exc)}), 502
        return jsonify({"store_id": store_id, **result})

    @app.post("/stores/<store_id>/rag")
    def post_rag(store_id: str):
        body = request.get_json(silent=True) or {}
        paths = _resolve(store_id, body)
        question = str(body.get("message") or "").strip()
        raw_stream = body.get("stream", True)
        if not isinstance(raw_stream, bool):
            abort(400, description="stream must be a boolean")
        stream = raw_stream
        try:
            events: Iterator[Tuple[str, Dict[str, Any]]] = service.answer_stream(paths, question)
        except QuestionTooShort:
            return jsonify({"ok": False, "error": "question_too_short",
                            "message": "Pregunta demasiado corta. Formula una pregunta concreta."}), 400
        except Exception as exc:
            logging.exception("retrieval failed for store %s", store_id)
            return jsonify({"ok": False, "error": "retrieval_failed", "message": str(exc)}), 502
        try:
            first = next(events, None)
        except QuestionTooShort:
            return jsonify({"ok": False, "error": "question_too_short",
                            "message": "Pregunta demasiado corta. Formula una pregunta concreta."}), 400
        except Exception as exc:
            # The service is a lazy generator: this first pull drives both
            # retrieval and the first generated token, so a failure here
            # cannot be attributed to retrieval alone. A neutral kind keeps
            # Daimon from reindexing on what may be a model failure.
            logging.exception("rag failed for store %s", store_id)
            return jsonify({"ok": False, "error": "rag_failed", "message": str(exc)}), 502
        if first is None or first[0] == "no_results":
            return jsonify({"ok": False, "error": "no_results",
                            "message": "No se encontró información relevante en los documentos."}), 200

        def replay() -> Iterator[Tuple[str, Dict[str, Any]]]:
            yield first
            yield from events

        if stream:
            def generate():
                try:
                    for kind, payload in replay():
                        yield _sse_event(kind, payload)
                except Exception as exc:
                    logging.exception("generation failed for store %s", store_id)
                    yield _sse_event("error", {"error": str(exc)})

            return Response(stream_with_context(generate()), mimetype="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        text = ""
        sources: Any = []
        metrics: Dict[str, Any] = {}
        try:
            for kind, payload in replay():
                if kind == "token":
                    text += payload["token"]
                elif kind == "done":
                    sources = payload["sources"]
                    metrics = payload.get("metrics", {})
        except Exception as exc:
            logging.exception("generation failed for store %s", store_id)
            return jsonify({"ok": False, "error": "generation_failed", "message": str(exc)}), 502
        return jsonify({"ok": True, "response": text, "sources": sources, "metrics": metrics})

    return app
