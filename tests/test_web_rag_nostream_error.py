"""/api/rag without streaming reports a generation failure as JSON (issue #460).

Loose test: importing rag.web.app needs the engine stack.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.web import app as web  # noqa: E402

_FRAGMENT = {"documento": "paper.pdf", "metadata": {"source": "paper.pdf", "page": 2}, "texto": "x"}


def test_a_generation_failure_is_a_json_error_with_a_stable_code(monkeypatch):
    monkeypatch.setattr(web, "_get_collection", lambda: object())
    monkeypatch.setattr(web.rag_engine, "realizar_busqueda_hibrida", lambda q, c: ([_FRAGMENT], None, {}))
    monkeypatch.setattr(web.rag_engine, "preparar_fragmentos_para_generacion", lambda f, c: (f, {}))
    monkeypatch.setattr(web, "_format_sources", lambda f: [])
    monkeypatch.setattr(web.rag_engine, "_preparar_mensaje_usuario_rag", lambda q, f: "prompt")

    def _fails(_mensaje):
        raise RuntimeError("Ollama generate failed for model 'x': connection refused")

    monkeypatch.setattr(web.rag_engine, "_generar_respuesta_stream", _fails)

    response = web.app.test_client().post(
        "/api/rag", json={"message": "una pregunta suficientemente larga", "stream": False}
    )

    assert response.is_json
    body = response.get_json()
    assert body["ok"] is False and body["error"] == "model_unavailable"
