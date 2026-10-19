"""/api/rag sends its sources before the first token (issue #376).

The sources are computed before generation starts, and Answer's split into
select_evidence / stream exists so the web can send them first. Sending them
only with ``done`` lost them whenever generation failed mid-answer.

Loose test: importing rag.web.app needs the engine stack.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.web import app as web  # noqa: E402

_FRAGMENT = {"documento": "paper.pdf", "metadata": {"source": "paper.pdf", "page": 2}, "texto": "x"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(web, "_get_collection", lambda: object())
    monkeypatch.setattr(web.rag_engine, "realizar_busqueda_hibrida", lambda q, c: ([_FRAGMENT], None, {}))
    monkeypatch.setattr(web.rag_engine, "preparar_fragmentos_para_generacion", lambda f, c: (f, {}))
    monkeypatch.setattr(web, "_format_sources", lambda f: [{"document": "paper.pdf", "pages": [2], "best_page": 2}])
    monkeypatch.setattr(web.rag_engine, "_preparar_mensaje_usuario_rag", lambda q, f: "prompt")
    monkeypatch.setattr(web.rag_engine, "guardar_debug_rag", lambda *a, **k: None)
    web.app.config["TESTING"] = True
    with web.app.test_client() as c:
        yield c


def _events(body: str):
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_sources_arrive_before_the_first_token(client, monkeypatch):
    monkeypatch.setattr(web.rag_engine, "generar_tokens_respuesta", lambda m: iter(["Hola", " mundo"]))

    body = client.post("/api/rag", json={"message": "una pregunta suficientemente larga"}).get_data(as_text=True)

    names = [name for name, _ in _events(body)]
    assert names == ["sources", "token", "token", "done"]
    assert _events(body)[0][1]["sources"][0]["document"] == "paper.pdf"


def test_sources_survive_a_generation_failure(client, monkeypatch):
    def _fails(_mensaje):
        yield "Hola"
        raise RuntimeError("Ollama generate failed for model 'x': connection reset")

    monkeypatch.setattr(web.rag_engine, "generar_tokens_respuesta", _fails)

    body = client.post("/api/rag", json={"message": "una pregunta suficientemente larga"}).get_data(as_text=True)

    events = _events(body)
    assert [name for name, _ in events] == ["sources", "token", "error"]
    assert events[0][1]["sources"]
