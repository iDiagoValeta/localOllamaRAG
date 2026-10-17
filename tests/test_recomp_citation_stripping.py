"""Citation-marker stripping reaches the legacy RECOMP facade.

``rag.engine.context.sintetizar_contexto_recomp`` is still part of the public
``rag.chat_pdfs`` surface, so it must clean fragment text the same way
``Answer`` does (issue #311). The HTTP call is stubbed -- no Ollama server is
needed -- but importing the engine stack is the point, so this lives outside
``tests/unit/`` where the stdlib-only CI job does not collect it.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag.chat_pdfs  # noqa: E402,F401  (breaks the context <-> chat_pdfs import cycle)
import rag.engine.context as context  # noqa: E402


class _FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": "## Facts relevant to the question\n- A grounded fact."}}


class _FakeConfig:
    USAR_RECOMP_SYNTHESIS = True
    USAR_OPTIMIZACION_CONTEXTO = False
    MODELO_RECOMP = "fake-recomp"
    OLLAMA_KEEP_ALIVE = "5m"
    OLLAMA_RECOMP_NUM_CTX = 4096
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_REQUEST_TIMEOUT = 10

    def construir_contexto_para_modelo(self, fragmentos):
        return ""


def test_recomp_facade_strips_citation_markers_but_keeps_numeric_notation(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["payload"] = json
        return _FakeResponse()

    monkeypatch.setattr(context, "cfg", _FakeConfig())
    monkeypatch.setattr(context.requests, "post", fake_post)

    fragmentos = [{"doc": "As shown in [12]. Prior work [3, 7, 9] shows x[0] and [0, 1]."}]
    context.sintetizar_contexto_recomp(fragmentos, "question?")

    prompt = captured["payload"]["messages"][1]["content"]
    assert "As shown in." in prompt
    assert "Prior work shows" in prompt
    assert "x[0]" in prompt
    assert "[0, 1]" in prompt


def test_recomp_facade_marks_truncated_fragments_without_optimization(monkeypatch):
    """The marker does not depend on ``USAR_OPTIMIZACION_CONTEXTO`` (issue
    #397 point 3); the legacy facade must match ``Answer``."""
    captured = {}

    def fake_post(url, json, timeout):
        captured["payload"] = json
        return _FakeResponse()

    monkeypatch.setattr(context, "cfg", _FakeConfig())
    monkeypatch.setattr(context.requests, "post", fake_post)

    context.sintetizar_contexto_recomp(
        [{"doc": "The list continues with a cut"}], "question?"
    )

    prompt = captured["payload"]["messages"][1]["content"]
    assert "[excerpt ends mid-sentence]" in prompt
