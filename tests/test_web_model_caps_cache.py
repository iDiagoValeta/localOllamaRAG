"""A failed /api/show is not cached as "no capabilities" (issue #459).

Loose test: importing rag.web.app needs the engine stack.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.web import app as web  # noqa: E402


def test_a_transient_show_failure_is_retried_on_the_next_listing(monkeypatch):
    answers = iter([None, ["completion", "vision"]])
    monkeypatch.setattr(web, "_ollama_capabilities", lambda name: next(answers))
    monkeypatch.setattr(web, "_model_caps_cache", {})

    class _Tags:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": [{"name": "m", "digest": "d1", "details": {}}]}

    monkeypatch.setattr(web.requests, "get", lambda *a, **k: _Tags())

    first = web._ollama_models()
    assert first[0]["vision"] is False
    assert web._model_caps_cache == {}

    second = web._ollama_models()
    assert second[0]["vision"] is True
    assert web._model_caps_cache == {"d1": ["completion", "vision"]}
