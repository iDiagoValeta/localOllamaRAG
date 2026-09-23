"""Warning on unauthenticated loopback startup (SECURITY-003, non-breaking)."""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rag.headless.app import create_app

EXPECTED = "running without auth on loopback; exposing beyond loopback unsupported"


def _service_double():
    def answer_stream(paths, question):
        yield ("no_results", {})

    def index_store(paths):
        return {}

    def status_store(paths):
        return {}

    import types
    return types.SimpleNamespace(
        answer_stream=answer_stream,
        index_store=index_store,
        status_store=status_store,
    )


def _health_double(config):
    from rag.headless.health import HealthReport
    return HealthReport(True, None, "abc", {}, {}, {})


def test_create_app_without_token_warns(caplog):
    with caplog.at_level(logging.WARNING):
        create_app(service=_service_double(), health=_health_double)
    assert EXPECTED in caplog.text


def test_create_app_with_token_stays_silent(caplog):
    with caplog.at_level(logging.WARNING):
        create_app(service=_service_double(), health=_health_double, token="s3cret")
    assert EXPECTED not in caplog.text
