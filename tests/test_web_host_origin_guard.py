"""Host and Origin guard on the web API (issue #346).

DNS rebinding arrives as a foreign Host; a cross-site form POST arrives with a
foreign Origin and needs no CORS preflight. Both must be refused before any
route runs, while loopback traffic (the desktop app, Vite dev, curl) is not
affected.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.web import app as web


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("MONKEYGRAB_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("MONKEYGRAB_HOST", raising=False)
    web.app.config["TESTING"] = True
    with web.app.test_client() as c:
        yield c


def test_a_rebound_host_is_refused_before_the_route_runs(client):
    resp = client.get("/api/ollama", headers={"Host": "evil.example:5000"})

    assert resp.status_code == 403
    assert resp.get_json()["error"] == "host_not_allowed"


@pytest.mark.parametrize("host", ["localhost:5000", "127.0.0.1:41233", "[::1]:5000", "localhost"])
def test_loopback_hosts_pass_the_guard(client, host):
    resp = client.get("/api/ollama", headers={"Host": host})

    assert resp.get_json(silent=True) is None or resp.get_json().get("error") != "host_not_allowed"


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://localhost.evil.example"])
def test_a_mutating_request_from_a_foreign_origin_is_refused(client, origin):
    resp = client.post("/api/reindex", headers={"Origin": origin})

    assert resp.status_code == 403
    assert resp.get_json()["error"] == "origin_not_allowed"


def test_a_foreign_origin_does_not_block_a_read(client):
    resp = client.get("/api/ollama", headers={"Origin": "https://evil.example"})

    assert resp.status_code != 403


def test_a_loopback_origin_is_not_refused_on_a_mutating_request(client):
    # An unknown store name is rejected by the route itself, so this reaches
    # the handler without touching any store or index.
    resp = client.post(
        "/api/stores/select",
        json={"name": "no-such-store"},
        headers={"Origin": "http://127.0.0.1:5000"},
    )

    assert resp.status_code == 404
    assert resp.get_json()["error"] == "unknown_store"


def test_allowed_hosts_env_admits_a_named_host(client, monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_ALLOWED_HOSTS", "box.lan")

    resp = client.get("/api/ollama", headers={"Host": "box.lan:5000"})

    assert resp.get_json(silent=True) is None or resp.get_json().get("error") != "host_not_allowed"
