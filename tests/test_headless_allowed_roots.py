"""Optional path allowlist for the headless service (issue #346).

The service indexes whatever absolute folder a caller names. With
MONKEYGRAB_HEADLESS_ALLOWED_ROOTS set, docs_folder and data_dir must resolve
(symlinks followed) under one of those roots; unset, nothing changes, so
Daimon's existing calls keep working.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tests.test_headless_routes import _health, _service  # noqa: E402
from rag.headless.app import create_app  # noqa: E402


def _client(tmp_path, roots):
    app = create_app(service=_service(), health=_health(), allowed_roots=roots)
    return app.test_client()


def test_paths_outside_the_roots_are_refused(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    client = _client(tmp_path, (str(allowed),))

    resp = client.post("/stores/s/index", json={"docs_folder": "/etc", "data_dir": str(allowed / "data")})

    assert resp.status_code == 403
    assert resp.get_json()["error"] == "path_not_allowed"
    assert "docs_folder" in resp.get_json()["message"]


def test_a_symlink_inside_a_root_cannot_escape_it(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "link").symlink_to("/etc")
    client = _client(tmp_path, (str(allowed),))

    resp = client.post("/stores/s/index", json={"docs_folder": str(allowed / "link"),
                                                 "data_dir": str(allowed / "data")})

    assert resp.status_code == 403


def test_paths_inside_the_roots_reach_the_service(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    client = _client(tmp_path, (str(allowed),))

    resp = client.post("/stores/s/index", json={"docs_folder": str(allowed / "docs"),
                                                 "data_dir": str(allowed / "data")})

    assert resp.status_code == 200


def test_without_roots_any_path_is_accepted(tmp_path, monkeypatch):
    monkeypatch.delenv("MONKEYGRAB_HEADLESS_ALLOWED_ROOTS", raising=False)
    client = _client(tmp_path, None)

    resp = client.post("/stores/s/index", json={"docs_folder": str(tmp_path / "anywhere"),
                                                 "data_dir": str(tmp_path / "data")})

    assert resp.status_code == 200


def test_roots_are_read_from_the_environment(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("MONKEYGRAB_HEADLESS_ALLOWED_ROOTS", str(allowed))
    client = _client(tmp_path, None)

    resp = client.post("/stores/s/index", json={"docs_folder": "/etc", "data_dir": str(allowed)})

    assert resp.status_code == 403
