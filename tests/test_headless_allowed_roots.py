"""Optional path allowlist for the headless service (issue #346).

The service indexes whatever absolute folder a caller names. With
MONKEYGRAB_HEADLESS_ALLOWED_ROOTS set, docs_folder and data_dir must resolve
(symlinks followed) under one of those roots; unset, nothing changes, so
Daimon's existing calls keep working. A root list that exists but is empty
refuses every path instead of quietly disabling the check (issue #490 review).
"""
import logging
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tests.test_headless_routes import _health, _service  # noqa: E402
from rag.headless.app import create_app  # noqa: E402


def _client(roots):
    app = create_app(service=_service(), health=_health(), allowed_roots=roots)
    return app.test_client()


def _payload(docs_folder, data_dir):
    return {"docs_folder": str(docs_folder), "data_dir": str(data_dir)}


def test_paths_outside_the_roots_are_refused(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    client = _client((str(allowed),))

    resp = client.post("/stores/s/index", json={"docs_folder": "/etc", "data_dir": str(allowed / "data")})

    assert resp.status_code == 403
    assert resp.get_json()["error"] == "path_not_allowed"
    assert "docs_folder" in resp.get_json()["message"]


def test_a_symlink_inside_a_root_cannot_escape_it(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "link").symlink_to("/etc")
    client = _client((str(allowed),))

    resp = client.post("/stores/s/index", json={"docs_folder": str(allowed / "link"),
                                                 "data_dir": str(allowed / "data")})

    assert resp.status_code == 403
    assert resp.get_json()["error"] == "path_not_allowed"


def test_paths_inside_the_roots_reach_the_service(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    client = _client((str(allowed),))

    resp = client.post("/stores/s/index", json={"docs_folder": str(allowed / "docs"),
                                                 "data_dir": str(allowed / "data")})

    assert resp.status_code == 200


def test_without_roots_any_path_is_accepted(tmp_path, monkeypatch):
    monkeypatch.delenv("MONKEYGRAB_HEADLESS_ALLOWED_ROOTS", raising=False)
    client = _client(None)

    resp = client.post("/stores/s/index", json={"docs_folder": str(tmp_path / "anywhere"),
                                                 "data_dir": str(tmp_path / "data")})

    assert resp.status_code == 200


def test_an_empty_variable_means_no_restriction(tmp_path, monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_HEADLESS_ALLOWED_ROOTS", "")
    client = _client(None)

    resp = client.post("/stores/s/index", json={"docs_folder": str(tmp_path / "anywhere"),
                                                 "data_dir": str(tmp_path / "data")})

    assert resp.status_code == 200


def test_roots_are_read_from_the_environment(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("MONKEYGRAB_HEADLESS_ALLOWED_ROOTS", str(allowed))
    client = _client(None)

    resp = client.post("/stores/s/index", json={"docs_folder": "/etc", "data_dir": str(allowed)})

    assert resp.status_code == 403


def test_a_relative_root_is_refused_with_json_instead_of_a_500(tmp_path):
    client = _client(("relative-root",))

    resp = client.post("/stores/s/index", json=_payload("/etc", tmp_path / "data"))

    assert resp.status_code == 403
    assert resp.get_json()["error"] == "path_not_allowed"


def test_an_empty_root_is_refused_with_json_instead_of_a_500(tmp_path):
    client = _client(("",))

    resp = client.post("/stores/s/index", json=_payload("/etc", tmp_path / "data"))

    assert resp.status_code == 403
    assert resp.get_json()["error"] == "path_not_allowed"


@pytest.mark.parametrize("field", ["docs_folder", "data_dir"])
def test_a_nul_byte_is_a_json_400_not_a_500(tmp_path, field):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    body = _payload(allowed / "docs", allowed / "data")
    body[field] += "\x00"

    resp = _client((str(allowed),)).post("/stores/s/index", json=body)

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "bad_request"


def test_an_explicitly_empty_root_list_refuses_every_path(tmp_path):
    client = _client(())

    resp = client.post("/stores/s/index", json=_payload(tmp_path / "docs", tmp_path / "data"))

    assert resp.status_code == 403
    assert resp.get_json()["error"] == "path_not_allowed"


def test_roots_set_to_only_separators_refuse_every_path(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("MONKEYGRAB_HEADLESS_ALLOWED_ROOTS", os.pathsep)
    with caplog.at_level(logging.WARNING):
        client = _client(None)

    assert "MONKEYGRAB_HEADLESS_ALLOWED_ROOTS" in caplog.text
    resp = client.post("/stores/s/index", json=_payload(tmp_path / "docs", tmp_path / "data"))

    assert resp.status_code == 403
    assert resp.get_json()["error"] == "path_not_allowed"


def test_a_tilde_root_is_expanded(tmp_path, monkeypatch):
    (tmp_path / "allowed").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    client = _client(("~/allowed",))

    resp = client.post("/stores/s/index", json=_payload(tmp_path / "allowed" / "docs",
                                                         tmp_path / "allowed" / "data"))

    assert resp.status_code == 200


def test_dotdot_cannot_escape_the_root(tmp_path):
    allowed = tmp_path / "allowed"
    (allowed / "sub").mkdir(parents=True)
    client = _client((str(allowed),))

    resp = client.post("/stores/s/index", json=_payload(allowed / "sub" / ".." / ".." / "etc",
                                                         allowed / "data"))

    assert resp.status_code == 403
    assert resp.get_json()["error"] == "path_not_allowed"


def test_dotdot_that_stays_inside_the_root_is_accepted(tmp_path):
    allowed = tmp_path / "allowed"
    (allowed / "sub").mkdir(parents=True)
    client = _client((str(allowed),))

    resp = client.post("/stores/s/index", json=_payload(allowed / "sub" / ".." / "docs",
                                                         allowed / "data"))

    assert resp.status_code == 200


def test_a_sibling_folder_sharing_the_prefix_is_refused(tmp_path):
    allowed = tmp_path / "data"
    allowed.mkdir()
    client = _client((str(allowed),))

    resp = client.post("/stores/s/index", json=_payload(tmp_path / "data2", tmp_path / "data2" / "dir"))

    assert resp.status_code == 403
    assert resp.get_json()["error"] == "path_not_allowed"


def test_the_second_of_several_roots_is_enough(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv("MONKEYGRAB_HEADLESS_ALLOWED_ROOTS", os.pathsep.join([str(first), str(second)]))
    client = _client(None)

    resp = client.post("/stores/s/index", json=_payload(second / "docs", second / "data"))

    assert resp.status_code == 200
