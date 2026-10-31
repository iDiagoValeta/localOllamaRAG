"""Desktop boot: embedder liveness warning and MinerU cache placement.

Covers #342 and #343. ``rag/web/desktop.py`` imports only the standard
library at module scope -- pywebview, werkzeug and the backend are deferred
to the functions that need them -- so these tests run in the dependency-free
fast gate. ``_boot`` is exercised with a fake window and a stubbed server: no
pywebview, no Flask, no listening socket.
"""

import json
import os
import sys
import types
import urllib.error

import rag.web.desktop as desktop


class _Response:
    """Minimal stand-in for ``urllib.request.urlopen``'s context manager."""

    def __init__(self, status=200, body=b"{}"):
        self.status = status
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


_VISION = {"modalities": {"vision": True}}


def _stub_urlopen(monkeypatch, *, status=200, error=None, props=_VISION):
    """Replace ``urlopen`` and return the ``(url, timeout)`` calls it saw.

    ``/props`` answers ``props`` (a server started with ``--mmproj`` by default).
    """
    calls = []

    def _urlopen(url, timeout):
        calls.append((url, timeout))
        if error is not None:
            raise error
        return _Response(status, json.dumps(props).encode("utf-8"))

    monkeypatch.setattr(desktop.urllib.request, "urlopen", _urlopen)
    return calls


# EMBEDDER HEALTH CHECK (#342)


def test_the_health_probe_uses_the_appconfig_embedder_url(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_EMBEDDER_URL", "http://embedder.example:9000/")
    calls = _stub_urlopen(monkeypatch, status=200)

    assert desktop._embedder_warning() == ""

    timeout = desktop._EMBEDDER_HEALTH_TIMEOUT_SECONDS
    assert calls == [
        ("http://embedder.example:9000/health", timeout),
        ("http://embedder.example:9000/props", timeout),
    ]


def test_the_probe_timeout_is_short(monkeypatch):
    # A dead llama-server must not hold the desktop boot for long; the whole
    # point of checking at boot is a fast, actionable message.
    _stub_urlopen(monkeypatch, status=200)

    desktop._embedder_warning()

    assert desktop._EMBEDDER_HEALTH_TIMEOUT_SECONDS <= 3.0


def test_a_dead_embedder_yields_an_actionable_message(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_EMBEDDER_URL", "http://embedder.example:9000")
    _stub_urlopen(monkeypatch, error=urllib.error.URLError("Connection refused"))

    message = desktop._embedder_warning()

    assert "http://embedder.example:9000/health" in message
    assert "llama-server" in message
    assert "--embedding" in message
    assert "-b 2048 -ub 2048" in message
    assert "--mmproj mmproj-embeddinggemma-2-Q8_0.gguf" in message
    assert "MONKEYGRAB_EMBEDDER_URL" in message


# EMBEDDER VISION CHECK (#565)


def test_a_server_without_vision_warns_when_image_embeddings_are_on(monkeypatch):
    monkeypatch.setenv("USAR_EMBEDDINGS_IMAGEN", "1")
    _stub_urlopen(monkeypatch, props={"modalities": {"vision": False}})

    message = desktop._embedder_warning()

    assert "--mmproj mmproj-embeddinggemma-2-Q8_0.gguf" in message
    assert "USAR_EMBEDDINGS_IMAGEN" in message


def test_a_server_without_vision_is_fine_when_image_embeddings_are_off(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_EMBEDDER_URL", "http://embedder.example:9000")
    monkeypatch.setenv("USAR_EMBEDDINGS_IMAGEN", "0")
    calls = _stub_urlopen(monkeypatch, props={"modalities": {"vision": False}})

    assert desktop._embedder_warning() == ""
    assert [url for url, _timeout in calls] == ["http://embedder.example:9000/health"]


def test_a_server_with_vision_is_fine_when_image_embeddings_are_on(monkeypatch):
    monkeypatch.setenv("USAR_EMBEDDINGS_IMAGEN", "1")
    _stub_urlopen(monkeypatch, props={"modalities": {"vision": True}})

    assert desktop._embedder_warning() == ""


def test_a_non_200_health_response_counts_as_down(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_EMBEDDER_URL", "http://embedder.example:9000")
    _stub_urlopen(monkeypatch, status=503)

    message = desktop._embedder_warning()

    assert "503" in message


def test_boot_still_loads_the_window_when_the_embedder_is_down(monkeypatch):
    """A stopped embedder is a warning, never a boot failure (issue #342)."""
    loaded = {}
    logged = []

    class _Window:
        def load_url(self, url):
            loaded["url"] = url

        def load_html(self, html):
            loaded["html"] = html

    fake_app = types.ModuleType("rag.web.app")
    fake_app.start_ollama_if_needed = lambda: None

    monkeypatch.setattr(desktop, "_start_server", lambda port: object())
    monkeypatch.setattr(desktop, "_wait_until_ready", lambda port, timeout=30.0: True)
    monkeypatch.setattr(desktop, "_embedder_warning", lambda: "embedder (llama-server) is down")
    monkeypatch.setattr(desktop, "_log_boot_warning", logged.append)
    monkeypatch.setattr(desktop, "_SERVER", None, raising=False)
    monkeypatch.setitem(sys.modules, "rag.web.app", fake_app)

    desktop._boot(_Window(), 1234)

    assert loaded == {"url": "http://127.0.0.1:1234/"}
    assert logged == ["embedder (llama-server) is down"]


def test_the_warning_is_persisted_because_a_windowed_build_has_no_stderr(tmp_path, monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_DATA_DIR", str(tmp_path))

    path = desktop._log_boot_warning("embedder (llama-server) is down")

    assert path == str(tmp_path / "monkeygrab_boot.log")
    with open(path, encoding="utf-8") as f:
        assert "embedder (llama-server) is down" in f.read()


# MINERU CACHE (#343)


def test_the_mineru_cache_is_pinned_under_the_data_dir_without_overriding_an_export(
    tmp_path, monkeypatch
):
    data_dir = str(tmp_path / "data")
    monkeypatch.setenv("MONKEYGRAB_DATA_DIR", data_dir)
    monkeypatch.delenv("MINERU_CACHE_DIR", raising=False)

    desktop._ensure_data_dir()

    assert os.environ["MINERU_CACHE_DIR"] == os.path.join(data_dir, "mineru_cache")

    # setdefault, not assignment: a user who points the cache elsewhere keeps
    # it, the same precedence MONKEYGRAB_DATA_DIR itself follows.
    custom = str(tmp_path / "custom-cache")
    monkeypatch.setenv("MINERU_CACHE_DIR", custom)

    desktop._ensure_data_dir()

    assert os.environ["MINERU_CACHE_DIR"] == custom


# DATA DIRECTORY (#400)


def test_a_source_run_without_the_variable_uses_the_engine_data_dir(monkeypatch):
    """A source run keeps repo-relative paths, exactly as ``rag/chat_pdfs.py`` does.

    Log and port file used to land in the per-user folder while the engine wrote
    its vector DBs and history under ``rag/``: two data dirs in one run.
    """
    monkeypatch.delenv("MONKEYGRAB_DATA_DIR", raising=False)
    monkeypatch.setattr(desktop.sys, "frozen", False, raising=False)

    assert desktop._data_dir() == os.path.dirname(os.path.dirname(os.path.abspath(desktop.__file__)))


def test_a_frozen_run_without_the_variable_uses_the_per_user_dir(monkeypatch):
    monkeypatch.delenv("MONKEYGRAB_DATA_DIR", raising=False)
    monkeypatch.setattr(desktop.sys, "frozen", True, raising=False)

    assert desktop._data_dir() == desktop._default_data_dir()


def test_an_exported_variable_wins_in_both_modes(tmp_path, monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(desktop.sys, "frozen", True, raising=False)
    assert desktop._data_dir() == str(tmp_path)

    monkeypatch.setattr(desktop.sys, "frozen", False, raising=False)
    assert desktop._data_dir() == str(tmp_path)


def test_the_log_and_the_port_file_land_where_the_engine_writes(tmp_path, monkeypatch):
    """One resolution feeds both files: no second, diverging path for either."""
    monkeypatch.delenv("MONKEYGRAB_DATA_DIR", raising=False)
    monkeypatch.setattr(desktop.sys, "frozen", False, raising=False)
    # Point the source-run data root at tmp so the test writes nothing into rag/.
    monkeypatch.setattr(desktop, "_RAG_DIR", str(tmp_path))
    monkeypatch.setattr(
        desktop, "_start_server", lambda port: types.SimpleNamespace(shutdown=lambda: None)
    )

    def _interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(desktop.time, "sleep", _interrupt)

    desktop._run_headless(4321)
    log = desktop._log_boot_warning("x")

    assert log == str(tmp_path / "monkeygrab_boot.log")
    assert (tmp_path / "monkeygrab_server.port").read_text(encoding="utf-8") == "4321"
