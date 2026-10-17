"""Desktop boot: embedder liveness warning and MinerU cache placement.

Covers #342 and #343. ``rag/web/desktop.py`` imports only the standard
library at module scope -- pywebview, werkzeug and the backend are deferred
to the functions that need them -- so these tests run in the dependency-free
fast gate. ``_boot`` is exercised with a fake window and a stubbed server: no
pywebview, no Flask, no listening socket.
"""

import os
import sys
import types
import urllib.error

import rag.web.desktop as desktop


class _Response:
    """Minimal stand-in for ``urllib.request.urlopen``'s context manager."""

    def __init__(self, status=200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _stub_urlopen(monkeypatch, *, status=200, error=None):
    """Replace ``urlopen`` and return the ``(url, timeout)`` calls it saw."""
    calls = []

    def _urlopen(url, timeout):
        calls.append((url, timeout))
        if error is not None:
            raise error
        return _Response(status)

    monkeypatch.setattr(desktop.urllib.request, "urlopen", _urlopen)
    return calls


# EMBEDDER HEALTH CHECK (#342)


def test_the_health_probe_uses_the_appconfig_embedder_url(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_EMBEDDER_URL", "http://embedder.example:9000/")
    calls = _stub_urlopen(monkeypatch, status=200)

    assert desktop._embedder_warning() == ""

    assert calls == [("http://embedder.example:9000/health", desktop._EMBEDDER_HEALTH_TIMEOUT_SECONDS)]


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
    assert "MONKEYGRAB_EMBEDDER_URL" in message


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
