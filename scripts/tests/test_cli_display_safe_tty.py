import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _reload_display(monkeypatch, backend=None):
    if backend is None:
        monkeypatch.delenv("MONKEYGRAB_CLI_BACKEND", raising=False)
    else:
        monkeypatch.setenv("MONKEYGRAB_CLI_BACKEND", backend)

    import rag.cli.display as display_module

    return importlib.reload(display_module)


def test_backend_defaults_to_prompt_toolkit_on_windows(monkeypatch):
    display_module = _reload_display(monkeypatch, None)
    monkeypatch.setattr(display_module.os, "name", "nt")
    monkeypatch.setattr(display_module, "PTK_AVAILABLE", True)
    monkeypatch.setattr(display_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(display_module.sys.stdout, "isatty", lambda: True)

    assert display_module._select_backend() == "prompt_toolkit"


def test_backend_defaults_to_rich_off_windows(monkeypatch):
    display_module = _reload_display(monkeypatch, None)
    monkeypatch.setattr(display_module.os, "name", "posix")
    monkeypatch.setattr(display_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(display_module.sys.stdout, "isatty", lambda: True)

    assert display_module._select_backend() == "rich"


def test_backend_env_override_to_plain(monkeypatch):
    display_module = _reload_display(monkeypatch, "plain")
    monkeypatch.setattr(display_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(display_module.sys.stdout, "isatty", lambda: True)

    assert display_module._select_backend() == "plain"


def test_backend_env_override_to_prompt_toolkit_falls_back_when_missing(monkeypatch):
    display_module = _reload_display(monkeypatch, "prompt_toolkit")
    monkeypatch.setattr(display_module, "PTK_AVAILABLE", False)
    monkeypatch.setattr(display_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(display_module.sys.stdout, "isatty", lambda: True)

    assert display_module._select_backend() == "plain"
