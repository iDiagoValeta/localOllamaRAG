import importlib
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _reload_display(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("MONKEYGRAB_SAFE_TTY", raising=False)
    else:
        monkeypatch.setenv("MONKEYGRAB_SAFE_TTY", value)

    import rag.cli.display as display_module

    return importlib.reload(display_module)


def test_safe_tty_defaults_on_windows(monkeypatch):
    display_module = _reload_display(monkeypatch, None)
    monkeypatch.setattr(display_module.os, "name", "nt")

    assert display_module._safe_tty_enabled() is True


def test_safe_tty_env_override_off(monkeypatch):
    display_module = _reload_display(monkeypatch, "0")

    assert display_module._safe_tty_enabled() is False


def test_safe_tty_env_override_on(monkeypatch):
    display_module = _reload_display(monkeypatch, "1")

    assert display_module._safe_tty_enabled() is True
