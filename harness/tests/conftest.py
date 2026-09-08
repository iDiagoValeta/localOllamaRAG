"""Shared fixtures for the harness tests.

These tests measure the harness's own logic -- comparability, the ratchet, the
ledger -- against fake evaluators. None of them is about which model this
machine has configured, so none may depend on it.

Two things would otherwise make them depend on it, and both are invisible
when they bite:

- ``settings.json`` is gitignored, so CI has no such file and a developer's
  machine does. A test reading it passes in one place and fails in the other,
  with nothing on either side saying why.
- ``rag/web/app.py`` applies that file onto ``rag.chat_pdfs``'s globals at
  import time, so merely importing the web app anywhere earlier in the run
  changes what a later test sees. Test order becomes load-bearing.

Pinning the reference to ``AppConfig.from_env()`` closes both. The path that
does read ``settings.json`` is the product's real one and is covered by
``tests/test_harness_reference_honours_settings.py``, which controls the file
instead of inheriting it.
"""

import pytest

from harness import cli


@pytest.fixture(autouse=True)
def reference_from_env_only(monkeypatch):
    """Build the campaign reference from the environment alone."""
    from monkeygrab.config.app_config import AppConfig

    monkeypatch.setattr(cli, "_build_reference", AppConfig.from_env)
