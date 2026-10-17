"""LOG_LEVEL defaults to WARNING and honours MONKEYGRAB_LOG_LEVEL (issue #398).

``rag/chat_pdfs.py`` pinned ``logging.ERROR``, so every ``logging.warning``
in the pipeline -- documents that could not be indexed, ``settings.json``
flags that were ignored, the headless service's unauthenticated-startup
notice -- never reached the terminal. These tests pin the new default (the
warnings are the point) and the environment override, including the
hard-fail on an unknown level.

``rag.chat_pdfs`` is only ever imported in a subprocess: LOG_LEVEL is read at
import time, so an in-process test could not observe a changed environment
without reloading the module and disturbing every other test holding
references to it. The subprocess also keeps this file importable in the
dependency-free fast CI gate -- a missing engine dependency skips the checks
instead of failing them.

Dependencies: stdlib only plus pytest.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = (
    "import json, logging, sys\n"
    f"sys.path.insert(0, {str(ROOT)!r})\n"
    f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
    "import rag.chat_pdfs as rag\n"
    "print(json.dumps({'log_level': rag.LOG_LEVEL, "
    "'root_level': logging.getLogger().level}))\n"
)


def _run(extra_env: dict) -> subprocess.CompletedProcess:
    """Import rag.chat_pdfs in a subprocess with a fixed log-level environment."""
    environ = {k: v for k, v in os.environ.items() if k != "MONKEYGRAB_LOG_LEVEL"}
    environ.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
        env=environ,
    )


def _levels(extra_env: dict) -> dict:
    result = _run(extra_env)
    if result.returncode != 0:
        if "ModuleNotFoundError" in result.stderr:
            pytest.skip(
                "rag.chat_pdfs cannot be imported here (missing engine "
                "dependency). Agreement check skipped, not passed."
            )
        raise AssertionError(f"subprocess failed:\n{result.stderr}")
    return json.loads(result.stdout)


def test_default_log_level_keeps_warnings_visible():
    """An unset variable leaves the pipeline's warnings on screen."""
    assert _levels({}) == {
        "log_level": logging.WARNING,
        "root_level": logging.WARNING,
    }


def test_monkeygrab_log_level_overrides_the_default():
    assert _levels({"MONKEYGRAB_LOG_LEVEL": "INFO"}) == {
        "log_level": logging.INFO,
        "root_level": logging.INFO,
    }


def test_unknown_log_level_hard_fails():
    """A typo aborts startup instead of silently falling back to a default."""
    result = _run({"MONKEYGRAB_LOG_LEVEL": "VERBOSE"})

    assert result.returncode != 0
    assert "MONKEYGRAB_LOG_LEVEL" in result.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
