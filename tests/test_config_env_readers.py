"""Contract tests for the environment readers of both configuration paths.

These live outside ``tests/unit`` because importing ``rag.chat_pdfs`` pulls the
engine stack in (``tests/conftest.py`` skips this file when that stack is
absent), while ``monkeygrab.config.env`` alone would not need it. That is the
point: the web startup path (``rag.chat_pdfs`` module globals) and the core
(``monkeygrab.config.env``) must agree on what a bad or empty export means.
Issues #355 (hard-fail on unparseable numbers) and #356 (empty data dir).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_PREAMBLE = (
    "import sys\n"
    f"sys.path.insert(0, {str(ROOT)!r})\n"
    f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
    "import rag.chat_pdfs as rag\n"
)

_CLEAN_VARS = ("MONKEYGRAB_TEST_READER", "MONKEYGRAB_DATA_DIR")


def _run(script: str, extra_env: dict) -> subprocess.CompletedProcess:
    """Import ``rag.chat_pdfs`` in a subprocess with only ``extra_env`` added."""
    environ = {name: value for name, value in os.environ.items() if name not in _CLEAN_VARS}
    environ.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", _PREAMBLE + script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
        env=environ,
    )


@pytest.mark.parametrize(
    "reader,label",
    [("_leer_env_int", "Invalid integer"), ("_leer_env_float", "Invalid float")],
)
def test_chat_pdfs_readers_hard_fail_on_unparseable_value(reader, label):
    """A set-but-unparseable number aborts instead of becoming the default.

    Before the fix both helpers caught ``ValueError`` and returned ``default``,
    so startup succeeded under a configuration nobody declared and only the
    core reader failed later.
    """
    result = _run(
        f"rag.{reader}('MONKEYGRAB_TEST_READER', 5)",
        {"MONKEYGRAB_TEST_READER": "not-a-number"},
    )

    assert result.returncode != 0
    assert f"{label} for MONKEYGRAB_TEST_READER" in result.stderr
    assert "not-a-number" in result.stderr


def test_chat_pdfs_readers_keep_the_default_when_unset():
    result = _run(
        "print(rag._leer_env_int('MONKEYGRAB_TEST_READER', 5))\n"
        "print(rag._leer_env_float('MONKEYGRAB_TEST_READER', 1.5))",
        {},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["5", "1.5"]


def test_chat_pdfs_data_dir_treats_empty_env_as_unset():
    """``MONKEYGRAB_DATA_DIR=`` must not resolve to the process cwd."""
    result = _run("print(rag.DATA_DIR)", {"MONKEYGRAB_DATA_DIR": ""})

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == ROOT / "rag"
    assert Path(result.stdout.strip()) != ROOT
