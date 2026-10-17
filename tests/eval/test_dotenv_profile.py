"""A MONKEYGRAB_PROFILE written in .env must reach the gate (issue #303).

``run_eval`` imports ``monkeygrab.config`` at module level, and that package
applies the profile at import time (``src/monkeygrab/config/env.py``). The
module that loads ``.env`` for the product (``rag.chat_pdfs``) is only
imported later, inside ``evaluate()`` -- so a ``MONKEYGRAB_PROFILE=lite`` in
``.env`` was silently ignored and the gate measured the code defaults.

The check runs in a subprocess because the defect is import order: in the
pytest process ``monkeygrab.config`` is already imported and its one-time
profile application has already happened, so no monkeypatch can reproduce
it. ``python-dotenv`` is optional here exactly as it is for
``rag/chat_pdfs.py``; without it the file is never loaded.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from monkeygrab.config.env import PROFILES

pytest.importorskip("dotenv")

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "tests" / "eval"

_LITE_RAG_MODEL = PROFILES["lite"]["OLLAMA_RAG_MODEL"]

# Variables the parent shell may export that would otherwise mask the .env
# file (load_dotenv never overrides an existing variable) and let this test
# pass even without the fix.
_MASKED = {"MONKEYGRAB_PROFILE", *PROFILES["lite"]}

_PROBE = """
import json, os, sys
sys.path.insert(0, {eval_dir!r})
import run_eval
print("RESULT " + json.dumps({{
    "file": run_eval.__file__,
    "embedder_dim": os.environ.get("MONKEYGRAB_EMBEDDER_DIM"),
    "rag_model": os.environ.get("OLLAMA_RAG_MODEL"),
    "default_models": run_eval.DEFAULT_MODELS,
}}))
"""


def _probe_env() -> dict:
    """Inherit the test environment without anything a profile would seed."""
    env = {name: value for name, value in os.environ.items() if name not in _MASKED}
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (src, env.get("PYTHONPATH")) if part
    )
    return env


def test_profile_in_dotenv_is_applied_before_run_eval_imports_monkeygrab(tmp_path):
    repo = tmp_path / "repo"
    eval_dir = repo / "tests" / "eval"
    eval_dir.mkdir(parents=True)
    shutil.copyfile(EVAL_DIR / "run_eval.py", eval_dir / "run_eval.py")
    shutil.copyfile(EVAL_DIR / "grade.py", eval_dir / "grade.py")
    (repo / ".env").write_text("MONKEYGRAB_PROFILE=lite\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(eval_dir=str(eval_dir))],
        cwd=repo,
        env=_probe_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    line = next(
        line for line in result.stdout.splitlines() if line.startswith("RESULT ")
    )
    observed = json.loads(line[len("RESULT "):])

    # Prove the probe imported the copy under the temporary repo, not the
    # real module pytest may already have imported.
    assert Path(observed["file"]).resolve() == (eval_dir / "run_eval.py").resolve()
    # Seeded by apply_profile() when monkeygrab.config was imported.
    assert observed["embedder_dim"] == "512"
    # And the profile's generator is what the gate runs by default, because
    # DEFAULT_MODELS reads OLLAMA_RAG_MODEL right after that import.
    assert observed["rag_model"] == _LITE_RAG_MODEL
    assert observed["default_models"] == [_LITE_RAG_MODEL]
