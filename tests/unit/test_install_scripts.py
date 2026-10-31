"""The Linux install, run and uninstall scripts must at least parse, and the
two decisions that can hurt someone must hold (issue #559).

The scripts themselves are exercised end to end in a clean container (they
download gigabytes and need a GPU, so they cannot run in CI). What a unit test
can pin without side effects: every script is syntactically valid bash, the
entry points are executable, and ``uninstall.sh`` refuses a manifest entry
that would aim ``rm -rf`` at the filesystem root or the home directory, and
reads the llama.cpp build number from both ``--version`` spellings.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPO_ROOT / "tools"
ENTRY_POINTS = ("install.sh", "run.sh", "uninstall.sh")
SCRIPTS = (*ENTRY_POINTS, "_common.sh")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_is_valid_bash(script):
    result = subprocess.run(
        ["bash", "-n", str(TOOLS / script)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_entry_point_is_executable(script):
    assert os.access(TOOLS / script, os.X_OK)


def _env(tmp_path):
    env = dict(os.environ)
    env["MONKEYGRAB_STATE_DIR"] = str(tmp_path / "state")
    env["MONKEYGRAB_MODELS_DIR"] = str(tmp_path / "models")
    return env


def test_uninstall_refuses_paths_that_could_wipe_the_machine(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    (state / "install-manifest.tsv").write_text(
        f"repo\t{REPO_ROOT}\npath\t/\npath\t{Path.home()}\npath\t{victim}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [str(TOOLS / "uninstall.sh"), "--dry-run"],
        capture_output=True, text=True, env=_env(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert "REFUSED (unsafe path in manifest): /\n" in result.stdout
    assert f"REFUSED (unsafe path in manifest): {Path.home()}\n" in result.stdout
    assert str(victim) in result.stdout
    assert victim.exists(), "--dry-run must never delete"


def test_uninstall_refuses_another_checkouts_manifest(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "install-manifest.tsv").write_text("repo\t/somewhere/else\n", encoding="utf-8")
    result = subprocess.run(
        [str(TOOLS / "uninstall.sh"), "--yes"],
        capture_output=True, text=True, env=_env(tmp_path),
    )
    assert result.returncode != 0
    assert "belongs to /somewhere/else" in result.stderr


@pytest.mark.parametrize(
    "banner, expected",
    [
        ("version: 0.3.0-dev (build 10751, commit 3466812d1)", "10751"),
        ("version: 11669 (abc1234)", "11669"),
        ("not a version line", ""),
    ],
)
def test_llama_build_reads_both_version_spellings(tmp_path, banner, expected):
    fake = tmp_path / "llama-server"
    fake.write_text(f"#!/usr/bin/env bash\necho '{banner}'\n", encoding="utf-8")
    fake.chmod(0o755)
    env = _env(tmp_path)
    env["LLAMA_SERVER_BIN"] = str(fake)
    result = subprocess.run(
        ["bash", "-c", f". '{TOOLS / '_common.sh'}'; llama_build"],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected
