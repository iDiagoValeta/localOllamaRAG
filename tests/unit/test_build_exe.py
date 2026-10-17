"""``packaging/build_exe.py`` must refuse to freeze a bundle with a partial corpus.

Issue #345. The desktop bundle ships the built-in corpora from ``rag/docs``, but
the PDFs themselves are gitignored (``AGENTS.md`` section 8): a fresh clone has
the manifest and only the documents committed before the fetcher existed.
Packaging such a clone succeeds and produces an app that cannot answer a
question, and nothing in the artifact says why.

``tools/fetch_corpus.py --check`` already knows which documents the manifest
requires and exits 1 when any is missing, so the build delegates the verdict
instead of keeping a second copy of the list.

The script is loaded by path rather than imported: the repo-root ``packaging``
directory has no ``__init__.py``, and the installed PyPA ``packaging`` package
would win the name. Standard library only, so this runs in the fast gate's
dependency-free ``architecture`` job.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "packaging" / "build_exe.py"

_spec = importlib.util.spec_from_file_location("build_exe_under_test", _SCRIPT)
assert _spec is not None and _spec.loader is not None
build_exe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_exe)


class _Completed:
    def __init__(self, returncode):
        self.returncode = returncode


def _record_run(monkeypatch, returncode):
    """Double ``subprocess.run`` and return the calls it recorded."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _Completed(returncode)

    monkeypatch.setattr(build_exe.subprocess, "run", fake_run)
    return calls


class TestCorpusCheck:
    def test_uses_the_running_interpreter_and_the_check_flag(self, monkeypatch):
        calls = _record_run(monkeypatch, 0)
        build_exe.check_corpus()

        assert len(calls) == 1
        cmd, kwargs = calls[0]
        assert cmd == [sys.executable, build_exe.FETCH_CORPUS, "--check"]
        assert kwargs["cwd"] == build_exe.PROJ

    def test_a_missing_document_aborts_and_names_the_fetch_command(self, monkeypatch):
        _record_run(monkeypatch, 1)

        with pytest.raises(RuntimeError, match=r"python tools/fetch_corpus\.py"):
            build_exe.check_corpus()

    def test_the_check_runs_before_anything_is_built(self, monkeypatch, tmp_path):
        events = []
        monkeypatch.setattr(build_exe, "check_corpus", lambda: events.append("check"))
        monkeypatch.setattr(build_exe, "build_frontend", lambda: events.append("frontend"))
        monkeypatch.setattr(build_exe, "build_icon", lambda: events.append("icon"))
        monkeypatch.setattr(build_exe, "run_pyinstaller", lambda dist: events.append("pyinstaller"))
        monkeypatch.setattr(build_exe, "_parse_args", lambda: argparse.Namespace(
            skip_frontend=True, dist=str(tmp_path)
        ))
        monkeypatch.setattr(build_exe, "DIST_UI", str(tmp_path))

        build_exe.main()

        assert events == ["check", "icon", "pyinstaller"]
