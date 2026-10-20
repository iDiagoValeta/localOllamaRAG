"""``MONKEYGRAB_DOCS_ROOT`` resolution for the packaged desktop app (issue #344).

The frozen bundle keeps ``rag/docs`` inside the read-only install, so the
desktop entry point points this variable at a per-user copy. Unset must keep
the caller's bundle-relative default exactly as before.
"""

import os

from monkeygrab.config.env import read_env_docs_root


def test_unset_falls_back_to_the_caller_default(monkeypatch):
    monkeypatch.delenv("MONKEYGRAB_DOCS_ROOT", raising=False)

    assert read_env_docs_root("/bundle/rag/docs") == os.path.abspath("/bundle/rag/docs")


def test_an_empty_value_counts_as_unset(monkeypatch):
    # Shell exports and .env files spell "not configured" as an empty value;
    # abspath("") would silently turn the docs root into the process cwd.
    monkeypatch.setenv("MONKEYGRAB_DOCS_ROOT", "   ")

    assert read_env_docs_root("/bundle/rag/docs") == os.path.abspath("/bundle/rag/docs")


def test_a_relative_value_is_returned_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MONKEYGRAB_DOCS_ROOT", "custom-docs")

    assert read_env_docs_root("/bundle/rag/docs") == str(tmp_path / "custom-docs")
