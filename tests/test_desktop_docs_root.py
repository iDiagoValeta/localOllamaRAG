"""Frozen desktop runs point the docs root at a writable, seeded copy (#344).

``rag/web/desktop.py`` is stdlib-only at import time, so this test runs in the
dependency-free fast gate like ``tests/unit/test_desktop_boot.py``.
"""

import os

import pytest

import rag.web.desktop as desktop


@pytest.fixture(autouse=True)
def _isolated_environ(monkeypatch):
    """Give each test a private ``os.environ`` copy.

    ``_ensure_data_dir`` pins variables with ``os.environ.setdefault``, which
    monkeypatch cannot undo: ``delenv(name, raising=False)`` registers no
    undo when the name is already absent. Replacing the mapping restores every
    pin at teardown, so no later test sees a docs root or data dir from here.
    """
    monkeypatch.setattr(os, "environ", dict(os.environ))


def _fake_bundle(tmp_path):
    """Create a bundle dir shaped like the frozen ``_RAG_DIR``."""
    bundle = tmp_path / "bundle"
    docs = bundle / "docs"
    (docs / "en").mkdir(parents=True)
    (docs / "en" / "a.pdf").write_bytes(b"%PDF-1.4\n")
    (docs / "corpus_manifest.json").write_text("{}", encoding="utf-8")
    return bundle


def _frozen_run(tmp_path, monkeypatch, bundle):
    """Simulate the frozen entry point with an isolated data dir and bundle."""
    data_dir = tmp_path / "data"
    monkeypatch.setattr(desktop.sys, "frozen", True, raising=False)
    monkeypatch.setattr(desktop, "_RAG_DIR", str(bundle))
    monkeypatch.setenv("MONKEYGRAB_DATA_DIR", str(data_dir))
    monkeypatch.delenv("MONKEYGRAB_DOCS_ROOT", raising=False)
    return data_dir


def test_a_frozen_run_seeds_the_docs_root_from_the_bundle(tmp_path, monkeypatch):
    bundle = _fake_bundle(tmp_path)
    data_dir = _frozen_run(tmp_path, monkeypatch, bundle)

    desktop._ensure_data_dir()

    docs_root = data_dir / "docs"
    assert os.environ["MONKEYGRAB_DOCS_ROOT"] == str(docs_root)
    assert (docs_root / "en" / "a.pdf").read_bytes() == b"%PDF-1.4\n"
    assert (docs_root / "corpus_manifest.json").is_file()


def test_a_second_start_does_not_overwrite_a_modified_store(tmp_path, monkeypatch):
    bundle = _fake_bundle(tmp_path)
    data_dir = _frozen_run(tmp_path, monkeypatch, bundle)
    desktop._ensure_data_dir()
    edited = data_dir / "docs" / "en" / "a.pdf"
    edited.write_bytes(b"user edit")

    desktop._ensure_data_dir()

    assert edited.read_bytes() == b"user edit"


def test_a_second_start_does_not_restore_a_deleted_bundled_pdf(tmp_path, monkeypatch):
    # Deleting a bundled PDF is a legitimate way to shrink a store; seeding
    # must never merge missing files back in.
    bundle = _fake_bundle(tmp_path)
    data_dir = _frozen_run(tmp_path, monkeypatch, bundle)
    desktop._ensure_data_dir()
    (data_dir / "docs" / "en" / "a.pdf").unlink()

    desktop._ensure_data_dir()

    assert not (data_dir / "docs" / "en" / "a.pdf").exists()


def test_a_source_run_does_not_pin_the_docs_root(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop.sys, "frozen", False, raising=False)
    monkeypatch.setenv("MONKEYGRAB_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("MONKEYGRAB_DOCS_ROOT", raising=False)

    desktop._ensure_data_dir()

    assert "MONKEYGRAB_DOCS_ROOT" not in os.environ
