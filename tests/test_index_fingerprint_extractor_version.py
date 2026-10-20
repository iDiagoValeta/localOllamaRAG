"""The product writes and checks the fingerprint with MinerU's version (issue #426).

Both sides go through the same helper, so a machine without MinerU writes and
checks the version-less fingerprint consistently instead of reading every
store as stale. The write side is exercised through ``indexar_documentos``
itself, not only through the helper: that seam is where the version-less
fingerprint could regress unnoticed.
"""

from types import SimpleNamespace

import pytest

import rag.chat_pdfs  # noqa: F401  (loads the engine package in its supported order)
from monkeygrab.adapters.extraction import mineru_extractor
from monkeygrab.application.index_fingerprint import compute_index_fingerprint
from monkeygrab.config.app_config import AppConfig
from rag.engine import indexing, wiring


class _Store:
    def __init__(self, stored):
        self._stored = stored

    def read_fingerprint(self):
        return self._stored


class _WritableStore:
    """Same double as tests/unit/test_indexing_fingerprint.py: only the
    fingerprint sidecar matters for the write path."""

    def __init__(self, fingerprint=None):
        self._fingerprint = fingerprint

    def read_fingerprint(self):
        return self._fingerprint

    def write_fingerprint(self, value):
        self._fingerprint = value


class _FakeIndexCorpus:
    """Stands in for real extraction/embedding, which the fingerprint
    bookkeeping around it does not depend on."""

    def __init__(self, *_a, **_kw):
        pass

    def run(self, *_a, **_kw):
        return SimpleNamespace(chunks_indexed=1)


def _config():
    # Both flags off: otherwise indexar_documentos builds a real MinerU image
    # extractor or a chat model, neither of which this test is about.
    return AppConfig().with_overrides(
        **{
            "flags.usar_contextual_retrieval": False,
            "flags.usar_embeddings_imagen": False,
        }
    )


def _wire_fakes(monkeypatch, config):
    monkeypatch.setattr(wiring, "app_config_from_runtime", lambda: config)
    monkeypatch.setattr(indexing, "IndexCorpus", _FakeIndexCorpus)
    monkeypatch.setattr(indexing, "build_extractor", lambda _config: None)
    monkeypatch.setattr(wiring, "embedder", lambda _config: None)


def _version(monkeypatch, value):
    def fake():
        if value is None:
            raise RuntimeError("MinerU CLI not found")
        return value
    monkeypatch.setattr(indexing, "mineru_installed_version", fake)


def test_a_store_built_by_another_mineru_version_is_stale(monkeypatch):
    config = wiring.app_config_from_runtime()
    _version(monkeypatch, "3.5.0")

    assert indexing.index_fingerprint_mismatch(_Store(compute_index_fingerprint(config, "3.4.5")))


def test_the_same_version_is_not_stale(monkeypatch):
    config = wiring.app_config_from_runtime()
    _version(monkeypatch, "3.5.0")

    assert not indexing.index_fingerprint_mismatch(_Store(compute_index_fingerprint(config, "3.5.0")))


def test_without_mineru_the_version_is_left_out_on_both_sides(tmp_path, monkeypatch):
    config = _config()
    _version(monkeypatch, None)
    _wire_fakes(monkeypatch, config)
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4")
    store = _WritableStore()

    indexing.indexar_documentos(str(tmp_path), store, silent=True)

    assert indexing._extractor_version() is None
    # The real write path, not a hand-built fingerprint: a version added on
    # write only would make every store built without MinerU read as stale.
    assert store.read_fingerprint() == compute_index_fingerprint(config)
    assert not indexing.index_fingerprint_mismatch(store)


def test_installed_version_turns_a_vanishing_binary_into_runtime_error(monkeypatch, tmp_path):
    """The docstring promises RuntimeError for "cannot report its version".

    _extractor_version only catches RuntimeError, so a bare OSError from the
    stat would escape it: during indexing that aborts the run after every
    chunk was embedded, and during the staleness check it silently reads as
    "not stale".
    """
    binary = tmp_path / "mineru"
    binary.write_text("fake binary", encoding="utf-8")
    real_os = mineru_extractor.os

    class _OsWithMissingStat:
        """Delegates to the real os, failing only the binary's stat.

        Patching os.stat itself would also break pytest's own tmp_path
        teardown, which stats directories with ``follow_symlinks``.
        """

        def __getattr__(self, name):
            if name == "stat":
                def stat_that_misses(*_args, **_kwargs):
                    raise FileNotFoundError("mineru vanished")

                return stat_that_misses
            return getattr(real_os, name)

    monkeypatch.setattr(mineru_extractor, "os", _OsWithMissingStat())

    with pytest.raises(RuntimeError, match="became unreadable"):
        mineru_extractor.installed_version(str(binary))


def test_the_version_probe_uses_a_short_timeout_of_its_own(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="mineru, version 3.4.5\n", stderr="")

    monkeypatch.setattr(mineru_extractor.subprocess, "run", fake_run)

    mineru_extractor._mineru_version.__wrapped__("/x/mineru", 1)

    assert seen["timeout"] == mineru_extractor._VERSION_PROBE_TIMEOUT_SECONDS
    # This probe runs inside HTTP requests; the extraction-sized timeout would
    # hold /api/init or /stores/<id>/status for minutes on a hung CLI.
    assert seen["timeout"] < 120
