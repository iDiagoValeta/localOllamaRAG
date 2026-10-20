"""The product writes and checks the fingerprint with MinerU's version (issue #426).

Both sides go through the same helper, so a machine without MinerU writes and
checks the version-less fingerprint consistently instead of reading every
store as stale.
"""

import rag.chat_pdfs  # noqa: F401  (loads the engine package in its supported order)
from monkeygrab.application.index_fingerprint import compute_index_fingerprint
from rag.engine import indexing, wiring


class _Store:
    def __init__(self, stored):
        self._stored = stored

    def read_fingerprint(self):
        return self._stored


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


def test_without_mineru_the_version_is_left_out_on_both_sides(monkeypatch):
    config = wiring.app_config_from_runtime()
    _version(monkeypatch, None)

    assert indexing._extractor_version() is None
    assert not indexing.index_fingerprint_mismatch(_Store(compute_index_fingerprint(config)))
