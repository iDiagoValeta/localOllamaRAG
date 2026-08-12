"""Tests for wiring the index fingerprint into the product's own indexing path
(issue #36): ``indexar_documentos`` must write it, and
``index_fingerprint_mismatch`` must read it back correctly, including the
"index built before fingerprinting existed" case.

Imports ``rag.engine.indexing``, so this file is skipped by the dependency-free
fast CI gate and runs where the full engine stack is installed -- see
tests/conftest.py.
"""

from types import SimpleNamespace

from monkeygrab.application.index_fingerprint import compute_index_fingerprint
from monkeygrab.config.app_config import AppConfig

# rag.chat_pdfs must finish importing before rag.engine.wiring does: wiring's
# get_runtime() imports rag.chat_pdfs, which re-exports rag.engine.indexing at
# its own module level -- importing rag.engine.indexing first (indirectly, via
# wiring) makes it the entry point instead and that re-export then finds a
# partially initialized module. Every real caller already goes through
# rag.chat_pdfs first (rag/web/app.py, rag/cli/app.py), so this mirrors that,
# not a workaround for a bug this change introduced.
import rag.chat_pdfs  # noqa: E402,F401
from rag.engine import indexing, wiring  # noqa: E402


class _FakeStore:
    """Minimal VectorStore double: only the fingerprint sidecar matters here."""

    def __init__(self, fingerprint=None):
        self._fingerprint = fingerprint

    def read_fingerprint(self):
        return self._fingerprint

    def write_fingerprint(self, value):
        self._fingerprint = value


class _FakeIndexCorpus:
    """Stands in for the real IndexCorpus -- no extraction/embedding needed
    to exercise the fingerprint bookkeeping around it."""

    def __init__(self, *_a, **_kw):
        pass

    def run(self, *_a, **_kw):
        return SimpleNamespace(chunks_indexed=1)


def _wire_fakes(monkeypatch, config):
    """Patch indexar_documentos's dependencies down to a no-op indexing pass."""
    monkeypatch.setattr(wiring, "app_config_from_runtime", lambda: config)
    monkeypatch.setattr(indexing, "IndexCorpus", _FakeIndexCorpus)
    monkeypatch.setattr(indexing, "build_extractor", lambda _config: None)
    monkeypatch.setattr(wiring, "embedder", lambda _config: None)


def _config(**overrides):
    base = {"flags.usar_contextual_retrieval": False, "flags.usar_embeddings_imagen": False}
    base.update(overrides)
    return AppConfig().with_overrides(**base)


def test_full_index_writes_the_fingerprint(tmp_path, monkeypatch):
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4")
    config = _config()
    _wire_fakes(monkeypatch, config)
    store = _FakeStore()

    indexing.indexar_documentos(str(tmp_path), store, silent=True)

    assert store.read_fingerprint() == compute_index_fingerprint(config)


def test_partial_add_does_not_touch_the_fingerprint(tmp_path, monkeypatch):
    # solo_archivos means "add these files to an existing store" -- the rest
    # of that store may have been indexed under a different recipe, so this
    # call cannot vouch for the whole thing. Writing here would let a partial
    # add silently launder a real mismatch away (see index_fingerprint_mismatch
    # below): the exact "engañable" failure mode this feature exists to close.
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4")
    config = _config()
    _wire_fakes(monkeypatch, config)
    store = _FakeStore(fingerprint="pre-existing-value")

    indexing.indexar_documentos(str(tmp_path), store, solo_archivos=["paper.pdf"], silent=True)

    assert store.read_fingerprint() == "pre-existing-value"


def test_empty_folder_does_not_touch_the_fingerprint(tmp_path, monkeypatch):
    config = _config()
    _wire_fakes(monkeypatch, config)
    store = _FakeStore(fingerprint="pre-existing-value")

    total = indexing.indexar_documentos(str(tmp_path), store, silent=True)

    assert total == 0
    assert store.read_fingerprint() == "pre-existing-value"


def test_mismatch_detected_when_stored_fingerprint_disagrees(monkeypatch):
    stored_under = _config()
    now_in_force = _config(**{"chunking.chunk_size": 999})
    monkeypatch.setattr(wiring, "app_config_from_runtime", lambda: now_in_force)
    store = _FakeStore(fingerprint=compute_index_fingerprint(stored_under))

    assert indexing.index_fingerprint_mismatch(store) is True


def test_no_mismatch_when_stored_fingerprint_matches(monkeypatch):
    config = _config()
    monkeypatch.setattr(wiring, "app_config_from_runtime", lambda: config)
    store = _FakeStore(fingerprint=compute_index_fingerprint(config))

    assert indexing.index_fingerprint_mismatch(store) is False


def test_no_mismatch_when_fingerprint_is_unknown(monkeypatch):
    # Every index built before this feature existed has no fingerprint.txt.
    # That must read as "unknown", not "mismatch" -- otherwise upgrading the
    # product would warn every existing user at their very next launch.
    config = _config()
    monkeypatch.setattr(wiring, "app_config_from_runtime", lambda: config)
    store = _FakeStore(fingerprint=None)

    assert indexing.index_fingerprint_mismatch(store) is False


def test_mismatch_check_never_blocks_startup_on_a_read_failure(monkeypatch):
    # A diagnostic must not be able to take the app down with it: an
    # antivirus-locked sidecar file on Windows (the platform the packaged
    # .exe ships on) must not abort the CLI before the prompt or poison
    # /api/init. Same fallback shape as obtener_documentos_indexados below.
    config = _config()
    monkeypatch.setattr(wiring, "app_config_from_runtime", lambda: config)

    class _BrokenStore:
        def read_fingerprint(self):
            raise OSError("sidecar file is locked")

    assert indexing.index_fingerprint_mismatch(_BrokenStore()) is False
