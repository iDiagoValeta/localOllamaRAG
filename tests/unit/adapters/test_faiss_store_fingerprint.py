"""Tests for the FAISS store's opaque fingerprint sidecar."""

from monkeygrab.adapters.vectorstore.faiss_store import FaissVectorStore
from monkeygrab.config.paths import PathsConfig
from monkeygrab.domain.chunk import Chunk
from monkeygrab.domain.chunk_metadata import ChunkMetadata


def _store(tmp_path):
    return FaissVectorStore(PathsConfig(path_db=str(tmp_path), collection_name="c"))


def _chunk():
    # Chunk.id is derived from source/page/chunk, so metadata is all it needs.
    return Chunk(
        text="hello",
        metadata=ChunkMetadata(source="a.pdf", page=1, chunk=0, section_header=""),
    )


def test_absent_fingerprint_reads_as_none(tmp_path):
    assert _store(tmp_path).read_fingerprint() is None


def test_written_fingerprint_survives_reopen(tmp_path):
    store = _store(tmp_path)
    store.write_fingerprint("abc123")
    assert _store(tmp_path).read_fingerprint() == "abc123"


def test_blank_fingerprint_file_reads_as_none(tmp_path):
    store = _store(tmp_path)
    store.write_fingerprint("   ")
    assert _store(tmp_path).read_fingerprint() is None


def test_clear_removes_the_fingerprint(tmp_path):
    store = _store(tmp_path)
    store.add(_chunk(), [0.1, 0.2, 0.3])
    store.write_fingerprint("abc123")
    store.clear()
    assert store.read_fingerprint() is None
    assert _store(tmp_path).read_fingerprint() is None


def test_a_store_without_a_fingerprint_still_loads(tmp_path):
    # Every index written before fingerprinting existed lacks the file. It must
    # load as a valid store with an unknown recipe, not as corruption.
    store = _store(tmp_path)
    store.add(_chunk(), [0.1, 0.2, 0.3])
    reopened = _store(tmp_path)
    assert reopened.count() == 1
    assert reopened.read_fingerprint() is None
