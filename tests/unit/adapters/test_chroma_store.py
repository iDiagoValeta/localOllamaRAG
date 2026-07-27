"""Unit tests for monkeygrab.adapters.vectorstore.chroma_store.ChromaVectorStore.

Stubs chromadb.PersistentClient entirely -- no on-disk vector store, no
chromadb server -- so these run in milliseconds.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from monkeygrab.adapters.vectorstore import chroma_store as module
from monkeygrab.adapters.vectorstore.chroma_store import ChromaVectorStore
from monkeygrab.config.paths import PathsConfig
from monkeygrab.domain.chunk import Chunk
from monkeygrab.domain.chunk_metadata import ChunkMetadata
from monkeygrab.domain.fragment import Fragment


class FakeCollection:
    def __init__(self):
        self.add_calls = []
        self.query_calls = []
        self.get_calls = []
        self.count_value = 0
        self.query_result = {"documents": [[]], "distances": [[]], "metadatas": [[]]}
        self.get_result = {"documents": [], "metadatas": []}
        self.raise_on_query = None

    def add(self, ids, embeddings, documents, metadatas):
        self.add_calls.append(
            {"ids": ids, "embeddings": embeddings, "documents": documents, "metadatas": metadatas}
        )

    def query(self, query_embeddings, n_results, include):
        self.query_calls.append({"query_embeddings": query_embeddings, "n_results": n_results})
        if self.raise_on_query:
            raise self.raise_on_query
        return self.query_result

    def get(self, include, ids=None, limit=None, offset=None):
        self.get_calls.append({"ids": ids, "limit": limit, "offset": offset})
        return self.get_result

    def count(self):
        return self.count_value


class FakeClient:
    def __init__(self, path):
        self.path = path
        self.collection = FakeCollection()
        self.get_or_create_collection_calls = []

    def get_or_create_collection(self, name):
        self.get_or_create_collection_calls.append(name)
        return self.collection


def _make_store(monkeypatch, path_db="db/path", collection_name="my_collection"):
    fake_clients = []

    def fake_persistent_client(path):
        client = FakeClient(path)
        fake_clients.append(client)
        return client

    monkeypatch.setattr(module.chromadb, "PersistentClient", fake_persistent_client)
    paths = PathsConfig(path_db=path_db, collection_name=collection_name)
    store = ChromaVectorStore(paths)
    return store, fake_clients[0]


def test_constructor_uses_the_injected_paths_not_a_frozen_default(monkeypatch):
    """Two stores built from different PathsConfig must open different DB paths/collections."""
    store_a, client_a = _make_store(monkeypatch, path_db="db/a", collection_name="coll_a")
    store_b, client_b = _make_store(monkeypatch, path_db="db/b", collection_name="coll_b")

    assert client_a.path == "db/a"
    assert client_a.get_or_create_collection_calls == ["coll_a"]
    assert client_b.path == "db/b"
    assert client_b.get_or_create_collection_calls == ["coll_b"]


def test_add_serializes_chunk_and_omits_unset_optional_metadata_fields(monkeypatch):
    store, client = _make_store(monkeypatch)
    chunk = Chunk(
        text="chunk body",
        metadata=ChunkMetadata(source="paper.pdf", page=2, chunk=3, format="markdown", section_header="Intro"),
    )

    store.add(chunk, [0.1, 0.2])

    assert client.collection.add_calls == [{
        "ids": ["paper.pdf_pag2_chunk3"],
        "embeddings": [[0.1, 0.2]],
        "documents": ["chunk body"],
        "metadatas": [{
            "source": "paper.pdf", "page": 2, "chunk": 3,
            "section_header": "Intro", "format": "markdown",
        }],
    }]
    # total_chunks_in_page / image_width / image_height were never set on the
    # metadata -- they must not appear as None in the stored dict.
    stored_meta = client.collection.add_calls[0]["metadatas"][0]
    assert "total_chunks_in_page" not in stored_meta
    assert "image_width" not in stored_meta


def test_query_converts_chroma_rows_into_fragments_ordered_nearest_first(monkeypatch):
    store, client = _make_store(monkeypatch)
    client.collection.query_result = {
        "documents": [["doc one", "doc two"]],
        "distances": [[0.5, 0.9]],
        "metadatas": [[
            {"source": "a.pdf", "page": 0, "chunk": 0},
            {"source": "a.pdf", "page": 0, "chunk": 1},
        ]],
    }

    fragments = store.query([0.1, 0.2, 0.3], n_results=2)

    assert client.collection.query_calls[0] == {"query_embeddings": [[0.1, 0.2, 0.3]], "n_results": 2}
    assert fragments == [
        Fragment(doc="doc one", metadata=ChunkMetadata(source="a.pdf", page=0, chunk=0), distancia=0.5),
        Fragment(doc="doc two", metadata=ChunkMetadata(source="a.pdf", page=0, chunk=1), distancia=0.9),
    ]


def test_get_by_ids_returns_empty_list_without_calling_chroma_for_empty_input(monkeypatch):
    store, client = _make_store(monkeypatch)

    assert store.get_by_ids([]) == []
    assert client.collection.get_calls == []


def test_get_page_passes_limit_none_through_for_a_full_corpus_scan(monkeypatch):
    store, client = _make_store(monkeypatch)
    client.collection.get_result = {"documents": ["d"], "metadatas": [{"source": "a.pdf", "page": 0, "chunk": 0}]}

    fragments = store.get_page(limit=None, offset=0)

    assert client.collection.get_calls == [{"ids": None, "limit": None, "offset": 0}]
    assert fragments == [Fragment(doc="d", metadata=ChunkMetadata(source="a.pdf", page=0, chunk=0))]


def test_count_delegates_to_the_collection(monkeypatch):
    store, client = _make_store(monkeypatch)
    client.collection.count_value = 42

    assert store.count() == 42


def test_query_hard_fails_when_chroma_raises(monkeypatch):
    """No fallback to an empty result list -- a storage failure must propagate."""
    store, client = _make_store(monkeypatch)
    client.collection.raise_on_query = RuntimeError("chroma index corrupted")

    with pytest.raises(RuntimeError, match="chroma index corrupted"):
        store.query([0.1], n_results=5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
