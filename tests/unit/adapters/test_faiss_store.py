"""Unit tests for monkeygrab.adapters.vectorstore.faiss_store.FaissVectorStore.

Round-trips a real faiss.IndexFlatIP through tmp_path directories -- unlike
test_chroma_store.py (which stubs chromadb.PersistentClient), there is no
thin client seam to stub here: the whole point of this adapter is what it
does with a real FAISS index, so these tests exercise it directly. The one
exception is test_add_hard_fails_when_persistence_write_fails, which injects
a write failure that cannot be reliably reproduced with a real disk.
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from monkeygrab.adapters.vectorstore.chroma_store import ChromaVectorStore
from monkeygrab.adapters.vectorstore.faiss_store import FaissVectorStore
from monkeygrab.config.paths import PathsConfig
from monkeygrab.domain.chunk import Chunk
from monkeygrab.domain.chunk_metadata import ChunkMetadata


def _chunk(source, page, chunk_idx, text="body"):
    return Chunk(text=text, metadata=ChunkMetadata(source=source, page=page, chunk=chunk_idx))


def _paths(tmp_path, name="store"):
    return PathsConfig(path_db=str(tmp_path / "db"), collection_name=name)


def _unit(vector):
    """Normalize a vector to unit L2 norm.

    Fixture vectors across this file use unit length so that Chroma's L2
    ranking and FAISS's cosine ranking provably agree -- see the
    interchangeability test below for the full argument.
    """
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


def test_add_query_get_by_ids_get_page_count_round_trip(tmp_path):
    store = FaissVectorStore(_paths(tmp_path))

    c0 = _chunk("a.pdf", 0, 0, "alpha body")
    c1 = _chunk("a.pdf", 0, 1, "beta body")
    c2 = _chunk("a.pdf", 1, 0, "gamma body")
    store.add(c0, _unit([1.0, 0.0, 0.0]))
    store.add(c1, _unit([0.0, 1.0, 0.0]))
    store.add(c2, _unit([0.9, 0.1, 0.0]))

    assert store.count() == 3

    # c0 is an exact match (distancia ~ 0); c2 is closer to the query than
    # c1, which is orthogonal to it.
    results = store.query(_unit([1.0, 0.0, 0.0]), n_results=2)
    assert [r.id for r in results] == [c0.id, c2.id]
    assert results[0].doc == "alpha body"
    assert results[0].distancia == pytest.approx(0.0, abs=1e-6)

    by_ids = store.get_by_ids([c1.id, c2.id, "missing.pdf_pag9_chunk9"])
    assert {f.id for f in by_ids} == {c1.id, c2.id}

    page = store.get_page(limit=2, offset=1)
    assert [f.id for f in page] == [c1.id, c2.id]

    full_page = store.get_page(limit=None, offset=0)
    assert [f.id for f in full_page] == [c0.id, c1.id, c2.id]


def test_get_by_ids_returns_partial_list_for_missing_ids_without_raising(tmp_path):
    """Matches the VectorStore port contract: an absent id is not a failure --
    the engine relies on this for neighbor-context expansion at document
    edges, where the "next" or "previous" chunk legitimately does not exist."""
    store = FaissVectorStore(_paths(tmp_path))
    c0 = _chunk("w.pdf", 0, 0)
    store.add(c0, _unit([1.0, 0.0]))

    fragments = store.get_by_ids([c0.id, "missing.pdf_pag9_chunk9"])

    assert [f.id for f in fragments] == [c0.id]


def test_query_on_empty_store_returns_empty_list(tmp_path):
    store = FaissVectorStore(_paths(tmp_path))
    assert store.query([1.0, 0.0], n_results=5) == []


def test_query_breaks_score_ties_by_ascending_chunk_id(tmp_path):
    """Two chunks share an identical embedding -- an exact score tie. FAISS's
    own internal tie order (here, insertion order: cb before ca) must be
    overridden by the documented (score desc, id asc) re-sort."""
    store = FaissVectorStore(_paths(tmp_path))
    cb = _chunk("z.pdf", 0, 1)  # id "z.pdf_pag0_chunk1"
    ca = _chunk("z.pdf", 0, 0)  # id "z.pdf_pag0_chunk0" -- lexicographically smaller
    store.add(cb, _unit([1.0, 0.0]))
    store.add(ca, _unit([1.0, 0.0]))

    results = store.query(_unit([1.0, 0.0]), n_results=2)

    assert [r.id for r in results] == [ca.id, cb.id]


def test_repeated_identical_queries_return_identical_order(tmp_path):
    store = FaissVectorStore(_paths(tmp_path))
    for i, vec in enumerate([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]]):
        store.add(_chunk("rep.pdf", 0, i), _unit(vec))

    query_vector = _unit([0.6, 0.5, 0.1])
    first = [f.id for f in store.query(query_vector, n_results=3)]
    second = [f.id for f in store.query(query_vector, n_results=3)]

    assert first == second


def test_persists_to_disk_and_reopening_matches(tmp_path):
    paths = _paths(tmp_path, "persisted")
    store = FaissVectorStore(paths)
    store.add(_chunk("p.pdf", 0, 0, "one"), _unit([1.0, 0.0]))
    store.add(_chunk("p.pdf", 0, 1, "two"), _unit([0.0, 1.0]))

    store_dir = Path(paths.path_db) / paths.collection_name
    assert (store_dir / "index.faiss").exists()
    assert (store_dir / "meta.jsonl").exists()
    assert (store_dir / "version.txt").exists()

    reopened = FaissVectorStore(paths)

    assert reopened.count() == store.count() == 2
    assert reopened.get_page(limit=None, offset=0) == store.get_page(limit=None, offset=0)
    assert reopened.query(_unit([1.0, 0.0]), n_results=1) == store.query(_unit([1.0, 0.0]), n_results=1)


def test_version_mismatch_hard_fails_on_reopen(tmp_path):
    paths = _paths(tmp_path, "versioned")
    store = FaissVectorStore(paths)
    store.add(_chunk("v.pdf", 0, 0), _unit([1.0, 0.0]))

    version_path = Path(paths.path_db) / paths.collection_name / "version.txt"
    version_path.write_text("999", encoding="utf-8")

    with pytest.raises(RuntimeError, match="format version"):
        FaissVectorStore(paths)


def test_partial_store_directory_hard_fails(tmp_path):
    """A directory missing one of the three sidecar files (e.g. meta.jsonl
    deleted by hand) is not a fresh store -- it is corruption."""
    paths = _paths(tmp_path, "partial")
    store = FaissVectorStore(paths)
    store.add(_chunk("q.pdf", 0, 0), _unit([1.0, 0.0]))

    meta_path = Path(paths.path_db) / paths.collection_name / "meta.jsonl"
    meta_path.unlink()

    with pytest.raises(RuntimeError, match="corrupted"):
        FaissVectorStore(paths)


def test_corrupted_index_file_hard_fails(tmp_path):
    paths = _paths(tmp_path, "badindex")
    store = FaissVectorStore(paths)
    store.add(_chunk("r.pdf", 0, 0), _unit([1.0, 0.0]))

    index_path = Path(paths.path_db) / paths.collection_name / "index.faiss"
    index_path.write_bytes(b"not a real faiss index")

    with pytest.raises(RuntimeError, match="corrupted"):
        FaissVectorStore(paths)


def test_corrupted_meta_jsonl_hard_fails(tmp_path):
    paths = _paths(tmp_path, "badmeta")
    store = FaissVectorStore(paths)
    store.add(_chunk("s.pdf", 0, 0), _unit([1.0, 0.0]))

    meta_path = Path(paths.path_db) / paths.collection_name / "meta.jsonl"
    meta_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="corrupted"):
        FaissVectorStore(paths)


def test_meta_and_index_row_count_mismatch_hard_fails(tmp_path):
    paths = _paths(tmp_path, "mismatch")
    store = FaissVectorStore(paths)
    store.add(_chunk("t.pdf", 0, 0), _unit([1.0, 0.0]))

    meta_path = Path(paths.path_db) / paths.collection_name / "meta.jsonl"
    extra_row = (
        '{"id": "extra", "doc": "x", "metadata": {"source": "t.pdf", "page": 0, "chunk": 1, '
        '"section_header": "", "total_chunks_in_page": null, "format": null, '
        '"image_width": null, "image_height": null}}\n'
    )
    with meta_path.open("a", encoding="utf-8") as f:
        f.write(extra_row)

    with pytest.raises(RuntimeError, match="corrupted"):
        FaissVectorStore(paths)


def test_add_hard_fails_on_dimension_mismatch(tmp_path):
    store = FaissVectorStore(_paths(tmp_path))
    store.add(_chunk("u.pdf", 0, 0), _unit([1.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="dimension"):
        store.add(_chunk("u.pdf", 0, 1), _unit([1.0, 0.0]))


def test_query_hard_fails_on_dimension_mismatch(tmp_path):
    store = FaissVectorStore(_paths(tmp_path))
    store.add(_chunk("u.pdf", 0, 0), _unit([1.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="dimension"):
        store.query(_unit([1.0, 0.0]), n_results=1)


def test_add_hard_fails_on_duplicate_chunk_id(tmp_path):
    store = FaissVectorStore(_paths(tmp_path))
    chunk = _chunk("dup.pdf", 0, 0)
    store.add(chunk, _unit([1.0, 0.0]))

    with pytest.raises(ValueError, match="already exists"):
        store.add(chunk, _unit([0.0, 1.0]))


def test_add_hard_fails_when_persistence_write_fails(tmp_path, monkeypatch):
    from monkeygrab.adapters.vectorstore import faiss_store as module

    store = FaissVectorStore(_paths(tmp_path, "writefail"))

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(module.faiss, "write_index", _boom)

    with pytest.raises(RuntimeError, match="failed to persist"):
        store.add(_chunk("x.pdf", 0, 0), _unit([1.0, 0.0]))


def test_faiss_and_chroma_are_interchangeable_behind_the_port(tmp_path):
    """Same vectors and metadata through both real adapters -- the query
    order must match exactly, proving they are swappable behind VectorStore.

    Distance *values* are deliberately not compared: Chroma reports its own
    distance metric (squared L2 by default) over the raw vectors it was
    given, while FaissVectorStore reports cosine distance (1 - cosine
    similarity) over its own L2-normalized copy of those vectors (see
    FaissVectorStore.query's docstring note) -- the two numbers are not
    meant to agree. The fixture vectors below are pre-normalized to unit
    length specifically so the *orderings* agree regardless of which metric
    Chroma uses: for unit vectors a and b, ||a - b||^2 == 2 - 2*cos(a, b), a
    strictly monotonic function of cosine similarity, so "nearest by L2" and
    "nearest by cosine" always pick the same order.
    """
    chunks_and_vectors = [
        (_chunk("eq.pdf", 0, 0, "north"), _unit([1.0, 0.0, 0.0])),
        (_chunk("eq.pdf", 0, 1, "east"), _unit([0.0, 1.0, 0.0])),
        (_chunk("eq.pdf", 1, 0, "near-north"), _unit([0.9, 0.1, 0.0])),
        (_chunk("eq.pdf", 1, 1, "far corner"), _unit([-0.3, -0.3, 0.9])),
    ]

    chroma_store = ChromaVectorStore(_paths(tmp_path, "chroma_eq"))
    faiss_store = FaissVectorStore(
        PathsConfig(path_db=str(tmp_path / "faiss_db"), collection_name="faiss_eq")
    )
    for chunk, vector in chunks_and_vectors:
        chroma_store.add(chunk, vector)
        faiss_store.add(chunk, vector)

    query_vector = _unit([0.95, 0.05, 0.0])
    chroma_results = chroma_store.query(query_vector, n_results=4)
    faiss_results = faiss_store.query(query_vector, n_results=4)

    assert [f.id for f in chroma_results] == [f.id for f in faiss_results]

    # get_by_ids: mix existing and nonexistent ids. The port explicitly
    # allows "no particular order" here, so compare as id sets, not lists --
    # and this is exactly the case (a neighbor lookup that partially misses)
    # the two adapters must agree on to be truly interchangeable.
    existing_ids = {chunks_and_vectors[0][0].id, chunks_and_vectors[2][0].id}
    ids_to_fetch = list(existing_ids) + ["missing.pdf_pag9_chunk9"]
    chroma_by_ids = {f.id for f in chroma_store.get_by_ids(ids_to_fetch)}
    faiss_by_ids = {f.id for f in faiss_store.get_by_ids(ids_to_fetch)}
    assert chroma_by_ids == faiss_by_ids == existing_ids


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
