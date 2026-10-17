"""BM25 cache invalidation when a re-added source keeps the chunk count (#387).

Deleting one document and re-adding it with different text leaves the corpus
the same size. ``Bm25LexicalIndex`` used to key its cached snapshot on
``(count, k1, b)`` alone, so that edit was invisible to it: BM25 kept citing
the deleted text until a chunk count happened to change. The vector store now
exposes a monotonic version counter, bumped on every mutation, and BM25 folds
it into the cache key.

Uses a real ``FaissVectorStore`` on ``tmp_path`` (no GPU, no Ollama): the bug
is precisely about the store's own invalidation signal, so a hand-written
double with the same counter would test the fixture, not the adapter.
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monkeygrab.adapters.lexical.bm25_index import Bm25LexicalIndex
from monkeygrab.adapters.vectorstore.faiss_store import FaissVectorStore
from monkeygrab.config.paths import PathsConfig
from monkeygrab.config.retrieval import RetrievalConfig
from monkeygrab.domain.chunk import Chunk
from monkeygrab.domain.chunk_metadata import ChunkMetadata


def _chunk(source, page, chunk_idx, text):
    return Chunk(text=text, metadata=ChunkMetadata(source=source, page=page, chunk=chunk_idx))


def _unit(vector):
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


def test_readding_a_source_with_new_text_refreshes_the_bm25_snapshot(tmp_path):
    paths = PathsConfig(path_db=str(tmp_path / "db"), collection_name="bm25-version")
    store = FaissVectorStore(paths)
    store.add(_chunk("doc.pdf", 0, 0, "obsolete quokkas phrasing"), _unit([1.0, 0.0, 0.0]))
    # Two distractors keep the query term's BM25 idf strictly positive: at
    # N=2 with the term in exactly one document, idf is log(1) == 0 and the
    # result is filtered out as a non-match.
    store.add(_chunk("d1.pdf", 0, 0, "unrelated wombats material"), _unit([0.0, 1.0, 0.0]))
    store.add(_chunk("d2.pdf", 0, 0, "another captive paragraph"), _unit([0.0, 0.0, 1.0]))
    index = Bm25LexicalIndex(store, RetrievalConfig())

    first = index.search("quokkas", top_n=5)
    assert [f.doc for f in first] == ["obsolete quokkas phrasing"]

    assert store.delete_source("doc.pdf") == 1
    store.add(_chunk("doc.pdf", 0, 0, "replacement numbars wording"), _unit([1.0, 0.0, 0.0]))

    # The premises of the regression: same chunk count, different text.
    assert store.count() == 3

    assert index.search("quokkas", top_n=5) == []
    results = index.search("numbars", top_n=5)
    assert [f.doc for f in results] == ["replacement numbars wording"]
