"""Unit tests for monkeygrab.adapters.lexical.bm25_index.Bm25LexicalIndex.

Uses a hand-written fake VectorStore -- no ChromaDB, no chunk collection on
disk -- so these run in milliseconds and never touch real infrastructure.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from monkeygrab.adapters.lexical.bm25_index import Bm25LexicalIndex
from monkeygrab.config.retrieval import RetrievalConfig
from monkeygrab.domain.chunk_metadata import ChunkMetadata
from monkeygrab.domain.fragment import Fragment


class FakeVectorStore:
    """Minimal VectorStore double: only count()/get_page() are used by BM25."""

    def __init__(self, fragments):
        self.fragments = fragments
        self.get_page_calls = 0
        self.raise_on_get_page = None

    def count(self):
        return len(self.fragments)

    def get_page(self, limit, offset):
        self.get_page_calls += 1
        if self.raise_on_get_page:
            raise self.raise_on_get_page
        assert limit is None and offset == 0, "BM25 must scan the whole corpus in one call"
        return self.fragments


def _fragment(text, chunk=0):
    return Fragment(doc=text, metadata=ChunkMetadata(source="doc.pdf", page=0, chunk=chunk))


def test_search_ranks_the_corpus_by_bm25_relevance():
    store = FakeVectorStore([
        _fragment("the transformer architecture uses attention", chunk=0),
        _fragment("bananas are a good source of potassium", chunk=1),
        _fragment("attention attention attention mechanism transformer", chunk=2),
    ])
    index = Bm25LexicalIndex(store, RetrievalConfig())

    results = index.search("transformer attention mechanism", top_n=10)

    assert [f.metadata.chunk for f in results][:2] == [2, 0]  # most term overlap first
    assert all(f.score_keyword == 0.0 for f in results)  # port: ranking only, no fusion score


def test_search_returns_empty_list_for_a_query_with_no_tokens():
    store = FakeVectorStore([_fragment("some content")])

    results = Bm25LexicalIndex(store, RetrievalConfig()).search("   ", top_n=5)

    assert results == []
    assert store.get_page_calls == 0  # never even needs to build the index


def test_search_respects_top_n():
    # A distractor document without the query terms keeps their BM25 idf
    # positive -- if every document shared "keyword match", rank_bm25's idf
    # goes negative/zero (matching busqueda_lexica_bm25's documented "score
    # <= 0 -> excluded" behavior) and no results would be positive at all.
    store = FakeVectorStore(
        [_fragment(f"keyword match number {i}", chunk=i) for i in range(5)]
        + [_fragment("an unrelated distractor document", chunk=99)]
    )

    results = Bm25LexicalIndex(store, RetrievalConfig()).search("keyword match", top_n=2)

    assert len(results) == 2


def test_index_uses_the_injected_bm25_parameters_not_frozen_defaults(monkeypatch):
    """Two indexes built with different k1/b must construct BM25Okapi differently."""
    captured_kwargs = []

    import monkeygrab.adapters.lexical.bm25_index as module

    class RecordingBM25Okapi:
        def __init__(self, corpus, k1, b):
            captured_kwargs.append({"k1": k1, "b": b})
            self._corpus = corpus

        def get_scores(self, query_tokens):
            return [1.0] * len(self._corpus)

    monkeypatch.setattr(module, "BM25Okapi", RecordingBM25Okapi)

    store = FakeVectorStore([_fragment("some searchable content")])
    Bm25LexicalIndex(store, RetrievalConfig(bm25_k1=2.5, bm25_b=0.1)).search("content", top_n=1)
    Bm25LexicalIndex(store, RetrievalConfig(bm25_k1=1.0, bm25_b=0.9)).search("content", top_n=1)

    assert captured_kwargs == [{"k1": 2.5, "b": 0.1}, {"k1": 1.0, "b": 0.9}]


def test_index_is_cached_per_instance_and_rebuilt_only_when_the_corpus_changes():
    store = FakeVectorStore([_fragment("alpha beta gamma")])
    index = Bm25LexicalIndex(store, RetrievalConfig())

    index.search("alpha", top_n=5)
    index.search("beta", top_n=5)
    assert store.get_page_calls == 1  # second search reused the cached index

    store.fragments.append(_fragment("delta epsilon"))
    index.search("delta", top_n=5)
    assert store.get_page_calls == 2  # corpus size changed -> cache invalidated


def test_two_instances_never_share_a_cache():
    """rag/engine/lexical.py's module-global cache is shared process-wide;
    the adapter's replacement must not reproduce that."""
    # Each store needs >=2 distractor docs so the query term's idf stays
    # strictly positive under rank_bm25's Robertson-Sparck-Jones formula
    # (idf is exactly 0 when the term appears in exactly half the corpus,
    # and negative when in more than half -- see test_search_respects_top_n).
    store_a = FakeVectorStore([
        _fragment("alpha content"),
        _fragment("distractor one", chunk=1),
        _fragment("distractor two", chunk=2),
    ])
    store_b = FakeVectorStore([
        _fragment("beta content"),
        _fragment("distractor three", chunk=1),
        _fragment("distractor four", chunk=2),
    ])
    index_a = Bm25LexicalIndex(store_a, RetrievalConfig())
    index_b = Bm25LexicalIndex(store_b, RetrievalConfig())

    results_a = index_a.search("alpha", top_n=5)
    results_b = index_b.search("beta", top_n=5)

    assert len(results_a) == 1 and results_a[0].doc == "alpha content"
    assert len(results_b) == 1 and results_b[0].doc == "beta content"


def test_search_hard_fails_when_the_vector_store_fails():
    """LexicalIndex.search must propagate a storage failure, not fall back to []."""
    store = FakeVectorStore([_fragment("content")])
    store.raise_on_get_page = RuntimeError("vector store unavailable")

    with pytest.raises(RuntimeError, match="vector store unavailable"):
        Bm25LexicalIndex(store, RetrievalConfig()).search("content", top_n=5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
