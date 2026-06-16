import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag.chat_pdfs as rag
from rag.engine import lexical


class FakeCollection:
    def __init__(self, docs):
        self.name = "docs_cache_test"
        self.docs = list(docs)

    def count(self):
        return len(self.docs)

    def get(self, limit=None, offset=0, include=None, **kwargs):
        end = None if limit is None else offset + limit
        idx = list(range(len(self.docs)))[offset:end]
        return {
            "documents": self.docs[offset:end],
            "metadatas": [{"source": "a.pdf", "page": 0, "chunk": i} for i in idx],
            "ids": [f"a.pdf_pag0_chunk{i}" for i in idx],
        }


def test_bm25_index_is_cached_and_invalidated():
    """The BM25 index is built once per collection state, not per query.

    Counts how many times ``BM25Okapi`` is constructed: two queries against an
    unchanged collection must build it once (cache hit); changing the chunk
    count must trigger exactly one rebuild (cache invalidation).
    """
    builds = {"n": 0}
    real_bm25 = rag.BM25Okapi

    class CountingBM25(real_bm25):
        def __init__(self, *a, **k):
            builds["n"] += 1
            super().__init__(*a, **k)

    # Patch the runtime source: the per-call global sync copies BM25Okapi from
    # rag.chat_pdfs into rag.engine.lexical, so patching the module is overwritten.
    rag.BM25Okapi = CountingBM25
    lexical._bm25_index_cache.update(key=None, bm25=None, entries=None)
    try:
        coll = FakeCollection([
            "transformer attention mechanism encoder decoder",
            "convolutional neural network image classification",
            "bm25 lexical ranking term frequency document",
        ])

        rag.busqueda_lexica_bm25("transformer attention", coll)
        rag.busqueda_lexica_bm25("neural network image", coll)
        assert builds["n"] == 1, f"BM25 rebuilt {builds['n']}x; expected 1 cache miss"

        coll.docs.append("graph database vector store retrieval")
        rag.busqueda_lexica_bm25("graph database", coll)
        assert builds["n"] == 2, f"BM25 not rebuilt after size change; builds={builds['n']}"
    finally:
        rag.BM25Okapi = real_bm25
        lexical._bm25_index_cache.update(key=None, bm25=None, entries=None)


if __name__ == "__main__":
    test_bm25_index_is_cached_and_invalidated()
    print("ok: BM25 cache hit/invalidation")
