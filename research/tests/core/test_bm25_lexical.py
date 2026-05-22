import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag.chat_pdfs as rag


class FakeCollection:
    def __init__(self, docs):
        self.docs = docs
        self.metas = [
            {"source": f"doc{i}.pdf", "page": 0, "chunk": i}
            for i in range(len(docs))
        ]
        self.ids = [f"doc{i}.pdf_pag0_chunk{i}" for i in range(len(docs))]

    def count(self):
        return len(self.docs)

    def get(self, limit=None, offset=0, include=None, **kwargs):
        end = None if limit is None else offset + limit
        return {
            "documents": self.docs[offset:end],
            "metadatas": self.metas[offset:end],
            "ids": self.ids[offset:end],
        }


def test_bm25_returns_ranked_positive_match():
    collection = FakeCollection([
        "alpha beta beta",
        "gamma delta",
        "epsilon zeta",
    ])

    results, metrics = rag.busqueda_lexica_bm25("beta", collection)

    assert metrics["documentos_indexados"] == 3
    assert metrics["resultados_totales"] == 1
    assert results[0]["id"] == "doc0.pdf_pag0_chunk0"
    assert results[0]["bm25_score"] > 0


def test_bm25_ignores_empty_query_tokens():
    results, metrics = rag.busqueda_lexica_bm25("the and or", FakeCollection(["alpha beta"]))

    assert results == []
    assert metrics["terminos_query"] == 0


def test_bm25_handles_empty_tokenized_corpus():
    results, metrics = rag.busqueda_lexica_bm25("alpha", FakeCollection(["the and or"]))

    assert results == []
    assert metrics["documentos_indexados"] == 1
