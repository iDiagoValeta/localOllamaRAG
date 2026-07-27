"""Equivalence test: monkeygrab.application.rrf_fusion vs the fusion block
inside rag.engine.retrieval.realizar_busqueda_hibrida, over MULTIPLE semantic
query variants.

``test_rrf_fusion_equivalence.py`` only replays the single-query-variant
characterization fixture. Replays the multi-query fixture from
``tests/characterization/test_retrieval_rrf_multi_query.py`` instead, through
both the moved ``fuse_semantic_and_keyword`` and the untouched original, and
asserts identical order, per-fragment ``score_final`` and ``query_matches`` --
the accumulation-across-variants behavior that fixture exists to characterize.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

import rag.chat_pdfs as rag  # noqa: E402
from rag.engine import retrieval as retrieval_mod  # noqa: E402

from monkeygrab.application.rrf_fusion import fuse_semantic_and_keyword  # noqa: E402
from monkeygrab.domain.chunk_metadata import ChunkMetadata  # noqa: E402
from monkeygrab.domain.fragment import Fragment  # noqa: E402

_SUBQUERY_A = "sub-query A: a different phrasing of the same underlying question"
_SUBQUERY_B = "sub-query B: yet another angle on the same underlying question"


class FakeCollection:
    """Same double as the multi-query characterization test: one fixed,
    pre-ranked semantic result set per call, in call order."""

    _VARIANT_RESULTS = [
        {
            "documents": [["content X", "content Y"]],
            "metadatas": [[
                {"source": "fileX.pdf", "page": 0, "chunk": 0},
                {"source": "fileY.pdf", "page": 0, "chunk": 0},
            ]],
            "distances": [[0.10, 0.20]],
        },
        {
            "documents": [["content Z", "content X"]],
            "metadatas": [[
                {"source": "fileZ.pdf", "page": 0, "chunk": 0},
                {"source": "fileX.pdf", "page": 0, "chunk": 0},
            ]],
            "distances": [[0.15, 0.25]],
        },
        {
            "documents": [["content X", "content W"]],
            "metadatas": [[
                {"source": "fileX.pdf", "page": 0, "chunk": 0},
                {"source": "fileW.pdf", "page": 0, "chunk": 0},
            ]],
            "distances": [[0.05, 0.30]],
        },
    ]

    def __init__(self):
        self.call_count = 0

    def query(self, query_embeddings, n_results, include):
        result = self._VARIANT_RESULTS[self.call_count]
        self.call_count += 1
        return result


def _fake_embeddings(model, prompt, keep_alive):
    return {"embedding": [0.1, 0.2, 0.3]}


def _semantic_fragments_per_variant():
    """The Fragment-shaped equivalent of FakeCollection's three result sets,
    as three separate VectorStore.query() calls would return them
    (nearest-first, rank order, one list per query variant)."""
    return [
        [
            Fragment(doc="content X", metadata=ChunkMetadata(source="fileX.pdf", page=0, chunk=0), distancia=0.10),
            Fragment(doc="content Y", metadata=ChunkMetadata(source="fileY.pdf", page=0, chunk=0), distancia=0.20),
        ],
        [
            Fragment(doc="content Z", metadata=ChunkMetadata(source="fileZ.pdf", page=0, chunk=0), distancia=0.15),
            Fragment(doc="content X", metadata=ChunkMetadata(source="fileX.pdf", page=0, chunk=0), distancia=0.25),
        ],
        [
            Fragment(doc="content X", metadata=ChunkMetadata(source="fileX.pdf", page=0, chunk=0), distancia=0.05),
            Fragment(doc="content W", metadata=ChunkMetadata(source="fileW.pdf", page=0, chunk=0), distancia=0.30),
        ],
    ]


def test_multi_query_fusion_order_scores_and_query_matches_match_the_original_pipeline(monkeypatch):
    long_question = "A" * 61

    monkeypatch.setattr(rag, "USAR_LLM_QUERY_DECOMPOSITION", True)
    monkeypatch.setattr(rag, "USAR_RERANKER", False)
    monkeypatch.setattr(rag, "USAR_BUSQUEDA_HIBRIDA", False)
    monkeypatch.setattr(rag, "generar_queries_con_llm", lambda pregunta: [_SUBQUERY_A, _SUBQUERY_B])
    monkeypatch.setattr(rag, "extraer_keywords", lambda texto: [])
    monkeypatch.setattr(rag, "_validar_coherencia_query", lambda q: True)
    monkeypatch.setattr(retrieval_mod.ollama, "embeddings", _fake_embeddings)

    original_fragmentos, original_mejor_score, _metricas = rag.realizar_busqueda_hibrida(
        long_question, FakeCollection()
    )

    fused = fuse_semantic_and_keyword(
        semantic_hits_per_query=_semantic_fragments_per_variant(),
        keyword_hits=[],
        rrf_k=rag.RRF_K,
        weight_semantic=rag.PESO_SEMANTICO_RRF,
        weight_keyword=rag.PESO_BM25_RRF,
    )

    assert [f.id for f in fused] == [f["id"] for f in original_fragmentos]
    for ours, theirs in zip(fused, original_fragmentos):
        assert ours.score_final == pytest.approx(theirs["score_final"])
        assert ours.distancia == pytest.approx(theirs["distancia"])
        assert sorted(ours.query_matches) == sorted(theirs["query_matches"])
    assert fused[0].score_final == pytest.approx(original_mejor_score)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
