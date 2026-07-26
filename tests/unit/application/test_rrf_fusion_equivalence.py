"""Equivalence test: monkeygrab.application.rrf_fusion vs the fusion block
inside rag.engine.retrieval.realizar_busqueda_hibrida.

Replays the exact fixture from
``tests/characterization/test_retrieval_rrf.py`` through both the moved
``fuse_semantic_and_keyword`` and the untouched original, and asserts
identical order and per-fragment ``score_final``. The original's fusion
block is not independently callable (it is inline in
``realizar_busqueda_hibrida``), so this test drives the original through its
public entry point with semantic search and BM25 doubled out exactly as the
characterization test does, and drives the new function with the same two
ranked lists built directly (skipping the doubled Chroma/BM25 calls that
exist only to feed the fusion block).
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


class FakeCollection:
    """Same double as the characterization test: one fixed, pre-ranked semantic result set."""

    def query(self, query_embeddings, n_results, include):
        return {
            "documents": [[
                "gamma delta content B",
                "alpha beta content A",
                "omega chi content D",
            ]],
            "metadatas": [[
                {"source": "fileB.pdf", "page": 0, "chunk": 0},
                {"source": "fileA.pdf", "page": 0, "chunk": 0},
                {"source": "fileD.pdf", "page": 0, "chunk": 0},
            ]],
            "distances": [[0.10, 0.20, 0.30]],
        }


def _fake_bm25(pregunta, collection):
    results = [
        {
            "doc": "alpha epsilon content C", "distancia": 0.5, "bm25_score": 3.0,
            "id": "fileC.pdf_pag0_chunk0",
            "metadata": {"source": "fileC.pdf", "page": 0, "chunk": 0},
        },
        {
            "doc": "alpha beta content A", "distancia": 0.5, "bm25_score": 2.0,
            "id": "fileA.pdf_pag0_chunk0",
            "metadata": {"source": "fileA.pdf", "page": 0, "chunk": 0},
        },
        {
            "doc": "omega chi content D", "distancia": 0.5, "bm25_score": 1.0,
            "id": "fileD.pdf_pag0_chunk0",
            "metadata": {"source": "fileD.pdf", "page": 0, "chunk": 0},
        },
    ]
    metrics = {
        "disponible": True, "documentos_indexados": 3,
        "terminos_query": 1, "resultados_totales": 3, "mejor_score": 3.0,
    }
    return results, metrics


def _fake_embeddings(model, prompt, keep_alive):
    return {"embedding": [0.1, 0.2, 0.3]}


def _semantic_fragments_single_query():
    """The Fragment-shaped equivalent of FakeCollection.query's single result,
    as a VectorStore.query() call would return it (nearest-first, rank order)."""
    return [
        Fragment(doc="gamma delta content B", metadata=ChunkMetadata(source="fileB.pdf", page=0, chunk=0), distancia=0.10),
        Fragment(doc="alpha beta content A", metadata=ChunkMetadata(source="fileA.pdf", page=0, chunk=0), distancia=0.20),
        Fragment(doc="omega chi content D", metadata=ChunkMetadata(source="fileD.pdf", page=0, chunk=0), distancia=0.30),
    ]


def _keyword_fragments():
    """The Fragment-shaped equivalent of _fake_bm25's result, as a
    LexicalIndex.search() call would return it (best-first)."""
    return [
        Fragment(doc="alpha epsilon content C", metadata=ChunkMetadata(source="fileC.pdf", page=0, chunk=0)),
        Fragment(doc="alpha beta content A", metadata=ChunkMetadata(source="fileA.pdf", page=0, chunk=0)),
        Fragment(doc="omega chi content D", metadata=ChunkMetadata(source="fileD.pdf", page=0, chunk=0)),
    ]


def test_fusion_order_and_scores_match_the_original_pipeline(monkeypatch):
    monkeypatch.setattr(rag, "USAR_LLM_QUERY_DECOMPOSITION", False)
    monkeypatch.setattr(rag, "USAR_RERANKER", False)
    monkeypatch.setattr(rag, "USAR_BUSQUEDA_HIBRIDA", True)
    monkeypatch.setattr(rag, "extraer_keywords", lambda texto: [])
    monkeypatch.setattr(rag, "busqueda_lexica_bm25", _fake_bm25)
    monkeypatch.setattr(retrieval_mod.ollama, "embeddings", _fake_embeddings)

    original_fragmentos, original_mejor_score, _metricas = rag.realizar_busqueda_hibrida(
        "alpha query", FakeCollection()
    )

    fused = fuse_semantic_and_keyword(
        semantic_hits_per_query=[_semantic_fragments_single_query()],
        keyword_hits=_keyword_fragments(),
        rrf_k=rag.RRF_K,
        weight_semantic=rag.PESO_SEMANTICO_RRF,
        weight_keyword=rag.PESO_BM25_RRF,
    )

    assert [f.id for f in fused] == [f["id"] for f in original_fragmentos]
    for ours, theirs in zip(fused, original_fragmentos):
        assert ours.score_final == pytest.approx(theirs["score_final"])
    assert fused[0].score_final == pytest.approx(original_mejor_score)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
