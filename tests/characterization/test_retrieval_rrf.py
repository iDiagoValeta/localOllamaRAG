"""Characterization test for the RRF fusion stage of hybrid retrieval.

Only the fusion math and the resulting ORDER are under test here: semantic
retrieval and embeddings are doubled out, LLM query decomposition and the
Cross-Encoder reranker are disabled via pipeline flags, and BM25 lexical
search is doubled with a canned ranked list (BM25 itself has dedicated
coverage in tests/test_bm25_lexical.py). This isolates RRF scoring and merge
order given fixed semantic and lexical rankings.

The doubles are injected at the port boundary, which is where retrieval now
takes its dependencies. The expected order and scores are the contract and
predate that change: do not edit them. If this test starts failing, the
fusion behavior changed, which is the regression this suite exists to catch.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

import rag.chat_pdfs as rag
from monkeygrab.domain.chunk_metadata import ChunkMetadata
from monkeygrab.domain.fragment import Fragment
from rag.engine import wiring


class FakeCollection:
    """Chroma collection double: returns a fixed, pre-ranked semantic result set
    regardless of the embedding vector it's queried with."""

    def query(self, query_embeddings, n_results, include):
        # Rank 1: docB (semantic-only signal). Rank 2: docA (also hit by BM25).
        # Rank 3: docD (also hit by BM25, weaker on both branches).
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


class FakeEmbedder:
    def embed(self, text, *, keep_alive=None):
        return [0.1, 0.2, 0.3]


class FakeLexicalIndex:
    """BM25 double: fixed ranking with docC as a lexical-only hit."""

    def search(self, query, top_n):
        return [
            Fragment(
                doc="alpha epsilon content C",
                metadata=ChunkMetadata(source="fileC.pdf", page=0, chunk=0),
                distancia=0.5,
            ),
            Fragment(
                doc="alpha beta content A",
                metadata=ChunkMetadata(source="fileA.pdf", page=0, chunk=0),
                distancia=0.5,
            ),
            Fragment(
                doc="omega chi content D",
                metadata=ChunkMetadata(source="fileD.pdf", page=0, chunk=0),
                distancia=0.5,
            ),
        ]


def test_rrf_fusion_favors_cross_branch_consensus_over_single_branch_top_rank(monkeypatch):
    """RRF score = score_semantic * PESO_SEMANTICO_RRF + score_keyword * PESO_BM25_RRF,
    with score_x = sum(1 / (rank + RRF_K)) per branch a document appears in.

    docA and docD each rank #2/#3 on BOTH branches; docB ranks #1 on semantic
    only; docC ranks #1 on BM25 only. Today's fusion formula ranks the
    cross-validated docA above the single-branch top-ranked docB -- this is
    the exact behavior the migration must preserve (or consciously change).
    """
    # Isolate to a single semantic query: no LLM decomposition, no keyword-driven
    # second query variant. Reranker off so no CrossEncoder/torch is touched.
    monkeypatch.setattr(rag, "USAR_LLM_QUERY_DECOMPOSITION", False)
    monkeypatch.setattr(rag, "USAR_RERANKER", False)
    monkeypatch.setattr(rag, "USAR_BUSQUEDA_HIBRIDA", True)
    monkeypatch.setattr("monkeygrab.application.retrieve.extract_keywords", lambda texto: [])
    monkeypatch.setattr(wiring, "embedder", lambda config: FakeEmbedder())
    monkeypatch.setattr(wiring, "lexical_index", lambda collection, config: FakeLexicalIndex())

    fragmentos, mejor_score, metricas = rag.realizar_busqueda_hibrida(
        "alpha query", FakeCollection()
    )

    assert [f["id"] for f in fragmentos] == [
        "fileA.pdf_pag0_chunk0",
        "fileD.pdf_pag0_chunk0",
        "fileB.pdf_pag0_chunk0",
        "fileC.pdf_pag0_chunk0",
    ]

    # Expected scores from today's constants, computed independently of the
    # magic numbers so the test still catches a fusion-formula regression
    # even if RRF_K / the branch weights are re-tuned via env vars.
    k = rag.RRF_K
    w_sem, w_kw = rag.PESO_SEMANTICO_RRF, rag.PESO_BM25_RRF
    rrf = lambda rank: 1.0 / (rank + k)

    scores = {f["id"]: f["score_final"] for f in fragmentos}
    assert scores["fileA.pdf_pag0_chunk0"] == pytest.approx(rrf(2) * w_sem + rrf(2) * w_kw)
    assert scores["fileD.pdf_pag0_chunk0"] == pytest.approx(rrf(3) * w_sem + rrf(3) * w_kw)
    assert scores["fileB.pdf_pag0_chunk0"] == pytest.approx(rrf(1) * w_sem)
    assert scores["fileC.pdf_pag0_chunk0"] == pytest.approx(rrf(1) * w_kw)

    assert mejor_score == scores["fileA.pdf_pag0_chunk0"]
    assert metricas["candidatos_fusion"] == 4
    assert metricas["fase_fusion"]["ambas_ramas"] == 2  # docA, docD
    assert metricas["fase_fusion"]["solo_semantica"] == 1  # docB
    assert metricas["fase_fusion"]["solo_lexica"] == 1  # docC


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
