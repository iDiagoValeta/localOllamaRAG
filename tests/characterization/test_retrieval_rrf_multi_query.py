"""Characterization test for RRF fusion over MULTIPLE semantic query variants
in ``rag.engine.retrieval.realizar_busqueda_hibrida`` -- pre-migration snapshot.

``tests/characterization/test_retrieval_rrf.py`` disables LLM query
decomposition entirely to isolate the fusion math with a SINGLE semantic
query variant. That leaves the accumulation behavior across MULTIPLE query
variants uncharacterized: the sum of ``1 / (rank + RRF_K)`` for a fragment
that appears in more than one variant's semantic results, the ``min()`` of
``distancia`` taken across variants, and the population of ``query_matches``
with every variant index that hit a fragment (``rag/engine/retrieval.py``,
the loop starting at "for q_idx, query in enumerate(queries):"). This is
exactly the part a reimplementation could get wrong without any test
noticing: BM25 and the Cross-Encoder reranker both have their own dedicated
suites, but nothing exercised multi-variant semantic accumulation directly.

Do not edit the expected scores/order/query_matches during a migration: if
this test starts failing, the multi-query fusion behavior changed, which is
the regression this suite exists to catch.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

import rag.chat_pdfs as rag
from rag.engine import retrieval as retrieval_mod

# Three semantic query variants: the original question (variant 1) plus two
# LLM-decomposed sub-queries (variants 2 and 3). Design:
#   - fragX: hit by all three variants (rank 1, rank 2, rank 1) -- exercises
#     RRF accumulation across variants AND the running min() of distancia
#     (0.10 / 0.25 / 0.05 -> min is 0.05, from variant 3, not variant 1).
#   - fragZ: hit only by variant 2 (rank 1).
#   - fragY / fragW: each hit by exactly one variant at rank 2 (variants 1
#     and 3 respectively) -- same score, exercises the fusion's stable-sort
#     tie-break (first-seen order wins).
_VARIANT_RESULTS = [
    {  # variant 1: the original question
        "documents": [["content X", "content Y"]],
        "metadatas": [[
            {"source": "fileX.pdf", "page": 0, "chunk": 0},
            {"source": "fileY.pdf", "page": 0, "chunk": 0},
        ]],
        "distances": [[0.10, 0.20]],
    },
    {  # variant 2: sub-query A
        "documents": [["content Z", "content X"]],
        "metadatas": [[
            {"source": "fileZ.pdf", "page": 0, "chunk": 0},
            {"source": "fileX.pdf", "page": 0, "chunk": 0},
        ]],
        "distances": [[0.15, 0.25]],
    },
    {  # variant 3: sub-query B
        "documents": [["content X", "content W"]],
        "metadatas": [[
            {"source": "fileX.pdf", "page": 0, "chunk": 0},
            {"source": "fileW.pdf", "page": 0, "chunk": 0},
        ]],
        "distances": [[0.05, 0.30]],
    },
]

_SUBQUERY_A = "sub-query A: a different phrasing of the same underlying question"
_SUBQUERY_B = "sub-query B: yet another angle on the same underlying question"


class FakeCollection:
    """Returns one fixed, pre-ranked result set per call, in call order --
    one call per semantic query variant, matching realizar_busqueda_hibrida's
    per-query-variant loop."""

    def __init__(self):
        self.call_count = 0

    def query(self, query_embeddings, n_results, include):
        result = _VARIANT_RESULTS[self.call_count]
        self.call_count += 1
        return result


def _fake_embeddings(model, prompt, keep_alive):
    return {"embedding": [0.1, 0.2, 0.3]}


def test_rrf_accumulates_across_multiple_semantic_query_variants(monkeypatch):
    long_question = "A" * 61  # > 60 chars: required to trigger query decomposition

    monkeypatch.setattr(rag, "USAR_LLM_QUERY_DECOMPOSITION", True)
    monkeypatch.setattr(rag, "USAR_RERANKER", False)
    monkeypatch.setattr(rag, "USAR_BUSQUEDA_HIBRIDA", False)
    monkeypatch.setattr(rag, "generar_queries_con_llm", lambda pregunta: [_SUBQUERY_A, _SUBQUERY_B])
    monkeypatch.setattr(rag, "extraer_keywords", lambda texto: [])
    monkeypatch.setattr(rag, "_validar_coherencia_query", lambda q: True)
    monkeypatch.setattr(retrieval_mod.ollama, "embeddings", _fake_embeddings)

    fragmentos, mejor_score, metricas = rag.realizar_busqueda_hibrida(long_question, FakeCollection())

    # Three query variants reached the semantic search loop: original + 2 sub-queries.
    assert metricas["queries_semanticas"] == [long_question, _SUBQUERY_A, _SUBQUERY_B]

    by_id = {f["id"]: f for f in fragmentos}
    k = rag.RRF_K
    w_sem = rag.PESO_SEMANTICO_RRF
    rrf = lambda rank: 1.0 / (rank + k)

    # fragX: variant1 rank1, variant2 rank2, variant3 rank1.
    assert by_id["fileX.pdf_pag0_chunk0"]["score_semantic"] == pytest.approx(2 * rrf(1) + rrf(2))
    assert by_id["fileX.pdf_pag0_chunk0"]["distancia"] == pytest.approx(0.05)  # min across all 3 hits
    assert sorted(by_id["fileX.pdf_pag0_chunk0"]["query_matches"]) == [1, 2, 3]

    # fragZ: variant2 rank1 only.
    assert by_id["fileZ.pdf_pag0_chunk0"]["score_semantic"] == pytest.approx(rrf(1))
    assert by_id["fileZ.pdf_pag0_chunk0"]["query_matches"] == [2]

    # fragY: variant1 rank2 only. fragW: variant3 rank2 only. Same score.
    assert by_id["fileY.pdf_pag0_chunk0"]["score_semantic"] == pytest.approx(rrf(2))
    assert by_id["fileY.pdf_pag0_chunk0"]["query_matches"] == [1]
    assert by_id["fileW.pdf_pag0_chunk0"]["score_semantic"] == pytest.approx(rrf(2))
    assert by_id["fileW.pdf_pag0_chunk0"]["query_matches"] == [3]

    # Hybrid search is off, so score_final is purely the weighted semantic score.
    for frag in fragmentos:
        assert frag["score_final"] == pytest.approx(frag["score_semantic"] * w_sem)

    # Order: fragX (highest) > fragZ (single rank-1 hit) > fragY/fragW tied at
    # rank-2 -- ties keep first-seen order (fragY inserted during variant 1,
    # before fragW is inserted during variant 3).
    assert [f["id"] for f in fragmentos] == [
        "fileX.pdf_pag0_chunk0", "fileZ.pdf_pag0_chunk0",
        "fileY.pdf_pag0_chunk0", "fileW.pdf_pag0_chunk0",
    ]
    assert mejor_score == by_id["fileX.pdf_pag0_chunk0"]["score_final"]
    assert metricas["candidatos_fusion"] == 4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
