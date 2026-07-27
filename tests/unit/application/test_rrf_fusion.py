"""Unit tests for monkeygrab.application.rrf_fusion.fuse_semantic_and_keyword.

Reciprocal Rank Fusion is the step that makes hybrid retrieval worth having:
it has to reward a chunk both branches found over a chunk that merely topped
one of them. These tests drive the pure function directly, with hand-built
rankings, so a scoring or ordering change is attributed here rather than
surfacing as a mysterious retrieval regression.

The same two fixtures are exercised end-to-end through the retrieval pipeline
in tests/characterization/test_retrieval_rrf*.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from monkeygrab.application.rrf_fusion import fuse_semantic_and_keyword  # noqa: E402
from monkeygrab.domain.chunk_metadata import ChunkMetadata  # noqa: E402
from monkeygrab.domain.fragment import Fragment  # noqa: E402

RRF_K = 60
W_SEMANTIC = 0.55
W_KEYWORD = 0.45


def rrf(rank: int) -> float:
    """Reciprocal-rank contribution of a hit at ``rank`` (1-based)."""
    return 1.0 / (rank + RRF_K)


def _fragment(name: str, doc: str, distancia: float = 0.5) -> Fragment:
    return Fragment(
        doc=doc,
        metadata=ChunkMetadata(source=f"file{name}.pdf", page=0, chunk=0),
        distancia=distancia,
    )


def _fuse(semantic_hits_per_query, keyword_hits):
    return fuse_semantic_and_keyword(
        semantic_hits_per_query, keyword_hits, RRF_K, W_SEMANTIC, W_KEYWORD
    )


def test_cross_branch_consensus_outranks_a_single_branch_top_hit():
    """docA and docD rank #2/#3 on BOTH branches; docB is #1 on semantic only
    and docC is #1 on BM25 only. Fusion must put the cross-validated docA
    first: agreement between two independent retrievers is stronger evidence
    than one retriever's top rank."""
    semantic = [[
        _fragment("B", "gamma delta content B", 0.10),
        _fragment("A", "alpha beta content A", 0.20),
        _fragment("D", "omega chi content D", 0.30),
    ]]
    keyword = [
        _fragment("C", "alpha epsilon content C"),
        _fragment("A", "alpha beta content A"),
        _fragment("D", "omega chi content D"),
    ]

    fused = _fuse(semantic, keyword)

    assert [f.id for f in fused] == [
        "fileA.pdf_pag0_chunk0",
        "fileD.pdf_pag0_chunk0",
        "fileB.pdf_pag0_chunk0",
        "fileC.pdf_pag0_chunk0",
    ]

    scores = {f.id: f.score_final for f in fused}
    assert scores["fileA.pdf_pag0_chunk0"] == pytest.approx(rrf(2) * W_SEMANTIC + rrf(2) * W_KEYWORD)
    assert scores["fileD.pdf_pag0_chunk0"] == pytest.approx(rrf(3) * W_SEMANTIC + rrf(3) * W_KEYWORD)
    assert scores["fileB.pdf_pag0_chunk0"] == pytest.approx(rrf(1) * W_SEMANTIC)
    assert scores["fileC.pdf_pag0_chunk0"] == pytest.approx(rrf(1) * W_KEYWORD)


def test_scores_accumulate_across_query_variants_and_keep_the_best_distance():
    """A fragment retrieved by several query variants accumulates one
    reciprocal-rank contribution per variant, records every variant that hit
    it, and keeps the smallest distance seen -- not the first."""
    semantic = [
        # Variant 1: fragX at rank 1, fragY at rank 2.
        [_fragment("X", "content X", 0.10), _fragment("Y", "content Y", 0.20)],
        # Variant 2: fragZ at rank 1, fragX again at rank 2.
        [_fragment("Z", "content Z", 0.15), _fragment("X", "content X", 0.25)],
        # Variant 3: fragX again at rank 1, fragW at rank 2.
        [_fragment("X", "content X", 0.05), _fragment("W", "content W", 0.30)],
    ]

    fused = _fuse(semantic, [])
    by_id = {f.id: f for f in fused}

    frag_x = by_id["fileX.pdf_pag0_chunk0"]
    assert frag_x.score_semantic == pytest.approx(2 * rrf(1) + rrf(2))
    assert frag_x.distancia == pytest.approx(0.05)  # best across all three hits
    assert sorted(frag_x.query_matches) == [1, 2, 3]

    assert by_id["fileZ.pdf_pag0_chunk0"].score_semantic == pytest.approx(rrf(1))
    assert by_id["fileZ.pdf_pag0_chunk0"].query_matches == (2,)
    assert by_id["fileY.pdf_pag0_chunk0"].score_semantic == pytest.approx(rrf(2))
    assert by_id["fileY.pdf_pag0_chunk0"].query_matches == (1,)
    assert by_id["fileW.pdf_pag0_chunk0"].score_semantic == pytest.approx(rrf(2))
    assert by_id["fileW.pdf_pag0_chunk0"].query_matches == (3,)

    # With no lexical branch, score_final is purely the weighted semantic score.
    for fragment in fused:
        assert fragment.score_final == pytest.approx(fragment.score_semantic * W_SEMANTIC)

    # fragY and fragW tie on score; ties keep first-seen order, and fragY was
    # inserted during variant 1 while fragW only appears in variant 3.
    assert [f.id for f in fused] == [
        "fileX.pdf_pag0_chunk0", "fileZ.pdf_pag0_chunk0",
        "fileY.pdf_pag0_chunk0", "fileW.pdf_pag0_chunk0",
    ]


def test_empty_input_fuses_to_nothing():
    assert _fuse([], []) == []
    assert _fuse([[]], []) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
