"""Characterization tests for the two post-retrieval filters.

The relevance threshold decides what evidence is good enough to answer from,
and the character budget decides how much of it fits in the prompt. Both are
pure functions of their inputs, so no doubles are needed.

They used to live behind dict-shaped wrappers in ``rag/engine/generation.py``;
the wrappers are gone and the functions are called directly. The boundaries
asserted here are the contract and predate that move: do not edit them.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monkeygrab.application.answer import _limit_by_char_budget
from monkeygrab.application.retrieve import _filter_by_reranker_threshold
from monkeygrab.domain.chunk_metadata import ChunkMetadata
from monkeygrab.domain.fragment import Fragment

THRESHOLD = 0.65


def _fragment(name, doc="", **scores):
    """A fragment whose id starts with ``name``, so assertions stay readable."""
    return Fragment(doc=doc, metadata=ChunkMetadata(source=name, page=0, chunk=0), **scores)


def _names(fragments):
    return [f.metadata.source for f in fragments]


def test_reranker_threshold_keeps_scores_strictly_at_or_above_threshold():
    """Boundary is inclusive (``>=``): a fragment scoring exactly the
    threshold is kept, one just below it is dropped."""
    fragmentos = [
        _fragment("above", score_reranker=0.9),
        _fragment("at_threshold", score_reranker=0.65),
        _fragment("just_below", score_reranker=0.649999),
        _fragment("far_below", score_reranker=0.1),
    ]

    kept = _filter_by_reranker_threshold(fragmentos, True, THRESHOLD)

    assert _names(kept) == ["above", "at_threshold"]


def test_reranker_threshold_falls_back_to_score_final_when_score_reranker_absent():
    """A fragment that was never reranked is still evaluated against the
    threshold, using ``score_final`` (the RRF score) as the stand-in relevance
    signal."""
    fragmentos = [
        _fragment("no_reranker_score_ok", score_final=0.7),
        _fragment("no_reranker_score_low", score_final=0.2),
    ]

    kept = _filter_by_reranker_threshold(fragmentos, True, THRESHOLD)

    assert _names(kept) == ["no_reranker_score_ok"]


def test_reranker_threshold_disabled_flag_passes_everything_through_unfiltered():
    """With reranking off there are no reranker scores to threshold on, so the
    filter is a strict no-op -- even fragments scoring 0 survive."""
    fragmentos = [
        _fragment("a", score_reranker=0.9),
        _fragment("b"),
        _fragment("c", score_final=0.0),
    ]

    kept = _filter_by_reranker_threshold(fragmentos, False, THRESHOLD)

    assert _names(kept) == ["a", "b", "c"]


def test_reranker_threshold_all_below_returns_empty_list():
    fragmentos = [_fragment("a", score_reranker=0.1), _fragment("b", score_reranker=0.2)]

    assert _filter_by_reranker_threshold(fragmentos, True, THRESHOLD) == []


def test_char_budget_no_truncation_when_total_fits():
    fragmentos = [_fragment("a", doc="x" * 50), _fragment("b", doc="y" * 50)]

    kept, n_discarded = _limit_by_char_budget(fragmentos, 100)

    assert kept == fragmentos
    assert n_discarded == 0


def test_char_budget_boundary_is_inclusive_then_drops_the_next_fragment():
    """A running total landing EXACTLY on the budget is kept; the fragment that
    would push it over is dropped, along with everything after it. Fragments
    are consumed in rank order, never re-sorted by size to pack the budget
    better -- the best evidence goes in first."""
    fragmentos = [
        _fragment("a", doc="x" * 60),
        _fragment("b", doc="y" * 40),  # 60 + 40 == 100: exactly at budget, kept
        _fragment("c", doc="z" * 1),   # would push to 101: dropped
    ]

    kept, n_discarded = _limit_by_char_budget(fragmentos, 100)

    assert _names(kept) == ["a", "b"]
    assert n_discarded == 1


def test_char_budget_drops_from_first_fragment_that_overflows_onward():
    """Once one fragment overflows the budget the loop stops; later fragments
    that would individually still fit are NOT backfilled."""
    fragmentos = [
        _fragment("a", doc="x" * 60),  # overflows immediately: 60 > 50
        _fragment("b", doc="y" * 10),  # would fit alone, but never considered
    ]

    kept, n_discarded = _limit_by_char_budget(fragmentos, 50)

    assert kept == []
    assert n_discarded == 2


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
