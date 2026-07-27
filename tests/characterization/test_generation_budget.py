"""Characterization tests for the post-retrieval filters in
rag/engine/generation.py -- pre-migration snapshot.

Covers ``_filtrar_por_umbral_reranker`` (relevance threshold) and
``_limitar_fragmentos_por_chars`` (context character budget). Both are pure
functions of their inputs plus a couple of cfg flags, so no doubles are
needed beyond monkeypatching those flags.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag.chat_pdfs as rag
from rag.engine.generation import _filtrar_por_umbral_reranker, _limitar_fragmentos_por_chars


def test_reranker_threshold_keeps_scores_strictly_at_or_above_threshold(monkeypatch):
    """Boundary is inclusive (``>=``): a fragment scoring exactly the
    threshold is kept, one just below it is dropped."""
    monkeypatch.setattr(rag, "USAR_RERANKER", True)
    monkeypatch.setattr(rag, "UMBRAL_SCORE_RERANKER", 0.65)

    fragmentos = [
        {"id": "above", "score_reranker": 0.9},
        {"id": "at_threshold", "score_reranker": 0.65},
        {"id": "just_below", "score_reranker": 0.649999},
        {"id": "far_below", "score_reranker": 0.1},
    ]

    kept = _filtrar_por_umbral_reranker(fragmentos)

    assert [f["id"] for f in kept] == ["above", "at_threshold"]


def test_reranker_threshold_falls_back_to_score_final_when_score_reranker_absent(monkeypatch):
    """A fragment that was never reranked (e.g. reranker skipped upstream for
    this candidate) is still evaluated against the threshold, using
    ``score_final`` (RRF score) as the stand-in relevance signal."""
    monkeypatch.setattr(rag, "USAR_RERANKER", True)
    monkeypatch.setattr(rag, "UMBRAL_SCORE_RERANKER", 0.65)

    fragmentos = [
        {"id": "no_reranker_score_ok", "score_final": 0.7},
        {"id": "no_reranker_score_low", "score_final": 0.2},
    ]

    kept = _filtrar_por_umbral_reranker(fragmentos)

    assert [f["id"] for f in kept] == ["no_reranker_score_ok"]


def test_reranker_threshold_disabled_flag_passes_everything_through_unfiltered(monkeypatch):
    """When USAR_RERANKER is off, the threshold filter is a strict no-op --
    even fragments with a score of 0 or missing scores survive."""
    monkeypatch.setattr(rag, "USAR_RERANKER", False)
    monkeypatch.setattr(rag, "UMBRAL_SCORE_RERANKER", 0.65)

    fragmentos = [{"id": "a", "score_reranker": 0.9}, {"id": "b"}, {"id": "c", "score_final": 0.0}]

    kept = _filtrar_por_umbral_reranker(fragmentos)

    assert [f["id"] for f in kept] == ["a", "b", "c"]


def test_reranker_threshold_all_below_returns_empty_list(monkeypatch):
    monkeypatch.setattr(rag, "USAR_RERANKER", True)
    monkeypatch.setattr(rag, "UMBRAL_SCORE_RERANKER", 0.65)

    fragmentos = [{"id": "a", "score_reranker": 0.1}, {"id": "b", "score_reranker": 0.2}]

    assert _filtrar_por_umbral_reranker(fragmentos) == []


def test_char_budget_no_truncation_when_total_fits(monkeypatch):
    monkeypatch.setattr(rag, "MAX_CONTEXTO_CHARS", 100)

    fragmentos = [{"id": "a", "doc": "x" * 50}, {"id": "b", "doc": "y" * 50}]

    kept, n_discarded = _limitar_fragmentos_por_chars(fragmentos)

    assert kept == fragmentos
    assert n_discarded == 0


def test_char_budget_boundary_is_inclusive_then_drops_the_next_fragment(monkeypatch):
    """A running total that lands EXACTLY on MAX_CONTEXTO_CHARS is kept; the
    fragment that would push it over the limit is dropped, along with
    everything after it (fragments are consumed in input order, not
    re-sorted by size to better fill the budget)."""
    monkeypatch.setattr(rag, "MAX_CONTEXTO_CHARS", 100)

    fragmentos = [
        {"id": "a", "doc": "x" * 60},
        {"id": "b", "doc": "y" * 40},  # 60 + 40 == 100: exactly at budget, kept
        {"id": "c", "doc": "z" * 1},   # would push to 101: dropped
    ]

    kept, n_discarded = _limitar_fragmentos_por_chars(fragmentos)

    assert [f["id"] for f in kept] == ["a", "b"]
    assert n_discarded == 1


def test_char_budget_drops_from_first_fragment_that_overflows_onward(monkeypatch):
    """Once one fragment overflows the budget, the loop ``break``s -- later
    fragments that would individually still fit are NOT backfilled."""
    monkeypatch.setattr(rag, "MAX_CONTEXTO_CHARS", 50)

    fragmentos = [
        {"id": "a", "doc": "x" * 60},  # overflows immediately: 60 > 50
        {"id": "b", "doc": "y" * 10},  # would fit alone, but never considered
    ]

    kept, n_discarded = _limitar_fragmentos_por_chars(fragmentos)

    assert kept == []
    assert n_discarded == 2


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
