"""Tests for harness/search_space.py -- issue #31 spec section 6, tests 1-4.

No GPU, no Ollama, no model downloads: AppConfig (src/monkeygrab/config/) is
pure stdlib, which is what makes this whole file collectible in the fast CI
gate.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pytest

from monkeygrab.config.app_config import AppConfig
from harness import search_space


# TEST 1 -- space validates against real AppConfig; an unknown field raises.


def test_declared_space_imports_cleanly_against_real_appconfig():
    # Importing this module already ran _validate_declared_space(SEARCH_SPACE)
    # at module load time; getting here at all is half the proof. The other
    # half: every declared key really does resolve on a real AppConfig.
    base = AppConfig()
    for key, values, _why, _stage in search_space.SEARCH_SPACE:
        for value in values:
            base.with_overrides(**{key: value})  # must not raise


def test_unknown_field_raises():
    with pytest.raises(ValueError):
        search_space._validate_declared_space((("retrieval.not_a_real_field", (1,), "bogus", "retrieval"),))


def test_unknown_section_raises():
    with pytest.raises(ValueError):
        search_space._validate_declared_space((("not_a_real_section.foo", (1,), "bogus", "retrieval"),))


# TEST 2 -- every index-time key is rejected as a tunable.


@pytest.mark.parametrize("index_time_key", sorted(search_space.INDEX_TIME_KEYS))
def test_index_time_key_rejected_as_tunable(index_time_key):
    with pytest.raises(ValueError, match="index-time"):
        search_space._validate_declared_space(
            (("retrieval.top_k_final", (8,), "control", "retrieval"), (index_time_key, (1,), "should be rejected", "generation"))
        )


def test_declared_space_contains_no_index_time_key():
    declared = {key for key, _values, _why, _stage in search_space.SEARCH_SPACE}
    assert declared.isdisjoint(search_space.INDEX_TIME_KEYS)


# TEST 3 -- feasibility predicate.


def test_feasibility_rejects_top_k_final_above_rerank_candidates():
    base = AppConfig()
    assert search_space.is_feasible(
        {"retrieval.top_k_final": 16, "retrieval.top_k_rerank_candidates": 12}, base
    ) is False


def test_feasibility_rejects_expansion_wider_than_top_k_final():
    base = AppConfig()
    assert search_space.is_feasible(
        {"retrieval.top_k_final": 4, "retrieval.n_top_for_expansion": 6}, base
    ) is False


def test_feasibility_rejects_expansion_when_flag_is_off():
    base = AppConfig().with_overrides(**{"flags.expandir_contexto": False})
    assert search_space.is_feasible({"retrieval.n_top_for_expansion": 3}, base) is False


def test_feasibility_accepts_a_plain_valid_change():
    base = AppConfig()
    assert search_space.is_feasible({"retrieval.top_k_final": 12}, base) is True


def test_feasibility_rejects_unknown_key():
    base = AppConfig()
    assert search_space.is_feasible({"retrieval.not_a_field": 1}, base) is False


# TEST 4 -- weight_bm25_rrf is always derived as 1 - weight_semantic_rrf.


@pytest.mark.parametrize("weight", [0.3, 0.45, 0.55, 0.7])
def test_weight_bm25_rrf_is_always_one_minus_semantic(weight):
    expanded = search_space.expand_overrides({"retrieval.weight_semantic_rrf": weight})
    assert expanded["retrieval.weight_bm25_rrf"] == pytest.approx(1.0 - weight)


def test_expand_overrides_does_not_override_an_explicit_bm25_weight():
    expanded = search_space.expand_overrides({
        "retrieval.weight_semantic_rrf": 0.3, "retrieval.weight_bm25_rrf": 0.9,
    })
    assert expanded["retrieval.weight_bm25_rrf"] == 0.9


def test_expand_overrides_is_a_no_op_without_the_semantic_weight():
    overrides = {"retrieval.top_k_final": 12}
    assert search_space.expand_overrides(overrides) == overrides


def test_expand_overrides_does_not_mutate_its_argument():
    overrides = {"retrieval.weight_semantic_rrf": 0.3}
    search_space.expand_overrides(overrides)
    assert overrides == {"retrieval.weight_semantic_rrf": 0.3}
