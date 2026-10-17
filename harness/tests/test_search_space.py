"""Tests for harness/search_space.py -- issue #31 spec section 6, tests 1-4.

No GPU, no Ollama, no model downloads: AppConfig (src/monkeygrab/config/) is
pure stdlib, which is what makes this whole file collectible in the fast CI
gate. Test 5 (issue #340) derives its key set from ``index_recipe`` instead of
a copied list, so INDEX_TIME_KEYS cannot drift from the real index recipe.
"""

import dataclasses
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pytest

from monkeygrab.application.index_fingerprint import index_recipe
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


# TEST 5 -- every config key that changes index_recipe is excluded from stage 1.


def _perturbed(value):
    """Return a different value of the same scalar type.

    Every field ``with_overrides`` accepts as a single tunable is a bool, int,
    float or str; nested dataclasses are whole sub-configs, not tunables. Any
    other type means the field surface changed and this probe must be extended
    rather than silently skipping the key.

    Args:
        value: The current value of a scalar config field.

    Returns:
        A value guaranteed to differ from ``value``.

    Raises:
        AssertionError: ``value`` is not a scalar this probe knows how to move.
    """
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.1
    if isinstance(value, str):
        return value + "-probe"
    raise AssertionError(f"no perturbation defined for {value!r}")


def _scalar_config_keys(config):
    """Every ``section.field`` key holding a scalar, derived from the dataclasses.

    Derived, not declared: a hand-written list here would be the same drift the
    test exists to catch. Nested dataclass fields (``models.ollama`` and
    friends) are skipped because a candidate override names a scalar field, and
    ``with_overrides``'s one-level nesting puts their inner fields out of reach
    -- see ``tests/eval/run_eval.py``'s ``_UNREACHABLE_CONFIG_OVERRIDE_REASONS``.

    Args:
        config: Config whose section dataclasses are walked.

    Returns:
        ``{dotted_key: current_value}`` for every scalar field.
    """
    keys = {}
    for section_field in dataclasses.fields(config):
        section = getattr(config, section_field.name)
        for field in dataclasses.fields(section):
            value = getattr(section, field.name)
            if dataclasses.is_dataclass(value):
                continue
            keys[f"{section_field.name}.{field.name}"] = value
    return keys


def _configs_with_each_flag_flipped(config):
    """``config`` plus one copy per boolean pipeline flag flipped.

    Some recipe members are conditional on a flag (contextual summaries, figure
    descriptions), and the default is off for the latter: probing only the
    default config would miss exactly the key #340 was filed about.

    Args:
        config: The reference config.

    Returns:
        A list starting with ``config``, followed by one variant per bool flag.
    """
    configs = [config]
    for flag in dataclasses.fields(config.flags):
        value = getattr(config.flags, flag.name)
        if isinstance(value, bool):
            configs.append(config.with_overrides(**{f"flags.{flag.name}": not value}))
    return configs


def test_every_key_that_changes_index_recipe_is_index_time():
    """Guard against INDEX_TIME_KEYS drifting from the real index recipe.

    Derived from ``index_recipe``'s behaviour, not from a copied key list:
    every scalar key on ``AppConfig`` is probed by perturbing it and asking
    whether the recipe moved, and a conditional member is probed under each
    flag it might depend on. Any key that moves the recipe must be excluded
    from stage 1, or ``slow_tier`` prices its reindex as if it were free
    (issue #340, where ``flags.usar_descripcion_imagen`` was missing).
    """
    bases = _configs_with_each_flag_flipped(AppConfig())
    recipe_moving = set()
    for key in _scalar_config_keys(AppConfig()):
        for base in bases:
            changed = base.with_overrides(**{key: _perturbed(_scalar_config_keys(base)[key])})
            if index_recipe(base) != index_recipe(changed):
                recipe_moving.add(key)
                break

    assert recipe_moving, "the probe moved no key at all -- its derivation is broken, not the space"
    missing = recipe_moving - set(search_space.INDEX_TIME_KEYS)
    assert not missing, (
        f"config key(s) change index_recipe but are not in INDEX_TIME_KEYS: {sorted(missing)}. "
        "slow_tier would bill a candidate that overrides one as fast tier, though every stored "
        "chunk must be rebuilt."
    )
