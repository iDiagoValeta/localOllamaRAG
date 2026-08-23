"""Tests for the index fingerprint: what must and must not invalidate an index."""

import dataclasses

from monkeygrab.application import index_fingerprint
from monkeygrab.application.index_fingerprint import (
    compute_index_fingerprint,
    fingerprint_is_stale,
    index_recipe,
)
from monkeygrab.config.app_config import AppConfig


def _with_contextual_num_ctx(config: AppConfig, value: int) -> AppConfig:
    # models.ollama.* is not part of with_overrides's runtime-mutable surface
    # (see app_config.py's _SECTION_NAMES comment), so this bypasses it and
    # rebuilds the nested dataclass directly.
    ollama = dataclasses.replace(config.models.ollama, contextual_num_ctx=value)
    models = dataclasses.replace(config.models, ollama=ollama)
    return dataclasses.replace(config, models=models)


def test_same_config_gives_same_fingerprint():
    config = AppConfig()
    assert compute_index_fingerprint(config) == compute_index_fingerprint(AppConfig())


def test_chunk_size_change_invalidates():
    base = AppConfig()
    changed = base.with_overrides(**{"chunking.chunk_size": 1000})
    assert compute_index_fingerprint(base) != compute_index_fingerprint(changed)


def test_chunk_overlap_change_invalidates():
    base = AppConfig()
    changed = base.with_overrides(**{"chunking.chunk_overlap": 100})
    assert compute_index_fingerprint(base) != compute_index_fingerprint(changed)


def test_min_chunk_length_change_invalidates():
    base = AppConfig()
    changed = base.with_overrides(**{"chunking.min_chunk_length": 50})
    assert compute_index_fingerprint(base) != compute_index_fingerprint(changed)


def test_index_time_flag_change_invalidates():
    base = AppConfig()
    changed = base.with_overrides(**{"flags.usar_embeddings_imagen": False})
    assert compute_index_fingerprint(base) != compute_index_fingerprint(changed)


def test_retrieval_parameters_do_not_invalidate():
    # Retrieval fan-out is read at query time and never reaches stored chunks.
    # If this ever flips, the loop would reindex on every candidate it tries.
    base = AppConfig()
    changed = base.with_overrides(**{"retrieval.top_k_final": 3, "retrieval.weight_bm25_rrf": 0.9})
    assert compute_index_fingerprint(base) == compute_index_fingerprint(changed)


def test_answer_model_does_not_invalidate():
    base = AppConfig()
    changed = base.with_overrides(**{"models.rag": "some-other-model"})
    assert compute_index_fingerprint(base) == compute_index_fingerprint(changed)


def test_contextual_model_only_counts_when_the_stage_runs():
    off = AppConfig().with_overrides(**{"flags.usar_contextual_retrieval": False})
    off_other_model = off.with_overrides(**{"models.contextual": "some-other-model"})
    assert compute_index_fingerprint(off) == compute_index_fingerprint(off_other_model)

    on = AppConfig().with_overrides(**{"flags.usar_contextual_retrieval": True})
    on_other_model = on.with_overrides(**{"models.contextual": "some-other-model"})
    assert compute_index_fingerprint(on) != compute_index_fingerprint(on_other_model)


def test_contextual_doc_chars_only_counts_when_the_stage_runs():
    off = AppConfig().with_overrides(**{"flags.usar_contextual_retrieval": False})
    off_other_chars = off.with_overrides(**{"chunking.contextual_doc_chars": 500})
    assert compute_index_fingerprint(off) == compute_index_fingerprint(off_other_chars)

    on = AppConfig().with_overrides(**{"flags.usar_contextual_retrieval": True})
    on_other_chars = on.with_overrides(**{"chunking.contextual_doc_chars": 500})
    assert compute_index_fingerprint(on) != compute_index_fingerprint(on_other_chars)


def test_contextual_num_ctx_only_counts_when_the_stage_runs():
    off = AppConfig().with_overrides(**{"flags.usar_contextual_retrieval": False})
    off_other_ctx = _with_contextual_num_ctx(off, 9999)
    assert compute_index_fingerprint(off) == compute_index_fingerprint(off_other_ctx)

    on = AppConfig().with_overrides(**{"flags.usar_contextual_retrieval": True})
    on_other_ctx = _with_contextual_num_ctx(on, 9999)
    assert compute_index_fingerprint(on) != compute_index_fingerprint(on_other_ctx)


def test_description_model_only_counts_when_the_stage_runs():
    off = AppConfig()
    assert off.flags.usar_descripcion_imagen is False
    off_other_model = off.with_overrides(**{"models.chat": "some-other-model"})
    assert compute_index_fingerprint(off) == compute_index_fingerprint(off_other_model)

    on = off.with_overrides(**{"flags.usar_descripcion_imagen": True})
    on_other_model = on.with_overrides(**{"models.chat": "some-other-model"})
    assert compute_index_fingerprint(on) != compute_index_fingerprint(on_other_model)
    # Flipping the flag itself invalidates: stored image-chunk text differs.
    assert compute_index_fingerprint(off) != compute_index_fingerprint(on)


def test_recipe_version_is_omitted_at_v1_and_matches_pre_change_main():
    """recipe_version must not move a single existing fingerprint at v1.

    Uses a contextual-off config specifically so this test isolates
    recipe_version's own effect from the deliberate, unrelated
    contextual_num_ctx addition tested above (issue #37 item 2): that
    addition legitimately changes the fingerprint of a *contextual-on*
    config -- including the all-defaults AppConfig, since
    usar_contextual_retrieval defaults to True. No product code reads or
    writes fingerprint.txt today (only tests/eval/run_eval.py does --
    wiring it into rag/engine/indexing.py and the CLI/web is #36, still
    open), so that particular invalidation costs nothing right now; it will
    once #36 lands. This test is only about recipe_version, which must move
    nothing on its own, ever.

    Value computed on main @ 8b4f16d (before recipe_version or
    contextual_num_ctx existed in the recipe) via:
        compute_index_fingerprint(
            AppConfig().with_overrides(**{"flags.usar_contextual_retrieval": False})
        )
    """
    config = AppConfig().with_overrides(**{"flags.usar_contextual_retrieval": False})
    assert compute_index_fingerprint(config) == "3802ef9e11d5e813"
    assert "recipe_version" not in index_recipe(config)


def test_recipe_version_bump_actually_changes_the_recipe_and_the_fingerprint(monkeypatch):
    """Exercises the mechanism itself, not just its at-rest state at v1.

    test_recipe_version_is_omitted_at_v1_and_matches_pre_change_main above
    only proves the key is absent while _RECIPE_VERSION == 1 -- equally true
    of a tree where the two lines
        if _RECIPE_VERSION != 1:
            recipe["recipe_version"] = _RECIPE_VERSION
    were deleted entirely, or never written. Deleting them leaves the full
    suite green, since nothing else in this module bumps the constant. This
    test bumps it and checks the key actually appears and the fingerprint
    actually moves, so a later refactor that "simplifies" those two lines
    away fails here instead of shipping silently.
    """
    config = AppConfig()
    v1_fingerprint = compute_index_fingerprint(config)

    monkeypatch.setattr(index_fingerprint, "_RECIPE_VERSION", 2)

    assert index_recipe(config)["recipe_version"] == 2
    assert compute_index_fingerprint(config) != v1_fingerprint


def test_default_app_config_fingerprint_is_pinned():
    """Pins the branch every product user actually takes -- contextual ON.

    The pin above (test_recipe_version_is_omitted_at_v1_and_matches_pre_change_main)
    only exercises the contextual-OFF path, which is the eval gate's own
    config, not the product default: usar_contextual_retrieval defaults to
    True, so a stray field added inside
    `if config.flags.usar_contextual_retrieval:` would move every real
    user's fingerprint while that other test stayed green. This PR's own
    contextual_num_ctx addition is exactly such a change -- it is the reason
    this value differs from main's 70b88d55bb078bf9 -- and only a pin on the
    default config, not the contextual-off one, would have caught it.

    Value computed on this branch, after recipe_version + contextual_num_ctx:
        compute_index_fingerprint(AppConfig()) == "b4bc32189fbd3efe"
    """
    assert compute_index_fingerprint(AppConfig()) == "b4bc32189fbd3efe"


def test_fingerprint_is_short_hex():
    value = compute_index_fingerprint(AppConfig())
    assert len(value) == 16
    assert all(c in "0123456789abcdef" for c in value)


def test_recipe_is_json_serializable_and_names_the_fixed_stack():
    recipe = index_recipe(AppConfig())
    assert recipe["extractor"] == "mineru"
    assert recipe["embedding"].startswith("jinaai/jina-clip-v2")


# fingerprint_is_stale -- the product-facing tri-state read of a comparison
# (match / mismatch / unknown), as opposed to run_eval.should_rebuild's binary
# "rebuild unless proven identical" (tests/eval/run_eval.py). The product must
# not warn about an index nobody has touched under this feature yet.


def test_stale_when_stored_disagrees_with_expected():
    assert fingerprint_is_stale("old-recipe", "new-recipe") is True


def test_not_stale_when_stored_matches_expected():
    assert fingerprint_is_stale("same-recipe", "same-recipe") is False


def test_missing_fingerprint_is_unknown_not_stale():
    # Every index built before this feature existed has no fingerprint.txt at
    # all -- that must read as "unknown", never as "mismatch", or upgrading
    # the product would warn every single existing user at their next launch
    # for a config they never touched.
    assert fingerprint_is_stale(None, "any-recipe") is False
