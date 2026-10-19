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


def test_recipe_version_bump_actually_changes_the_recipe_and_the_fingerprint(monkeypatch):
    """Exercises the mechanism itself, not just the constant's current value.

    Bumps _RECIPE_VERSION past whatever it is now and checks the key follows
    and the fingerprint actually moves, so a later refactor that "simplifies"
    the two lines
        if _RECIPE_VERSION != 1:
            recipe["recipe_version"] = _RECIPE_VERSION
    away fails here instead of shipping silently.
    """
    config = AppConfig()
    current_fingerprint = compute_index_fingerprint(config)
    bumped = index_fingerprint._RECIPE_VERSION + 1

    monkeypatch.setattr(index_fingerprint, "_RECIPE_VERSION", bumped)

    assert index_recipe(config)["recipe_version"] == bumped
    assert compute_index_fingerprint(config) != current_fingerprint


def test_default_app_config_fingerprint_is_pinned():
    """Pins the branch every product user actually takes -- contextual ON.

    The other pins here exercise the contextual-OFF path, which is the eval gate's own
    config, not the product default: usar_contextual_retrieval defaults to
    True, so a stray field added inside
    `if config.flags.usar_contextual_retrieval:` would move every real
    user's fingerprint while that other test stayed green. This PR's own
    contextual_num_ctx addition is exactly such a change -- it is the reason
    this value differs from main's 70b88d55bb078bf9 -- and only a pin on the
    default config, not the contextual-off one, would have caught it.

    Value computed on this branch after the default embedder became
    embeddinggemma-2 (issue #292):
        compute_index_fingerprint(AppConfig()) == "41c129b913131f7d"

    and after _RECIPE_VERSION became 2 (code blocks indexed, #366; 1-based
    figure placeholder, #375):
        compute_index_fingerprint(AppConfig()) == "2d135aadd64b8c13"
    """
    assert compute_index_fingerprint(AppConfig()) == "2d135aadd64b8c13"


def test_fingerprint_is_short_hex():
    value = compute_index_fingerprint(AppConfig())
    assert len(value) == 16
    assert all(c in "0123456789abcdef" for c in value)


def test_recipe_is_json_serializable_and_names_the_fixed_stack():
    recipe = index_recipe(AppConfig())
    assert recipe["extractor"] == "mineru"
    assert recipe["embedding"].startswith("google/embeddinggemma-2")


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
