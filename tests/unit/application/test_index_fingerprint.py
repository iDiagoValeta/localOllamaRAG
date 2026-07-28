"""Tests for the index fingerprint: what must and must not invalidate an index."""

from monkeygrab.application.index_fingerprint import (
    compute_index_fingerprint,
    index_recipe,
)
from monkeygrab.config.app_config import AppConfig


def test_same_config_gives_same_fingerprint():
    config = AppConfig()
    assert compute_index_fingerprint(config) == compute_index_fingerprint(AppConfig())


def test_chunk_size_change_invalidates():
    base = AppConfig()
    changed = base.with_overrides(**{"chunking.chunk_size": 1000})
    assert compute_index_fingerprint(base) != compute_index_fingerprint(changed)


def test_index_time_flag_change_invalidates():
    base = AppConfig()
    changed = base.with_overrides(**{"flags.usar_embeddings_imagen": False})
    assert compute_index_fingerprint(base) != compute_index_fingerprint(changed)


def test_retrieval_parameters_do_not_invalidate():
    # Retrieval fan-out is read at query time and never reaches stored chunks.
    # If this ever flips, the loop would reindex on every candidate it tries.
    base = AppConfig()
    changed = base.with_overrides(
        **{"retrieval.top_k_final": 3, "retrieval.weight_bm25_rrf": 0.9}
    )
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


def test_fingerprint_is_short_hex():
    value = compute_index_fingerprint(AppConfig())
    assert len(value) == 16
    assert all(c in "0123456789abcdef" for c in value)


def test_recipe_is_json_serializable_and_names_the_fixed_stack():
    recipe = index_recipe(AppConfig())
    assert recipe["extractor"] == "mineru"
    assert recipe["embedding"].startswith("jinaai/jina-clip-v2")
