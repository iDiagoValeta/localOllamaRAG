"""The slow tier's accounting, and the two things it refuses to guess.

Issue #102. The arithmetic is easy; what these pin is the economics. Batching
by *recipe* rather than by candidate is the whole reason the tier could ever be
affordable — two candidates differing only in a retrieval knob read the same
index — and a cost model that invents its own constants would be a measurement
that lies, so ``estimate_cost`` takes both costs from the caller and refuses a
non-positive one instead of falling back to a plausible default.
"""

import pytest

from harness import slow_tier
from harness.search_space import INDEX_TIME_KEYS, SEARCH_SPACE


class TestWhichTierACandidateBelongsTo:
    def test_a_retrieval_only_candidate_needs_no_reindex(self):
        assert slow_tier.requires_reindex({"retrieval.top_k_final": 8}) is False

    def test_touching_one_index_time_key_is_enough(self):
        assert slow_tier.requires_reindex({"chunking.chunk_size": 800}) is True

    def test_a_mixed_candidate_is_slow_tier(self):
        assert slow_tier.requires_reindex(
            {"retrieval.top_k_final": 8, "flags.usar_embeddings_imagen": True}
        ) is True

    def test_an_empty_candidate_needs_no_reindex(self):
        assert slow_tier.requires_reindex({}) is False


class TestBatchingIsByIndexNotByCandidate:
    def test_candidates_sharing_an_index_share_a_batch(self):
        # The economics of the whole tier: these two read one index.
        batches = slow_tier.batch_by_recipe([
            {"chunking.chunk_size": 800, "retrieval.top_k_final": 4},
            {"chunking.chunk_size": 800, "retrieval.top_k_final": 8},
        ])
        assert len(batches) == 1
        assert len(batches[0][1]) == 2

    def test_different_indexes_are_different_batches(self):
        batches = slow_tier.batch_by_recipe([
            {"chunking.chunk_size": 800},
            {"chunking.chunk_size": 1200},
        ])
        assert len(batches) == 2

    def test_key_order_does_not_split_a_batch(self):
        batches = slow_tier.batch_by_recipe([
            {"chunking.chunk_size": 800, "chunking.chunk_overlap": 100},
            {"chunking.chunk_overlap": 100, "chunking.chunk_size": 800},
        ])
        assert len(batches) == 1

    def test_fast_tier_candidates_get_their_own_empty_recipe(self):
        # So a caller can hand over a mixed list and still get a true bill.
        batches = slow_tier.batch_by_recipe([
            {"retrieval.top_k_final": 4},
            {"chunking.chunk_size": 800},
        ])
        recipes = [recipe for recipe, _ in batches]
        assert () in recipes
        assert len(recipes) == 2

    def test_first_seen_order_is_preserved(self):
        # Re-ordering here would quietly override the proposer's own decision.
        batches = slow_tier.batch_by_recipe([
            {"chunking.chunk_size": 1200},
            {"chunking.chunk_size": 800},
        ])
        assert [recipe[0][1] for recipe, _ in batches] == [1200, 800]


class TestTheBill:
    def test_one_reindex_is_charged_per_recipe_not_per_candidate(self):
        batches = slow_tier.batch_by_recipe([
            {"chunking.chunk_size": 800, "retrieval.top_k_final": 4},
            {"chunking.chunk_size": 800, "retrieval.top_k_final": 8},
        ])
        cost = slow_tier.estimate_cost(batches, reindex_seconds=215, evaluation_seconds=10)
        assert cost.reindexes == 1
        assert cost.evaluations == 2
        assert cost.seconds == 215 + 20

    def test_a_fast_tier_batch_is_charged_no_rebuild(self):
        batches = slow_tier.batch_by_recipe([{"retrieval.top_k_final": 4}])
        cost = slow_tier.estimate_cost(batches, reindex_seconds=215, evaluation_seconds=10)
        assert cost.reindexes == 0
        assert cost.seconds == 10

    def test_the_bill_is_expressed_in_the_unit_the_budget_uses(self):
        # 215 s of rebuild plus one 10 s evaluation is 22.5 ordinary
        # evaluations -- the number an operator weighs against a patience of 3.
        batches = slow_tier.batch_by_recipe([{"chunking.chunk_size": 800}])
        cost = slow_tier.estimate_cost(batches, reindex_seconds=215, evaluation_seconds=10)
        assert cost.fast_tier_equivalents == 22.5

    def test_the_measured_2026_08_23_numbers_come_out_of_the_model(self):
        # Two stores at 215 s each, as the issue recorded, plus the four
        # evaluations that flip-flopped over them.
        batches = slow_tier.batch_by_recipe([
            {"flags.usar_embeddings_imagen": False},
            {"flags.usar_embeddings_imagen": False},
            {"flags.usar_embeddings_imagen": True},
            {"flags.usar_embeddings_imagen": True},
        ])
        cost = slow_tier.estimate_cost(batches, reindex_seconds=215, evaluation_seconds=10)
        assert cost.reindexes == 2
        assert cost.seconds == 2 * 215 + 4 * 10


class TestWhatItRefusesToGuess:
    @pytest.mark.parametrize("reindex, evaluation", [(0, 10), (-1, 10), (215, 0), (215, -5)])
    def test_a_non_positive_cost_raises_rather_than_defaulting(self, reindex, evaluation):
        # A zero evaluation cost makes every batch look free; a default anyone
        # could have picked is a measurement that lies.
        batches = slow_tier.batch_by_recipe([{"chunking.chunk_size": 800}])
        with pytest.raises(slow_tier.SlowTierError):
            slow_tier.estimate_cost(
                batches, reindex_seconds=reindex, evaluation_seconds=evaluation
            )

    def test_no_default_reindex_cost_is_importable(self):
        # If a constant appeared here, someone would use it as a measurement.
        assert not any(
            "SECONDS" in name and not name.startswith("_") for name in dir(slow_tier)
        )


class TestItStaysOutOfStageOne:
    def test_every_declared_slow_tier_key_is_an_index_time_key(self):
        assert set(slow_tier.SLOW_TIER_SPACE) <= set(INDEX_TIME_KEYS)

    def test_no_slow_tier_key_leaks_into_the_searched_space(self):
        # Declaring a knob here must not make stage 1 try to search it.
        assert set(slow_tier.SLOW_TIER_SPACE).isdisjoint(set(SEARCH_SPACE))


def test_the_cost_line_names_both_units():
    batches = slow_tier.batch_by_recipe([{"chunking.chunk_size": 800}])
    line = slow_tier.describe_cost(
        slow_tier.estimate_cost(batches, reindex_seconds=215, evaluation_seconds=10)
    )
    assert "reindex" in line and "fast-tier" in line
