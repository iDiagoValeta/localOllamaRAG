"""Tests for harness/evaluator.py -- issue #31 spec section 6, tests 5-6,
plus the objective/latency helpers those tests and loop.py both rely on.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pytest

from harness import evaluator as ev


# TEST 5 -- no arxiv-sourced case id can reach the objective function.


def test_blind_set_is_disjoint_from_the_search_set():
    assert set(ev.blind_set_case_ids()).isdisjoint(ev.search_set_case_ids())


def test_blind_set_is_disjoint_from_the_fast_tier():
    assert set(ev.blind_set_case_ids()).isdisjoint(ev.load_fast_tier())


def test_search_set_and_blind_set_partition_all_gold_cases():
    all_ids = {c["id"] for c in ev._gold_cases()}
    assert set(ev.search_set_case_ids()) | set(ev.blind_set_case_ids()) == all_ids
    assert set(ev.search_set_case_ids()).isdisjoint(ev.blind_set_case_ids())


def test_every_gold_case_source_is_corpus_or_arxiv():
    # If a third `source` value ever appears, the two functions above would
    # silently stop partitioning all cases -- this pins the assumption they
    # both depend on.
    sources = {c["source"] for c in ev._gold_cases()}
    assert sources == {"corpus", "arxiv"}


def test_fast_tier_loader_rejects_a_blind_set_id(tmp_path):
    bogus = tmp_path / "fast_tier.txt"
    bogus.write_text("resnet-top1-34layer\n", encoding="utf-8")  # a real arxiv-sourced id
    with pytest.raises(ValueError, match="outside the search set"):
        ev.load_fast_tier(bogus)


# TEST 6 -- fast tier is a strict subset of the search set, and every
# case_type in the search set appears in it.


def test_fast_tier_is_a_strict_subset_of_the_search_set():
    fast = set(ev.load_fast_tier())
    search = set(ev.search_set_case_ids())
    assert fast <= search
    assert fast != search


def test_fast_tier_covers_every_case_type_in_the_search_set():
    case_type_by_id = {c["id"]: c["case_type"] for c in ev._gold_cases()}
    search_case_types = {case_type_by_id[i] for i in ev.search_set_case_ids()}
    fast_case_types = {case_type_by_id[i] for i in ev.load_fast_tier()}
    assert fast_case_types == search_case_types


def test_fast_tier_has_no_duplicate_ids():
    fast = ev.load_fast_tier()
    assert len(fast) == len(set(fast))


# UNREACHABLE CASES -- starts empty (see harness/unreachable_cases.txt header).


def test_unreachable_cases_file_starts_empty():
    assert ev.load_unreachable_ids() == ()


def test_unreachable_loader_rejects_an_id_outside_the_search_set(tmp_path):
    bogus = tmp_path / "unreachable_cases.txt"
    bogus.write_text("not-a-real-case-id  fabricated for this test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the search set"):
        ev.load_unreachable_ids(bogus)


# OBJECTIVE: raw vs. unreachable-adjusted.


def _rec(case_id, case_type="factual_number", passed=True, elapsed=200.0):
    return ev.CaseRecord(id=case_id, case_type=case_type, passed=passed, elapsed_seconds=elapsed)


def test_objective_raw_and_adjusted_are_equal_with_no_unreachable_ids():
    records = [_rec("a", passed=True), _rec("b", passed=False), _rec("c", passed=True)]
    raw, adjusted = ev.compute_objective(records, unreachable_ids=())
    assert raw == adjusted == 2


def test_objective_adjusted_excludes_unreachable_cases_from_both_pass_and_total():
    records = [_rec("a", passed=True), _rec("b", passed=True), _rec("c", passed=False)]
    raw, adjusted = ev.compute_objective(records, unreachable_ids=("b",))
    assert raw == 2  # "b" still counts as a pass in the raw count
    assert adjusted == 1  # "b" is dropped entirely from the adjusted count


# LATENCY BUCKETS.


def test_median_latency_by_bucket_splits_answered_from_retrieval_only():
    records = [
        _rec("a", case_type="factual_number", elapsed=200.0),
        _rec("b", case_type="factual_concept", elapsed=210.0),
        _rec("c", case_type="figure_retrieval", elapsed=4.0),
        _rec("d", case_type="table_retrieval", elapsed=6.0),
    ]
    buckets = ev.median_latency_by_bucket(records)
    assert buckets["answered"] == pytest.approx(205.0)
    assert buckets["retrieval_only"] == pytest.approx(5.0)


def test_median_latency_by_bucket_is_none_for_an_empty_bucket():
    records = [_rec("a", case_type="figure_retrieval", elapsed=4.0)]
    buckets = ev.median_latency_by_bucket(records)
    assert buckets["answered"] is None
    assert buckets["retrieval_only"] == pytest.approx(4.0)


def test_retrieval_only_case_types_match_run_eval():
    """Guards against drift between this module's copy and run_eval's private constant.

    run_eval.py imports only argparse/contextlib/json/math/os/shutil/sys/time/
    datetime/pathlib/typing/grade at module level (grade itself is pure stdlib
    regex), so importing it costs nothing extra in the fast CI gate.
    """
    eval_dir = REPO_ROOT / "tests" / "eval"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    import run_eval

    assert ev.RETRIEVAL_ONLY_CASE_TYPES == tuple(run_eval._RETRIEVAL_ONLY_CASE_TYPES)
