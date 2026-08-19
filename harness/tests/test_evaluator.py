"""Tests for harness/evaluator.py -- issue #31 spec section 6, tests 5-6,
plus the objective/latency helpers those tests and loop.py both rely on.
"""

import sys
import types
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


def test_every_declared_search_space_key_is_honoured_by_the_real_evaluate_contract():
    """Cross-checks search_space.DECLARED_KEYS against tests/eval/run_eval.py's
    real, merged evaluate() contract (issue #31 sibling PR #56, merged as
    commit 8069724) -- not a stub this time, the actual module. #56 built
    describe_config_overrides()/validate_config_overrides() as public,
    pure, side-effect-free functions specifically so this harness could
    check its declared space against evaluate()'s real reachable surface
    (see that function's own docstring) -- both do a plain dict/set lookup,
    no I/O, so this needs no GPU, no Ollama, no model downloads.
    """
    eval_dir = REPO_ROOT / "tests" / "eval"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    import run_eval

    from harness import search_space

    honoured = set(run_eval.describe_config_overrides()["honoured"])
    unreachable = run_eval.describe_config_overrides()["unreachable"]

    missing = {key: unreachable.get(key, "not in evaluate()'s honoured set") for key in search_space.DECLARED_KEYS if key not in honoured}
    assert not missing, f"declared key(s) evaluate() does not honour: {missing}"

    probe_overrides = {key: search_space.ALLOWED_VALUES[key][0] for key in search_space.DECLARED_KEYS}
    run_eval.validate_config_overrides(probe_overrides)  # must not raise


# real_evaluate() -- HIGH 1 regression (#65 PR review): mapped raw["results"]/
# raw.get("effective_config") from an earlier sketch of #56's API. The
# actual contract #56 returns is {"records": [...], "config": {"dev": ...,
# "blind": ...}}; the old mapping raised KeyError on the very first real
# call. A stub `run_eval` module carrying the real shape is injected into
# sys.modules so this is exercised without landing #56 or touching Ollama
# (this round's constraint: no GPU/Ollama -- the full eval gate owns the card).


def _fake_run_eval_module(records, config_dev, seen=None):
    """A stub `run_eval` module whose evaluate() carries #56's real return shape."""
    module = types.ModuleType("run_eval")
    module.DEFAULT_MODELS = ["gemma4:e2b"]

    def evaluate(*, models, case_ids=None, config_overrides=None, update_baseline=False, write_report=True):
        del models, case_ids, config_overrides, update_baseline
        if seen is not None:
            seen["write_report"] = write_report
        return {"records": records, "config": {"dev": config_dev, "blind": {}}}

    module.evaluate = evaluate
    return module


def test_real_evaluate_maps_the_actual_56_contract(monkeypatch):
    records = [
        {"id": "a", "case_type": "factual_number", "passed": True, "elapsed_seconds": 200.0},
        {
            "id": "b", "case_type": "figure_retrieval", "passed": False, "elapsed_seconds": 4.0,
            "infrastructure_error": True,
        },
    ]
    config_dev = {"retrieval": {"top_k_final": 8}}
    seen = {}
    monkeypatch.setitem(sys.modules, "run_eval", _fake_run_eval_module(records, config_dev, seen))

    result = ev.real_evaluate({"retrieval.top_k_final": 8}, ("a", "b"))

    assert result.effective_config == config_dev
    by_id = {r.id: r for r in result.records}
    assert by_id["a"].passed is True
    assert by_id["a"].infrastructure_error is False
    assert by_id["b"].infrastructure_error is True
    # The harness ledger is the evidence; the first real run (issue #71)
    # wrote five extra JSONs under tests/eval/runs/ because this defaulted
    # to True -- including a 0-case reachability probe.
    assert seen["write_report"] is False


def test_real_evaluate_raises_not_implemented_without_an_evaluate_function(monkeypatch):
    stub = types.ModuleType("run_eval")  # no .evaluate attribute -- matches pre-#56 main
    monkeypatch.setitem(sys.modules, "run_eval", stub)

    with pytest.raises(NotImplementedError):
        ev.real_evaluate({}, ())


# CRITERION 7 -- reconstruct-and-rerun pass vector.


def test_replay_identical_when_every_passed_bit_matches():
    stored = [
        {"id": "a", "passed": True},
        {"id": "b", "passed": False},
    ]

    def evaluate(overrides, case_ids):
        assert list(case_ids) == ["a", "b"]
        assert dict(overrides) == {"retrieval.top_k_final": 4}
        return ev.EvaluationResult(
            records=(
                _rec("a", passed=True),
                _rec("b", passed=False),
            ),
            effective_config={},
        )

    result = ev.replay(evaluate, {"retrieval.top_k_final": 4}, stored)
    assert result.identical is True
    assert result.flips == ()
    assert result.missing == ()
    assert result.extra == ()
    assert result.infrastructure_errors == ()


def test_replay_reports_a_classification_flip():
    stored = [{"id": "a", "passed": True}, {"id": "b", "passed": False}]

    def evaluate(overrides, case_ids):
        del overrides, case_ids
        return ev.EvaluationResult(
            records=(_rec("a", passed=True), _rec("b", passed=True)),
            effective_config={},
        )

    result = ev.replay(evaluate, {}, stored)
    assert result.identical is False
    assert result.flips == ("b",)


def test_replay_reports_missing_and_extra_ids():
    stored = [{"id": "a", "passed": True}]

    def evaluate(overrides, case_ids):
        del overrides, case_ids
        return ev.EvaluationResult(records=(_rec("b", passed=True),), effective_config={})

    result = ev.replay(evaluate, {}, stored)
    assert result.missing == ("a",)
    assert result.extra == ("b",)
    assert result.identical is False


def test_replay_treats_an_infrastructure_error_as_not_identical():
    stored = [{"id": "a", "passed": True}]

    def evaluate_broken(overrides, case_ids):
        del overrides, case_ids
        return ev.EvaluationResult(
            records=(
                ev.CaseRecord(
                    id="a", case_type="factual_number", passed=True,
                    elapsed_seconds=1.0, infrastructure_error=True,
                ),
            ),
            effective_config={},
        )

    result = ev.replay(evaluate_broken, {}, stored)
    assert result.identical is False
    assert result.infrastructure_errors == ("a",)
