"""Tests for harness/loop.py -- issue #31 spec section 6, tests 9 and 11,
plus regression/acceptance coverage for the same state machine.

Every evaluator here is a small deterministic in-test double -- exactly the
"deterministic fake evaluator" spec section 6's preamble asks for -- so none
of this needs a GPU, Ollama or PDFs.
"""

import itertools
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pytest

from monkeygrab.config.app_config import AppConfig
from harness import evaluator as ev
from harness import loop


def _rec(case_id, case_type, passed, elapsed, infra_error=False):
    return ev.CaseRecord(
        id=case_id, case_type=case_type, passed=passed, elapsed_seconds=elapsed,
        infrastructure_error=infra_error,
    )


class _FixedProposer:
    """Always proposes the same overrides -- for tests where the proposer's
    own search behavior is not what is under test."""

    def __init__(self, overrides):
        self._overrides = overrides
        self.last_meta = {"proposer": "grid", "rationale": None, "model": None,
                           "fallback": False, "fallback_reason": None}

    def propose(self, space, ledger_history, last_result):
        return dict(self._overrides)


class _CyclingProposer:
    def __init__(self, options):
        self._it = itertools.cycle(options)
        self.last_meta = {"proposer": "grid", "rationale": None, "model": None,
                           "fallback": False, "fallback_reason": None}

    def propose(self, space, ledger_history, last_result):
        return dict(next(self._it))


# TEST 9 -- a candidate that scores higher but exceeds the latency ceiling
# is rejected_latency and does not move the ratchet (criterion 6).


def test_candidate_scoring_higher_but_breaching_latency_is_rejected(tmp_path):
    search_ids = ("a", "b", "c")
    fast_ids = ("a",)

    def evaluate(overrides, case_ids):
        boosted = overrides.get("retrieval.top_k_final") == 16
        records = []
        for cid in case_ids:
            if boosted:
                records.append(_rec(cid, "factual_number", True, 500.0))  # passes everything, but slow
            else:
                records.append(_rec(cid, "factual_number", cid != "c", 200.0))  # reference fails "c"
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    proposer = _FixedProposer({"retrieval.top_k_final": 16})
    report = loop.run_loop(
        reference=AppConfig(), evaluate=evaluate, proposer=proposer,
        search_set_ids=search_ids, fast_tier_ids=fast_ids, unreachable_ids=(),
        max_iterations=1, patience=None, ledger_dir=tmp_path, verify_reachability=False,
    )

    assert len(report["iterations"]) == 1
    entry = report["iterations"][0]
    assert entry["verdict"] == "rejected_latency"
    assert entry["objective_adjusted"] > report["reference"]["objective_adjusted"]  # scored higher...
    assert report["ratchet"] == report["reference"]["objective_adjusted"]  # ...but ratchet did not move
    assert report["best_iteration"] is None


def test_latency_breach_is_checked_per_bucket_independently(tmp_path):
    # A candidate that blows the retrieval-only budget but keeps the
    # answered budget must still be rejected -- a blended median would hide
    # this (retrieval-only is ~50x cheaper, so it barely moves a blend).
    search_ids = ("fig",)
    fast_ids = ("fig",)

    def evaluate(overrides, case_ids):
        slow_retrieval = "retrieval.n_semantic_results" in overrides
        elapsed = 50.0 if slow_retrieval else 4.0  # reference retrieval-only median is 4.0s
        records = [_rec(cid, "figure_retrieval", True, elapsed) for cid in case_ids]
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    proposer = _FixedProposer({"retrieval.n_semantic_results": 120})
    report = loop.run_loop(
        reference=AppConfig(), evaluate=evaluate, proposer=proposer,
        search_set_ids=search_ids, fast_tier_ids=fast_ids, unreachable_ids=(),
        max_iterations=1, patience=None, ledger_dir=tmp_path, verify_reachability=False,
    )
    assert report["iterations"][0]["verdict"] == "rejected_latency"
    assert "retrieval_only" in report["iterations"][0]["reason"]


# TEST 11 -- termination fires on budget, and on patience, and writes a
# report in both cases.


def test_termination_fires_on_max_iterations(tmp_path):
    proposer = _CyclingProposer([{"retrieval.top_k_final": 4}, {"retrieval.top_k_final": 6}])

    def evaluate(overrides, case_ids):
        records = [_rec(cid, "factual_number", True, 200.0) for cid in case_ids]
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    report = loop.run_loop(
        reference=AppConfig(), evaluate=evaluate, proposer=proposer,
        search_set_ids=("a", "b"), fast_tier_ids=("a",), unreachable_ids=(),
        max_iterations=2, patience=None, ledger_dir=tmp_path, verify_reachability=False,
    )

    assert report is not None
    assert report["termination_reason"] == "max_iterations"
    assert report["iterations_run"] == 2
    assert len(report["iterations"]) == 2


def test_termination_fires_on_patience(tmp_path):
    proposer = _CyclingProposer([{"retrieval.top_k_final": 4}, {"retrieval.top_k_final": 6}])

    def evaluate(overrides, case_ids):
        # Candidate always matches the reference exactly -> always rejected_no_gain.
        records = [_rec(cid, "factual_number", True, 200.0) for cid in case_ids]
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    report = loop.run_loop(
        reference=AppConfig(), evaluate=evaluate, proposer=proposer,
        search_set_ids=("a", "b"), fast_tier_ids=("a",), unreachable_ids=(),
        max_iterations=None, patience=2, ledger_dir=tmp_path, verify_reachability=False,
    )

    assert report is not None
    assert report["termination_reason"] == "patience"
    assert report["iterations_run"] == 2
    assert all(entry["verdict"] == "rejected_no_gain" for entry in report["iterations"])


def test_run_loop_requires_a_termination_condition():
    with pytest.raises(ValueError):
        loop.run_loop(
            reference=AppConfig(), evaluate=lambda o, c: None, proposer=_FixedProposer({}),
            search_set_ids=(), fast_tier_ids=(), max_iterations=None, patience=None,
        )


# REGRESSION AND ACCEPTANCE (state-machine coverage beyond 9/11).


def test_fast_tier_regression_is_rejected_before_paying_for_the_full_set(tmp_path):
    search_ids = ("a", "b", "c", "d")
    fast_ids = ("a",)
    full_eval_calls = {"count": 0}

    def evaluate(overrides, case_ids):
        if set(case_ids) == set(search_ids):
            full_eval_calls["count"] += 1
        regress = "retrieval.bm25_k1" in overrides
        # Reference: "a" passes. Candidate: "a" now fails -> regression.
        records = [_rec(cid, "factual_number", not (regress and cid == "a"), 200.0) for cid in case_ids]
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    proposer = _FixedProposer({"retrieval.bm25_k1": 2.0})
    report = loop.run_loop(
        reference=AppConfig(), evaluate=evaluate, proposer=proposer,
        search_set_ids=search_ids, fast_tier_ids=fast_ids, unreachable_ids=(),
        max_iterations=1, patience=None, ledger_dir=tmp_path, verify_reachability=False,
    )

    entry = report["iterations"][0]
    assert entry["verdict"] == "rejected_regression"
    assert entry["evaluated_case_set"] == "fast_tier"
    assert "a" in entry["reason"]
    # The reference's own full-set evaluation happens once regardless (to
    # seed the ratchet); the candidate's full-set evaluation must never run.
    assert full_eval_calls["count"] == 1


def test_a_real_improvement_is_accepted_and_moves_the_ratchet(tmp_path):
    search_ids = ("a", "b", "c", "d", "e", "f", "g", "h")
    fast_ids = ("a",)

    def evaluate(overrides, case_ids):
        better = overrides.get("retrieval.top_k_final") == 12
        # Reference passes 6/8; the improved candidate passes all 8 -- a
        # clear gain, well past the noise floor of 0.
        records = [_rec(cid, "factual_number", better or cid not in ("g", "h"), 200.0) for cid in case_ids]
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    proposer = _FixedProposer({"retrieval.top_k_final": 12})
    report = loop.run_loop(
        reference=AppConfig(), evaluate=evaluate, proposer=proposer,
        search_set_ids=search_ids, fast_tier_ids=fast_ids, unreachable_ids=(),
        max_iterations=1, patience=None, ledger_dir=tmp_path, verify_reachability=False,
    )

    entry = report["iterations"][0]
    assert entry["verdict"] == "accepted"
    assert report["ratchet"] == 8
    assert report["best_iteration"] == entry["iteration"]


def test_resolution_warning_is_always_present_in_the_report(tmp_path):
    def evaluate(overrides, case_ids):
        records = [_rec(cid, "factual_number", True, 200.0) for cid in case_ids]
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    report = loop.run_loop(
        reference=AppConfig(), evaluate=evaluate, proposer=_FixedProposer({"retrieval.top_k_final": 4}),
        search_set_ids=("a",), fast_tier_ids=("a",), max_iterations=1, patience=None,
        ledger_dir=tmp_path, verify_reachability=False,
    )
    assert report["resolution_warning"] == loop.RESOLUTION_WARNING
    assert report["resolution_warning"]["available_search_set_failures"] == 5
    assert report["resolution_warning"]["demonstrability_flip_threshold"] == 6


# MEDIUM 3 (#65 PR review): a blended objective can hide a candidate that
# trades retrieval quality for factual-answer luck. rejected_regression now
# also fires at the search-set stage when the retrieval-only bucket loses
# cases net, even if the blended objective_adjusted went up.


def test_retrieval_only_net_loss_is_rejected_despite_a_higher_blended_objective(tmp_path):
    search_ids = ("fig1", "fig2", "fig3", "ans1", "ans2", "ans3", "ans4", "ans5")

    def evaluate(overrides, case_ids):
        traded = overrides.get("retrieval.top_k_final") == 16
        records = []
        for cid in case_ids:
            is_fig = cid.startswith("fig")
            if is_fig:
                passed = not traded  # reference: all 3 figures pass; traded: all 3 fail
            else:
                passed = traded or cid == "ans1"  # reference: only ans1 passes; traded: all 5 pass
            records.append(_rec(cid, "figure_retrieval" if is_fig else "factual_number", passed,
                                 4.0 if is_fig else 200.0))
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    proposer = _FixedProposer({"retrieval.top_k_final": 16})
    report = loop.run_loop(
        reference=AppConfig(), evaluate=evaluate, proposer=proposer,
        search_set_ids=search_ids, fast_tier_ids=("ans1",), unreachable_ids=(),
        max_iterations=1, patience=None, ledger_dir=tmp_path, verify_reachability=False,
    )

    entry = report["iterations"][0]
    # Blended: reference passes 3+1=4, candidate passes 0+5=5 -- a plain
    # objective check alone would call this a +1 improvement.
    assert entry["objective_adjusted"] == 5
    assert entry["objective_adjusted"] > report["reference"]["objective_adjusted"]
    assert entry["verdict"] == "rejected_regression"
    assert "retrieval-only" in entry["reason"]
    assert report["ratchet"] == report["reference"]["objective_adjusted"]  # unmoved
    assert report["best_iteration"] is None


def test_summary_carries_a_per_case_type_breakdown(tmp_path):
    def evaluate(overrides, case_ids):
        records = [
            _rec("fig", "figure_retrieval", True, 4.0),
            _rec("num", "factual_number", overrides.get("retrieval.top_k_final") == 12, 200.0),
        ]
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    report = loop.run_loop(
        reference=AppConfig(), evaluate=evaluate, proposer=_FixedProposer({"retrieval.top_k_final": 12}),
        search_set_ids=("fig", "num"), fast_tier_ids=("fig",), max_iterations=1, patience=None,
        ledger_dir=tmp_path, verify_reachability=False,
    )
    summary = report["iterations"][0]["summary"]
    assert summary["by_case_type"]["figure_retrieval"] == {"total": 1, "passed": 1, "pass_rate": 1.0}
    assert summary["by_case_type"]["factual_number"] == {"total": 1, "passed": 1, "pass_rate": 1.0}


# HIGH 2 (#65 PR review): an infrastructure_error record (dead/overloaded
# Ollama, a retrieval exception -- see run_eval.run_factual_case) must never
# read as an ordinary quality failure. loop.py refuses to score it.


def test_reference_with_infrastructure_error_raises_before_the_loop_starts(tmp_path):
    def evaluate(overrides, case_ids):
        infra = not overrides  # only the reference call (empty overrides) errors
        records = [_rec(cid, "factual_number", False, 200.0, infra_error=infra) for cid in case_ids]
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    with pytest.raises(ev.InconclusiveEvaluationError):
        loop.run_loop(
            reference=AppConfig(), evaluate=evaluate, proposer=_FixedProposer({"retrieval.top_k_final": 4}),
            search_set_ids=("a", "b"), fast_tier_ids=("a",), max_iterations=1, patience=None,
            ledger_dir=tmp_path, verify_reachability=False,
        )


def test_fast_tier_infrastructure_error_yields_inconclusive_not_regression(tmp_path):
    def evaluate(overrides, case_ids):
        is_fast_tier_call = set(case_ids) == {"a"}
        infra = bool(overrides) and is_fast_tier_call  # only the candidate's fast-tier call errors
        records = [_rec(cid, "factual_number", True, 200.0, infra_error=infra) for cid in case_ids]
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    proposer = _FixedProposer({"retrieval.top_k_final": 4})
    report = loop.run_loop(
        reference=AppConfig(), evaluate=evaluate, proposer=proposer,
        search_set_ids=("a", "b"), fast_tier_ids=("a",), max_iterations=1, patience=None,
        ledger_dir=tmp_path, verify_reachability=False,
    )

    entry = report["iterations"][0]
    assert entry["verdict"] == "inconclusive"
    assert entry["evaluated_case_set"] == "fast_tier"
    assert report["ratchet"] == report["reference"]["objective_adjusted"]


def test_search_set_infrastructure_error_yields_inconclusive_and_no_ratchet_move(tmp_path):
    def evaluate(overrides, case_ids):
        is_full_set_call = set(case_ids) == {"a", "b"}
        infra = bool(overrides) and is_full_set_call  # only the candidate's full evaluation errors
        records = [_rec(cid, "factual_number", True, 200.0, infra_error=infra) for cid in case_ids]
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    proposer = _FixedProposer({"retrieval.top_k_final": 4})
    report = loop.run_loop(
        reference=AppConfig(), evaluate=evaluate, proposer=proposer,
        search_set_ids=("a", "b"), fast_tier_ids=("a",), max_iterations=1, patience=None,
        ledger_dir=tmp_path, verify_reachability=False,
    )

    entry = report["iterations"][0]
    assert entry["verdict"] == "inconclusive"
    assert entry["evaluated_case_set"] == "search_set"
    assert report["ratchet"] == report["reference"]["objective_adjusted"]
    assert report["best_iteration"] is None


def test_inconclusive_counts_toward_patience(tmp_path):
    def evaluate(overrides, case_ids):
        records = [_rec(cid, "factual_number", True, 200.0, infra_error=bool(overrides)) for cid in case_ids]
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    proposer = _CyclingProposer([{"retrieval.top_k_final": 4}, {"retrieval.top_k_final": 6}])
    report = loop.run_loop(
        reference=AppConfig(), evaluate=evaluate, proposer=proposer,
        search_set_ids=("a",), fast_tier_ids=("a",), max_iterations=None, patience=2,
        ledger_dir=tmp_path, verify_reachability=False,
    )

    assert report["termination_reason"] == "patience"
    assert report["iterations_run"] == 2
    assert all(entry["verdict"] == "inconclusive" for entry in report["iterations"])
