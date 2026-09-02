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
        id=case_id,
        case_type=case_type,
        passed=passed,
        elapsed_seconds=elapsed,
        infrastructure_error=infra_error,
    )


class _FixedProposer:
    """Always proposes the same overrides -- for tests where the proposer's
    own search behavior is not what is under test."""

    def __init__(self, overrides):
        self._overrides = overrides
        self.last_meta = {
            "proposer": "grid",
            "rationale": None,
            "model": None,
            "fallback": False,
            "fallback_reason": None,
        }

    def propose(self, space, ledger_history, last_result):
        return dict(self._overrides)


class _CyclingProposer:
    def __init__(self, options):
        self._it = itertools.cycle(options)
        self.last_meta = {
            "proposer": "grid",
            "rationale": None,
            "model": None,
            "fallback": False,
            "fallback_reason": None,
        }

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
                records.append(
                    _rec(cid, "factual_number", True, 500.0)
                )  # passes everything, but slow
            else:
                records.append(
                    _rec(cid, "factual_number", cid != "c", 200.0)
                )  # reference fails "c"
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    proposer = _FixedProposer({"retrieval.top_k_final": 16})
    report = loop.run_loop(
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=proposer,
        search_set_ids=search_ids,
        fast_tier_ids=fast_ids,
        unreachable_ids=(),
        max_iterations=1,
        patience=None,
        ledger_dir=tmp_path,
        verify_reachability=False,
    )

    assert len(report["iterations"]) == 1
    entry = report["iterations"][0]
    assert entry["verdict"] == "rejected_latency"
    assert (
        entry["objective_adjusted"] > report["reference"]["objective_adjusted"]
    )  # scored higher...
    assert (
        report["ratchet"] == report["reference"]["objective_adjusted"]
    )  # ...but ratchet did not move
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
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=proposer,
        search_set_ids=search_ids,
        fast_tier_ids=fast_ids,
        unreachable_ids=(),
        max_iterations=1,
        patience=None,
        ledger_dir=tmp_path,
        verify_reachability=False,
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
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=proposer,
        search_set_ids=("a", "b"),
        fast_tier_ids=("a",),
        unreachable_ids=(),
        max_iterations=2,
        patience=None,
        ledger_dir=tmp_path,
        verify_reachability=False,
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
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=proposer,
        search_set_ids=("a", "b"),
        fast_tier_ids=("a",),
        unreachable_ids=(),
        max_iterations=None,
        patience=2,
        ledger_dir=tmp_path,
        verify_reachability=False,
    )

    assert report is not None
    assert report["termination_reason"] == "patience"
    assert report["iterations_run"] == 2
    assert all(entry["verdict"] == "rejected_no_gain" for entry in report["iterations"])


def test_run_loop_requires_a_termination_condition():
    with pytest.raises(ValueError):
        loop.run_loop(
            reference=AppConfig(),
            evaluate=lambda o, c: None,
            proposer=_FixedProposer({}),
            search_set_ids=(),
            fast_tier_ids=(),
            max_iterations=None,
            patience=None,
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
        records = [
            _rec(cid, "factual_number", not (regress and cid == "a"), 200.0) for cid in case_ids
        ]
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    proposer = _FixedProposer({"retrieval.bm25_k1": 2.0})
    report = loop.run_loop(
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=proposer,
        search_set_ids=search_ids,
        fast_tier_ids=fast_ids,
        unreachable_ids=(),
        max_iterations=1,
        patience=None,
        ledger_dir=tmp_path,
        verify_reachability=False,
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
        records = [
            _rec(cid, "factual_number", better or cid not in ("g", "h"), 200.0) for cid in case_ids
        ]
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    proposer = _FixedProposer({"retrieval.top_k_final": 12})
    report = loop.run_loop(
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=proposer,
        search_set_ids=search_ids,
        fast_tier_ids=fast_ids,
        unreachable_ids=(),
        max_iterations=1,
        patience=None,
        ledger_dir=tmp_path,
        verify_reachability=False,
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
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=_FixedProposer({"retrieval.top_k_final": 4}),
        search_set_ids=("a",),
        fast_tier_ids=("a",),
        max_iterations=1,
        patience=None,
        ledger_dir=tmp_path,
        verify_reachability=False,
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
            records.append(
                _rec(
                    cid,
                    "figure_retrieval" if is_fig else "factual_number",
                    passed,
                    4.0 if is_fig else 200.0,
                )
            )
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    proposer = _FixedProposer({"retrieval.top_k_final": 16})
    report = loop.run_loop(
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=proposer,
        search_set_ids=search_ids,
        fast_tier_ids=("ans1",),
        unreachable_ids=(),
        max_iterations=1,
        patience=None,
        ledger_dir=tmp_path,
        verify_reachability=False,
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
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=_FixedProposer({"retrieval.top_k_final": 12}),
        search_set_ids=("fig", "num"),
        fast_tier_ids=("fig",),
        max_iterations=1,
        patience=None,
        ledger_dir=tmp_path,
        verify_reachability=False,
    )
    summary = report["iterations"][0]["summary"]
    assert summary["by_case_type"]["figure_retrieval"] == {
        "total": 1,
        "passed": 1,
        "pass_rate": 1.0,
    }
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
            reference=AppConfig(),
            evaluate=evaluate,
            proposer=_FixedProposer({"retrieval.top_k_final": 4}),
            search_set_ids=("a", "b"),
            fast_tier_ids=("a",),
            max_iterations=1,
            patience=None,
            ledger_dir=tmp_path,
            verify_reachability=False,
        )


def test_fast_tier_infrastructure_error_yields_inconclusive_not_regression(tmp_path):
    def evaluate(overrides, case_ids):
        is_fast_tier_call = set(case_ids) == {"a"}
        infra = bool(overrides) and is_fast_tier_call  # only the candidate's fast-tier call errors
        records = [_rec(cid, "factual_number", True, 200.0, infra_error=infra) for cid in case_ids]
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    proposer = _FixedProposer({"retrieval.top_k_final": 4})
    report = loop.run_loop(
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=proposer,
        search_set_ids=("a", "b"),
        fast_tier_ids=("a",),
        max_iterations=1,
        patience=None,
        ledger_dir=tmp_path,
        verify_reachability=False,
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
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=proposer,
        search_set_ids=("a", "b"),
        fast_tier_ids=("a",),
        max_iterations=1,
        patience=None,
        ledger_dir=tmp_path,
        verify_reachability=False,
    )

    entry = report["iterations"][0]
    assert entry["verdict"] == "inconclusive"
    assert entry["evaluated_case_set"] == "search_set"
    assert report["ratchet"] == report["reference"]["objective_adjusted"]
    assert report["best_iteration"] is None


def test_inconclusive_counts_toward_patience(tmp_path):
    def evaluate(overrides, case_ids):
        records = [
            _rec(cid, "factual_number", True, 200.0, infra_error=bool(overrides))
            for cid in case_ids
        ]
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    proposer = _CyclingProposer([{"retrieval.top_k_final": 4}, {"retrieval.top_k_final": 6}])
    report = loop.run_loop(
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=proposer,
        search_set_ids=("a",),
        fast_tier_ids=("a",),
        max_iterations=None,
        patience=2,
        ledger_dir=tmp_path,
        verify_reachability=False,
    )

    assert report["termination_reason"] == "patience"
    assert report["iterations_run"] == 2
    assert all(entry["verdict"] == "inconclusive" for entry in report["iterations"])


# RECOVERY MODE (issue #92): against a degraded reference whose extra pass is
# sabotage luck (it passes a case every healthy configuration fails), pairing
# regressions against the reference blocks recovery forever. When the ledger's
# prior history holds a comparable state with a strictly higher objective, the
# loop pairs against that high-water pass vector instead.


def _seed_entry(
    tmp_path_ledger,
    iteration,
    objective,
    case_records,
    verdict="accepted",
    effective_config=None,
    evaluated_case_set="search_set",
    environment_fingerprint=None,
):
    import dataclasses

    from harness import ledger as ledger_mod

    entry = ledger_mod.LedgerEntry(
        schema_version=ledger_mod.SCHEMA_VERSION,
        iteration=iteration,
        parent_iteration=None if iteration == 1 else iteration - 1,
        git_commit=None,
        config_overrides={},
        effective_config=effective_config or dataclasses.asdict(AppConfig()),
        proposer="grid",
        proposer_rationale=None,
        proposer_model=None,
        proposer_fallback=False,
        proposer_fallback_reason=None,
        evaluated_case_set=evaluated_case_set,
        case_records=case_records,
        summary={
            "total": len(case_records),
            "passed": objective,
            "pass_rate": objective / len(case_records),
        },
        objective_raw=objective,
        objective_adjusted=objective,
        median_latency_answered_s=200.0,
        median_latency_retrieval_only_s=None,
        reference_median_latency_answered_s=200.0,
        reference_median_latency_retrieval_only_s=None,
        latency_ceiling_multiplier=loop.LATENCY_CEILING_MULTIPLIER,
        verdict=verdict,
        reason="seeded",
        environment_fingerprint=environment_fingerprint,
    )
    ledger_mod.write_entry(entry, tmp_path_ledger)


def test_recovery_candidate_reaches_the_search_set_when_reference_is_degraded(tmp_path):
    """The #92 scenario end to end: a degraded reference passes 'lucky' only
    because the sabotage lucked into it; the healthy high-water state in the
    ledger fails it. Recovery pairing must let a candidate that restores
    health through to acceptance instead of rejecting it at the fast tier."""
    search_ids = ("lucky", "a", "b", "c", "d")
    fast_ids = ("lucky",)
    # High water (healthy): fails 'lucky', passes everything else -> 4.
    _seed_entry(
        tmp_path,
        1,
        4,
        [
            {
                "id": "lucky",
                "case_type": "factual_number",
                "passed": False,
                "elapsed_seconds": 200.0,
            },
            *[
                {"id": cid, "case_type": "factual_number", "passed": True, "elapsed_seconds": 200.0}
                for cid in ("a", "b", "c", "d")
            ],
        ],
    )

    def evaluate(overrides, case_ids):
        healthy = overrides.get("retrieval.top_k_final") == 12
        records = []
        for cid in case_ids:
            if cid == "lucky":
                passed = not healthy and not overrides  # only the degraded reference passes it
            else:
                passed = healthy  # the degraded reference fails these too
            records.append(_rec(cid, "factual_number", passed, 200.0))
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    proposer = _FixedProposer({"retrieval.top_k_final": 12})
    report = loop.run_loop(
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=proposer,
        search_set_ids=search_ids,
        fast_tier_ids=fast_ids,
        unreachable_ids=(),
        max_iterations=1,
        patience=None,
        ledger_dir=tmp_path,
        verify_reachability=False,
    )

    assert report["recovery_mode"]["active"] is True
    assert report["recovery_mode"]["baseline_iteration"] == 1
    assert report["recovery_mode"]["baseline_objective_adjusted"] == 4
    entry = report["iterations"][0]
    assert entry["evaluated_case_set"] == "search_set"  # got past the fast tier
    assert entry["verdict"] == "accepted"  # restored health: 4 > ratchet 1
    assert entry["regression_baseline_iteration"] == 1


def test_recovery_candidate_losing_earned_fast_tier_cases_is_still_rejected(tmp_path):
    """Recovery pairs against earned passes, not against nothing: a candidate
    that re-fails a case the high-water state EARNED is still a regression,
    even though the degraded reference never passed that case either."""
    search_ids = ("lucky", "a", "b", "c", "d")
    fast_ids = ("lucky", "a")
    _seed_entry(
        tmp_path,
        1,
        4,
        [
            {
                "id": "lucky",
                "case_type": "factual_number",
                "passed": False,
                "elapsed_seconds": 200.0,
            },
            *[
                {"id": cid, "case_type": "factual_number", "passed": True, "elapsed_seconds": 200.0}
                for cid in ("a", "b", "c", "d")
            ],
        ],
    )

    def evaluate(overrides, case_ids):
        if not overrides:  # degraded reference: passes only the lucky case -> 1
            records = [_rec(cid, "factual_number", cid == "lucky", 200.0) for cid in case_ids]
            return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))
        # Candidate: keeps the lucky case, restores b/c/d, but loses earned 'a' -> 4.
        records = [_rec(cid, "factual_number", cid != "a", 200.0) for cid in case_ids]
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    proposer = _FixedProposer({"retrieval.top_k_final": 12})
    report = loop.run_loop(
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=proposer,
        search_set_ids=search_ids,
        fast_tier_ids=fast_ids,
        unreachable_ids=(),
        max_iterations=1,
        patience=None,
        ledger_dir=tmp_path,
        verify_reachability=False,
    )

    entry = report["iterations"][0]
    assert entry["verdict"] == "rejected_regression"
    assert "'a'" in entry["reason"]
    assert entry["regression_baseline_iteration"] == 1


def test_without_comparable_history_reference_pairing_is_kept(tmp_path):
    """A prior entry from a different model is NOT comparable: no recovery
    mode, so a candidate re-failing the lucky case is rejected exactly as
    before the fix."""
    import dataclasses

    search_ids = ("lucky", "a")
    fast_ids = ("lucky",)
    other_model = dataclasses.asdict(AppConfig())
    other_model["models"]["rag"] = "some-other-model"
    _seed_entry(
        tmp_path,
        1,
        2,
        [
            {
                "id": "lucky",
                "case_type": "factual_number",
                "passed": False,
                "elapsed_seconds": 200.0,
            },
            {"id": "a", "case_type": "factual_number", "passed": True, "elapsed_seconds": 200.0},
        ],
        effective_config=other_model,
    )

    def evaluate(overrides, case_ids):
        healthy = bool(overrides)
        records = []
        for cid in case_ids:
            if cid == "lucky":
                passed = not healthy  # candidate re-fails what the sabotage lucked into
            else:
                passed = healthy
            records.append(_rec(cid, "factual_number", passed, 200.0))
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    proposer = _FixedProposer({"retrieval.top_k_final": 12})
    report = loop.run_loop(
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=proposer,
        search_set_ids=search_ids,
        fast_tier_ids=fast_ids,
        unreachable_ids=(),
        max_iterations=1,
        patience=None,
        ledger_dir=tmp_path,
        verify_reachability=False,
    )

    assert report["recovery_mode"]["active"] is False
    entry = report["iterations"][0]
    assert entry["verdict"] == "rejected_regression"
    assert entry["regression_baseline_iteration"] is None


def test_pre_flag_ledger_entries_stay_comparable(tmp_path):
    """Entries written before an index-time flag existed lack the key; they
    must still count as comparable history (absent == off), or every old
    ledger stops arming recovery mode the moment a new flag ships (issue
    #92's own validation campaign hit this against 2026-08 entries)."""
    import dataclasses

    stripped = dataclasses.asdict(AppConfig())
    del stripped["flags"]["usar_descripcion_imagen"]  # pre-flag writer
    _seed_entry(
        tmp_path, 1, 2,
        [
            {"id": "lucky", "case_type": "factual_number", "passed": False, "elapsed_seconds": 200.0},
            {"id": "a", "case_type": "factual_number", "passed": True, "elapsed_seconds": 200.0},
        ],
        effective_config=stripped,
    )

    def evaluate(overrides, case_ids):
        healthy = bool(overrides)
        records = [
            _rec(cid, "factual_number", healthy if cid != "lucky" else not healthy, 200.0)
            for cid in case_ids
        ]
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    proposer = _FixedProposer({"retrieval.top_k_final": 12})
    report = loop.run_loop(
        reference=AppConfig(), evaluate=evaluate, proposer=proposer,
        search_set_ids=("lucky", "a"), fast_tier_ids=("lucky",), unreachable_ids=(),
        max_iterations=1, patience=None, ledger_dir=tmp_path, verify_reachability=False,
    )

    assert report["recovery_mode"]["active"] is True


def test_inconclusive_history_never_provides_the_high_water(tmp_path):
    """An inconclusive entry may be measuring a dead Ollama, not quality; its
    higher objective must not activate recovery mode."""
    search_ids = ("lucky", "a")
    fast_ids = ("lucky",)
    _seed_entry(
        tmp_path,
        1,
        2,
        [
            {
                "id": "lucky",
                "case_type": "factual_number",
                "passed": False,
                "elapsed_seconds": 200.0,
            },
            {"id": "a", "case_type": "factual_number", "passed": True, "elapsed_seconds": 200.0},
        ],
        verdict="inconclusive",
    )

    def evaluate(overrides, case_ids):
        healthy = bool(overrides)
        records = []
        for cid in case_ids:
            passed = healthy if cid != "lucky" else not healthy
            records.append(_rec(cid, "factual_number", passed, 200.0))
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    proposer = _FixedProposer({"retrieval.top_k_final": 12})
    report = loop.run_loop(
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=proposer,
        search_set_ids=search_ids,
        fast_tier_ids=fast_ids,
        unreachable_ids=(),
        max_iterations=1,
        patience=None,
        ledger_dir=tmp_path,
        verify_reachability=False,
    )

    assert report["recovery_mode"]["active"] is False
    assert report["iterations"][0]["verdict"] == "rejected_regression"


def test_healthy_reference_at_or_above_high_water_keeps_reference_pairing(tmp_path):
    """Recovery mode needs the reference STRICTLY below the high water: an
    equal-or-better reference is healthy, and its own pass vector governs."""
    search_ids = ("x", "y")
    fast_ids = ("x",)
    _seed_entry(
        tmp_path,
        1,
        2,
        [
            {"id": "x", "case_type": "factual_number", "passed": False, "elapsed_seconds": 200.0},
            {"id": "y", "case_type": "factual_number", "passed": True, "elapsed_seconds": 200.0},
        ],
    )

    def evaluate(overrides, case_ids):
        traded = bool(overrides)  # candidate loses 'x' (reference passes it; high water failed it)
        records = [
            _rec(cid, "factual_number", not (traded and cid == "x"), 200.0) for cid in case_ids
        ]
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    proposer = _FixedProposer({"retrieval.top_k_final": 12})
    report = loop.run_loop(
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=proposer,
        search_set_ids=search_ids,
        fast_tier_ids=fast_ids,
        unreachable_ids=(),
        max_iterations=1,
        patience=None,
        ledger_dir=tmp_path,
        verify_reachability=False,
    )

    assert report["recovery_mode"]["active"] is False
    entry = report["iterations"][0]
    # Rejected against the REFERENCE's pass vector ('x' was earned here).
    assert entry["verdict"] == "rejected_regression"
    assert entry["regression_baseline_iteration"] is None


# COMPARABILITY AT LAUNCH (issue #100): recovery mode pairs against comparable
# prior history, and a silently incomparable ledger is how a campaign spends
# hours fabricating evidence against its own fix. describe_ledger_comparability
# is what the CLI prints BEFORE any evaluation is paid.


def test_describe_comparability_on_an_empty_ledger():
    info = loop.describe_ledger_comparability(AppConfig(), [])
    assert info == {
        "history_entries": 0,
        "comparable_search_set_states": 0,
        "high_water_objective_adjusted": None,
        # Issue #107: no comparable state, so none verified and none
        # unverified -- the split is over what pairing would actually use --
        # and no high water whose stack could be verified either way.
        "high_water_environment_verified": None,
        "environment_verified_states": 0,
        "environment_unverified_states": 0,
        "incomparable_reasons": [],
    }


def test_describe_comparability_counts_a_matching_search_set_entry(tmp_path):
    _seed_entry(tmp_path, 1, 27, [{"id": "a", "case_type": "factual_number", "passed": True, "elapsed_seconds": 200.0}])
    from harness import ledger as ledger_mod

    entries = list(ledger_mod.read_history(tmp_path))
    info = loop.describe_ledger_comparability(AppConfig(), entries)
    assert info["history_entries"] == 1
    assert info["comparable_search_set_states"] == 1
    assert info["high_water_objective_adjusted"] == 27
    assert info["incomparable_reasons"] == []


def test_describe_comparability_names_the_role_that_drifted(tmp_path):
    """The 2026-08-23 trap: settings.json moved chat off the code default since
    the healthy sibling ran; the launch line must name that field before hours
    are paid."""
    import dataclasses

    drifted = dataclasses.asdict(AppConfig())
    drifted["models"]["chat"] = "gemma4:e2b"
    _seed_entry(
        tmp_path, 1, 27,
        [{"id": "a", "case_type": "factual_number", "passed": True, "elapsed_seconds": 200.0}],
        effective_config=drifted,
    )
    from harness import ledger as ledger_mod

    entries = list(ledger_mod.read_history(tmp_path))
    info = loop.describe_ledger_comparability(AppConfig(), entries)
    assert info["comparable_search_set_states"] == 0
    assert info["incomparable_reasons"] == ["models.chat: this launch 'gemma4:e4b', ledger 'gemma4:e2b'"]


def test_describe_comparability_explains_fast_tier_only_history(tmp_path):
    """A config that matches but never produced a full search-set pass vector
    is not usable as a high water -- say that instead of a config diff."""
    _seed_entry(
        tmp_path, 1, 8,
        [{"id": "a", "case_type": "factual_number", "passed": True, "elapsed_seconds": 200.0}],
        verdict="rejected_regression",
        evaluated_case_set="fast_tier",
    )
    from harness import ledger as ledger_mod

    info = loop.describe_ledger_comparability(AppConfig(), list(ledger_mod.read_history(tmp_path)))
    assert info["comparable_search_set_states"] == 0
    assert info["incomparable_reasons"] == [
        "config matches the latest entry but it carries no complete search-set pass vector"
    ]


# REFERENCE OVERRIDES (issue #101, caught live 2026-08-23): --set pins used to
# exist only on the harness's bookkeeping AppConfig, while the measured
# pipeline re-derived its config from environment and settings.json -- a
# sabotaged reference that scores healthy. The pins must reach every evaluate()
# call this run makes.


def test_reference_overrides_reach_every_evaluation(tmp_path):
    """The reference measurement applies the pins; each candidate's evaluation
    merges them under its own overrides."""
    seen: list = []

    def evaluate(overrides, case_ids):
        seen.append(dict(overrides))
        records = [_rec(cid, "factual_number", True, 200.0) for cid in case_ids]
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    report = loop.run_loop(
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=_FixedProposer({"retrieval.top_k_final": 12}),
        search_set_ids=("a", "b", "c"),
        fast_tier_ids=("a",),
        unreachable_ids=(),
        max_iterations=1,
        patience=None,
        ledger_dir=tmp_path,
        verify_reachability=False,
        reference_overrides={"retrieval.top_k_final": 1},
    )

    # Calls in order: reference full, reference fast, candidate fast, candidate full.
    assert seen[0] == {"retrieval.top_k_final": 1}
    assert seen[1] == {"retrieval.top_k_final": 1}
    # Candidate evaluations carry the pins under the candidate's own value.
    assert seen[2] == {"retrieval.top_k_final": 12}
    assert seen[3] == {"retrieval.top_k_final": 12}
    # Provenance lands in the report so a pinned campaign is auditable.
    assert report["reference"]["overrides_applied"] == {"retrieval.top_k_final": 1}


def test_candidate_overrides_win_over_reference_pins(tmp_path):
    """A candidate key colliding with a pin is the candidate's to decide --
    the proposer owns the searched axes even when the reference pins one."""

    def evaluate(overrides, case_ids):
        records = [_rec(cid, "factual_number", True, 200.0) for cid in case_ids]
        return ev.EvaluationResult(records=tuple(records), effective_config=dict(overrides))

    report = loop.run_loop(
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=_FixedProposer({"retrieval.top_k_final": 8}),
        search_set_ids=("a",),
        fast_tier_ids=("a",),
        unreachable_ids=(),
        max_iterations=1,
        patience=None,
        ledger_dir=tmp_path,
        verify_reachability=False,
        reference_overrides={"retrieval.top_k_final": 1},
    )

    entry = report["iterations"][0]
    # The candidate's own value reached the ledger's provenance.
    assert entry["config_overrides"] == {"retrieval.top_k_final": 8}


# ENVIRONMENT DRIFT (issue #107): comparable-view equality covers models,
# chunking and index-time flags, and nothing else. A healthy August baseline
# can therefore hold passes today's MinerU/jina-clip cannot reach -- recovery
# pairs against a high water nobody can climb back to, every campaign ends
# rejected_regression on ghost passes, and there is no bug to fix. The stack
# each iteration ran on is recorded per entry, and a KNOWN-different stack
# makes an entry incomparable. Unknown does not: pre-v3 entries carry no
# fingerprint, and calling them different would disarm recovery mode across
# the whole existing ledger, discarding the evidence #92 was built on.

_STACK_AUGUST = {"schema": 1, "packages": {"isolated:mineru": "2.6.3"}}
_STACK_TODAY = {"schema": 1, "packages": {"isolated:mineru": "3.4.5"}}

_ONE_PASS = [{"id": "a", "case_type": "factual_number", "passed": True, "elapsed_seconds": 200.0}]


def test_an_entry_measured_on_a_different_stack_is_not_comparable(tmp_path):
    from harness import ledger as ledger_mod

    _seed_entry(tmp_path, 1, 27, _ONE_PASS, environment_fingerprint=_STACK_AUGUST)
    info = loop.describe_ledger_comparability(
        AppConfig(), list(ledger_mod.read_history(tmp_path)), launch_environment=_STACK_TODAY
    )
    assert info["comparable_search_set_states"] == 0
    assert info["high_water_objective_adjusted"] is None
    assert info["incomparable_reasons"] == [
        "isolated:mineru: this launch '3.4.5', ledger '2.6.3'"
    ]


def test_an_entry_from_the_same_stack_is_comparable_and_reported_as_verified(tmp_path):
    from harness import ledger as ledger_mod

    _seed_entry(tmp_path, 1, 27, _ONE_PASS, environment_fingerprint=_STACK_TODAY)
    info = loop.describe_ledger_comparability(
        AppConfig(), list(ledger_mod.read_history(tmp_path)), launch_environment=_STACK_TODAY
    )
    assert info["comparable_search_set_states"] == 1
    assert info["environment_verified_states"] == 1
    assert info["environment_unverified_states"] == 0


def test_an_entry_without_a_fingerprint_stays_comparable_but_counts_as_unverified(tmp_path):
    """Every entry written before schema v3 is this case. Disarming them would
    solve #107 by throwing away #92's evidence -- so they still pair, and the
    launch line says how many are unverified."""
    from harness import ledger as ledger_mod

    _seed_entry(tmp_path, 1, 27, _ONE_PASS, environment_fingerprint=None)
    info = loop.describe_ledger_comparability(
        AppConfig(), list(ledger_mod.read_history(tmp_path)), launch_environment=_STACK_TODAY
    )
    assert info["comparable_search_set_states"] == 1
    assert info["environment_verified_states"] == 0
    assert info["environment_unverified_states"] == 1
    assert info["incomparable_reasons"] == []


def test_an_unreadable_launch_environment_verifies_nothing_and_rejects_nothing(tmp_path):
    """environment_fingerprint() returns None when it cannot read the stack.
    That is a gap in provenance, not evidence that anything drifted."""
    from harness import ledger as ledger_mod

    _seed_entry(tmp_path, 1, 27, _ONE_PASS, environment_fingerprint=_STACK_AUGUST)
    info = loop.describe_ledger_comparability(
        AppConfig(), list(ledger_mod.read_history(tmp_path)), launch_environment=None
    )
    assert info["comparable_search_set_states"] == 1
    assert info["environment_unverified_states"] == 1


def test_the_high_water_skips_an_entry_measured_on_a_different_stack(tmp_path):
    """The #107 failure in one assertion: the drifted entry scores higher, and
    pairing against it is what makes recovery structurally unreachable."""
    from harness import ledger as ledger_mod

    _seed_entry(tmp_path, 1, 30, _ONE_PASS, environment_fingerprint=_STACK_AUGUST)
    _seed_entry(tmp_path, 2, 27, _ONE_PASS, environment_fingerprint=_STACK_TODAY)
    entries = list(ledger_mod.read_history(tmp_path))
    import dataclasses

    view = loop._comparable_config_view(dataclasses.asdict(AppConfig()))

    high_water = loop._historical_high_water(entries, view, launch_environment=_STACK_TODAY)
    assert high_water is not None and high_water.iteration == 2

    both = loop._historical_high_water(entries, view, launch_environment=None)
    assert both is not None and both.iteration == 1


def test_run_loop_stamps_the_launch_environment_into_every_entry(tmp_path):
    from harness import ledger as ledger_mod

    search_ids = ["a", "b"]

    def evaluate(overrides, case_ids):
        return ev.EvaluationResult(
            records=tuple(
                ev.CaseRecord(id=cid, case_type="factual_number", passed=True, elapsed_seconds=1.0)
                for cid in case_ids
            ),
            effective_config=dict(overrides),
        )

    report = loop.run_loop(
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=_FixedProposer({"retrieval.top_k_final": 12}),
        search_set_ids=search_ids,
        fast_tier_ids=["a"],
        unreachable_ids=(),
        max_iterations=1,
        patience=None,
        ledger_dir=tmp_path,
        verify_reachability=False,
        launch_environment=_STACK_TODAY,
    )

    written = list(ledger_mod.read_history(tmp_path))
    assert written and all(e.environment_fingerprint == _STACK_TODAY for e in written)
    assert report["environment"]["fingerprint"] == _STACK_TODAY


def test_the_report_says_when_the_launch_environment_could_not_be_read(tmp_path):
    def evaluate(overrides, case_ids):
        return ev.EvaluationResult(
            records=tuple(
                ev.CaseRecord(id=cid, case_type="factual_number", passed=True, elapsed_seconds=1.0)
                for cid in case_ids
            ),
            effective_config=dict(overrides),
        )

    report = loop.run_loop(
        reference=AppConfig(),
        evaluate=evaluate,
        proposer=_FixedProposer({"retrieval.top_k_final": 12}),
        search_set_ids=["a", "b"],
        fast_tier_ids=["a"],
        unreachable_ids=(),
        max_iterations=1,
        patience=None,
        ledger_dir=tmp_path,
        verify_reachability=False,
        launch_environment=None,
    )
    assert report["environment"]["fingerprint"] is None
    assert report["environment"]["readable"] is False


def test_a_refreshed_baseline_takes_over_from_a_higher_aged_one(tmp_path):
    """The exact #107 situation, now resolved rather than only reported.

    This used to assert the opposite: the aged 30 won and the launch line said
    so, which was accurate and useless -- the operator was told the baseline
    was unreachable and had no way to replace it. Option 1 is that way: once a
    comparable state has been measured on this stack, the entry that never said
    what it ran on stops being the baseline, even though it scores higher.

    The count of comparable states does not change. Both still pair; only one
    is eligible to be the high water.
    """
    from harness import ledger as ledger_mod

    _seed_entry(tmp_path, 1, 30, _ONE_PASS, environment_fingerprint=None)
    _seed_entry(tmp_path, 2, 25, _ONE_PASS, environment_fingerprint=_STACK_TODAY)
    info = loop.describe_ledger_comparability(
        AppConfig(), list(ledger_mod.read_history(tmp_path)), launch_environment=_STACK_TODAY
    )
    assert info["comparable_search_set_states"] == 2
    assert info["high_water_objective_adjusted"] == 25
    assert info["high_water_environment_verified"] is True


def test_the_launch_line_says_when_the_high_water_itself_is_unverified(tmp_path):
    """With nothing measured on this stack there is nothing to displace with,
    so the aged entry is still the baseline -- and the launch line has to say
    that it was never verified, because that is the warning telling the
    operator a refresh campaign is what they need."""
    from harness import ledger as ledger_mod

    _seed_entry(tmp_path, 1, 30, _ONE_PASS, environment_fingerprint=None)
    _seed_entry(tmp_path, 2, 25, _ONE_PASS, environment_fingerprint=None)
    info = loop.describe_ledger_comparability(
        AppConfig(), list(ledger_mod.read_history(tmp_path)), launch_environment=_STACK_TODAY
    )
    assert info["comparable_search_set_states"] == 2
    assert info["high_water_objective_adjusted"] == 30
    assert info["high_water_environment_verified"] is False


def test_a_verified_high_water_is_reported_as_verified(tmp_path):
    from harness import ledger as ledger_mod

    _seed_entry(tmp_path, 1, 25, _ONE_PASS, environment_fingerprint=None)
    _seed_entry(tmp_path, 2, 30, _ONE_PASS, environment_fingerprint=_STACK_TODAY)
    info = loop.describe_ledger_comparability(
        AppConfig(), list(ledger_mod.read_history(tmp_path)), launch_environment=_STACK_TODAY
    )
    assert info["high_water_objective_adjusted"] == 30
    assert info["high_water_environment_verified"] is True


# ---------------------------------------------------------------------------
# Issue #107, option 1: a refreshed baseline displaces the aged one.
#
# Option 2 (the fingerprint) landed already and stops an entry measured on a
# *provably different* stack from pairing. It does nothing about the entries
# that actually aged: everything written before schema v3 carries no
# fingerprint at all, compares UNKNOWN, and keeps pairing forever. Those are
# exactly the August entries holding passes this stack cannot reach.
#
# So the rule these tests pin: once this stack has measured a comparable state
# of its own, history that never said what it ran on stops being the baseline.
# Before that first verified entry nothing changes, because discarding the
# unverified history with nothing to put in its place is how #92's evidence
# would be thrown away for a stack drift nobody has demonstrated.
# ---------------------------------------------------------------------------

_STACK_NOW = {
    "schema": 1,
    "packages": {"isolated": {"mineru": "3.4.5"}, "product": {"torch": "2.9.0"}},
}


def _high_water_of(tmp_path, launch_environment):
    import dataclasses

    from harness import ledger as ledger_mod

    entries = list(ledger_mod.read_history(tmp_path))
    view = loop._comparable_config_view(dataclasses.asdict(AppConfig()))
    return loop._historical_high_water(entries, view, launch_environment)


def _records(passed_ids, failed_ids=()):
    return (
        [{"id": i, "case_type": "factual_number", "passed": True, "elapsed_seconds": 200.0}
         for i in passed_ids]
        + [{"id": i, "case_type": "factual_number", "passed": False, "elapsed_seconds": 200.0}
           for i in failed_ids]
    )


def test_unverified_history_still_pairs_when_nothing_has_been_verified(tmp_path):
    # Unchanged behaviour, and the reason the displacement is conditional:
    # this is #92's evidence and there is nothing measured to replace it with.
    _seed_entry(tmp_path, 1, 2, _records(("a", "b")), environment_fingerprint=None)

    assert _high_water_of(tmp_path, _STACK_NOW).iteration == 1


def test_a_verified_entry_displaces_a_higher_unverified_one(tmp_path):
    # The whole point: the aged entry scores better and still loses, because
    # it never said which stack produced that score.
    _seed_entry(tmp_path, 1, 2, _records(("a", "b")), environment_fingerprint=None)
    _seed_entry(tmp_path, 2, 1, _records(("a",), ("b",)), environment_fingerprint=_STACK_NOW)

    high_water = _high_water_of(tmp_path, _STACK_NOW)
    assert high_water.iteration == 2
    assert high_water.objective_adjusted == 1


def test_among_verified_entries_the_best_still_wins(tmp_path):
    # Displacement changes which pool is eligible, not how the best is chosen.
    _seed_entry(tmp_path, 1, 1, _records(("a",), ("b",)), environment_fingerprint=_STACK_NOW)
    _seed_entry(tmp_path, 2, 2, _records(("a", "b")), environment_fingerprint=_STACK_NOW)

    assert _high_water_of(tmp_path, _STACK_NOW).iteration == 2


def test_an_entry_from_a_different_stack_verifies_nothing(tmp_path):
    # DIFFERS was already excluded from pairing; it must also not count as the
    # verified evidence that displaces unverified history, or a campaign run
    # on the wrong stack would silently discard the ledger.
    other_stack = {
        "schema": 1,
        "packages": {"isolated": {"mineru": "2.1.0"}, "product": {"torch": "2.9.0"}},
    }
    _seed_entry(tmp_path, 1, 2, _records(("a", "b")), environment_fingerprint=None)
    _seed_entry(tmp_path, 2, 3, _records(("a", "b")), environment_fingerprint=other_stack)

    assert _high_water_of(tmp_path, _STACK_NOW).iteration == 1


def test_a_launch_that_cannot_read_its_own_stack_verifies_nothing(tmp_path):
    # With no launch fingerprint every comparison is UNKNOWN, so there is no
    # verified pool and the ledger must behave exactly as it did before #107.
    _seed_entry(tmp_path, 1, 2, _records(("a", "b")), environment_fingerprint=None)
    _seed_entry(tmp_path, 2, 1, _records(("a",), ("b",)), environment_fingerprint=_STACK_NOW)

    assert _high_water_of(tmp_path, None).iteration == 1
