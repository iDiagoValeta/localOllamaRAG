"""Tests for the run-to-run comparison used to measure the gate's noise floor."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_runs import compare  # noqa: E402


def _report(outcomes):
    results = [
        {"id": case_id, "model": "m", "passed": passed}
        for case_id, passed in outcomes.items()
    ]
    return {"results": results}


def test_identical_runs_have_no_flips():
    report = _report({"a": True, "b": False})
    result = compare(report, report)
    assert result["flipped_to_pass"] == []
    assert result["flipped_to_fail"] == []
    assert result["stable"] == 2
    assert result["pass_rate_delta"] == 0.0


def test_flips_are_reported_in_both_directions():
    a = _report({"x": False, "y": True, "z": True})
    b = _report({"x": True, "y": False, "z": True})
    result = compare(a, b)
    assert result["flipped_to_pass"] == ["x / m"]
    assert result["flipped_to_fail"] == ["y / m"]
    assert result["stable"] == 1
    assert result["pass_rate_delta"] == 0.0


def test_pass_rate_delta_is_b_minus_a():
    a = _report({"x": False, "y": False})
    b = _report({"x": True, "y": False})
    assert compare(a, b)["pass_rate_delta"] == 0.5


def test_cases_missing_from_one_run_are_rejected():
    # A run that crashed mid-way must not be silently compared as if it were
    # complete -- that would read as a huge improvement or regression.
    a = _report({"x": True, "y": True})
    b = _report({"x": True})
    with pytest.raises(ValueError, match="y / m"):
        compare(a, b)


def test_cases_missing_from_one_run_are_rejected_the_other_way_too():
    # Mirror of the above with the extra case on b instead of a -- the
    # symmetric-difference check must catch either direction, not just "b is
    # missing something a has".
    a = _report({"x": True})
    b = _report({"x": True, "y": True})
    with pytest.raises(ValueError, match="y / m"):
        compare(a, b)


def test_inconclusive_run_is_rejected():
    # A record with infrastructure_error is a case that never ran -- an
    # Ollama timeout or dead server, not a real pass/fail. Comparing it as an
    # ordinary failure would inflate the measured noise floor with a fluke.
    a = _report({"x": True, "y": True})
    b = _report({"x": True, "y": True})
    b["results"][1]["infrastructure_error"] = True
    with pytest.raises(ValueError, match="report_b.*inconclusive"):
        compare(a, b)


def test_inconclusive_run_is_rejected_on_the_other_side_too():
    # Mirror of the above with the infrastructure error on a instead of b --
    # _reject_unusable is called for both reports, but only one side had
    # coverage.
    a = _report({"x": True, "y": True})
    b = _report({"x": True, "y": True})
    a["results"][1]["infrastructure_error"] = True
    with pytest.raises(ValueError, match="report_a.*inconclusive"):
        compare(a, b)


def test_empty_run_is_rejected():
    # Zero cases is the most complete crash of all -- it must not compare as
    # "0 case(s) unchanged", which reads as a clean, noise-free result.
    a = _report({})
    b = _report({"x": True})
    with pytest.raises(ValueError, match="report_a.*no cases"):
        compare(a, b)


def test_empty_run_is_rejected_on_the_other_side_too():
    # Mirror of the above with the empty report on b instead of a.
    a = _report({"x": True})
    b = _report({})
    with pytest.raises(ValueError, match="report_b.*no cases"):
        compare(a, b)


def test_rejection_messages_name_the_files_when_labels_are_given():
    # compare_runs.main() passes the real report paths as labels so a
    # rejection names the actual file instead of the generic parameter name
    # "report_a"/"report_b".
    a = _report({})
    b = _report({"x": True})
    with pytest.raises(ValueError, match="runs/2026-01-01.json"):
        compare(a, b, label_a="runs/2026-01-01.json", label_b="runs/2026-01-02.json")


def test_an_added_conditions_sibling_key_does_not_break_the_comparison():
    # Issue #222 adds a top-level "conditions" key next to "run"/"results" --
    # compare() must keep reading "results" the same way regardless of what
    # else a real artifact carries.
    a = _report({"x": True, "y": False})
    b = _report({"x": True, "y": False})
    a["conditions"] = {"git_commit": {"hash": "aaa1111", "dirty": False}}
    b["conditions"] = {"git_commit": {"hash": "bbb2222", "dirty": True}}
    a["run"] = {"timestamp": "2026-01-01T00:00:00Z"}
    b["run"] = {"timestamp": "2026-01-02T00:00:00Z"}

    result = compare(a, b)

    assert result["flipped_to_pass"] == []
    assert result["flipped_to_fail"] == []
    assert result["stable"] == 2


# --exclude-infrastructure-errors (issue #240)


def _report_two_models(outcomes):
    """Like _report, but ``outcomes`` maps "case / model" keys, so a fixture
    can put an infrastructure error on one model without touching the
    other's record of the same case."""
    results = []
    for key, passed in outcomes.items():
        case_id, model = key.split(" / ")
        results.append({"id": case_id, "model": model, "passed": passed})
    return {"results": results}


def test_excluding_infrastructure_errors_drops_the_pair_from_both_sides():
    # Granite crashing on one study case in run b must not hide the other
    # model's measurements, and must not count as a flip either: the pair
    # is removed from a and b alike, and the removal is reported.
    a = _report_two_models({"x / m1": True, "x / m2": True, "y / m1": False, "y / m2": True})
    b = _report_two_models({"x / m1": True, "x / m2": True, "y / m1": True, "y / m2": True})
    b["results"][3]["infrastructure_error"] = True  # y / m2 never ran in b
    result = compare(a, b, exclude_infrastructure_errors=True)
    assert result["excluded"] == ["y / m2"]
    assert result["flipped_to_pass"] == ["y / m1"]
    assert result["flipped_to_fail"] == []
    assert result["stable"] == 2
    # 3 pairs compared, one of them flipped to pass.
    assert result["pass_rate_delta"] == round(1 / 3, 4)


def test_excluding_infrastructure_errors_still_refuses_a_run_that_measured_nothing():
    # The strict rule stays for the case it was written for: if every pair is
    # an infrastructure error there is nothing to compare, and "0 unchanged"
    # would read as a clean result.
    a = _report({"x": True})
    b = _report({"x": True})
    b["results"][0]["infrastructure_error"] = True
    with pytest.raises(ValueError, match="no comparable"):
        compare(a, b, exclude_infrastructure_errors=True)


def test_the_default_still_rejects_an_inconclusive_run():
    # Opt-in only: the gate's own comparison keeps refusing a report with
    # infrastructure errors unless the caller says otherwise.
    a = _report({"x": True, "y": True})
    b = _report({"x": True, "y": True})
    b["results"][1]["infrastructure_error"] = True
    with pytest.raises(ValueError, match="inconclusive"):
        compare(a, b)
    assert compare(a, b, exclude_infrastructure_errors=True)["excluded"] == ["y / m"]


def test_without_exclusions_the_result_carries_an_empty_excluded_list():
    report = _report({"a": True})
    assert compare(report, report)["excluded"] == []
