"""The summary must not blend retrieval-only cases with answered ones."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import build_summary  # noqa: E402


def _record(case_type, passed, model=None):
    return {
        "id": f"{case_type}-{passed}", "paper": "p", "case_type": case_type,
        "lang": "en", "model": model, "passed": passed, "reason": "",
    }


def test_retrieval_and_answer_are_reported_apart():
    records = [
        _record("figure_retrieval", True),
        _record("table_retrieval", True),
        _record("factual_number", False, "m"),
        _record("factual_concept", True, "m"),
    ]
    summary = build_summary(records)

    assert summary["retrieval_only"] == {"total": 2, "passed": 2, "pass_rate": 1.0}
    assert summary["answer"] == {"total": 2, "passed": 1, "pass_rate": 0.5}


def test_overall_keeps_its_existing_meaning():
    # The baseline ratchet is calibrated against this number. Changing what it
    # counts would silently change what the gate accepts.
    records = [
        _record("figure_retrieval", True),
        _record("factual_number", False, "m"),
    ]
    assert build_summary(records)["overall"] == {"total": 2, "passed": 1, "pass_rate": 0.5}


def test_empty_buckets_do_not_divide_by_zero():
    summary = build_summary([_record("figure_retrieval", True)])
    assert summary["answer"] == {"total": 0, "passed": 0, "pass_rate": 0.0}
