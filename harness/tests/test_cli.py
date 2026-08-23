"""Tests for harness.cli -- --replay (criterion 7) and --set wiring."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from harness import cli
from harness import evaluator as ev
from harness import ledger


def _entry_from_result(result, *, iteration=1, overrides=None):
    records = [
        {
            "id": r.id, "case_type": r.case_type,
            "passed": r.passed, "elapsed_seconds": r.elapsed_seconds,
        }
        for r in result.records
    ]
    passed = sum(1 for r in result.records if r.passed)
    return ledger.LedgerEntry(
        schema_version=ledger.SCHEMA_VERSION,
        iteration=iteration,
        parent_iteration=None,
        git_commit="deadbeef",
        config_overrides=overrides or {},
        effective_config=result.effective_config,
        proposer="grid",
        proposer_rationale=None,
        proposer_model=None,
        proposer_fallback=False,
        proposer_fallback_reason=None,
        evaluated_case_set="search_set",
        case_records=records,
        summary={"total": len(records), "passed": passed, "pass_rate": 0.0},
        objective_raw=passed,
        objective_adjusted=passed,
        median_latency_answered_s=None,
        median_latency_retrieval_only_s=None,
        reference_median_latency_answered_s=None,
        reference_median_latency_retrieval_only_s=None,
        latency_ceiling_multiplier=1.20,
        verdict="rejected_no_gain",
        reason="test",
    )


def test_parse_args_replay():
    args = cli.parse_args(["--replay", "2", "--ledger-dir", "somewhere"])
    assert args.replay == 2
    assert args.ledger_dir == Path("somewhere")


def test_parse_set_overrides_decodes_json_and_keeps_raw_strings():
    """Ints and bools must survive the CLI; model pins stay strings."""
    overrides = cli.parse_set_overrides([
        "retrieval.top_k_final=1", "flags.usar_reranker=false", "models.rag=gemma4:e4b",
    ])
    assert overrides == {
        "retrieval.top_k_final": 1,
        "flags.usar_reranker": False,
        "models.rag": "gemma4:e4b",
    }


def test_parse_set_overrides_rejects_a_pair_without_equals():
    import pytest

    for bad in ("retrieval.top_k_final", "=1"):
        with pytest.raises(ValueError, match="KEY=VALUE"):
            cli.parse_set_overrides([bad])


def test_set_overrides_reach_the_reference_and_unknown_keys_fail_hard(tmp_path, capsys):
    """--set feeds AppConfig.with_overrides: a known key changes the campaign's
    reference, an unknown key aborts before any evaluation is paid."""
    code = cli.main([
        "--dry-run", "--max-iterations", "1", "--ledger-dir", str(tmp_path),
        "--set", "retrieval.top_k_final=6",
    ])
    assert code == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    # The pin reached the MEASURED pipeline, not only the bookkeeping object:
    # the demo evaluator echoes its applied overrides as effective_config.
    assert report["reference"]["overrides_applied"] == {"retrieval.top_k_final": 6}

    code = cli.main([
        "--dry-run", "--max-iterations", "1", "--ledger-dir", str(tmp_path),
        "--set", "retrieval.no_such_field=6",
    ])
    assert code == 2
    assert "no_such_field" in capsys.readouterr().err


def test_cli_replay_exits_zero_when_the_demo_evaluator_matches(tmp_path):
    """End-to-end criterion 7 against the in-process demo landscape.

    Write a ledger entry from a demo evaluation, then ``--replay`` that
    iteration with the same demo evaluator -- the pass vector must match,
    which is the reconstruct-and-rerun path the first real #71 run never
    had as a command.
    """
    evaluate = ev.build_demo_evaluator()
    ids = ev.search_set_case_ids()[:3]
    overrides = {"retrieval.top_k_final": 8}
    result = evaluate(overrides, ids)
    ledger.write_entry(_entry_from_result(result, overrides=overrides), ledger_dir=tmp_path)

    code = cli.main(["--dry-run", "--replay", "1", "--ledger-dir", str(tmp_path)])
    assert code == 0


def test_cli_replay_exits_nonzero_when_the_iteration_is_missing(tmp_path, capsys):
    code = cli.main(["--dry-run", "--replay", "9", "--ledger-dir", str(tmp_path)])
    assert code == 1
    assert "iteration 9" in capsys.readouterr().err
