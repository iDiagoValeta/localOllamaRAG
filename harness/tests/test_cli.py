"""Tests for harness.cli -- --replay (criterion 7), --set wiring, --status."""

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


# THE LAUNCH LINE (issues #100 and #107). Its whole job is carrying facts a
# campaign depends on before hours are paid, so the lines are tested rather
# than trusted: a mistyped key would drop the warning that matters most while
# everything still ran green.


def _comparability(**overrides):
    base = {
        "history_entries": 3,
        "comparable_search_set_states": 2,
        "high_water_objective_adjusted": 30,
        "high_water_environment_verified": True,
        "environment_verified_states": 2,
        "environment_unverified_states": 0,
        "incomparable_reasons": [],
    }
    base.update(overrides)
    return base


def test_launch_line_reports_counts_and_the_high_water():
    lines = cli.format_comparability_lines(_comparability())
    assert lines[0] == "ledger history: 3 entry(ies), 2 comparable search-set state(s)"
    assert "historical high water: 30" in lines[1]
    assert not any("WARNING" in line for line in lines)


def test_launch_line_warns_when_the_high_water_itself_is_unverified():
    """Issue #107: the one entry pairing will use is the one nobody's stack
    matched. That is the fact that decides whether a refresh is due."""
    lines = cli.format_comparability_lines(
        _comparability(high_water_environment_verified=False,
                       environment_verified_states=1, environment_unverified_states=1)
    )
    warning = [line for line in lines if "high-water entry was NOT measured" in line]
    assert len(warning) == 1
    assert "#107" in warning[0]


def test_launch_line_counts_unverified_states_without_calling_them_wrong():
    lines = cli.format_comparability_lines(
        _comparability(environment_verified_states=1, environment_unverified_states=1)
    )
    unverified = [line for line in lines if "carry no comparable stack fingerprint" in line]
    assert len(unverified) == 1
    assert "unverified, not wrong" in unverified[0]


def test_launch_line_says_nothing_about_the_stack_when_everything_is_verified():
    lines = cli.format_comparability_lines(_comparability())
    assert not any("fingerprint" in line for line in lines)


def test_launch_line_still_names_a_config_mismatch():
    """Issue #100's own warning must survive #107's additions."""
    lines = cli.format_comparability_lines(
        _comparability(
            comparable_search_set_states=0,
            high_water_objective_adjusted=None,
            high_water_environment_verified=None,
            environment_verified_states=0,
            environment_unverified_states=0,
            incomparable_reasons=["models.chat: this launch 'a', ledger 'b'"],
        )
    )
    assert lines[-1] == (
        "  WARNING recovery-mode history mismatch -- models.chat: this launch 'a', ledger 'b'"
    )


# --status (issue #114): the ledger's state without paying a campaign. The
# launch line used to be reachable only after the reference measurement
# (~20 min), or from --dry-run, whose demo evaluator writes a partial
# effective_config that can never match a real AppConfig -- so it reported 0
# comparable states whatever the ledger held.


def _seed_search_set_entry(ledger_dir, iteration, objective, *, fingerprint=None):
    import dataclasses

    from monkeygrab.config.app_config import AppConfig

    ledger.write_entry(
        ledger.LedgerEntry(
            schema_version=ledger.SCHEMA_VERSION,
            iteration=iteration,
            parent_iteration=None,
            git_commit=None,
            config_overrides={},
            effective_config=dataclasses.asdict(AppConfig()),
            proposer="grid",
            proposer_rationale=None,
            proposer_model=None,
            proposer_fallback=False,
            proposer_fallback_reason=None,
            evaluated_case_set="search_set",
            case_records=[
                {"id": "a", "case_type": "factual_number", "passed": True, "elapsed_seconds": 1.0}
            ],
            summary={"total": 1, "passed": 1, "pass_rate": 1.0},
            objective_raw=objective,
            objective_adjusted=objective,
            median_latency_answered_s=1.0,
            median_latency_retrieval_only_s=None,
            reference_median_latency_answered_s=1.0,
            reference_median_latency_retrieval_only_s=None,
            latency_ceiling_multiplier=1.2,
            verdict="accepted",
            reason="seeded",
            environment_fingerprint=fingerprint,
        ),
        ledger_dir,
    )


def test_status_never_calls_the_evaluator(tmp_path, monkeypatch, capsys):
    """The whole point: an answer before the measurement, not after it."""

    def explode(*args, **kwargs):
        raise AssertionError("--status must not evaluate anything")

    monkeypatch.setattr(ev, "real_evaluate", explode)
    _seed_search_set_entry(tmp_path, 1, 27)

    assert cli.main(["--status", "--ledger-dir", str(tmp_path)]) == 0
    assert "ledger: 1 entry(ies)" in capsys.readouterr().out


def test_status_reads_the_ledger_without_writing_to_it(tmp_path, capsys):
    """The other half of "without paying a campaign": --dry-run pointed at a
    real ledger to inspect it APPENDS demo entries to an append-only file,
    indistinguishable from real ones afterwards. --status is read-only."""
    _seed_search_set_entry(tmp_path, 1, 27)
    before = sorted(p.name for p in tmp_path.iterdir())

    cli.main(["--status", "--ledger-dir", str(tmp_path)])

    assert sorted(p.name for p in tmp_path.iterdir()) == before
    out = capsys.readouterr().out
    assert "1 comparable search-set state(s)" in out
    assert "historical high water: 27" in out


def test_status_honours_set_so_it_answers_for_the_campaign_you_would_launch(tmp_path, capsys):
    """A --set pin changes comparability; --status computed under a different
    reference than the campaign would use is worse than no answer."""
    _seed_search_set_entry(tmp_path, 1, 27)

    cli.main(["--status", "--ledger-dir", str(tmp_path), "--set", "models.chat=other:model"])
    out = capsys.readouterr().out
    assert "chat=other:model" in out
    assert "0 comparable search-set state(s)" in out


def test_status_on_a_directory_that_does_not_exist_yet(tmp_path, capsys):
    assert cli.main(["--status", "--ledger-dir", str(tmp_path / "nope")]) == 0
    assert "directory does not exist yet" in capsys.readouterr().out


def test_ledger_summary_of_an_empty_history():
    assert cli.format_ledger_summary([]) == [
        "ledger: empty -- no prior campaign has written here"
    ]


def test_ledger_summary_counts_pre_v3_entries_separately(tmp_path):
    """How much of this history can never be verified against the current
    stack is the number that says whether a refresh campaign is due (#107)."""
    _seed_search_set_entry(tmp_path, 1, 27, fingerprint=None)
    _seed_search_set_entry(tmp_path, 2, 25, fingerprint={"schema": 1, "packages": {"a": "1"}})

    lines = cli.format_ledger_summary(list(ledger.read_history(tmp_path)))
    assert any("1 entry(ies) carry no stack fingerprint" in line for line in lines)
    assert any("iterations 1-2" in line for line in lines)
