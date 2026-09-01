"""Which evaluator produced an entry, and the refusal that keeps the two apart.

Issue #115. Pointing ``--dry-run`` at a real ledger to inspect it appended two
synthetic iterations to an append-only evidence file, indistinguishable from
measured ones afterwards. ``harness/README.md`` already claimed the protection
existed -- "``--dry-run`` deliberately does NOT default to this directory ...
precisely so demo runs never mix into that history" -- and the mechanism
covered the default while leaving the explicit ``--ledger-dir`` open, which is
the flag an operator reaches for when they want a *specific* ledger.

A demo entry is not inert once written. ``GridProposer`` skips points "already
tried" by reading history back, and ``_historical_high_water`` picks a pairing
baseline from it, so a synthetic iteration that happens to score well becomes
a baseline a real campaign is judged against.

Two decisions, both tested here:

1. **``evaluator`` is recorded per entry, defaulting to ``None``.** Not
   ``"real"``: entries written before this field cannot be known to be real
   (that is the whole defect), and asserting it would be the same mistake as
   treating an unknown stack fingerprint as a match (#107).
2. **A demo run refuses a ledger that holds anything not known to be demo**,
   rather than filtering or warning. The repo's hard-fail policy, and the
   only answer that cannot be missed in a log.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from harness import cli  # noqa: E402
from harness import ledger as ledger_mod  # noqa: E402


def _entry(iteration: int, evaluator):
    return ledger_mod.LedgerEntry(
        schema_version=ledger_mod.SCHEMA_VERSION,
        iteration=iteration,
        parent_iteration=None,
        git_commit=None,
        config_overrides={},
        effective_config={},
        proposer="grid",
        proposer_rationale=None,
        proposer_model=None,
        proposer_fallback=False,
        proposer_fallback_reason=None,
        evaluated_case_set="search_set",
        case_records=[],
        summary={"total": 0, "passed": 0, "pass_rate": 0.0},
        objective_raw=0,
        objective_adjusted=0,
        median_latency_answered_s=None,
        median_latency_retrieval_only_s=None,
        reference_median_latency_answered_s=None,
        reference_median_latency_retrieval_only_s=None,
        latency_ceiling_multiplier=1.2,
        verdict="accepted",
        reason="seeded",
        evaluator=evaluator,
    )


class TestTheFieldItself:
    def test_an_entry_defaults_to_an_unknown_evaluator(self):
        """Absence is not a claim. An entry written before this field may well
        have come from a dry-run -- that is the defect being fixed."""
        entry = ledger_mod.LedgerEntry.from_dict(
            {k: v for k, v in _entry(1, None).to_dict().items() if k != "evaluator"}
        )
        assert entry.evaluator is None

    def test_the_field_survives_a_write_and_read(self, tmp_path):
        ledger_mod.write_entry(_entry(1, ledger_mod.EVALUATOR_DEMO), tmp_path)
        assert ledger_mod.read_history(tmp_path)[0].evaluator == ledger_mod.EVALUATOR_DEMO


class TestTheRefusal:
    def test_a_demo_run_refuses_a_ledger_holding_real_entries(self, tmp_path, capsys):
        ledger_mod.write_entry(_entry(1, ledger_mod.EVALUATOR_REAL), tmp_path)

        code = cli.main(["--dry-run", "--max-iterations", "1", "--ledger-dir", str(tmp_path)])

        assert code == 2
        message = capsys.readouterr().err
        assert "1 entry(ies) this dry-run must not write beside" in message
        assert "--ledger-dir" in message

    def test_a_demo_run_refuses_a_ledger_of_unknown_provenance(self, tmp_path, capsys):
        """Pre-v4 entries. Unknown is not permission."""
        ledger_mod.write_entry(_entry(1, None), tmp_path)

        assert cli.main(["--dry-run", "--max-iterations", "1", "--ledger-dir", str(tmp_path)]) == 2
        assert "unknown" in capsys.readouterr().err

    def test_the_refusal_writes_nothing(self, tmp_path):
        """A refusal that appended first would be no protection at all."""
        ledger_mod.write_entry(_entry(1, ledger_mod.EVALUATOR_REAL), tmp_path)
        before = sorted(p.name for p in tmp_path.iterdir())

        cli.main(["--dry-run", "--max-iterations", "1", "--ledger-dir", str(tmp_path)])

        assert sorted(p.name for p in tmp_path.iterdir()) == before

    def test_a_demo_run_continues_into_a_ledger_of_its_own_demo_entries(self, tmp_path):
        """Iterating on the proposer against a scratch directory stays legal --
        the refusal is about mixing, not about reusing a demo ledger."""
        ledger_mod.write_entry(_entry(1, ledger_mod.EVALUATOR_DEMO), tmp_path)

        assert cli.main(["--dry-run", "--max-iterations", "1", "--ledger-dir", str(tmp_path)]) == 0
        assert len(ledger_mod.read_history(tmp_path)) == 2

    def test_a_demo_run_with_no_ledger_dir_is_unaffected(self):
        """The default already wrote to a temp dir; that path must not change."""
        assert cli.main(["--dry-run", "--max-iterations", "1"]) == 0


class TestTheReverseDirection:
    def test_a_real_campaign_reports_demo_entries_instead_of_pairing_against_them(self, tmp_path):
        """Filtering silently is what this repo does not do. The campaign runs;
        the line says what was set aside and why."""
        entries = [_entry(1, ledger_mod.EVALUATOR_DEMO), _entry(2, ledger_mod.EVALUATOR_REAL)]
        lines = cli.format_ledger_summary(entries)
        assert any("1 entry(ies) came from the demo evaluator" in line for line in lines)


def test_demo_entries_are_never_a_pairing_baseline(tmp_path):
    """The consequence that makes this a bug and not untidiness: a demo
    iteration scoring well would otherwise become the high water a real
    campaign is judged against."""
    from harness import loop

    demo = _entry(1, ledger_mod.EVALUATOR_DEMO)
    demo = ledger_mod.LedgerEntry.from_dict({**demo.to_dict(), "objective_adjusted": 99})
    ledger_mod.write_entry(demo, tmp_path)
    ledger_mod.write_entry(
        ledger_mod.LedgerEntry.from_dict(
            {**_entry(2, ledger_mod.EVALUATOR_REAL).to_dict(), "objective_adjusted": 27}
        ),
        tmp_path,
    )

    entries = list(ledger_mod.read_history(tmp_path))
    view = loop._comparable_config_view({})
    high_water = loop._historical_high_water(entries, view)
    assert high_water is not None and high_water.iteration == 2, (
        "a demo entry must never be selected as the pairing baseline"
    )


@pytest.mark.parametrize("value", [ledger_mod.EVALUATOR_DEMO, ledger_mod.EVALUATOR_REAL, None])
def test_every_accepted_evaluator_value_round_trips_through_json(value, tmp_path):
    ledger_mod.write_entry(_entry(1, value), tmp_path)
    assert ledger_mod.read_history(tmp_path)[0].evaluator == value
