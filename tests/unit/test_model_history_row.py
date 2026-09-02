"""The three numbers a model-history row reports, and what it refuses to say.

Issue #146. The interesting behaviour here is not the arithmetic, it is what
happens to records that carry no token statistics: they are skipped, never read
as zero. Counting a model's unreported generation as "0 tokens" would make it
the cheapest model in the table for the one reason that is not a measurement.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diagnostics.model_history_row import format_rows, summarise  # noqa: E402


def _record(model, passed=True, rate=None, count=None):
    record = {"model": model, "passed": passed}
    if rate is not None:
        record["tokens_per_second"] = rate
    if count is not None:
        record["eval_count"] = count
    return record


class TestTheThreeNumbers:
    def test_latency_is_the_tokens_divided_by_the_rate(self):
        # The whole point of #146: 96 tokens at 38.2 tok/s is 2.51 s, and the
        # table only ever showed the 38.2.
        summary = summarise([_record("m", rate=38.2, count=96)])["m"]
        assert summary["tokens_per_answer"] == 96
        assert round(summary["seconds_per_answer"], 2) == 2.51

    def test_a_faster_decoder_can_be_the_slower_model(self):
        summary = summarise(
            [_record("reasoner", rate=38.2, count=96), _record("terse", rate=43.5, count=5)]
        )
        assert summary["reasoner"]["tokens_per_second"] < summary["terse"]["tokens_per_second"]
        assert summary["reasoner"]["seconds_per_answer"] > summary["terse"]["seconds_per_answer"]

    def test_the_median_is_used_so_one_runaway_does_not_move_the_row(self):
        records = [_record("m", rate=40, count=10) for _ in range(4)]
        records.append(_record("m", rate=40, count=4000))
        assert summarise(records)["m"]["tokens_per_answer"] == 10


class TestWhatItRefusesToCount:
    def test_records_without_statistics_are_skipped_not_read_as_zero(self):
        summary = summarise([_record("m", rate=40, count=20), _record("m")])["m"]
        assert summary["tokens_per_answer"] == 20
        assert summary["answered"] == 2
        assert summary["measured"] == 1

    def test_a_model_that_reported_nothing_yields_no_figures(self):
        summary = summarise([_record("m"), _record("m")])["m"]
        assert summary["measured"] == 0
        assert summary["tokens_per_answer"] is None
        assert summary["seconds_per_answer"] is None

    def test_a_rate_without_a_count_gives_no_latency(self):
        # Half a pair cannot produce the number this tool exists to report.
        summary = summarise([_record("m", rate=40)])["m"]
        assert summary["measured"] == 0

    def test_records_with_no_model_are_ignored(self):
        assert summarise([{"passed": True}]) == {}


class TestTheRow:
    def test_an_unmeasured_column_says_so_rather_than_showing_a_number(self):
        rows = format_rows(summarise([_record("m")]), "run-1")
        assert "not recorded" in rows
        assert "0" not in rows.split("|")[4]

    def test_a_partial_denominator_is_shown(self):
        # Three medians over 1 of 2 records deserve less trust than over 23,
        # and the row has to say which it is.
        rows = format_rows(summarise([_record("m", rate=40, count=20), _record("m")]), "run-1")
        assert "*(of 1)*" in rows

    def test_pass_counts_use_every_answered_case_not_only_measured_ones(self):
        rows = format_rows(
            summarise([_record("m", passed=True, rate=40, count=20), _record("m", passed=False)]),
            "run-1",
        )
        assert "1 / 2" in rows

    def test_a_model_with_nothing_measured_does_not_say_of_zero(self):
        # "not recorded *(of 0)*" states the same absence twice.
        rows = format_rows(summarise([_record("m")]), "run-1")
        assert "*(of 0)*" not in rows
