"""What a model-history row reports, and what it refuses to say.

Issue #146, corrected by #241. `seconds_per_answer` must be the wall time of
the generation call (`generation_seconds`), not tokens divided by the decode
rate -- the old formula understated a measured 7.89 s wait as 0.22 s. The
other running theme is what happens to a record that carries no statistics,
no wall time, or no VRAM sample: it is skipped, never read as zero. Counting
an absence as zero would make the unmeasured case look like the cheapest or
fastest one in the table, which is not a measurement.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diagnostics.model_history_row import format_rows, main, summarise  # noqa: E402


def _record(
    model,
    passed=True,
    rate=None,
    count=None,
    generation_seconds=None,
    eval_duration_s=None,
    budget_exceeded=False,
    infrastructure_error=False,
    vram_fraction=None,
):
    record = {"model": model, "passed": passed}
    if rate is not None:
        record["tokens_per_second"] = rate
    if count is not None:
        record["eval_count"] = count
    if generation_seconds is not None:
        record["generation_seconds"] = generation_seconds
    if eval_duration_s is not None:
        record["eval_duration_s"] = eval_duration_s
    if budget_exceeded:
        record["budget_exceeded"] = True
    if infrastructure_error:
        record["infrastructure_error"] = True
    if vram_fraction is not None:
        record["vram_fraction"] = vram_fraction
    return record


class TestTheWait:
    def test_seconds_per_answer_is_the_wall_time_not_tokens_over_rate(self):
        # The point of #241: count/rate is 22/100 = 0.22 s, which is exactly
        # what the old formula would have reported and exactly the decode
        # time below it -- neither is the 7.89 s the user actually waited.
        record = _record(
            "m", rate=100.0, count=22, generation_seconds=7.89, eval_duration_s=0.22
        )
        summary = summarise([record])[0]["m"]
        assert summary["seconds_per_answer"] == 7.89
        assert summary["decode_seconds_per_answer"] == 0.22

    def test_a_record_with_tokens_but_no_wall_time_skips_only_the_wait(self):
        summary = summarise([_record("m", rate=40, count=10)])[0]["m"]
        assert summary["tokens_per_answer"] == 10
        assert summary["measured"] == 1
        assert summary["measured_wall"] == 0
        assert summary["seconds_per_answer"] is None

    def test_the_median_is_used_so_one_runaway_does_not_move_the_row(self):
        records = [_record("m", rate=40, count=10) for _ in range(4)]
        records.append(_record("m", rate=40, count=4000))
        assert summarise(records)[0]["m"]["tokens_per_answer"] == 10


class TestWhatItRefusesToCount:
    def test_records_without_statistics_are_skipped_not_read_as_zero(self):
        summary = summarise([_record("m", rate=40, count=20), _record("m")])[0]["m"]
        assert summary["tokens_per_answer"] == 20
        assert summary["answered"] == 2
        assert summary["measured"] == 1

    def test_a_model_that_reported_nothing_yields_no_figures(self):
        summary = summarise([_record("m"), _record("m")])[0]["m"]
        assert summary["measured"] == 0
        assert summary["tokens_per_answer"] is None
        assert summary["seconds_per_answer"] is None

    def test_a_rate_without_a_count_gives_no_latency(self):
        # Half a pair cannot produce the number this tool exists to report.
        summary = summarise([_record("m", rate=40)])[0]["m"]
        assert summary["measured"] == 0

    def test_records_with_no_model_are_tallied_as_shared_not_per_model(self):
        per_model, shared = summarise([{"passed": True}])
        assert per_model == {}
        assert shared == {"passed": 1, "total": 1}


class TestBudgetAndInfra:
    def test_budget_and_infra_are_counted_independently(self):
        records = [
            _record("m", budget_exceeded=True),
            _record("m", infrastructure_error=True),
            _record("m"),
        ]
        summary = summarise(records)[0]["m"]
        assert summary["answered"] == 3
        assert summary["budget"] == 1
        assert summary["infra"] == 1


class TestRetrievalOnlyIsSharedNotPerModel:
    def test_shared_cases_are_counted_once_and_never_into_a_models_answered(self):
        records = [
            _record(None, passed=True),
            _record(None, passed=False),
            _record("m", passed=True),
        ]
        per_model, shared = summarise(records)
        assert shared == {"passed": 1, "total": 2}
        assert per_model["m"]["answered"] == 1
        assert per_model["m"]["passed"] == 1


class TestPlacement:
    def test_no_sample_is_not_recorded(self):
        assert summarise([_record("m")])[0]["m"]["placement"] == "not recorded"

    def test_every_sample_fully_on_gpu(self):
        records = [_record("m", vram_fraction=1.0), _record("m", vram_fraction=1.0)]
        assert summarise(records)[0]["m"]["placement"] == "GPU"

    def test_every_sample_fully_on_cpu(self):
        assert summarise([_record("m", vram_fraction=0.0)])[0]["m"]["placement"] == "CPU"

    def test_mixed_samples_report_the_cpu_share_of_the_median_and_that_it_varies(self):
        records = [_record("m", vram_fraction=1.0), _record("m", vram_fraction=0.0)]
        assert summarise(records)[0]["m"]["placement"] == "~50% CPU (varies)"

    def test_identical_partial_samples_report_the_share_without_varies(self):
        records = [_record("m", vram_fraction=0.5), _record("m", vram_fraction=0.5)]
        assert summarise(records)[0]["m"]["placement"] == "~50% CPU"


class TestTheRow:
    def test_an_unmeasured_column_says_so_rather_than_showing_a_number(self):
        rows = format_rows(*summarise([_record("m")]), "run-1")
        assert "not recorded" in rows

    def test_a_partial_denominator_is_shown(self):
        # A median over 1 of 2 records deserves less trust than over 23, and
        # the row has to say which it is.
        rows = format_rows(
            *summarise([_record("m", rate=40, count=20), _record("m")]), "run-1"
        )
        assert "*(of 1)*" in rows

    def test_a_model_with_nothing_measured_does_not_say_of_zero(self):
        # "not recorded *(of 0)*" states the same absence twice.
        rows = format_rows(*summarise([_record("m")]), "run-1")
        assert "*(of 0)*" not in rows

    def test_pass_counts_use_every_answered_case_not_only_measured_ones(self):
        rows = format_rows(
            *summarise(
                [_record("m", passed=True, rate=40, count=20), _record("m", passed=False)]
            ),
            "run-1",
        )
        assert "1 / 2" in rows

    def test_header_and_a_full_row_cell_for_cell(self):
        records = [
            _record(
                "m",
                passed=True,
                rate=40.0,
                count=20,
                generation_seconds=8.0,
                eval_duration_s=0.5,
                vram_fraction=1.0,
            ),
            _record("m", passed=False),
            _record(None, passed=True),
            _record(None, passed=False),
        ]
        rows = format_rows(*summarise(records), "run-42").splitlines()

        assert rows[0] == (
            "| Model | Answered | Overall | tokens/s | tokens/answer | s/answer | "
            "decode s/answer | Budget hit | Infra | Placement | Run |"
        )
        # answered: 1/2 passed; overall folds in the 2 shared cases (1 passed):
        # 2/4. tokens/answer and s/answer are each measured on 1 of 2 records.
        assert rows[2] == (
            "| `m` | 1 / 2 (50.0%) | 2 / 4 (50.0%) | 40.0 | 20 *(of 1)* | "
            "8.00 *(of 1)* | 0.50 | 0 | 0 | GPU | `run-42` |"
        )


class TestReadingARealArtifact:
    """Issue #222 adds a "conditions" block next to "run"/"summary"/"results"
    in the artifact this tool reads -- main() only ever touches "results" and
    "run"/"id", so an artifact carrying the new block must read exactly as it
    did before."""

    def _write_artifact(self, tmp_path):
        payload = {
            "run": {"timestamp": "20260101T000000Z", "stack": "s"},
            "summary": {"overall": {"total": 2, "passed": 1, "pass_rate": 0.5}},
            "results": [
                _record("m", passed=True, rate=40, count=20),
                _record("m", passed=False),
            ],
            "conditions": {
                "config": {"dev": None, "blind": None},
                "versions": {"torch": None},
                "hardware": {"name": None, "vram_total_mib": None},
                "git_commit": {"hash": "abc1234", "dirty": False},
                "gold_sha256": "0" * 64,
                "sampling": {"rag": {"temperature": 0.15}},
                "seed": None,
                "keep_alive_seconds": 120,
            },
        }
        artifact = tmp_path / "artifact.json"
        artifact.write_text(json.dumps(payload), encoding="utf-8")
        return artifact

    def test_main_prints_the_row_unaffected_by_the_conditions_block(self, tmp_path, capsys):
        artifact = self._write_artifact(tmp_path)

        exit_code = main([str(artifact)])

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "1 / 2" in out
        assert "*(of 1)*" in out
        # The Run cell is the timestamp the table cites, not the file's stem.
        assert "`20260101T000000Z`" in out
