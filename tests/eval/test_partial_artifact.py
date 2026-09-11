"""Tests for the eval runner's incremental partial artifact (issue #219).

Before this, every record lived only in memory until evaluate()'s single
``report_path.write_text(...)`` at the very end -- the last full run took 64
minutes for 156 cases and one model, so a crash, an OOM or a ^C partway
through a multi-hour sweep left tests/eval/runs/ empty, measurements and
all. _RecordSink streams each record to a JSONL sidecar as it lands, so a
killed run still leaves its already-computed records on disk, in a form
readable independently of the final report.

Pure: no ``rag`` import, no GPU, no Ollama, no network -- _RecordSink is a
plain list subclass and this exercises run_all_cases's phase 1 (retrieval)
with the same fakes test_phase1_one_corpus_at_a_time.py uses, so this file
stays collected in the dependency-free fast CI gate. evaluate()-level
coverage (a run that dies mid-generation, and cleanup on a completed run)
lives in test_evaluate_library_api.py instead, next to evaluate()'s other
tests, which already need the real engine installed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

import run_eval  # noqa: E402
from run_eval import _RecordSink  # noqa: E402


# _RecordSink


def test_a_sink_with_no_path_behaves_like_a_plain_list():
    sink = _RecordSink(None)
    sink.append({"id": "a"})
    sink.extend([{"id": "b"}, {"id": "c"}])
    assert list(sink) == [{"id": "a"}, {"id": "b"}, {"id": "c"}]


def test_append_writes_one_readable_json_line(tmp_path):
    path = tmp_path / "run.partial.jsonl"
    sink = _RecordSink(path)

    sink.append({"id": "a", "passed": True})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"id": "a", "passed": True}


def test_extend_writes_one_line_per_record(tmp_path):
    path = tmp_path / "run.partial.jsonl"
    sink = _RecordSink(path)

    sink.extend([{"id": "a"}, {"id": "b"}])

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [{"id": "a"}, {"id": "b"}]


def test_append_and_extend_both_land_in_the_same_file_in_order(tmp_path):
    path = tmp_path / "run.partial.jsonl"
    sink = _RecordSink(path)

    sink.append({"id": "a"})
    sink.extend([{"id": "b"}, {"id": "c"}])
    sink.append({"id": "d"})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["a", "b", "c", "d"]


def test_the_sink_still_carries_every_record_in_memory_too(tmp_path):
    """Nothing downstream of run_all_cases (build_summary, the final report's
    "results" list) may lose records just because a sink path was given."""
    path = tmp_path / "run.partial.jsonl"
    sink = _RecordSink(path)

    sink.append({"id": "a"})
    sink.extend([{"id": "b"}])

    assert list(sink) == [{"id": "a"}, {"id": "b"}]


def test_sort_reorders_the_in_memory_list_without_touching_the_file(tmp_path):
    """_restore_case_order sorts run_all_cases's records in place -- the file
    is an append log of production order, which is fine to leave as-is."""
    path = tmp_path / "run.partial.jsonl"
    sink = _RecordSink(path)
    sink.extend([{"id": "b"}, {"id": "a"}])

    sink.sort(key=lambda r: r["id"])

    assert list(sink) == [{"id": "a"}, {"id": "b"}]
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["b", "a"]


# run_all_cases + partial_sink (phase 1 only -- retrieval-only case types
# need no Ollama, matching test_phase1_one_corpus_at_a_time.py's fakes)


class _FakeResult:
    fragments = ()


class _FakeRetrieve:
    def run(self, question):
        return _FakeResult()


class _FakeEvidence:
    def select_evidence(self, fragments):
        return [], {}


def _case(case_id):
    return {
        "id": case_id, "paper": "p", "case_type": "figure_retrieval", "lang": "en",
        "source": "corpus", "question": case_id,
    }


@pytest.fixture(autouse=True)
def _no_gpu_release(monkeypatch):
    monkeypatch.setattr(run_eval, "_release_gpu_models", lambda *_a, **_kw: None)
    monkeypatch.setattr(run_eval, "_release_ollama_models", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        run_eval,
        "run_retrieval_case",
        lambda case, retrieved, elapsed: {
            "id": case["id"], "paper": case["paper"], "case_type": case["case_type"],
            "lang": case["lang"], "model": None, "passed": True, "reason": "fake",
            "elapsed_seconds": round(elapsed, 2),
        },
    )


def test_run_all_cases_streams_every_record_to_the_partial_sink(tmp_path):
    sink_path = tmp_path / "run.partial.jsonl"
    cases = [_case("c1"), _case("c2"), _case("c3")]

    records = run_eval.run_all_cases(
        None, cases, _FakeRetrieve(), None, _FakeEvidence(), None,
        models=[], partial_sink=sink_path,
    )

    assert len(records) == 3
    lines = sink_path.read_text(encoding="utf-8").splitlines()
    written = [json.loads(line) for line in lines]
    assert {r["id"] for r in written} == {"c1", "c2", "c3"}
    # Every record run_all_cases returned is on disk too -- a reader does not
    # need the process to still be alive to see them.
    assert sorted(written, key=lambda r: r["id"]) == sorted(records, key=lambda r: r["id"])


def test_run_all_cases_with_no_sink_writes_nothing_to_disk(tmp_path):
    """partial_sink defaults to None -- every caller that predates issue
    #219, and every test that stubs run_all_cases out entirely, keeps
    behaving exactly as before."""
    records = run_eval.run_all_cases(
        None, [_case("c1")], _FakeRetrieve(), None, _FakeEvidence(), None, models=[],
    )

    assert len(records) == 1
    assert list(tmp_path.iterdir()) == []
