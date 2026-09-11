"""Tests for run_study_case: the generation budget (issue #229) and the
missing elapsed-time print the issue's own follow-up comment asked for.

Measured on the campaign that filed #229: of 289 generations, the 34
study_* ones cost ~90 minutes between them (one, ricci-study-summary-es,
took 26 minutes alone) and none of them printed how long they took --
only the factual lines did, as "(6.8s)". That made the cost invisible
exactly where it was concentrated. Both are fixed here: a study generation
is now bounded by the same GENERATION_BUDGET_SECONDS as a factual one, and
every study_* console line -- BUDGET, FAIL (malformed), ERROR
(infrastructure) and PASS/FAIL (graded) -- now reports elapsed time the
same way the factual path does.

run_study_case imports rag.engine.wiring and monkeygrab.application.study
internally, which pulls in the real engine (Ollama client, torch via
sentence-transformers). The ``import rag.chat_pdfs`` below is what lets
tests/conftest.py's heuristic skip this file in the dependency-free fast
CI gate -- see test_evaluate_library_api.py's module docstring for the
same pattern. Study.summarize is monkeypatched directly in every test, so
no GPU, Ollama server or network access is needed to run these.
"""

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rag.chat_pdfs  # noqa: E402,F401  (see module docstring)
import run_eval  # noqa: E402
from monkeygrab.application.study import (  # noqa: E402
    DocumentSummary,
    MalformedSummaryError,
    Study,
    SummarySection,
)


def _case():
    return {
        "id": "study-1", "paper": "p", "case_type": "study_summary", "lang": "en",
    }


def _fragments():
    return [{"doc": "text", "metadata": {"source": "p.pdf", "page": 1}}]


_ELAPSED_PATTERN = re.compile(r"\(\d+\.\d+s\)")


def test_a_study_generation_that_overruns_the_budget_is_budget_exceeded(monkeypatch, capsys):
    monkeypatch.setattr(run_eval, "GENERATION_BUDGET_SECONDS", 0.05)

    def _slow_summarize(self, fragments, config, *, language=None):
        time.sleep(1)
        raise AssertionError("must have been abandoned by the budget")

    monkeypatch.setattr(Study, "summarize", _slow_summarize)

    records = run_eval.run_study_case(_case(), _fragments(), ["m"], retrieval_elapsed=0.0)

    assert len(records) == 1
    assert records[0]["budget_exceeded"] is True
    assert records[0]["passed"] is False
    assert "infrastructure_error" not in records[0]
    out = capsys.readouterr().out
    assert "[BUDGET]" in out
    assert _ELAPSED_PATTERN.search(out), f"no elapsed time printed for the BUDGET line: {out!r}"


def test_a_malformed_artifact_is_a_fail_not_budget_exceeded_and_prints_elapsed(monkeypatch, capsys):
    monkeypatch.setattr(run_eval, "GENERATION_BUDGET_SECONDS", 5)

    def _bad_summarize(self, fragments, config, *, language=None):
        raise MalformedSummaryError("not a list")

    monkeypatch.setattr(Study, "summarize", _bad_summarize)

    records = run_eval.run_study_case(_case(), _fragments(), ["m"], retrieval_elapsed=0.0)

    assert len(records) == 1
    assert records[0]["passed"] is False
    assert "budget_exceeded" not in records[0]
    assert "infrastructure_error" not in records[0]
    out = capsys.readouterr().out
    assert "[FAIL]" in out
    assert _ELAPSED_PATTERN.search(out), f"no elapsed time printed for the FAIL line: {out!r}"


def test_a_generation_failure_is_an_infrastructure_error_and_prints_elapsed(monkeypatch, capsys):
    monkeypatch.setattr(run_eval, "GENERATION_BUDGET_SECONDS", 5)

    def _dead_ollama(self, fragments, config, *, language=None):
        raise RuntimeError("simulated Ollama unreachable")

    monkeypatch.setattr(Study, "summarize", _dead_ollama)

    records = run_eval.run_study_case(_case(), _fragments(), ["m"], retrieval_elapsed=0.0)

    assert len(records) == 1
    assert records[0]["infrastructure_error"] is True
    assert "budget_exceeded" not in records[0]
    out = capsys.readouterr().out
    assert "[ERROR]" in out
    assert _ELAPSED_PATTERN.search(out), f"no elapsed time printed for the ERROR line: {out!r}"


def test_a_graded_result_within_budget_prints_elapsed(monkeypatch, capsys):
    monkeypatch.setattr(run_eval, "GENERATION_BUDGET_SECONDS", 5)

    def _fast_summarize(self, fragments, config, *, language=None):
        sections = tuple(
            SummarySection(heading=f"H{i}", body=f"body {i}", source_pages=(1,))
            for i in range(3)
        )
        return DocumentSummary(sections=sections, source_document="p.pdf")

    monkeypatch.setattr(Study, "summarize", _fast_summarize)

    records = run_eval.run_study_case(_case(), _fragments(), ["m"], retrieval_elapsed=0.0)

    assert len(records) == 1
    assert records[0]["passed"] is True
    assert "budget_exceeded" not in records[0]
    assert "infrastructure_error" not in records[0]
    out = capsys.readouterr().out
    assert "[PASS]" in out
    assert _ELAPSED_PATTERN.search(out), f"no elapsed time printed for the PASS line: {out!r}"
