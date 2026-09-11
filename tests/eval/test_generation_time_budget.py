"""Tests for the eval runner's per-generation wall-clock budget (issue #229).

Measured on a real campaign (2026-09-11): the rag generator runs with
num_predict -1 and nothing else bounded a single case, so one uncooperative
generation held the run -- and the GPU -- for up to 45 minutes. This budget
caps a single generation (one model x one case call) instead.

``_run_with_budget`` and ``run_factual_case`` need nothing from ``rag`` at
import time -- ``run_factual_case`` takes its ``rag`` module as a plain
argument, so a fake stands in for it here -- which keeps this file collected
in the dependency-free fast CI gate (tests/conftest.py's heuristic only skips
a file that imports ``rag`` itself). ``run_study_case``'s equivalent
coverage lives in test_run_study_case.py instead, because that function
imports rag.engine.wiring internally and needs the real engine installed.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

import run_eval  # noqa: E402
from run_eval import (  # noqa: E402
    GenerationBudgetExceeded,
    _run_with_budget,
    run_factual_case,
)


# _run_with_budget


def test_returns_the_result_when_fn_finishes_in_time():
    assert _run_with_budget(lambda: 42, budget_seconds=5) == 42


def test_raises_generation_budget_exceeded_when_fn_overruns():
    with pytest.raises(GenerationBudgetExceeded):
        _run_with_budget(lambda: time.sleep(1), budget_seconds=0.05)


def test_reraises_fns_own_exception_when_it_fails_within_budget():
    def _boom():
        raise ValueError("simulated model failure")

    with pytest.raises(ValueError, match="simulated model failure"):
        _run_with_budget(_boom, budget_seconds=5)


def test_an_overrun_does_not_block_the_caller_past_the_budget():
    """The abandoned thread is not joined again -- the whole point is that
    the caller gets its budget back, not the wall-clock time fn actually
    used."""
    start = time.perf_counter()
    with pytest.raises(GenerationBudgetExceeded):
        _run_with_budget(lambda: time.sleep(2), budget_seconds=0.05)
    assert time.perf_counter() - start < 1.0


# run_factual_case


class _FakeRagBudgetExceeded:
    """generar_respuesta_silenciosa that never returns in time."""

    def set_model_roles_runtime(self, roles):
        pass

    def generar_respuesta_silenciosa(self, question, fragments, stats=None):
        time.sleep(1)
        raise AssertionError("must have been abandoned by the budget")


class _FakeRagInfrastructureError:
    """generar_respuesta_silenciosa that fails fast, not by overrunning."""

    def set_model_roles_runtime(self, roles):
        pass

    def generar_respuesta_silenciosa(self, question, fragments, stats=None):
        raise ConnectionError("simulated Ollama unreachable")


def _fragment():
    return {"doc": "text", "metadata": {"source": "p.pdf", "page": 1}}


def _case():
    return {
        "id": "c1", "paper": "p", "case_type": "factual_number", "lang": "en",
        "question": "how many?",
    }


def test_a_generation_that_overruns_the_budget_is_marked_budget_exceeded(monkeypatch):
    monkeypatch.setattr(run_eval, "GENERATION_BUDGET_SECONDS", 0.05)

    records = run_factual_case(
        _FakeRagBudgetExceeded(), _case(), [_fragment()], ["m"], retrieval_elapsed=0.0
    )

    assert len(records) == 1
    record = records[0]
    assert record["budget_exceeded"] is True
    assert record["passed"] is False
    assert "infrastructure_error" not in record


def test_a_budget_exceeded_record_is_never_also_an_infrastructure_error(monkeypatch):
    """The distinction is the point of the issue: infrastructure_error makes
    the whole run inconclusive against the baseline, budget_exceeded must
    not."""
    monkeypatch.setattr(run_eval, "GENERATION_BUDGET_SECONDS", 0.05)

    records = run_factual_case(
        _FakeRagBudgetExceeded(), _case(), [_fragment()], ["m"], retrieval_elapsed=0.0
    )

    assert records[0].get("infrastructure_error") is not True


def test_a_real_failure_within_budget_is_still_an_infrastructure_error(monkeypatch):
    """Regression guard for the other direction: a fast failure (dead
    Ollama) must not start being misreported as budget_exceeded just because
    the budget machinery now wraps the call."""
    monkeypatch.setattr(run_eval, "GENERATION_BUDGET_SECONDS", 5)

    records = run_factual_case(
        _FakeRagInfrastructureError(), _case(), [_fragment()], ["m"], retrieval_elapsed=0.0
    )

    assert len(records) == 1
    record = records[0]
    assert record["infrastructure_error"] is True
    assert "budget_exceeded" not in record


def test_a_generation_within_budget_passes_through_untouched(monkeypatch):
    """A budget generous enough for a normal answer must not change the
    non-budget-exceeded outcome (point 5 of the verification plan: the
    budget must not break the healthy case)."""
    monkeypatch.setattr(run_eval, "GENERATION_BUDGET_SECONDS", 5)

    class _FakeRagFast:
        def set_model_roles_runtime(self, roles):
            pass

        def generar_respuesta_silenciosa(self, question, fragments, stats=None):
            return "42"

    records = run_factual_case(
        _FakeRagFast(), _case(), [_fragment()], ["m"], retrieval_elapsed=0.0
    )

    assert len(records) == 1
    assert "budget_exceeded" not in records[0]
    assert "infrastructure_error" not in records[0]
