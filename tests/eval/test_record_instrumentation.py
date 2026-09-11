"""Tests for what a generation record says about *when* and *where* it ran.

Issue #234's repeat showed 19 budget exhaustions that were two contiguous
windows of one run, not the models -- and the only way to see that was to
count record positions by hand, because a record carried no clock. Issue
#235 showed a model measured entirely from system RAM with nothing in the
artifact saying so. Three fields close that gap: ``started_at`` on every
record, ``stage_at_budget`` on a budget-exhausted one, and ``vram_fraction``
on every generation record.

No GPU or Ollama server: the rag module is a double and ``/api/ps`` is a
planted ``requests`` module, the same way test_generation_keep_alive.py
satisfies run_eval.py's lazy ``import requests`` in the fast CI gate.
"""

import importlib
import sys
import time
import types
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

import run_eval  # noqa: E402


def _case():
    return {
        "id": "c1", "paper": "p", "case_type": "factual_number", "lang": "en",
        "question": "how many?", "accepted_answers": ["42"],
    }


def _fragment():
    return {"doc": "text", "metadata": {"source": "p.pdf", "page": 1}}


class _FakeRag:
    """Answers "42" and reports what the real silent path reports: token
    counts, plus the stage marker generar_respuesta_silenciosa now sets."""

    def set_model_roles_runtime(self, roles):
        pass

    def generar_respuesta_silenciosa(self, question, fragments, stats=None):
        if stats is not None:
            stats["stage"] = "context"
            stats["stage"] = "generator"
            stats.update({"eval_count": 4, "eval_duration": 40_000_000})
        return "42"


class _FakeRagStuckInContext(_FakeRag):
    """Marks the context stage and never comes back -- RECOMP hanging."""

    def generar_respuesta_silenciosa(self, question, fragments, stats=None):
        if stats is not None:
            stats["stage"] = "context"
        time.sleep(1)
        raise AssertionError("must have been abandoned by the budget")


@pytest.fixture
def no_placement(monkeypatch):
    """Keep placement out of tests that are about the other fields."""
    monkeypatch.setattr(run_eval, "_placement", lambda model: {})


def _is_utc_iso(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0


# started_at (issue #234)


def test_a_generation_record_carries_its_utc_start_time(no_placement):
    record = run_eval._run_factual_case_for_model(_FakeRag(), _case(), [_fragment()], "m", 0.0)
    assert _is_utc_iso(record["started_at"])


def test_a_budget_exhausted_record_carries_its_start_time_too(monkeypatch, no_placement):
    monkeypatch.setattr(run_eval, "GENERATION_BUDGET_SECONDS", 0.05)
    record = run_eval._run_factual_case_for_model(
        _FakeRagStuckInContext(), _case(), [_fragment()], "m", 0.0
    )
    assert record["budget_exceeded"] is True
    assert _is_utc_iso(record["started_at"])


def test_a_retrieval_record_carries_its_start_time(monkeypatch):
    class _Retrieve:
        def run(self, question):
            raise RuntimeError("simulated retrieval failure")

    records, pending = [], []
    case = {**_case(), "source": "corpus"}
    run_eval._run_retrieval_for_corpus([case], _Retrieve(), None, records, pending)
    assert records[0]["infrastructure_error"] is True
    assert _is_utc_iso(records[0]["started_at"])


# stage_at_budget (issue #234)


def test_a_budget_exhausted_factual_record_names_the_stage_that_was_running(monkeypatch, no_placement):
    monkeypatch.setattr(run_eval, "GENERATION_BUDGET_SECONDS", 0.05)
    record = run_eval._run_factual_case_for_model(
        _FakeRagStuckInContext(), _case(), [_fragment()], "m", 0.0
    )
    assert record["stage_at_budget"] == "context"


def test_a_record_that_finished_within_budget_has_no_stage_field(no_placement):
    record = run_eval._run_factual_case_for_model(_FakeRag(), _case(), [_fragment()], "m", 0.0)
    assert "stage_at_budget" not in record


def test_a_budget_exhausted_record_says_unknown_when_nothing_marked_a_stage(monkeypatch, no_placement):
    class _Silent(_FakeRag):
        def generar_respuesta_silenciosa(self, question, fragments, stats=None):
            time.sleep(1)

    monkeypatch.setattr(run_eval, "GENERATION_BUDGET_SECONDS", 0.05)
    record = run_eval._run_factual_case_for_model(_Silent(), _case(), [_fragment()], "m", 0.0)
    assert record["stage_at_budget"] == "unknown"


# vram_fraction (issue #235)


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture
def ps(monkeypatch):
    """Plant a ``requests`` whose GET answers /api/ps with what the test sets."""
    state = {"payload": {"models": []}, "urls": []}

    def _get(url, timeout=None):
        state["urls"].append(url)
        if isinstance(state["payload"], Exception):
            raise state["payload"]
        return _Response(state["payload"])

    fake = types.ModuleType("requests")
    fake.get = _get
    monkeypatch.setitem(sys.modules, "requests", fake)
    return state


def test_placement_is_the_vram_share_ollama_reports_for_that_model(ps):
    ps["payload"] = {"models": [
        {"name": "other", "model": "other", "size": 100, "size_vram": 100},
        {"name": "m", "model": "m", "size": 1000, "size_vram": 640},
    ]}
    assert run_eval._placement("m") == {"vram_fraction": 0.64}
    assert ps["urls"] == [f"{run_eval.OLLAMA_BASE_URL}/api/ps"]


def test_placement_is_absent_when_ollama_no_longer_lists_the_model(ps):
    ps["payload"] = {"models": [{"name": "other", "model": "other", "size": 1, "size_vram": 1}]}
    assert run_eval._placement("m") == {}


def test_placement_is_absent_rather_than_fatal_when_ps_fails(ps):
    # Best-effort like the version lookup: a diagnostic must not cost the
    # run a record that already holds a real pass/fail.
    ps["payload"] = ConnectionError("simulated")
    assert run_eval._placement("m") == {}


def test_placement_lands_on_a_passing_generation_record(monkeypatch):
    monkeypatch.setattr(run_eval, "_placement", lambda model: {"vram_fraction": 1.0})
    record = run_eval._run_factual_case_for_model(_FakeRag(), _case(), [_fragment()], "m", 0.0)
    assert record["vram_fraction"] == 1.0


def test_placement_lands_on_a_budget_exhausted_record(monkeypatch):
    # The model is still loaded when the budget hits -- and where it was
    # loaded is exactly what #235 wants next to a slow record.
    monkeypatch.setattr(run_eval, "GENERATION_BUDGET_SECONDS", 0.05)
    monkeypatch.setattr(run_eval, "_placement", lambda model: {"vram_fraction": 0.3})
    record = run_eval._run_factual_case_for_model(
        _FakeRagStuckInContext(), _case(), [_fragment()], "m", 0.0
    )
    assert record["vram_fraction"] == 0.3


def test_placement_is_read_for_the_model_under_test(monkeypatch):
    seen = []
    monkeypatch.setattr(run_eval, "_placement", lambda model: seen.append(model) or {})
    run_eval._run_factual_case_for_model(_FakeRag(), _case(), [_fragment()], "the-model", 0.0)
    assert seen == ["the-model"]


# OLLAMA_BASE_URL (issue #244)


def test_the_gates_own_endpoint_follows_ollama_base_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://eval-host.test:9999/")
    try:
        importlib.reload(run_eval)
        assert run_eval.OLLAMA_BASE_URL == "http://eval-host.test:9999"
    finally:
        monkeypatch.delenv("OLLAMA_BASE_URL")
        importlib.reload(run_eval)
