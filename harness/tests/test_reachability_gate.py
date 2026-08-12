"""Tests for the reachability gate -- evaluator.verify_reachable and
loop.run_loop's startup use of it.

Added after a mid-review finding while the sibling PR (#56, the
tests/eval/run_eval.evaluate() library API) was being built: four declared
keys did not reach the generation stage through config_overrides at the
time. #56 has since closed that gap (search_space.PENDING_REACHABILITY_KEYS
is empty -- see its docstring for the full history), and a *different* key
turned out inert for a different reason and was removed from the space
entirely (flags.usar_llm_query_decomposition, issue #64). The gate itself
stays: it is what turns a FUTURE case of either failure class into a loud
startup failure naming the key, instead of the loop silently running on top
of a knob that does nothing. The fake evaluators below use hypothetical key
names to test the mechanism in isolation -- they do not assert anything is
currently broken.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pytest

from monkeygrab.config.app_config import AppConfig
from harness import evaluator as ev
from harness import loop


class _AlwaysHonoursEverything:
    def __call__(self, overrides, case_ids):
        records = tuple(
            ev.CaseRecord(id=c, case_type="factual_number", passed=True, elapsed_seconds=1.0) for c in case_ids
        )
        return ev.EvaluationResult(records=records, effective_config=dict(overrides))


class _RejectsOneKey:
    """Mimics a hard-failing evaluate(): raises, naming the unreachable key."""

    def __init__(self, bad_key):
        self._bad_key = bad_key

    def __call__(self, overrides, case_ids):
        if self._bad_key in overrides:
            raise ValueError(f"override not honoured end to end: {self._bad_key!r}")
        records = tuple(
            ev.CaseRecord(id=c, case_type="factual_number", passed=True, elapsed_seconds=1.0) for c in case_ids
        )
        return ev.EvaluationResult(records=records, effective_config=dict(overrides))


def test_verify_reachable_passes_when_evaluate_honours_every_declared_key():
    ev.verify_reachable(_AlwaysHonoursEverything())  # must not raise


def test_verify_reachable_names_the_offending_key_in_its_message():
    bad_key = "flags.usar_recomp_synthesis"
    with pytest.raises(ev.ReachabilityError, match=bad_key.replace(".", r"\.")):
        ev.verify_reachable(_RejectsOneKey(bad_key))


class _FixedProposer:
    def __init__(self):
        self.last_meta = {"proposer": "grid", "rationale": None, "model": None,
                           "fallback": False, "fallback_reason": None}
        self.called = False

    def propose(self, space, ledger_history, last_result):
        self.called = True
        return {"retrieval.top_k_final": 4}


def test_loop_startup_fails_loudly_instead_of_running_with_an_inert_knob(tmp_path):
    proposer = _FixedProposer()
    with pytest.raises(ev.ReachabilityError):
        loop.run_loop(
            reference=AppConfig(), evaluate=_RejectsOneKey("flags.expandir_contexto"),
            proposer=proposer, search_set_ids=("a",), fast_tier_ids=("a",),
            max_iterations=1, ledger_dir=tmp_path,
        )
    # The gate fires before the loop ever asks the proposer for a candidate,
    # or evaluates anything, or writes a ledger entry.
    assert proposer.called is False
    assert list(tmp_path.glob("*.json")) == []


def test_loop_skips_the_gate_when_verify_reachability_is_false(tmp_path):
    # Only ever meant for tests whose evaluator is intentionally too narrow
    # to answer the combined probe (see run_loop's docstring) -- exercised
    # here with an evaluator that would otherwise fail the gate.
    proposer = _FixedProposer()
    report = loop.run_loop(
        reference=AppConfig(), evaluate=_RejectsOneKey("flags.expandir_contexto"),
        proposer=proposer, search_set_ids=("a",), fast_tier_ids=("a",),
        max_iterations=1, ledger_dir=tmp_path, verify_reachability=False,
    )
    assert report["iterations_run"] == 1
