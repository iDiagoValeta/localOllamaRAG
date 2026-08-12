"""Criterion-5 simulated test -- issue #31 spec section 6, test 12.

Starting from a deliberately worsened configuration IN A SYNTHETIC LANDSCAPE,
the loop recovers to at least a known target objective without intervention,
run with BOTH proposers. This proves the SEARCH LOGIC recovers in a
controlled, synthetic setting -- it does NOT prove the real pipeline
improves. The real criterion 5 (design doc section 2) needs the GPU and
hours (~2.8 h/candidate) and is documented in harness/README.md as pending
measurement, never claimed here.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from monkeygrab.config.app_config import AppConfig
from harness import evaluator as ev
from harness import loop
from harness.proposers import GridProposer, LlmProposer

_KEY = "retrieval.top_k_final"
# Deliberately not 4 (the first value GridProposer's coordinate descent
# tries for this key) -- forces the grid run to actually search past a
# couple of wrong guesses instead of stumbling onto the answer immediately.
_GOOD_VALUE = 16
_SEARCH_IDS = tuple("abcdefgh")
_FAST_IDS = ("a",)
_WORSENED_OBJECTIVE = 2  # what overrides={} (the reference) scores, out of 8
_GOOD_TARGET = 7  # what {_KEY: _GOOD_VALUE} scores -- the recovery target


def _synthetic_evaluate(overrides, case_ids):
    """Pure, deterministic landscape: empty overrides (the reference) is a
    deliberately worsened start; only the one declared point
    {_KEY: _GOOD_VALUE} recovers. Every other point scores like the
    worsened start, so a proposer cannot stumble into the good point by
    coincidence -- it has to actually be the point proposed.
    """
    good = overrides.get(_KEY) == _GOOD_VALUE
    n_pass = _GOOD_TARGET if good else _WORSENED_OBJECTIVE
    passed_ids = set(_SEARCH_IDS[:n_pass])
    records = tuple(
        ev.CaseRecord(id=cid, case_type="factual_number", passed=cid in passed_ids, elapsed_seconds=200.0)
        for cid in case_ids
    )
    return ev.EvaluationResult(records=records, effective_config=dict(overrides))


def test_grid_proposer_recovers_from_a_worsened_start(tmp_path):
    proposer = GridProposer(AppConfig())
    report = loop.run_loop(
        reference=AppConfig(), evaluate=_synthetic_evaluate, proposer=proposer,
        search_set_ids=_SEARCH_IDS, fast_tier_ids=_FAST_IDS, unreachable_ids=(),
        max_iterations=6, patience=None, ledger_dir=tmp_path, verify_reachability=False,
    )

    assert report["reference"]["objective_adjusted"] == _WORSENED_OBJECTIVE  # confirms the worsened premise
    assert report["ratchet"] >= _GOOD_TARGET

    accepted = [e for e in report["iterations"] if e["verdict"] == "accepted"]
    assert accepted, "grid proposer never found the recovering point within its iteration budget"
    assert accepted[0]["config_overrides"] == {_KEY: _GOOD_VALUE}
    assert accepted[0]["proposer"] == "grid"


def test_llm_proposer_recovers_from_a_worsened_start(tmp_path):
    scripted_response = json.dumps({"overrides": {_KEY: _GOOD_VALUE}, "rationale": "widen top_k_final"})
    proposer = LlmProposer(AppConfig(), call=lambda _prompt: scripted_response)
    report = loop.run_loop(
        reference=AppConfig(), evaluate=_synthetic_evaluate, proposer=proposer,
        search_set_ids=_SEARCH_IDS, fast_tier_ids=_FAST_IDS, unreachable_ids=(),
        max_iterations=3, patience=None, ledger_dir=tmp_path, verify_reachability=False,
    )

    assert report["reference"]["objective_adjusted"] == _WORSENED_OBJECTIVE
    assert report["ratchet"] >= _GOOD_TARGET

    accepted = [e for e in report["iterations"] if e["verdict"] == "accepted"]
    assert accepted, "llm proposer never found the recovering point within its iteration budget"
    assert accepted[0]["config_overrides"] == {_KEY: _GOOD_VALUE}
    assert accepted[0]["proposer"] == "llm"
    assert accepted[0]["proposer_fallback"] is False
    assert accepted[0]["proposer_rationale"] == "widen top_k_final"
