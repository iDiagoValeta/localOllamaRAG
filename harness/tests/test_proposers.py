"""Tests for harness/proposers.py -- issue #31 spec section 6, tests 7-8."""

import json
import sys
from collections import namedtuple
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pytest

from monkeygrab.config.app_config import AppConfig
from harness import search_space
from harness.proposers import (
    GridProposer, LlmProposer, OllamaUnreachableError, SearchSpaceExhausted,
)

# A minimal stand-in for ledger.LedgerEntry: GridProposer/LlmProposer only
# ever read .config_overrides off a ledger_history entry, so tests do not
# need the full ledger schema to exercise the proposer logic in isolation.
_FakeEntry = namedtuple("_FakeEntry", ["config_overrides", "objective_adjusted", "iteration", "verdict"])


def _entry(overrides, objective=0, iteration=1, verdict="rejected_no_gain"):
    return _FakeEntry(config_overrides=overrides, objective_adjusted=objective, iteration=iteration, verdict=verdict)


class _TinySpace:
    """A tiny two-value declared space, for testing GridProposer's exhaustion path
    without walking the full (much larger) real search_space module."""

    ALLOWED_VALUES = {"retrieval.top_k_final": (4, 6)}

    @staticmethod
    def proposal_order():
        return ("retrieval.top_k_final",)

    @staticmethod
    def is_feasible(overrides, reference):
        return True


# TEST 7 -- GridProposer never repeats a ledger point; reproducible.


def test_grid_proposer_first_proposal_is_the_first_reachable_key_first_value():
    proposer = GridProposer(AppConfig())
    overrides = proposer.propose(search_space, [], None)
    first_key = search_space.proposal_order()[0]
    assert list(overrides.keys()) == [first_key]
    assert proposer.last_meta["proposer"] == "grid"


def test_grid_proposer_skips_points_already_in_the_ledger():
    base = AppConfig()
    first = GridProposer(base).propose(search_space, [], None)
    history = [_entry(first)]
    second = GridProposer(base).propose(search_space, history, None)
    assert second != first


def test_grid_proposer_is_reproducible_given_the_same_ledger():
    base = AppConfig()
    history = [_entry({"retrieval.top_k_final": 4}, objective=20)]
    a = GridProposer(base).propose(search_space, history, None)
    b = GridProposer(base).propose(search_space, history, None)
    assert a == b


def test_grid_proposer_skips_the_reference_value_itself():
    # AppConfig()'s default top_k_final is already 8, which is one of the
    # declared values -- proposing "change to 8" would not be a change.
    base = AppConfig()
    for _ in range(len(search_space.ALLOWED_VALUES["retrieval.top_k_final"])):
        overrides = GridProposer(base).propose(search_space, [], None)
        assert overrides.get("retrieval.top_k_final") != 8


def test_grid_proposer_raises_search_space_exhausted_once_every_point_is_tried():
    base = AppConfig()  # default top_k_final == 8, so both 4 and 6 are "changes"
    proposer = GridProposer(base)
    first = proposer.propose(_TinySpace(), [], None)
    history = [_entry(first)]
    second = proposer.propose(_TinySpace(), history, None)
    history.append(_entry(second))
    with pytest.raises(SearchSpaceExhausted):
        proposer.propose(_TinySpace(), history, None)


# TEST 8 -- LlmProposer validation, malformed-JSON rejection, fallback.


def test_llm_proposer_accepts_a_valid_response():
    response = json.dumps({"overrides": {"retrieval.top_k_final": 12}, "rationale": "widen final k"})
    proposer = LlmProposer(AppConfig(), call=lambda _prompt: response)
    overrides = proposer.propose(search_space, [], None)
    assert overrides == {"retrieval.top_k_final": 12}
    assert proposer.last_meta == {
        "proposer": "llm", "rationale": "widen final k", "model": proposer._model,
        "fallback": False, "fallback_reason": None,
    }


def test_llm_proposer_rejects_a_key_outside_the_declared_space():
    proposer = LlmProposer(AppConfig())
    assert proposer._validate({"not.a.real.key": 1}, search_space) is False


def test_llm_proposer_rejects_a_value_outside_the_declared_set():
    proposer = LlmProposer(AppConfig())
    assert proposer._validate({"retrieval.top_k_final": 999}, search_space) is False


def test_llm_proposer_rejects_an_infeasible_combination():
    # Both values are individually declared; the combination itself violates
    # "n_top_for_expansion <= top_k_final" (search_space.is_feasible).
    proposer = LlmProposer(AppConfig())
    assert proposer._validate({"retrieval.n_top_for_expansion": 6, "retrieval.top_k_final": 4}, search_space) is False


@pytest.mark.parametrize("malformed", [
    "not json",
    json.dumps({"overrides": {"retrieval.top_k_final": 8}}),  # missing rationale
    json.dumps({"rationale": "x"}),  # missing overrides
    json.dumps(["overrides", "rationale"]),  # not an object
    json.dumps({"overrides": "not-a-dict", "rationale": "x"}),
])
def test_llm_proposer_parse_rejects_malformed_json(malformed):
    assert LlmProposer._parse(malformed) is None


def test_llm_proposer_falls_back_to_grid_after_repeated_invalid_responses():
    responses = iter([
        "not json at all",
        json.dumps({"overrides": {"not.a.real.key": 1}, "rationale": "bad key"}),
        json.dumps({"overrides": {"retrieval.top_k_final": 999}, "rationale": "bad value"}),
    ])
    base = AppConfig()
    proposer = LlmProposer(base, max_retries=3, call=lambda _prompt: next(responses))
    overrides = proposer.propose(search_space, [], None)

    assert proposer.last_meta["proposer"] == "llm"
    assert proposer.last_meta["fallback"] is True
    assert proposer.last_meta["rationale"] is None
    assert "3 attempt" in proposer.last_meta["fallback_reason"]
    # The fallback overrides still came from GridProposer, so still a single
    # declared, feasible point.
    assert set(overrides.keys()) <= search_space.DECLARED_KEYS
    assert search_space.is_feasible(overrides, base)


def test_llm_proposer_falls_back_when_ollama_is_unreachable():
    def _raise(_prompt):
        raise OllamaUnreachableError("connection refused")

    proposer = LlmProposer(AppConfig(), call=_raise)
    overrides = proposer.propose(search_space, [], None)

    assert proposer.last_meta["fallback"] is True
    assert "unreachable" in proposer.last_meta["fallback_reason"].lower()
    assert set(overrides.keys()) <= search_space.DECLARED_KEYS


def test_llm_proposer_fallback_does_not_hang_or_retry_past_max_retries():
    calls = {"count": 0}

    def _always_malformed(_prompt):
        calls["count"] += 1
        return "still not json"

    proposer = LlmProposer(AppConfig(), max_retries=3, call=_always_malformed)
    proposer.propose(search_space, [], None)
    assert calls["count"] == 3


# _real_call -- test gap #65 PR review flagged: the bounded timeout (what
# stops a stalled Ollama from hanging the loop -- this repo has a
# documented history of exactly that failure mode) was asserted only by a
# docstring. A fake `requests` module is injected into sys.modules so this
# is exercised without touching a real Ollama server (this round's
# constraint: no GPU/Ollama -- the full eval gate owns the card).


class _FakeRequestsModule:
    """Enough of the `requests` surface for LlmProposer._real_call."""

    class RequestException(Exception):
        pass

    def __init__(self, response=None, raises=None):
        self._response = response
        self._raises = raises
        self.calls = []

    def post(self, url, json, timeout):  # noqa: A002 -- matches requests.post's real signature
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self._raises is not None:
            raise self._raises
        return self._response


class _FakeResponse:
    def __init__(self, body="", raise_for_status_error=None, json_error=None):
        self._body = body
        self._raise_for_status_error = raise_for_status_error
        self._json_error = json_error

    def raise_for_status(self):
        if self._raise_for_status_error is not None:
            raise self._raise_for_status_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return {"response": self._body}


def test_llm_proposer_real_call_passes_the_bounded_timeout(monkeypatch):
    fake_requests = _FakeRequestsModule(response=_FakeResponse(body='{"overrides": {}, "rationale": "x"}'))
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    proposer = LlmProposer(AppConfig(), timeout=17.5)
    result = proposer._real_call("prompt text")

    assert fake_requests.calls[0]["timeout"] == 17.5
    assert "/api/generate" in fake_requests.calls[0]["url"]
    assert result == '{"overrides": {}, "rationale": "x"}'


def test_llm_proposer_real_call_wraps_a_connection_failure(monkeypatch):
    fake_requests = _FakeRequestsModule(raises=_FakeRequestsModule.RequestException("connection refused"))
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    proposer = LlmProposer(AppConfig())
    with pytest.raises(OllamaUnreachableError):
        proposer._real_call("prompt")


def test_llm_proposer_real_call_wraps_a_non_2xx_status(monkeypatch):
    fake_requests = _FakeRequestsModule(
        response=_FakeResponse(raise_for_status_error=_FakeRequestsModule.RequestException("500 Server Error"))
    )
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    proposer = LlmProposer(AppConfig())
    with pytest.raises(OllamaUnreachableError):
        proposer._real_call("prompt")


def test_llm_proposer_real_call_wraps_a_non_json_response_body(monkeypatch):
    # A 200 whose body is not JSON (a proxy error page, an empty body) --
    # the fix this test guards: response.json() used to sit outside the
    # try/except that converts failures to OllamaUnreachableError.
    fake_requests = _FakeRequestsModule(response=_FakeResponse(json_error=ValueError("not valid JSON")))
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    proposer = LlmProposer(AppConfig())
    with pytest.raises(OllamaUnreachableError):
        proposer._real_call("prompt")


def test_llm_proposer_real_call_wraps_a_missing_requests_dependency(monkeypatch):
    # sys.modules[name] = None is how CPython represents "this import
    # previously failed" -- `import requests` then raises ImportError,
    # simulating requests not being installed at all.
    monkeypatch.setitem(sys.modules, "requests", None)

    proposer = LlmProposer(AppConfig())
    with pytest.raises(OllamaUnreachableError):
        proposer._real_call("prompt")
