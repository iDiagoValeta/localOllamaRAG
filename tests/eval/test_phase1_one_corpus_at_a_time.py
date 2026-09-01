"""Phase 1 walks one corpus at a time, releasing each before touching the next.

Issue #123. The loop used to pick a stack per case from `case["source"]`, so
an interleaved case list kept both corpora's stacks resident for the whole
phase. Each jina-clip worker costs ~2.8 GiB; on an 8 GB card the second one
could not start, and every blind-set case came back `infrastructure_error` --
19 of 51, after paying the full indexing time, with the gate correctly
refusing to compare the run against the baseline. Measured 2026-09-01 on an
RTX 4060 Laptop.

Grouping is enough because the workers start lazily on first use: nothing has
to be constructed later than it already is, only released earlier.

These tests use fakes. The point is the *order of operations* -- which stack
each case is retrieved through, and when each is released -- which is exactly
what a real run cannot demonstrate cheaply and what a reader cannot verify by
eye.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

import run_eval  # noqa: E402


class _FakeResult:
    fragments = ()


class _FakeRetrieve:
    """Records which cases it was asked for, and whether it was released."""

    def __init__(self, name, log):
        self.name = name
        self._log = log
        self.released = False

    def run(self, question):
        self._log.append(("retrieve", self.name, question))
        return _FakeResult()


class _FakeEvidence:
    def select_evidence(self, fragments):
        return [], {}


def _case(case_id, source, case_type="figure_retrieval"):
    return {
        "id": case_id,
        "paper": "p",
        "case_type": case_type,
        "lang": "en",
        "source": source,
        "question": case_id,
    }


@pytest.fixture
def harness(monkeypatch):
    """Fakes for both stacks, plus a shared log of retrievals and releases."""
    log = []
    dev = _FakeRetrieve("dev", log)
    blind = _FakeRetrieve("blind", log)

    def fake_release(*retrievers):
        for retrieve in retrievers:
            retrieve.released = True
            log.append(("release", retrieve.name))

    monkeypatch.setattr(run_eval, "_release_gpu_models", fake_release)
    monkeypatch.setattr(
        run_eval,
        "run_retrieval_case",
        lambda case, retrieved, elapsed: {
            "id": case["id"],
            "passed": True,
            "reason": "fake",
            "case_type": case["case_type"],
        },
    )
    return log, dev, blind


def test_all_of_one_corpus_runs_before_any_of_the_other(harness):
    """Interleaved input, grouped execution."""
    log, dev, blind = harness
    cases = [
        _case("dev-1", "corpus"),
        _case("blind-1", "arxiv"),
        _case("dev-2", "corpus"),
        _case("blind-2", "arxiv"),
    ]

    run_eval.run_all_cases(
        None, cases, dev, blind, _FakeEvidence(), _FakeEvidence(), models=[]
    )

    order = [(kind, name) for kind, name, *_ in log]
    assert order == [
        ("retrieve", "dev"),
        ("retrieve", "dev"),
        ("release", "dev"),
        ("retrieve", "blind"),
        ("retrieve", "blind"),
        ("release", "blind"),
    ]


def test_the_first_corpus_is_released_before_the_second_is_touched(harness):
    """The whole point: the second worker starts on a card the first freed."""
    log, dev, blind = harness
    cases = [_case("dev-1", "corpus"), _case("blind-1", "arxiv")]

    run_eval.run_all_cases(
        None, cases, dev, blind, _FakeEvidence(), _FakeEvidence(), models=[]
    )

    kinds = [(kind, name) for kind, name, *_ in log]
    release_dev = kinds.index(("release", "dev"))
    first_blind = kinds.index(("retrieve", "blind"))
    assert release_dev < first_blind


def test_every_case_is_retrieved_through_its_own_corpus_stack(harness):
    """Grouping must not send a case to the wrong index -- which would score a
    question against a corpus that does not contain its answer."""
    log, dev, blind = harness
    cases = [_case("dev-1", "corpus"), _case("blind-1", "arxiv")]

    run_eval.run_all_cases(
        None, cases, dev, blind, _FakeEvidence(), _FakeEvidence(), models=[]
    )

    retrievals = {
        entry[2]: entry[1] for entry in log if entry[0] == "retrieve"
    }
    assert retrievals == {"dev-1": "dev", "blind-1": "blind"}


def test_records_come_back_in_the_callers_case_order(harness):
    """Execution order changed; result order must not, or an artefact from this
    build invites a reader to diff it against an older one and find a
    difference that is not there."""
    _log, dev, blind = harness
    cases = [
        _case("dev-1", "corpus"),
        _case("blind-1", "arxiv"),
        _case("dev-2", "corpus"),
        _case("blind-2", "arxiv"),
    ]

    records = run_eval.run_all_cases(
        None, cases, dev, blind, _FakeEvidence(), _FakeEvidence(), models=[]
    )

    assert [r["id"] for r in records] == ["dev-1", "blind-1", "dev-2", "blind-2"]


def test_a_corpus_with_no_cases_is_neither_run_nor_released(harness):
    """A search-set-only run (what the harness asks for) must not touch the
    blind stack at all -- that is why a campaign fits on 8 GB."""
    log, dev, blind = harness

    run_eval.run_all_cases(
        None,
        [_case("dev-1", "corpus")],
        dev,
        blind,
        _FakeEvidence(),
        _FakeEvidence(),
        models=[],
    )

    assert not any(name == "blind" for _kind, name, *_ in log)
    assert blind.released is False
    assert dev.released is True


def test_a_missing_stack_is_skipped_rather_than_dereferenced(harness):
    """evaluate() passes None for a corpus it never indexed."""
    log, dev, _blind = harness

    run_eval.run_all_cases(
        None, [_case("dev-1", "corpus")], dev, None, _FakeEvidence(), None, models=[]
    )

    assert [(k, n) for k, n, *_ in log] == [("retrieve", "dev"), ("release", "dev")]
