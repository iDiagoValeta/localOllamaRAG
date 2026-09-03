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
    # Phase 2 releases the Ollama models on exit, which imports `requests`.
    # The fast gate's `architecture` job installs pytest and nothing else --
    # by design, to prove the pure layers need nothing else -- so a test that
    # reaches that import fails there and passes locally, which is the worst
    # way for a test to be wrong.
    monkeypatch.setattr(run_eval, "_release_ollama_models", lambda *_a, **_kw: None)
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


# RELEASING AFTER INDEXING (issue #123, second half).
#
# Indexing is a use: it starts each corpus's jina-clip worker and loads its
# reranker, and both stay resident. Phase 1 then opens the one it needs on top
# of them. Measured 2026-09-01 -- indexing three new blind papers left two
# workers alive, and every dev case failed reranking with CUDA OOM before
# retrieval had touched the second corpus at all. Grouping phase 1 by corpus
# did not help, because the problem was already there when phase 1 started.


def test_evaluate_releases_both_stacks_before_the_run_starts(monkeypatch):
    """The stacks built during indexing must not still be holding the card
    when phase 1 opens the one it needs."""
    import inspect

    source = inspect.getsource(run_eval.evaluate)
    release_call = source.index("_release_gpu_models(retrieve_dev, retrieve_blind)")
    run_call = source.index("records = run_all_cases(")
    assert release_call < run_call, (
        "evaluate() must release the indexing stacks BEFORE run_all_cases, or "
        "phase 1 starts on a card two embedders and two rerankers are already on"
    )


def test_the_release_is_safe_because_both_halves_reload():
    """This test exists because the fix is only correct if both components
    come back on next use. If either stops reloading, releasing here turns a
    memory fix into an outage, and that has to fail loudly here rather than
    at the first case of a real run."""
    # Read as text rather than imported: both adapters pull in torch and
    # sentence-transformers, which the fast gate's architecture job installs
    # nothing of, on purpose. The claim being checked is about what the source
    # says, so reading it is not a weaker check -- it is the same one without
    # dragging the stack in.
    root = Path(__file__).resolve().parents[2] / "src" / "monkeygrab" / "adapters"

    embedder = (root / "embedding" / "jina_clip_embedder.py").read_text(encoding="utf-8")
    ensure = embedder[embedder.index("def _ensure_worker") :]
    ensure = ensure[: ensure.index("\n    def ", 1)]
    assert "self._start_worker()" in ensure, (
        "JinaClipEmbedder._ensure_worker no longer restarts a closed worker; "
        "releasing after indexing would leave phase 1 with no embedder"
    )

    reranker = (root / "reranking" / "cross_encoder_reranker.py").read_text(encoding="utf-8")
    release = reranker[reranker.index("def release") :]
    release = release[: release.index("\n    def ", 1)] if "\n    def " in release[1:] else release
    assert "loads again" in release, (
        "CrossEncoderReranker.release no longer documents lazy reload; "
        "releasing after indexing would leave phase 1 with no reranker"
    )


def test_evaluate_releases_ollama_models_before_the_run_starts():
    """Any model preflight or earlier requests loaded in Ollama must be released
    before phase 1 begins, so phase 1 starts with 0 B Ollama VRAM on the card (issue #162)."""
    import inspect

    source = inspect.getsource(run_eval.evaluate)
    release_call = source.index("_release_ollama_models(required_models)")
    run_call = source.index("records = run_all_cases(")
    assert release_call < run_call, (
        "evaluate() must release Ollama models BEFORE run_all_cases, or "
        "phase 1 starts with resident Ollama models contending for 8 GB VRAM"
    )

