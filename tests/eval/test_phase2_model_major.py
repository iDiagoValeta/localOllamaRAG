"""Phase 2 loads each generator once per sweep, not once per case (issue #220).

Phase 2 used to loop cases on the outside and models on the inside --
``run_factual_case``'s ``for model in models: rag.set_model_roles_runtime(...)``
ran on every generation call, so an N-model sweep set the "rag" role
N x len(pending) times. On an 8 GB card that held one generator at a time,
each switch was a real unload + reload (issue #229's sibling measurement:
2.3 s for a 4.9 GB model, 7.9 s for a 10 GB one). ``run_all_cases`` now loops
model-major: fix the role once per model, then run every pending case under
it -- N switches total, not N x len(pending).

Phase 1 made this exact change for this exact reason (issue #123) and
``_restore_case_order`` puts its records back into the caller's case order
afterwards so the artefact stays comparable. These tests check the phase 2
equivalent: that the role is set N times (not N x M), and that
``_restore_generation_order`` regroups the model-major output back into the
same case order *and* the same per-case model order the previous case-major
loop produced -- built from ``git log``'s issue #220 diff, not assumed.

These tests use fakes -- the point is call counts and record order, exactly
like ``test_phase1_one_corpus_at_a_time.py`` does for phase 1.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

import run_eval  # noqa: E402


class _FakeRag:
    """Records every role assignment and generation call; answers a fixed
    literal so grading is deterministic without Ollama."""

    def __init__(self, log):
        self._log = log

    def set_model_roles_runtime(self, overrides):
        self._log.append(("set_role", overrides["rag"]))
        return {"rag": overrides["rag"]}

    def generar_respuesta_silenciosa(self, question, fragments, stats=None):
        self._log.append(("generate", question))
        return "42"


def _factual_case(case_id):
    """A minimal factual_number case: source "corpus" so phase 1 routes it
    through the (faked) dev-corpus retrieval, accepted_answers matching
    _FakeRag's canned answer so every generation call grades a clean pass."""
    return {
        "id": case_id,
        "paper": f"paper-{case_id}",
        "case_type": "factual_number",
        "lang": "en",
        "source": "corpus",
        "question": f"question for {case_id}",
        "accepted_answers": ["42"],
    }


def _fake_retrieval(cases, retrieve, evidence, records, pending, store=None):
    """Stand-in for ``_run_retrieval_for_corpus``: every case retrieves the
    same one fragment and lands in ``pending``, in the order given.

    Phase 1's own ordering and Fragment-conversion machinery is exercised by
    test_phase1_one_corpus_at_a_time.py already; these tests are about phase
    2's loop shape, so phase 1 is faked down to "every case has fragments".
    """
    for case in cases:
        pending.append({
            "case": case,
            "fragments": [{"content": "frag", "source": case["paper"]}],
            "elapsed": 0.0,
        })


@pytest.fixture
def harness(monkeypatch):
    """A fake rag double, phase 1 replaced by ``_fake_retrieval``, and a
    frozen clock so elapsed_seconds/generation_seconds are reproducible
    (needed for the full-record equality check in the order test below)."""
    monkeypatch.setattr(run_eval, "_run_retrieval_for_corpus", _fake_retrieval)
    monkeypatch.setattr(run_eval, "_release_gpu_models", lambda *_a, **_kw: None)
    monkeypatch.setattr(run_eval, "_release_ollama_models", lambda *_a, **_kw: None)
    monkeypatch.setattr(run_eval.time, "perf_counter", lambda: 0.0)
    log = []
    rag = _FakeRag(log)
    return log, rag


def test_role_is_set_once_per_model_not_once_per_case(harness):
    """(a) N models x M cases must set the "rag" role N times, not N x M.

    This is the closure criterion itself: with the previous case-major loop
    (case outer, model inner -- ``run_factual_case`` calling
    ``rag.set_model_roles_runtime`` once per (case, model) pair), 3 models x
    4 cases would set the role 12 times. Model-major sets it 3 times, once
    per model, regardless of how many cases follow.
    """
    log, rag = harness
    cases = [_factual_case(f"c{i}") for i in range(4)]
    models = ["model-a", "model-b", "model-c"]

    run_eval.run_all_cases(rag, cases, object(), None, None, None, models)

    role_sets = [model for kind, model in log if kind == "set_role"]
    assert role_sets == models, (
        f"expected exactly one set_model_roles_runtime call per model, in "
        f"order, got {role_sets}"
    )


def test_generation_calls_are_grouped_by_model(harness):
    """The property the role count is a proxy for: every pending case for a
    model must run before the next model's role switch, not interleaved."""
    log, rag = harness
    cases = [_factual_case(f"c{i}") for i in range(3)]
    models = ["model-a", "model-b"]

    run_eval.run_all_cases(rag, cases, object(), None, None, None, models)

    kinds = [kind for kind, _value in log]
    assert kinds == [
        "set_role", "generate", "generate", "generate",
        "set_role", "generate", "generate", "generate",
    ]


def test_records_match_the_previous_case_major_loop_order(harness):
    """(b) Execution order changed (model-major); the artefact's record set
    and order must not.

    The reference below reconstructs the *previous* case-major loop --
    ``for item in pending: for model in models: rag.set_model_roles_runtime
    (...); <per-model body>`` -- reading straight off the #220 diff: the old
    ``run_all_cases`` called ``run_factual_case(rag, item["case"],
    item["fragments"], models, item["elapsed"])`` per pending item, and
    ``run_factual_case`` itself looped ``for model in models:
    rag.set_model_roles_runtime(...)`` then ran the per-model body now
    extracted, unchanged, into ``_run_factual_case_for_model`` (still true
    today: ``run_factual_case`` is kept case-major on purpose for
    run_probe_lang.py/run_probe_domain.py, so this is not a description of
    dead code). The reference is executed for real against the same fakes,
    not hand-typed, and compared record-for-record -- including
    elapsed_seconds/generation_seconds, made reproducible by the harness's
    frozen clock -- with what the new model-major run_all_cases returns.
    """
    log, rag = harness
    cases = [_factual_case(f"c{i}") for i in range(3)]
    models = ["model-a", "model-b"]
    pending = [
        {"case": c, "fragments": [{"content": "frag", "source": c["paper"]}], "elapsed": 0.0}
        for c in cases
    ]

    expected = []
    for item in pending:
        for model in models:
            rag.set_model_roles_runtime({"rag": model})
            expected.append(
                run_eval._run_factual_case_for_model(
                    rag, item["case"], item["fragments"], model, item["elapsed"]
                )
            )
    log.clear()  # the reference run above must not leak into the real assertion

    actual = run_eval.run_all_cases(rag, cases, object(), None, None, None, models)

    assert [(r["id"], r["model"]) for r in actual] == [
        (r["id"], r["model"]) for r in expected
    ]
    assert actual == expected


def test_model_major_records_still_stream_to_the_partial_sink_once_each(harness, tmp_path):
    """Issue #219's guarantee survives the reorder: every phase 2 record is
    on disk the moment it exists (so the file is in model-major execution
    order), each exactly once (the in-memory regroup must not go through
    the sink's append/extend), while the returned list is in case order."""
    log, rag = harness
    cases = [_factual_case(f"c{i}") for i in range(3)]
    models = ["model-a", "model-b"]
    sink_path = tmp_path / "run.partial.jsonl"

    returned = run_eval.run_all_cases(
        rag, cases, object(), None, None, None, models, partial_sink=sink_path,
    )

    on_disk = [json.loads(line) for line in sink_path.read_text().splitlines()]
    assert [(r["id"], r["model"]) for r in on_disk] == [
        ("c0", "model-a"), ("c1", "model-a"), ("c2", "model-a"),
        ("c0", "model-b"), ("c1", "model-b"), ("c2", "model-b"),
    ]
    assert [(r["id"], r["model"]) for r in returned] == [
        ("c0", "model-a"), ("c0", "model-b"),
        ("c1", "model-a"), ("c1", "model-b"),
        ("c2", "model-a"), ("c2", "model-b"),
    ]
    assert sorted(on_disk, key=lambda r: (r["id"], r["model"])) == sorted(
        returned, key=lambda r: (r["id"], r["model"])
    )
