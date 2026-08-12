"""Equivalence tests for run_eval.evaluate() -- the library API extracted
from main()'s body so a harness can drive an evaluation in-process (issue
#31, docs/design/2026-07-28-loop-automejorable.md section 6.1).

evaluate() imports the real rag.chat_pdfs internally (the one step no
amount of faking run_eval's own helpers avoids), so it needs the full engine
installed -- the ``import rag.chat_pdfs`` below is what lets
tests/conftest.py's heuristic skip this file in the dependency-free fast
gate. Everything downstream of that import (Ollama preflight, corpus
staging, indexing, case execution) is faked, so no GPU, Ollama server or
network access is needed to run these.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

import rag.chat_pdfs  # noqa: E402,F401  (see module docstring)
import run_eval  # noqa: E402
from run_eval import evaluate  # noqa: E402

_DEV_CASE_ID = "att-bleu-ende"  # source: "corpus"
_BLIND_CASE_ID = "resnet-top1-34layer"  # source: "arxiv"


def _fake_stack():
    return SimpleNamespace(
        vector_store=SimpleNamespace(get_page=lambda *_a, **_kw: []),
        embedder=None,
    )


def _capture_ensure_indexed(monkeypatch, calls):
    """Replace ensure_indexed with a double that records every call and
    returns a stack cheap enough that evaluate()'s teardown is a no-op."""

    def _fake(rag, carpeta, required_pdfs, label, path_label=None, config_overrides=None):
        calls.append({
            "carpeta": carpeta,
            "required_pdfs": set(required_pdfs),
            "label": label,
            "path_label": path_label,
            "config_overrides": config_overrides,
        })
        return (object(), object(), _fake_stack())

    monkeypatch.setattr(run_eval, "ensure_indexed", _fake)


def _capture_run_all_cases(monkeypatch, seen_cases):
    """Replace run_all_cases with a double that records which cases it was
    asked to run and grades every one of them a pass, so a run this test
    file drives never needs Ollama."""

    def _fake(rag, cases, retrieve_dev, retrieve_blind, evidence_dev, evidence_blind, models):
        seen_cases.extend(cases)
        return [
            {
                "id": c["id"], "paper": c["paper"], "case_type": c["case_type"],
                "lang": c["lang"], "model": None, "passed": True, "reason": "stub",
                "elapsed_seconds": 0.0,
            }
            for c in cases
        ]

    monkeypatch.setattr(run_eval, "run_all_cases", _fake)


@pytest.fixture(autouse=True)
def _stub_pipeline(monkeypatch):
    """Fake Ollama preflight, blind-set staging and the post-index paper
    check, and neutralize the one call that mutates rag.chat_pdfs's real
    module globals unconditionally -- evaluate() always calls
    set_model_roles_runtime, and leaving that mutation in place would leak
    into whatever test runs next in the same process (e.g.
    tests/unit/test_app_config_defaults.py, which reads those same globals).

    Pipeline flags (set_pipeline_flags) are NOT neutralized here -- several
    tests below exercise _generation_config_overrides, which calls the real
    set_pipeline_flags on purpose. Snapshotting and restoring them instead
    means a bug in that restoration shows up as THIS test's own assertions
    failing, not as unrelated flakiness two files later.
    """
    monkeypatch.setattr(run_eval, "preflight_ollama", lambda *_a, **_kw: None)
    monkeypatch.setattr(run_eval, "stage_blind_papers", lambda cases: {})
    monkeypatch.setattr(run_eval, "verify_all_papers_indexed", lambda *_a, **_kw: None)
    monkeypatch.setattr(rag.chat_pdfs, "set_model_roles_runtime", lambda *_a, **_kw: None)
    previous_flags = rag.chat_pdfs.get_pipeline_flags()
    yield
    rag.chat_pdfs.set_pipeline_flags(previous_flags)


def test_case_ids_runs_exactly_those_cases(monkeypatch):
    _capture_ensure_indexed(monkeypatch, [])
    seen = []
    _capture_run_all_cases(monkeypatch, seen)
    wanted = [_DEV_CASE_ID, "att-bleu-enfr"]

    result = evaluate(models=["m"], case_ids=wanted, write_report=False)

    assert {c["id"] for c in seen} == set(wanted)
    assert result["num_cases"] == len(wanted)


def test_case_ids_none_selects_every_gold_case(monkeypatch):
    _capture_ensure_indexed(monkeypatch, [])
    seen = []
    _capture_run_all_cases(monkeypatch, seen)
    all_cases = run_eval.load_gold_cases()

    result = evaluate(models=["m"], write_report=False)

    assert len(seen) == len(all_cases)
    assert {c["id"] for c in seen} == {c["id"] for c in all_cases}
    assert result["num_cases"] == len(all_cases)


def test_unknown_case_id_raises_value_error(monkeypatch):
    _capture_ensure_indexed(monkeypatch, [])
    _capture_run_all_cases(monkeypatch, [])

    with pytest.raises(ValueError):
        evaluate(models=["m"], case_ids=["not-a-real-case"], write_report=False)


def test_dev_only_subset_never_stages_or_indexes_blind(monkeypatch):
    ensure_calls = []
    _capture_ensure_indexed(monkeypatch, ensure_calls)
    _capture_run_all_cases(monkeypatch, [])
    staged = []
    monkeypatch.setattr(run_eval, "stage_blind_papers", lambda cases: staged.append(cases) or {})

    evaluate(models=["m"], case_ids=[_DEV_CASE_ID], write_report=False)

    assert {c["label"] for c in ensure_calls} == {"dev set"}
    assert [c["id"] for c in staged[0]] == [_DEV_CASE_ID]


def test_blind_only_subset_never_indexes_dev(monkeypatch):
    ensure_calls = []
    _capture_ensure_indexed(monkeypatch, ensure_calls)
    _capture_run_all_cases(monkeypatch, [])
    # A real stage_blind_papers would hit the network (fetch_papers); this
    # double reproduces only its {"<paper>.pdf": arxiv_id} return shape.
    monkeypatch.setattr(
        run_eval, "stage_blind_papers",
        lambda cases: {f"{c['paper']}.pdf": c.get("arxiv_id", "") for c in cases},
    )

    evaluate(models=["m"], case_ids=[_BLIND_CASE_ID], write_report=False)

    assert {c["label"] for c in ensure_calls} == {"blind set"}


def test_write_report_false_writes_nothing_to_disk(monkeypatch, tmp_path):
    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path)
    _capture_ensure_indexed(monkeypatch, [])
    _capture_run_all_cases(monkeypatch, [])

    result = evaluate(models=["m"], case_ids=[_DEV_CASE_ID], write_report=False)

    assert result["report_path"] is None
    assert list(tmp_path.iterdir()) == []


def test_write_report_true_writes_the_expected_json(monkeypatch, tmp_path):
    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path)
    _capture_ensure_indexed(monkeypatch, [])
    _capture_run_all_cases(monkeypatch, [])

    result = evaluate(models=["m"], case_ids=[_DEV_CASE_ID], write_report=True)

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert result["report_path"] == str(files[0])
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["run"]["num_cases"] == 1


def test_update_baseline_false_leaves_the_baseline_file_byte_identical(monkeypatch, tmp_path):
    baseline_file = tmp_path / "baseline_min_pass_rate.txt"
    baseline_file.write_text("0.61\n", encoding="utf-8")
    monkeypatch.setattr(run_eval, "BASELINE_FILE", baseline_file)
    _capture_ensure_indexed(monkeypatch, [])
    _capture_run_all_cases(monkeypatch, [])
    before = baseline_file.read_bytes()

    result = evaluate(
        models=["m"], case_ids=[_DEV_CASE_ID], write_report=False, update_baseline=False
    )

    assert baseline_file.read_bytes() == before
    assert result["baseline_updated"] is False


def test_update_baseline_true_raises_it_when_the_run_is_conclusive(monkeypatch, tmp_path):
    baseline_file = tmp_path / "baseline_min_pass_rate.txt"
    baseline_file.write_text("0.10\n", encoding="utf-8")
    monkeypatch.setattr(run_eval, "BASELINE_FILE", baseline_file)
    _capture_ensure_indexed(monkeypatch, [])
    _capture_run_all_cases(monkeypatch, [])  # every stubbed case passes

    result = evaluate(
        models=["m"], case_ids=[_DEV_CASE_ID], write_report=False, update_baseline=True
    )

    assert result["baseline_updated"] is True
    assert baseline_file.read_text(encoding="utf-8").strip() != "0.10"


def test_config_overrides_reach_the_appconfig_and_ensure_indexed(monkeypatch):
    ensure_calls = []
    _capture_ensure_indexed(monkeypatch, ensure_calls)
    _capture_run_all_cases(monkeypatch, [])
    overrides = {"retrieval.top_k_final": 3}

    result = evaluate(
        models=["m"], case_ids=[_DEV_CASE_ID], config_overrides=overrides, write_report=False
    )

    assert result["config"]["dev"]["retrieval"]["top_k_final"] == 3
    assert ensure_calls[0]["config_overrides"] == overrides


def test_config_is_none_for_a_corpus_never_staged(monkeypatch):
    _capture_ensure_indexed(monkeypatch, [])
    _capture_run_all_cases(monkeypatch, [])

    result = evaluate(models=["m"], case_ids=[_DEV_CASE_ID], write_report=False)

    assert result["config"]["dev"] is not None
    assert result["config"]["blind"] is None


def test_config_overrides_rejects_a_single_unreachable_key(monkeypatch):
    ensure_calls = []
    _capture_ensure_indexed(monkeypatch, ensure_calls)
    seen = []
    _capture_run_all_cases(monkeypatch, seen)

    with pytest.raises(ValueError, match="models.rag"):
        evaluate(
            models=["m"], case_ids=[_DEV_CASE_ID],
            config_overrides={"models.rag": "other-model"}, write_report=False,
        )

    # Fails before any staging/indexing/execution -- a caller-usage error,
    # not a partial run.
    assert ensure_calls == []
    assert seen == []


def test_config_overrides_rejects_every_unreachable_key_by_name(monkeypatch):
    _capture_ensure_indexed(monkeypatch, [])
    _capture_run_all_cases(monkeypatch, [])

    with pytest.raises(ValueError) as excinfo:
        evaluate(
            models=["m"], case_ids=[_DEV_CASE_ID],
            config_overrides={
                "models.rag": "x",
                "paths.docs_folder": "/tmp/y",
                "models.ollama.keep_alive": 30,
            },
            write_report=False,
        )

    message = str(excinfo.value)
    assert "models.rag" in message
    assert "paths.docs_folder" in message
    assert "models.ollama.keep_alive" in message


def test_config_overrides_accepts_every_key_evaluate_claims_to_honour(monkeypatch):
    """_APPCONFIG_OVERRIDE_KEYS / _GENERATION_FLAG_OVERRIDE_KEYS is the map
    reported alongside this test; this pins that _validate_config_overrides
    agrees with it (a key present in the map but rejected here would be a
    silent regression the map itself would not catch)."""
    _capture_ensure_indexed(monkeypatch, [])
    _capture_run_all_cases(monkeypatch, [])
    honored = run_eval._APPCONFIG_OVERRIDE_KEYS | set(run_eval._GENERATION_FLAG_OVERRIDE_KEYS)

    run_eval._validate_config_overrides({key: None for key in honored})  # must not raise


def test_generation_flag_override_reaches_rag_chat_pdfs_during_case_execution(monkeypatch):
    """The AppConfig path is already covered by
    test_config_overrides_reach_the_appconfig_and_ensure_indexed; this is the
    runtime-setter path's equivalent proof, for the two flags
    generar_respuesta_silenciosa can only see through rag.chat_pdfs's live
    globals (see evaluate()'s docstring)."""
    observed = {}

    def _fake_run_all_cases(rag_module, cases, retrieve_dev, retrieve_blind, evidence_dev, evidence_blind, models):
        observed["usar_recomp_synthesis"] = rag_module.USAR_RECOMP_SYNTHESIS
        observed["usar_optimizacion_contexto"] = rag_module.USAR_OPTIMIZACION_CONTEXTO
        return []

    monkeypatch.setattr(run_eval, "run_all_cases", _fake_run_all_cases)
    _capture_ensure_indexed(monkeypatch, [])
    before = (rag.chat_pdfs.USAR_RECOMP_SYNTHESIS, rag.chat_pdfs.USAR_OPTIMIZACION_CONTEXTO)

    evaluate(
        models=["m"], case_ids=[_DEV_CASE_ID],
        config_overrides={
            "flags.usar_recomp_synthesis": not before[0],
            "flags.usar_optimizacion_contexto": not before[1],
        },
        write_report=False,
    )

    assert observed == {
        "usar_recomp_synthesis": not before[0],
        "usar_optimizacion_contexto": not before[1],
    }
    # Restored once evaluate() returns.
    assert (rag.chat_pdfs.USAR_RECOMP_SYNTHESIS, rag.chat_pdfs.USAR_OPTIMIZACION_CONTEXTO) == before


def test_generation_flag_override_restores_on_exception_mid_evaluation(monkeypatch):
    _capture_ensure_indexed(monkeypatch, [])
    before = rag.chat_pdfs.get_pipeline_flags()

    def _raise(*_a, **_kw):
        raise RuntimeError("simulated failure mid-evaluation")

    monkeypatch.setattr(run_eval, "run_all_cases", _raise)

    with pytest.raises(RuntimeError, match="simulated failure"):
        evaluate(
            models=["m"], case_ids=[_DEV_CASE_ID],
            config_overrides={"flags.usar_recomp_synthesis": not before["USAR_RECOMP_SYNTHESIS"]},
            write_report=False,
        )

    assert rag.chat_pdfs.get_pipeline_flags() == before
