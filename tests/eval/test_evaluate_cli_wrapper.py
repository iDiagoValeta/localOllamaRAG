"""Pins main() as a thin wrapper over evaluate() (issue #31).

evaluate() itself needs the full engine to run a real evaluation (see
test_evaluate_library_api.py), but main()'s own job -- parse argv, call
evaluate() with the right arguments, translate its result into stdout/exit
code -- does not. These tests patch evaluate() away entirely, so nothing
here imports rag or monkeygrab.adapters and the file participates in the
dependency-free fast CI gate (tests/conftest.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_eval  # noqa: E402


def test_main_forwards_parsed_models_and_update_baseline(monkeypatch):
    captured = {}

    def _fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"exit_code": 0}

    monkeypatch.setattr(run_eval, "evaluate", _fake_evaluate)

    run_eval.main(["--models", "modelA", "modelB", "--update-baseline"])

    assert captured == {"models": ["modelA", "modelB"], "update_baseline": True}


def test_main_default_args_route_through_evaluate(monkeypatch):
    captured = {}

    def _fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"exit_code": 0}

    monkeypatch.setattr(run_eval, "evaluate", _fake_evaluate)

    run_eval.main([])

    assert captured == {"models": run_eval.DEFAULT_MODELS, "update_baseline": False}


def test_main_returns_evaluates_exit_code(monkeypatch):
    monkeypatch.setattr(run_eval, "evaluate", lambda **_kwargs: {"exit_code": 1})

    assert run_eval.main([]) == 1

    monkeypatch.setattr(run_eval, "evaluate", lambda **_kwargs: {"exit_code": 0})

    assert run_eval.main([]) == 0


def test_main_prints_setup_failed_and_returns_1_on_eval_setup_error(monkeypatch, capsys):
    def _raise(**_kwargs):
        raise run_eval.EvalSetupError("Ollama is not reachable")

    monkeypatch.setattr(run_eval, "evaluate", _raise)

    code = run_eval.main([])

    assert code == 1
    assert "SETUP FAILED: Ollama is not reachable" in capsys.readouterr().err
