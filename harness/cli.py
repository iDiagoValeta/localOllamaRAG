"""cli -- entry point for the block-C configuration search harness.

Usage:
    python -m harness.cli --dry-run --max-iterations 3
    python -m harness.cli --proposer llm --max-iterations 8 --patience 3
    python -m harness.cli --replay 1 --ledger-dir /path/to/ledger

``--dry-run`` always works: it wires in a small deterministic in-process
evaluator (``evaluator.build_demo_evaluator``) so the whole loop -- proposer,
feasibility, ledger, latency constraint, termination -- can be exercised
with no GPU, no Ollama and no PDFs, writing its ledger to a fresh temporary
directory unless ``--ledger-dir`` is given (so a demo run never litters
``harness/ledger/``). Without it, the CLI uses the real evaluator
(``evaluator.real_evaluate``), which depends on the sibling PR (issue #31
spec section 5.2, #56) landing ``tests/eval/run_eval.evaluate()`` -- it
fails with an actionable message, not a stack trace, until that lands.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from harness import evaluator as evaluator_mod
from harness import ledger as ledger_mod
from harness import loop as loop_mod
from harness import proposers as proposers_mod


def _build_reference():
    """The ``AppConfig`` the loop's ratchet, feasibility and latency ceiling measure against."""
    import sys as _sys
    from pathlib import Path as _Path

    src = _Path(__file__).resolve().parents[1] / "src"
    if str(src) not in _sys.path:
        _sys.path.insert(0, str(src))
    from monkeygrab.config.app_config import AppConfig

    return AppConfig.from_env()


def _build_proposer(name: str, reference, *, model: str):
    if name == "grid":
        return proposers_mod.GridProposer(reference)
    if name == "llm":
        return proposers_mod.LlmProposer(reference, model=model)
    raise ValueError(f"unknown proposer {name!r}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block-C configuration search harness (issue #31).")
    parser.add_argument("--proposer", choices=("grid", "llm"), default="grid",
                         help="grid (deterministic control) or llm (Ollama-backed). Default: grid.")
    parser.add_argument("--max-iterations", type=int, default=None,
                         help="Hard iteration budget. Default: none (patience alone decides).")
    parser.add_argument("--patience", type=int, default=3,
                         help="Stop after this many consecutive non-accepted iterations. Default: 3.")
    parser.add_argument("--ledger-dir", type=Path, default=None,
                         help="Default: harness/ledger/, or a fresh temp dir under --dry-run.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Use the deterministic in-process demo evaluator instead of the real gate.")
    parser.add_argument("--llm-model", default=proposers_mod.LlmProposer.DEFAULT_MODEL,
                         help=f"Ollama model for --proposer llm. Default: {proposers_mod.LlmProposer.DEFAULT_MODEL}.")
    parser.add_argument("--replay", type=int, metavar="ITERATION", default=None,
                         help=(
                             "Re-run ledger iteration ITERATION's overrides and case ids "
                             "(criterion 7). Skips the search loop. Needs --ledger-dir "
                             "pointing at the ledger that holds that entry."
                         ))
    return parser.parse_args(argv)


def _run_replay(iteration: int, evaluate, ledger_dir: Path) -> int:
    """Criterion 7: reconstruct one ledger entry and compare the pass vector."""
    try:
        entry = ledger_mod.read_entry_by_iteration(ledger_dir, iteration)
    except FileNotFoundError as exc:
        print(f"REPLAY FAILED: {exc}", file=sys.stderr)
        return 1
    try:
        result = evaluator_mod.replay(evaluate, entry.config_overrides, entry.case_records)
    except NotImplementedError as exc:
        print(f"NOT AVAILABLE: {exc}", file=sys.stderr)
        return 1
    except evaluator_mod.ReachabilityError as exc:
        print(f"REACHABILITY GATE FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"replay iteration {entry.iteration} ({entry.evaluated_case_set}, {len(entry.case_records)} cases)")
    print(f"overrides: {entry.config_overrides}")
    print(f"identical: {result.identical}")
    if result.flips:
        print(f"flips: {list(result.flips)}")
    if result.missing:
        print(f"missing from replay: {list(result.missing)}")
    if result.extra:
        print(f"extra in replay: {list(result.extra)}")
    if result.infrastructure_errors:
        print(f"infrastructure errors: {list(result.infrastructure_errors)}")
    return 0 if result.identical else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    reference = _build_reference()
    proposer = _build_proposer(args.proposer, reference, model=args.llm_model)

    if args.dry_run:
        evaluate = evaluator_mod.build_demo_evaluator()
        ledger_dir = args.ledger_dir or Path(tempfile.mkdtemp(prefix="harness_dry_run_"))
    else:
        evaluate = evaluator_mod.real_evaluate
        ledger_dir = args.ledger_dir or ledger_mod.LEDGER_DIR

    if args.replay is not None:
        return _run_replay(args.replay, evaluate, ledger_dir)

    search_set_ids = evaluator_mod.search_set_case_ids()
    fast_tier_ids = evaluator_mod.load_fast_tier()
    unreachable_ids = evaluator_mod.load_unreachable_ids()

    # Issue #100: recovery mode pairs against comparable prior history, and a
    # silently incomparable ledger is how a campaign spends hours fabricating
    # evidence against its own fix. Say what the ledger offers BEFORE the
    # reference measurement is paid; arming itself still depends on that
    # measurement, so nothing here promises it.
    if ledger_dir.exists():
        comparability = loop_mod.describe_ledger_comparability(
            reference, list(ledger_mod.read_history(ledger_dir))
        )
        print(
            f"ledger history: {comparability['history_entries']} entry(ies), "
            f"{comparability['comparable_search_set_states']} comparable search-set state(s)"
        )
        if comparability["high_water_objective_adjusted"] is not None:
            print(
                f"  historical high water: {comparability['high_water_objective_adjusted']} "
                "(recovery mode arms only if the measured reference scores lower)"
            )
        for reason in comparability["incomparable_reasons"]:
            print(f"  WARNING recovery-mode history mismatch -- {reason}")

    try:
        report = loop_mod.run_loop(
            reference=reference, evaluate=evaluate, proposer=proposer,
            search_set_ids=search_set_ids, fast_tier_ids=fast_tier_ids,
            unreachable_ids=unreachable_ids,
            max_iterations=args.max_iterations, patience=args.patience,
            ledger_dir=ledger_dir,
        )
    except evaluator_mod.ReachabilityError as exc:
        print(f"REACHABILITY GATE FAILED: {exc}", file=sys.stderr)
        return 1
    except NotImplementedError as exc:
        print(f"NOT AVAILABLE: {exc}", file=sys.stderr)
        return 1

    ledger_dir.mkdir(parents=True, exist_ok=True)
    report_path = ledger_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"iterations run: {report['iterations_run']} (stopped: {report['termination_reason']})")
    print(f"reference objective_adjusted: {report['reference']['objective_adjusted']}")
    print(f"ratchet: {report['ratchet']} (best iteration: {report['best_iteration']})")
    print(f"ledger: {ledger_dir}")
    print(f"report: {report_path}")
    print(f"resolution warning: {report['resolution_warning']['message']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
