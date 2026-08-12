"""cli -- entry point for the block-C configuration search harness.

Usage:
    python -m harness.cli --dry-run --max-iterations 3
    python -m harness.cli --proposer llm --max-iterations 8 --patience 3

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
    return parser.parse_args(argv)


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

    search_set_ids = evaluator_mod.search_set_case_ids()
    fast_tier_ids = evaluator_mod.load_fast_tier()
    unreachable_ids = evaluator_mod.load_unreachable_ids()

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
