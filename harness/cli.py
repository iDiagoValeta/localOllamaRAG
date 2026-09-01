"""cli -- entry point for the block-C configuration search harness.

Usage:
    python -m harness.cli --dry-run --max-iterations 3
    python -m harness.cli --proposer llm --max-iterations 8 --patience 3
    python -m harness.cli --replay 1 --ledger-dir /path/to/ledger
    python -m harness.cli --set retrieval.top_k_final=1 --ledger-dir tests/eval/runs/harness-loop

``--set KEY=VALUE`` (repeatable) pins fields of the reference config the
ratchet measures against, so a campaign that must be comparable with prior
ledger history -- a criterion-5 recovery run, say -- does not need
environment variables or a hand-written wrapper script. Unknown keys fail
at launch (the repo's hard-fail policy); values are JSON-decoded when they
parse as JSON and kept as raw strings otherwise.

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
from typing import Any, Dict, Optional, Sequence

from harness import environment as environment_mod
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


def parse_set_overrides(pairs: Sequence[str]) -> Dict[str, Any]:
    """Parse repeated ``KEY=VALUE`` strings into a dotted-key overrides dict.

    Values are JSON-decoded so integers, floats and booleans survive the
    command line; anything that is not valid JSON stays a raw string (which
    is what every model-role pin needs).

    Raises:
        ValueError: A pair carries no ``=`` or an empty key.
    """
    overrides: Dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set expects KEY=VALUE, got {pair!r}")
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        overrides[key.strip()] = value
    return overrides


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
    parser.add_argument("--status", action="store_true",
                         help=(
                             "Print what the ledger holds and how much of it this launch "
                             "could pair against, evaluate nothing, exit 0. Reads the same "
                             "ledger the run would (--ledger-dir), under the same reference "
                             "config (--set is honoured)."
                         ))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                         help=(
                             "Pin a reference-config field for this campaign (repeatable), "
                             "e.g. --set retrieval.top_k_final=1 --set models.chat=gemma4:e2b. "
                             "Applied to the reference the ratchet measures against; unknown "
                             "keys fail at launch."
                         ))
    return parser.parse_args(argv)


def format_comparability_lines(comparability: Dict[str, Any]) -> list:
    """The launch line an operator reads before any evaluation is paid.

    A function rather than a run of ``print`` calls so the lines themselves are
    testable: their whole job is to carry facts a campaign depends on (issues
    #100 and #107), and a mistyped key would silently drop the warning that
    matters most while everything still ran green.

    Args:
        comparability: ``loop.describe_ledger_comparability``'s return value.

    Returns:
        Lines to print, in order.
    """
    lines = [
        f"ledger history: {comparability['history_entries']} entry(ies), "
        f"{comparability['comparable_search_set_states']} comparable search-set state(s)"
    ]
    if comparability["high_water_objective_adjusted"] is not None:
        lines.append(
            f"  historical high water: {comparability['high_water_objective_adjusted']} "
            "(recovery mode arms only if the measured reference scores lower)"
        )
        # Issue #107's exact situation: the entry that will be paired against
        # is itself the one whose stack nobody recorded. The count below is
        # about the history; this is about the single entry that decides.
        if comparability["high_water_environment_verified"] is False:
            lines.append(
                "  WARNING that high-water entry was NOT measured on a stack "
                "comparable to this launch's -- it can hold passes this stack "
                "cannot reach, which reads as a regression no candidate can fix. "
                "A refresh campaign on the current stack is what replaces it (#107)"
            )
    # An unverified state still pairs. Saying so is the difference between
    # "12 comparable states" read as "12 measured on this stack" and read as
    # what it is: a stack an old entry never recorded is a stack nobody checked.
    if comparability["environment_unverified_states"]:
        lines.append(
            f"  {comparability['environment_unverified_states']} of those carry no "
            f"comparable stack fingerprint ({comparability['environment_verified_states']} "
            "verified against this launch) -- pairing against them is unverified, "
            "not wrong; a refresh campaign on this stack replaces them (#107)"
        )
    for reason in comparability["incomparable_reasons"]:
        lines.append(f"  WARNING recovery-mode history mismatch -- {reason}")
    return lines


def format_ledger_summary(entries: Sequence[ledger_mod.LedgerEntry]) -> list:
    """What a ledger holds, for an operator deciding whether to launch.

    Answers the questions the per-entry JSONs make expensive: how far the
    history goes, what it decided, how much of it predates the stack
    fingerprint, and what the last iteration did. Comparability is NOT
    answered here -- that depends on this launch's config and stack, and
    ``format_comparability_lines`` is the one place that decides it.

    Args:
        entries: Ledger history, oldest first (``ledger.read_history``).

    Returns:
        Lines to print, in order. A single line when the ledger is empty.
    """
    if not entries:
        return ["ledger: empty -- no prior campaign has written here"]

    verdicts: Dict[str, int] = {}
    for entry in entries:
        verdicts[entry.verdict] = verdicts.get(entry.verdict, 0) + 1
    by_verdict = ", ".join(f"{name} {count}" for name, count in sorted(verdicts.items()))

    latest = entries[-1]
    search_set = [e for e in entries if e.evaluated_case_set == "search_set"]
    best = max(search_set, key=lambda e: e.objective_adjusted, default=None)
    # Entries predating schema v3 carry no stack fingerprint at all, so a
    # count of them is a count of how much of this history can never be
    # verified against the current stack (#107) -- the number that says
    # whether a refresh campaign is what the ledger needs.
    without_fingerprint = sum(1 for e in entries if e.environment_fingerprint is None)

    lines = [
        f"ledger: {len(entries)} entry(ies), iterations "
        f"{entries[0].iteration}-{latest.iteration} ({by_verdict})",
        f"  last: iteration {latest.iteration} {latest.verdict} "
        f"on the {latest.evaluated_case_set} -- {latest.reason}",
    ]
    if best is not None:
        lines.append(
            f"  best search-set objective in this ledger: {best.objective_adjusted} "
            f"(iteration {best.iteration}) -- comparability decides whether this launch can use it"
        )
    if without_fingerprint:
        lines.append(
            f"  {without_fingerprint} entry(ies) carry no stack fingerprint "
            "(written before ledger schema v3) -- their stack can never be verified (#107)"
        )
    return lines


def _run_status(reference, ledger_dir: Path) -> int:
    """Report the ledger without evaluating anything (issue #114).

    The launch line was previously reachable only from a real campaign, which
    pays the reference measurement (~20 min) before printing it, or from
    ``--dry-run``, whose demo evaluator writes a partial ``effective_config``
    that can never match a real ``AppConfig`` -- so it reported 0 comparable
    states whatever the ledger held. An agent deciding whether to launch
    needs the answer before either.

    Deliberately reuses ``format_comparability_lines``: what ``--status``
    reports and what a campaign pairs against must not come from two
    readings that can drift apart.
    """
    launch_environment = environment_mod.environment_fingerprint()
    print(f"ledger dir: {ledger_dir}")
    # The model roles are half of what decides comparability, and the half a
    # settings.json change moves without anyone noticing -- the 2026-08-23
    # campaign A failure. Printing them here means an agent never has to
    # guess which configuration its --status answer was computed under.
    roles = ", ".join(
        f"{role}={getattr(reference.models, role)}"
        for role in ("rag", "chat", "contextual", "recomp")
    )
    print(f"reference models: {roles}")
    if launch_environment is None:
        print("environment: stack versions unreadable (#107)")
    else:
        readable = ", ".join(
            f"{key}={value}"
            for key, value in sorted(launch_environment["packages"].items())
            if value is not None
        )
        print(f"environment: {readable}")

    if not ledger_dir.exists():
        print("ledger: directory does not exist yet -- a first campaign creates it")
        return 0

    entries = list(ledger_mod.read_history(ledger_dir))
    for line in format_ledger_summary(entries):
        print(line)
    for line in format_comparability_lines(
        loop_mod.describe_ledger_comparability(reference, entries, launch_environment)
    ):
        print(line)
    return 0


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

    try:
        set_overrides = parse_set_overrides(args.set)
    except ValueError as exc:
        print(f"USAGE: {exc}", file=sys.stderr)
        return 2

    reference = _build_reference()
    if set_overrides:
        # with_overrides raises ValueError on an unknown section or field:
        # a mistyped --set key must abort the campaign before any evaluation
        # is paid, never silently measure a config that ignores it.
        try:
            reference = reference.with_overrides(**set_overrides)
        except ValueError as exc:
            print(f"SET FAILED: {exc}", file=sys.stderr)
            return 2
    proposer = _build_proposer(args.proposer, reference, model=args.llm_model)

    if args.dry_run:
        evaluate = evaluator_mod.build_demo_evaluator()
        ledger_dir = args.ledger_dir or Path(tempfile.mkdtemp(prefix="harness_dry_run_"))
    else:
        evaluate = evaluator_mod.real_evaluate
        ledger_dir = args.ledger_dir or ledger_mod.LEDGER_DIR

    if args.status:
        return _run_status(reference, ledger_dir)

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
    # Issue #107: read the stack once and hand the same fingerprint to both
    # the launch line and the loop, so what the operator is told and what the
    # campaign pairs against cannot come from two different readings.
    launch_environment = environment_mod.environment_fingerprint()
    if launch_environment is None:
        print(
            "environment: stack versions unreadable -- entries will carry no "
            "fingerprint and no prior entry can be rejected for drift (#107)"
        )

    if ledger_dir.exists():
        comparability = loop_mod.describe_ledger_comparability(
            reference, list(ledger_mod.read_history(ledger_dir)), launch_environment
        )
        for line in format_comparability_lines(comparability):
            print(line)

    try:
        report = loop_mod.run_loop(
            reference=reference, evaluate=evaluate, proposer=proposer,
            search_set_ids=search_set_ids, fast_tier_ids=fast_tier_ids,
            unreachable_ids=unreachable_ids,
            max_iterations=args.max_iterations, patience=args.patience,
            ledger_dir=ledger_dir,
            reference_overrides=set_overrides,
            launch_environment=launch_environment,
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
