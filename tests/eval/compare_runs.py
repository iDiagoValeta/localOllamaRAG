"""compare_runs -- diff two eval run reports case by case.

The gate's headline pass rate cannot tell a real improvement from sampling
noise on its own: a two-point move is one case. What distinguishes them is
which cases changed state between two runs, so this compares reports rather
than rates.

Two uses, same tool. Run the same configuration twice and every flip is noise,
which measures the floor below which no delta means anything. Run a
deliberately sabotaged configuration against a healthy one and the flips are
the gate's sensitivity -- a measure that does not move under a known
degradation cannot detect an improvement either.

Pure over parsed JSON: no network, no GPU, no model.

Usage:
    python tests/eval/compare_runs.py runs/<a>.json runs/<b>.json
    python tests/eval/compare_runs.py --exclude-infrastructure-errors runs/<a>.json runs/<b>.json

The second form is for a pair where one model crashed on a few cases in
either run (issue #240): those (case, model) pairs are dropped from both
sides and counted in the output, so the rest of the sweep can still be
compared. The default refuses such a run outright.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _outcomes(report: Dict[str, Any]) -> Dict[str, bool]:
    """Map ``"<case id> / <model>"`` to its pass/fail for one report.

    The model is part of the key because a multi-model run grades the same case
    once per generator, and those are different measurements.
    """
    return {
        f"{r['id']} / {r['model'] or 'n-a'}": bool(r["passed"])
        for r in report["results"]
    }


def _infrastructure_keys(report: Dict[str, Any]) -> set:
    """The ``"<case id> / <model>"`` keys whose record never really ran."""
    return {
        f"{r['id']} / {r['model'] or 'n-a'}"
        for r in report["results"]
        if r.get("infrastructure_error")
    }


def _reject_unusable(report: Dict[str, Any], label: str) -> None:
    """Refuse a report that never produced a real measurement.

    ``run_eval.py`` draws this same distinction itself -- it prints
    INCONCLUSIVE and skips the baseline check for a run with infrastructure
    errors -- but it writes the report to disk before doing so, so an
    inconclusive or empty report is otherwise indistinguishable from a
    healthy one once it is sitting in ``runs/``.

    Args:
        report: The parsed report to check.
        label: Which of the two reports this is, for the error message.

    Raises:
        ValueError: The report has no cases, or has cases that never
            completed (an Ollama timeout, a dead server, a retrieval crash).
    """
    results = report["results"]
    if not results:
        raise ValueError(f"{label} has no cases and cannot be compared")
    broken = [r for r in results if r.get("infrastructure_error")]
    if broken:
        raise ValueError(
            f"{label} is inconclusive: {len(broken)} case(s) hit an "
            "infrastructure error and cannot be compared"
        )


def compare(
    report_a: Dict[str, Any],
    report_b: Dict[str, Any],
    *,
    label_a: str = "report_a",
    label_b: str = "report_b",
    exclude_infrastructure_errors: bool = False,
) -> Dict[str, Any]:
    """Compare two run reports case by case.

    Args:
        report_a: The earlier/reference run, parsed.
        report_b: The later/candidate run, parsed.
        label_a: Name for ``report_a`` in a rejection message -- ``main()``
            passes the actual file path so a failure names the file, not the
            generic parameter name.
        label_b: Same, for ``report_b``.
        exclude_infrastructure_errors: Drop every (case, model) pair that
            hit an infrastructure error in either run instead of refusing
            the run (issue #240). A pair that never ran is not a pass or a
            fail, so counting it as a flip would inflate the noise floor,
            and refusing the whole file throws away the hundreds of pairs
            that did run. Off by default: the gate's own comparison keeps
            the strict rule.

    Returns:
        ``flipped_to_pass`` and ``flipped_to_fail`` (sorted case keys),
        ``stable`` (count unchanged), ``pass_rate_delta`` (b minus a) and
        ``excluded`` (sorted keys dropped under the flag, empty otherwise).

    Raises:
        ValueError: The two runs do not cover the same cases (a partial run
            would otherwise compare as a large improvement or regression),
            or either run is inconclusive or empty -- a run that measured
            nothing must not be treated as a noise-free result. Under the
            flag, "inconclusive" narrows to "no comparable pair left".
    """
    excluded: List[str] = []
    if exclude_infrastructure_errors:
        excluded = sorted(_infrastructure_keys(report_a) | _infrastructure_keys(report_b))
        report_a = _without(report_a, excluded)
        report_b = _without(report_b, excluded)
        if not report_a["results"] or not report_b["results"]:
            raise ValueError(
                f"no comparable pair left: every record of {label_a} or {label_b} "
                "hit an infrastructure error"
            )
    _reject_unusable(report_a, label_a)
    _reject_unusable(report_b, label_b)
    a, b = _outcomes(report_a), _outcomes(report_b)
    if a.keys() != b.keys():
        difference = sorted(set(a) ^ set(b))
        raise ValueError(
            f"runs cover different cases and cannot be compared: {difference}"
        )

    flipped_to_pass = sorted(k for k in a if not a[k] and b[k])
    flipped_to_fail = sorted(k for k in a if a[k] and not b[k])
    total = len(a)
    # total is always >= 1 here: _reject_unusable already raised above if
    # report_a's results were empty, and a non-empty results list always
    # yields at least one outcome key -- so there is no zero-total case left
    # for this division to guard against.
    delta = (sum(b.values()) - sum(a.values())) / total
    return {
        "flipped_to_pass": flipped_to_pass,
        "flipped_to_fail": flipped_to_fail,
        "stable": total - len(flipped_to_pass) - len(flipped_to_fail),
        "pass_rate_delta": round(delta, 4),
        "excluded": excluded,
    }


def _without(report: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """A shallow copy of ``report`` with the records for ``keys`` removed."""
    dropped = set(keys)
    return {
        **report,
        "results": [
            r for r in report["results"]
            if f"{r['id']} / {r['model'] or 'n-a'}" not in dropped
        ],
    }


def _print(result: Dict[str, Any]) -> None:
    """Print the comparison, leading with the verdict a reader actually needs."""
    to_pass: List[str] = result["flipped_to_pass"]
    to_fail: List[str] = result["flipped_to_fail"]
    if not to_pass and not to_fail:
        print(f"identical: {result['stable']} case(s) unchanged")
    else:
        print(f"{len(to_pass)} flipped to PASS, {len(to_fail)} flipped to FAIL, "
              f"{result['stable']} unchanged")
    for key in to_pass:
        print(f"  + {key}")
    for key in to_fail:
        print(f"  - {key}")
    print(f"pass rate delta: {result['pass_rate_delta']:+.4f}")
    excluded: List[str] = result["excluded"]
    if excluded:
        print(f"{len(excluded)} pair(s) excluded (infrastructure error in one run):")
        for key in excluded:
            print(f"  ! {key}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report_a", type=Path)
    parser.add_argument("report_b", type=Path)
    parser.add_argument(
        "--exclude-infrastructure-errors",
        action="store_true",
        help="drop the (case, model) pairs that hit an infrastructure error in "
             "either run instead of refusing the run; the count is printed",
    )
    args = parser.parse_args(argv)

    report_a = json.loads(args.report_a.read_text(encoding="utf-8"))
    report_b = json.loads(args.report_b.read_text(encoding="utf-8"))
    _print(compare(
        report_a, report_b,
        label_a=str(args.report_a), label_b=str(args.report_b),
        exclude_infrastructure_errors=args.exclude_infrastructure_errors,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
