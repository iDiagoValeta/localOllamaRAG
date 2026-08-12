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
) -> Dict[str, Any]:
    """Compare two run reports case by case.

    Args:
        report_a: The earlier/reference run, parsed.
        report_b: The later/candidate run, parsed.
        label_a: Name for ``report_a`` in a rejection message -- ``main()``
            passes the actual file path so a failure names the file, not the
            generic parameter name.
        label_b: Same, for ``report_b``.

    Returns:
        ``flipped_to_pass`` and ``flipped_to_fail`` (sorted case keys),
        ``stable`` (count unchanged) and ``pass_rate_delta`` (b minus a).

    Raises:
        ValueError: The two runs do not cover the same cases (a partial run
            would otherwise compare as a large improvement or regression),
            or either run is inconclusive or empty -- a run that measured
            nothing must not be treated as a noise-free result.
    """
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


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report_a", type=Path)
    parser.add_argument("report_b", type=Path)
    args = parser.parse_args(argv)

    report_a = json.loads(args.report_a.read_text(encoding="utf-8"))
    report_b = json.loads(args.report_b.read_text(encoding="utf-8"))
    _print(compare(report_a, report_b, label_a=str(args.report_a), label_b=str(args.report_b)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
