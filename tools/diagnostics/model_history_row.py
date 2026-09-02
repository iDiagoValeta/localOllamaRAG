"""Turn a gate run artifact into the `docs/model-history.md` row for each model.

Issue #146. The history table had one speed column, tokens/s, and that column
ranks models in the wrong order.

Latency per answer is tokens divided by rate, and a model that reasons inline
spends tokens the flag was supposed to suppress. Measured 2026-09-01 on one
representative prompt with `think: false`, `num_predict: 96`: `qwen3:30b-a3b`
used 96 tokens at 38.2 tok/s (2.51 s) where `qwen3-coder-30b` used 5 at 43.5
(0.11 s). Fourteen per cent apart on the column the table shows, twenty-three
times apart on the wait the user actually sits through.

So this reports three numbers per model, and the third is the one to read:

    tokens/s          how fast it decodes
    tokens/answer     how much it decides to say
    s/answer          the product of the two -- the wait

Aggregation is the median, not the mean: one truncated or one runaway
generation should not move the figure that goes in a durable table.

Records missing the counts are skipped rather than read as zero, matching
`run_eval.record_generation_stats`, which omits the keys entirely when Ollama
reports none. Counting those as zero would drag tokens/answer toward "cheap"
for a reason that is not real.

Usage:
    python tools/diagnostics/model_history_row.py tests/eval/runs/<artifact>.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _median(values: List[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def summarise(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per-model figures for the history table.

    Args:
        results: The artifact's ``results`` list, one record per case per model.

    Returns:
        Model name -> figures. ``answered`` counts the records that reached a
        generator at all; ``measured`` counts the subset that also carried
        token statistics, and the two differ whenever a model reported none.
        Both are reported because a median over three records deserves less
        trust than one over twenty-three, and hiding the denominator is how a
        table stops saying that.
    """
    per_model: Dict[str, Dict[str, Any]] = {}
    for record in results:
        model = record.get("model")
        if not model:
            continue
        entry = per_model.setdefault(
            model,
            {"answered": 0, "passed": 0, "rates": [], "counts": [], "seconds": []},
        )
        entry["answered"] += 1
        if record.get("passed"):
            entry["passed"] += 1
        rate, count = record.get("tokens_per_second"), record.get("eval_count")
        # Both or neither: a rate without a count cannot give a latency, and
        # pairing them per record keeps the three medians describing the same
        # generations rather than three different subsets.
        if rate and count:
            entry["rates"].append(float(rate))
            entry["counts"].append(int(count))
            entry["seconds"].append(float(count) / float(rate))

    for entry in per_model.values():
        entry["measured"] = len(entry["counts"])
        entry["tokens_per_second"] = _median(entry["rates"])
        entry["tokens_per_answer"] = _median([float(c) for c in entry["counts"]])
        entry["seconds_per_answer"] = _median(entry["seconds"])
        for key in ("rates", "counts", "seconds"):
            del entry[key]
    return per_model


def _cell(value: Optional[float], digits: int) -> str:
    return "not recorded" if value is None else f"{value:.{digits}f}".rstrip("0").rstrip(".")


def format_rows(per_model: Dict[str, Dict[str, Any]], run_id: str) -> str:
    """Markdown rows, ready to paste under the generators table."""
    lines = [
        "| Model | Answered cases passed | tokens/s | tokens/answer | s/answer | Run |",
        "|---|---|---|---|---|---|",
    ]
    for model in sorted(per_model):
        e = per_model[model]
        # The denominator qualifies a median. With nothing measured there is no
        # median to qualify, and "not recorded *(of 0)*" says it twice.
        partial = 0 < e["measured"] < e["answered"]
        suffix = f" *(of {e['measured']})*" if partial else ""
        lines.append(
            f"| `{model}` | {e['passed']} / {e['answered']} | "
            f"{_cell(e['tokens_per_second'], 1)} | "
            f"{_cell(e['tokens_per_answer'], 0)}{suffix} | "
            f"{_cell(e['seconds_per_answer'], 2)} | `{run_id}` |"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("artifact", type=Path, help="Gate run JSON under tests/eval/runs/")
    args = parser.parse_args(argv)

    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    results = payload.get("results") or []
    per_model = summarise(results)
    if not per_model:
        print(f"{args.artifact}: no per-model records found", file=sys.stderr)
        return 1

    run_id = (payload.get("run") or {}).get("id") or args.artifact.stem
    print(format_rows(per_model, run_id))

    unmeasured = [m for m, e in per_model.items() if not e["measured"]]
    if unmeasured:
        # Said out loud rather than left as an empty cell: artifacts written
        # before #145 carry no token statistics at all, and an empty column
        # there means "not recorded", never "fast".
        print(
            f"\nNo token statistics for: {', '.join(sorted(unmeasured))}. "
            "Runs before 2026-09-01 predate them; leave those cells empty "
            "rather than back-filling a number nobody measured.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
