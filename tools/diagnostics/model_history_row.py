"""Turn a gate run artifact into the `docs/model-history.md` row for each model.

Issue #146, corrected by #241. `s/answer` is the wall time of the generation
call (`generation_seconds`), not tokens divided by the decode rate: issue
#233 measured that the old formula understated the wait by a factor of 36 --
0.22 s against a measured 7.89 s -- because dividing tokens by rate omits
prompt evaluation, which dominates when the prompt is retrieved context.
Decode time is still reported, but under its own name
(`decode_seconds_per_answer`, from `eval_duration_s`) so it cannot be
mistaken for the wait again.

Aggregation is the median, not the mean: one truncated or one runaway
generation should not move the figure that goes in a durable table.

Records missing a field are skipped rather than read as zero, matching
`run_eval._decoding_metrics`, which omits its keys entirely when Ollama
reports nothing (and matching every record written before a field existed,
such as `vram_fraction`). Counting an absence as zero would drag a median
toward "cheap" or "instant" for a reason that is not a measurement.

Retrieval-only cases (`model` is `None`) are shared by every model in the
run -- the gate runs them once, not once per model -- so they are tallied
once here too and handed back separately, letting a caller add them into
each row's `Overall` without counting the same 20 cases once per model.

Columns match the current table in `docs/model-history.md` ("Generators on
the current gold set"): Answered, Overall, tokens/s, tokens/answer,
s/answer, decode s/answer, Budget hit, Infra, Placement, Run.

Usage:
    python tools/diagnostics/model_history_row.py tests/eval/runs/<artifact>.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _median(values: List[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _placement(vram_fractions: List[float]) -> str:
    """Describe where a model's weights sat in memory, from `vram_fraction` samples.

    Args:
        vram_fractions: One sample per record that reported it, each the
            share of the model's bytes resident in VRAM right after that
            call (1.0 fully on the GPU, 0.0 fully in system RAM).

    Returns:
        "not recorded" with no samples (the artifact predates the field);
        "GPU" / "CPU" when every sample agrees; otherwise the CPU share of
        the median sample, with " (varies)" appended when samples disagree.
    """
    if not vram_fractions:
        return "not recorded"
    if all(v == 1.0 for v in vram_fractions):
        return "GPU"
    if all(v == 0.0 for v in vram_fractions):
        return "CPU"
    median = statistics.median(vram_fractions)
    label = f"~{round(100 * (1 - median))}% CPU"
    if min(vram_fractions) != max(vram_fractions):
        label += " (varies)"
    return label


def summarise(
    results: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Per-model figures for the history table, plus the shared retrieval-only tally.

    Args:
        results: The artifact's ``results`` list -- one record per case per
            model, plus the retrieval-only cases where ``model`` is falsy.

    Returns:
        ``(per_model, shared)``.

        ``per_model`` maps model name to a dict of: ``answered``/``passed``/
        ``budget``/``infra`` (counts); ``tokens_per_second``/
        ``tokens_per_answer`` (medians of ``tokens_per_second``/``eval_count``
        over records carrying both -- pairing them keeps the two medians
        describing the same generations); ``seconds_per_answer`` (median of
        ``generation_seconds``, the wait); ``decode_seconds_per_answer``
        (median of ``eval_duration_s``, decode time only); ``measured`` (how
        many records had both token fields) and ``measured_wall`` (how many
        had ``generation_seconds``) -- reported so a median over three
        records is not trusted like one over twenty-three; ``placement``,
        ``vram_fraction_min`` and ``vram_fraction_median`` (``None`` when no
        record carries ``vram_fraction``).

        ``shared`` is ``{"passed": int, "total": int}`` for the
        retrieval-only cases. They are identical for every model in one run,
        so folding them into each model's counts would count the same cases
        once per model instead of once; kept separate, a caller computes
        ``overall = (passed + shared["passed"]) / (answered +
        shared["total"])``.
    """
    per_model: Dict[str, Dict[str, Any]] = {}
    shared_total = 0
    shared_passed = 0
    for record in results:
        model = record.get("model")
        if not model:
            shared_total += 1
            if record.get("passed"):
                shared_passed += 1
            continue

        entry = per_model.setdefault(
            model,
            {
                "answered": 0,
                "passed": 0,
                "budget": 0,
                "infra": 0,
                "rates": [],
                "counts": [],
                "gen_seconds": [],
                "decode_seconds": [],
                "vram": [],
            },
        )
        entry["answered"] += 1
        if record.get("passed"):
            entry["passed"] += 1
        if record.get("budget_exceeded"):
            entry["budget"] += 1
        if record.get("infrastructure_error"):
            entry["infra"] += 1

        rate, count = record.get("tokens_per_second"), record.get("eval_count")
        # Both or neither: a rate without a count cannot give tokens/answer,
        # and pairing them per record keeps the two medians describing the
        # same generations rather than two different subsets.
        if rate and count:
            entry["rates"].append(float(rate))
            entry["counts"].append(int(count))

        gen_seconds = record.get("generation_seconds")
        if gen_seconds is not None:
            entry["gen_seconds"].append(float(gen_seconds))

        decode_seconds = record.get("eval_duration_s")
        if decode_seconds is not None:
            entry["decode_seconds"].append(float(decode_seconds))

        vram_fraction = record.get("vram_fraction")
        if vram_fraction is not None:
            entry["vram"].append(float(vram_fraction))

    for entry in per_model.values():
        entry["measured"] = len(entry["counts"])
        entry["measured_wall"] = len(entry["gen_seconds"])
        entry["tokens_per_second"] = _median(entry["rates"])
        entry["tokens_per_answer"] = _median([float(c) for c in entry["counts"]])
        entry["seconds_per_answer"] = _median(entry["gen_seconds"])
        entry["decode_seconds_per_answer"] = _median(entry["decode_seconds"])
        entry["vram_fraction_min"] = min(entry["vram"]) if entry["vram"] else None
        entry["vram_fraction_median"] = _median(entry["vram"])
        entry["placement"] = _placement(entry["vram"])
        for key in ("rates", "counts", "gen_seconds", "decode_seconds", "vram"):
            del entry[key]

    return per_model, {"passed": shared_passed, "total": shared_total}


def _cell(value: Optional[float], digits: int) -> str:
    return "not recorded" if value is None else f"{value:.{digits}f}"


def _fraction_cell(passed: int, total: int) -> str:
    pct = 100 * passed / total if total else 0.0
    return f"{passed} / {total} ({pct:.1f}%)"


def format_rows(
    per_model: Dict[str, Dict[str, Any]], shared: Dict[str, int], run_id: str
) -> str:
    """Markdown rows, ready to paste under the generators table.

    Args:
        per_model: as returned by ``summarise``.
        shared: as returned by ``summarise``; folded into each row's ``Overall``.
        run_id: identifier printed in the ``Run`` column.

    Returns:
        The header, the separator row and one data row per model, sorted by
        model name.
    """
    lines = [
        "| Model | Answered | Overall | tokens/s | tokens/answer | s/answer | "
        "decode s/answer | Budget hit | Infra | Placement | Run |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model in sorted(per_model):
        e = per_model[model]
        answered = e["answered"]
        overall_total = answered + shared["total"]
        overall_passed = e["passed"] + shared["passed"]

        # The denominator qualifies a median. With nothing measured there is
        # no median to qualify, and "not recorded *(of 0)*" says it twice.
        tokens_partial = 0 < e["measured"] < answered
        tokens_suffix = f" *(of {e['measured']})*" if tokens_partial else ""
        wall_partial = 0 < e["measured_wall"] < answered
        wall_suffix = f" *(of {e['measured_wall']})*" if wall_partial else ""

        lines.append(
            f"| `{model}` | {_fraction_cell(e['passed'], answered)} | "
            f"{_fraction_cell(overall_passed, overall_total)} | "
            f"{_cell(e['tokens_per_second'], 1)} | "
            f"{_cell(e['tokens_per_answer'], 0)}{tokens_suffix} | "
            f"{_cell(e['seconds_per_answer'], 2)}{wall_suffix} | "
            f"{_cell(e['decode_seconds_per_answer'], 2)} | "
            f"{e['budget']} | {e['infra']} | {e['placement']} | `{run_id}` |"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("artifact", type=Path, help="Gate run JSON under tests/eval/runs/")
    args = parser.parse_args(argv)

    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    results = payload.get("results") or []
    per_model, shared = summarise(results)
    if not per_model:
        print(f"{args.artifact}: no per-model records found", file=sys.stderr)
        return 1

    # The table cites the run by its timestamp (`20260911T230513Z`), which is
    # what run_eval.py writes; "id" is kept for artifacts that carried one.
    run = payload.get("run") or {}
    run_id = run.get("timestamp") or run.get("id") or args.artifact.stem
    print(format_rows(per_model, shared, run_id))

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
