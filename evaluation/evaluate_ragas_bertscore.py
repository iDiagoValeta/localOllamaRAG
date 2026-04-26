"""
Complement RAGAS outputs with BERTScore.

This script does not run RAGAS or RAG inference. It reads existing RAGAS CSV
files, compares response/reference pairs with BERTScore, and writes separate
artifacts under evaluation/runs/bertscore by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_RAGAS_SCORES_DIR = EVAL_DIR / "runs" / "ragas"
DEFAULT_OUTPUT_ROOT = EVAL_DIR / "runs" / "bertscore"
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"
BERTSCORE_LANG = "en"
BERTSCORE_BATCH_SIZE = 8

RESPONSE_COLUMNS = ("response", "answer", "generated_answer")
REFERENCE_COLUMNS = ("reference", "ground_truth", "ground_truths", "expected_answer")


@dataclass
class BertScoreConfig:
    model_type: str = BERTSCORE_MODEL
    lang: str = BERTSCORE_LANG
    rescale_with_baseline: bool = True
    batch_size: int = BERTSCORE_BATCH_SIZE


def _safe_label(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    return safe.strip("_") or "ragas_bertscore"


def _find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    lower_to_original = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def _is_ragas_result_csv(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".csv":
        return False
    if path.name.endswith("_bertscore.csv") or path.name.startswith("bertscore_summary"):
        return False
    try:
        fieldnames, _ = _read_csv(path)
    except OSError:
        return False
    return bool(_find_column(fieldnames, RESPONSE_COLUMNS) and _find_column(fieldnames, REFERENCE_COLUMNS))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _round_or_none(value: float | None, ndigits: int = 6) -> float | None:
    if value is None or math.isnan(value):
        return None
    return round(float(value), ndigits)


def _score_pairs(
    predictions: list[str],
    references: list[str],
    config: BertScoreConfig,
) -> tuple[list[float], list[float], list[float]]:
    try:
        from bert_score import score as bert_score_fn
        import bert_score.utils as _bsu  # noqa: PLC0415

        # DeBERTa tokenizer sets model_max_length to a Python bigint that overflows
        # a C int inside the HuggingFace Rust tokenizer on Windows (OverflowError).
        # Cap it to 512 before bert_score calls sent_encode.
        _orig_sent_encode = _bsu.sent_encode

        def _patched_sent_encode(tokenizer, sent):  # noqa: ANN001, ANN202
            if getattr(tokenizer, "model_max_length", 0) > 100_000:
                tokenizer.model_max_length = 512
            return _orig_sent_encode(tokenizer, sent)

        _bsu.sent_encode = _patched_sent_encode
    except ImportError as e:
        print("Install BERTScore before running this script:")
        print("   pip install bert-score")
        raise SystemExit(1) from e

    precision, recall, f1 = bert_score_fn(
        predictions,
        references,
        model_type=config.model_type,
        lang=config.lang,
        rescale_with_baseline=config.rescale_with_baseline,
        batch_size=config.batch_size,
        verbose=False,
    )
    return precision.tolist(), recall.tolist(), f1.tolist()


def evaluate_csv(
    input_csv: str | Path,
    output_dir: str | Path,
    config: BertScoreConfig | None = None,
) -> dict[str, Any]:
    config = config or BertScoreConfig()
    input_path = Path(input_csv)
    output_path = Path(output_dir) / f"{input_path.stem}_bertscore.csv"

    fieldnames, rows = _read_csv(input_path)
    if not fieldnames:
        raise ValueError(f"CSV without header: {input_path}")

    response_col = _find_column(fieldnames, RESPONSE_COLUMNS)
    reference_col = _find_column(fieldnames, REFERENCE_COLUMNS)
    if not response_col or not reference_col:
        raise ValueError(
            f"{input_path} must contain response/reference columns. "
            f"Accepted response columns: {', '.join(RESPONSE_COLUMNS)}. "
            f"Accepted reference columns: {', '.join(REFERENCE_COLUMNS)}."
        )

    valid_indexes: list[int] = []
    predictions: list[str] = []
    references: list[str] = []
    for idx, row in enumerate(rows):
        prediction = (row.get(response_col) or "").strip()
        reference = (row.get(reference_col) or "").strip()
        if prediction and reference:
            valid_indexes.append(idx)
            predictions.append(prediction)
            references.append(reference)

    for row in rows:
        row["bertscore_precision"] = ""
        row["bertscore_recall"] = ""
        row["bertscore_f1"] = ""
        row["bertscore_model"] = config.model_type
        row["bertscore_rescale_with_baseline"] = str(config.rescale_with_baseline)

    if predictions:
        precision, recall, f1 = _score_pairs(predictions, references, config)
        for local_idx, row_idx in enumerate(valid_indexes):
            rows[row_idx]["bertscore_precision"] = f"{precision[local_idx]:.6f}"
            rows[row_idx]["bertscore_recall"] = f"{recall[local_idx]:.6f}"
            rows[row_idx]["bertscore_f1"] = f"{f1[local_idx]:.6f}"
    else:
        precision, recall, f1 = [], [], []

    output_fieldnames = list(fieldnames)
    for col in (
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
        "bertscore_model",
        "bertscore_rescale_with_baseline",
    ):
        if col not in output_fieldnames:
            output_fieldnames.append(col)

    _write_csv(output_path, output_fieldnames, rows)

    return {
        "input_csv": str(input_path.resolve()),
        "output_csv": str(output_path.resolve()),
        "rows_total": len(rows),
        "rows_scored": len(valid_indexes),
        "rows_skipped": len(rows) - len(valid_indexes),
        "response_column": response_col,
        "reference_column": reference_col,
        "bertscore_model": config.model_type,
        "bertscore_lang": config.lang,
        "bertscore_rescale_with_baseline": config.rescale_with_baseline,
        "mean_bertscore_precision": _round_or_none(_mean(precision)),
        "mean_bertscore_recall": _round_or_none(_mean(recall)),
        "mean_bertscore_f1": _round_or_none(_mean(f1)),
    }


def _iter_ragas_csvs(comparison_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in comparison_dir.glob("*.csv")
        if _is_ragas_result_csv(path)
    )


def discover_completed_experiments(scores_root: str | Path = DEFAULT_RAGAS_SCORES_DIR) -> list[dict[str, Any]]:
    """Discover completed RAGAS score artifacts that can be scored with BERTScore."""
    root = Path(scores_root)
    experiments: list[dict[str, Any]] = []

    for csv_path in sorted(root.glob("*.csv")):
        if _is_ragas_result_csv(csv_path):
            experiments.append(
                {
                    "kind": "single",
                    "label": _safe_label(csv_path.stem),
                    "input_csv": csv_path,
                    "comparison_dir": None,
                }
            )

    for csv_path in sorted(root.glob("**/scores.csv")):
        if not _is_ragas_result_csv(csv_path):
            continue
        parent = csv_path.parent
        if "comparisons" in parent.relative_to(root).parts:
            continue
        label = _safe_label("_".join(parent.relative_to(root).parts))
        experiments.append(
            {
                "kind": "single",
                "label": label,
                "input_csv": csv_path,
                "comparison_dir": None,
            }
        )

    comparison_root = root / "comparisons"
    if comparison_root.exists():
        for comparison_dir in sorted(path for path in comparison_root.iterdir() if path.is_dir()):
            csvs = _iter_ragas_csvs(comparison_dir / "scores")
            if csvs:
                experiments.append(
                    {
                        "kind": "comparison",
                        "label": _safe_label(comparison_dir.name),
                        "input_csv": None,
                        "comparison_dir": comparison_dir / "scores",
                    }
                )

    legacy_comparison_root = root / "scores" / "comparison_runs"
    if legacy_comparison_root.exists():
        for comparison_dir in sorted(path for path in legacy_comparison_root.iterdir() if path.is_dir()):
            csvs = _iter_ragas_csvs(comparison_dir)
            if csvs:
                experiments.append(
                    {
                        "kind": "comparison",
                        "label": _safe_label(comparison_dir.name),
                        "input_csv": None,
                        "comparison_dir": comparison_dir,
                    }
                )

    return experiments


def evaluate_inputs(
    input_csv: str | Path | None = None,
    comparison_dir: str | Path | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    label: str | None = None,
    config: BertScoreConfig | None = None,
) -> dict[str, Any]:
    config = config or BertScoreConfig()
    if bool(input_csv) == bool(comparison_dir):
        raise ValueError("Provide exactly one of input_csv or comparison_dir.")

    if input_csv:
        source = Path(input_csv)
        run_label = _safe_label(label or source.stem)
        inputs = [source]
    else:
        source = Path(comparison_dir or "")
        run_label = _safe_label(label or source.name)
        inputs = _iter_ragas_csvs(source)
        if not inputs:
            raise ValueError(f"No RAGAS CSV files found in: {source}")

    output_dir = Path(output_root) / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [evaluate_csv(path, output_dir, config) for path in inputs]
    scored_means = [
        run["mean_bertscore_f1"]
        for run in runs
        if run.get("mean_bertscore_f1") is not None
    ]
    summary = {
        "label": run_label,
        "source": str(source.resolve()),
        "output_dir": str(output_dir.resolve()),
        "bertscore_model": config.model_type,
        "bertscore_lang": config.lang,
        "bertscore_rescale_with_baseline": config.rescale_with_baseline,
        "bertscore_batch_size": config.batch_size,
        "runs": runs,
        "mean_bertscore_f1_across_runs": _round_or_none(_mean(scored_means)),
    }

    summary_json = output_dir / "bertscore_summary.json"
    summary_csv = output_dir / "bertscore_summary.csv"
    summary["summary_json"] = str(summary_json.resolve())
    summary["summary_csv"] = str(summary_csv.resolve())
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(summary_csv, runs)
    return summary


def evaluate_all_completed(
    scores_root: str | Path = DEFAULT_RAGAS_SCORES_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    config: BertScoreConfig | None = None,
) -> dict[str, Any]:
    config = config or BertScoreConfig()
    experiments = discover_completed_experiments(scores_root)
    if not experiments:
        raise ValueError(f"No completed RAGAS CSV experiments found in: {scores_root}")

    summaries: list[dict[str, Any]] = []
    for experiment in experiments:
        print(f"\nProcessing completed experiment: {experiment['label']}")
        summaries.append(
            evaluate_inputs(
                input_csv=experiment["input_csv"],
                comparison_dir=experiment["comparison_dir"],
                output_root=output_root,
                label=experiment["label"],
                config=config,
            )
        )

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    scored_means = [
        summary["mean_bertscore_f1_across_runs"]
        for summary in summaries
        if summary.get("mean_bertscore_f1_across_runs") is not None
    ]
    master_summary = {
        "scores_root": str(Path(scores_root).resolve()),
        "output_root": str(output_root_path.resolve()),
        "bertscore_model": config.model_type,
        "bertscore_lang": config.lang,
        "bertscore_rescale_with_baseline": config.rescale_with_baseline,
        "bertscore_batch_size": config.batch_size,
        "experiments_count": len(summaries),
        "mean_bertscore_f1_across_experiments": _round_or_none(_mean(scored_means)),
        "experiments": summaries,
    }
    summary_json = output_root_path / "all_completed_bertscore_summary.json"
    summary_csv = output_root_path / "all_completed_bertscore_summary.csv"
    master_summary["summary_json"] = str(summary_json.resolve())
    master_summary["summary_csv"] = str(summary_csv.resolve())
    summary_json.write_text(json.dumps(master_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_all_completed_summary_csv(summary_csv, summaries)
    return master_summary


def _write_summary_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    fieldnames = [
        "input_csv",
        "output_csv",
        "rows_total",
        "rows_scored",
        "rows_skipped",
        "mean_bertscore_precision",
        "mean_bertscore_recall",
        "mean_bertscore_f1",
        "bertscore_model",
        "bertscore_rescale_with_baseline",
    ]
    _write_csv(path, fieldnames, runs)


def _write_all_completed_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    rows = []
    for summary in summaries:
        rows.append(
            {
                "label": summary["label"],
                "source": summary["source"],
                "output_dir": summary["output_dir"],
                "runs_count": len(summary["runs"]),
                "mean_bertscore_f1_across_runs": summary["mean_bertscore_f1_across_runs"],
                "bertscore_model": summary["bertscore_model"],
                "bertscore_rescale_with_baseline": summary["bertscore_rescale_with_baseline"],
            }
        )
    _write_csv(
        path,
        [
            "label",
            "source",
            "output_dir",
            "runs_count",
            "mean_bertscore_f1_across_runs",
            "bertscore_model",
            "bertscore_rescale_with_baseline",
        ],
        rows,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute BERTScore over existing RAGAS CSV outputs."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-csv", help="Single RAGAS CSV file to enrich with BERTScore.")
    group.add_argument("--comparison-dir", help="Directory with RAGAS CSV files from a comparison run.")
    group.add_argument(
        "--all-completed",
        action="store_true",
        help="Process every completed RAGAS score artifact under --scores-root.",
    )
    parser.add_argument("--scores-root", default=str(DEFAULT_RAGAS_SCORES_DIR), help="Root with completed RAGAS scores.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root directory for BERTScore artifacts.")
    parser.add_argument("--label", default=None, help="Output label. Defaults to input stem or comparison directory name.")
    parser.add_argument("--batch-size", type=int, default=BERTSCORE_BATCH_SIZE, help="BERTScore batch size.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BertScoreConfig(batch_size=args.batch_size)
    if args.all_completed:
        summary = evaluate_all_completed(
            scores_root=args.scores_root,
            output_root=args.output_root,
            config=config,
        )
        print("\nBERTScore completed for all completed experiments.")
        print(f"   model: {summary['bertscore_model']}")
        print(f"   experiments: {summary['experiments_count']}")
        print(f"   output: {summary['output_root']}")
        print(f"   summary: {summary['summary_json']}")
        return 0

    summary = evaluate_inputs(
        input_csv=args.input_csv,
        comparison_dir=args.comparison_dir,
        output_root=args.output_root,
        label=args.label,
        config=config,
    )
    print("\nBERTScore completed.")
    print(f"   model: {summary['bertscore_model']}")
    print(f"   output: {summary['output_dir']}")
    print(f"   summary: {summary['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
