"""Compute training-style metrics for stored RAG inference checkpoints.

Usage:
    python research/evaluation/training_metrics.py \
        --checkpoint-dir research/evaluation/runs/ragas/comparisons/reinferencia_v2_es/checkpoints

Dependencies:
    bert_score, pandas, torch/transformers stack from research/training requirements.

The metrics intentionally mirror the training scripts:
Token F1 uses SQuAD-style token overlap after EN/ES/CA article removal,
ROUGE-L uses token-level LCS with the same normalization, and BERTScore uses
``microsoft/deberta-xlarge-mnli`` with ``rescale_with_baseline=True``.

# ─────────────────────────────────────────────
# MODULE MAP
# ─────────────────────────────────────────────
#
#  1. Environment setup       import path and heavy-library knobs
#  2. Metric configuration    constants shared with training scripts
#  3. Token metrics           normalization, Token F1, ROUGE-L
#  4. Checkpoint evaluation   load generation payload, score, write per-row CSV
#  5. CLI                    argument parsing and multi-directory dispatch
#
# ─────────────────────────────────────────────
"""

from __future__ import annotations

# ─────────────────────────────────────────────
# SECTION 1: ENVIRONMENT SETUP
# ─────────────────────────────────────────────

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TRITON_DISABLE", "1")

_THIS_FILE = Path(__file__).resolve()
_PROJ_ROOT = _THIS_FILE.parent.parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import pandas as pd
from bert_score import score as bert_score_fn
import bert_score.utils as _bsu

from research.evaluation._lib.checkpoints import generation_from_checkpoint


# ─────────────────────────────────────────────
# SECTION 2: METRIC CONFIGURATION
# ─────────────────────────────────────────────

BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"
BERTSCORE_BATCH_SIZE = 32
BERTSCORE_LANG = "en"
BERTSCORE_RESCALE_WITH_BASELINE = True
DEFAULT_OUTPUT_DIR_NAME = "training_metrics"
DEFAULT_COMPARISON_FILENAME = "comparison_training_metrics.csv"
DEFAULT_GLOBAL_SUMMARY_FILENAME = "training_metrics_comparison_all.csv"


def _safe_sent_encode(tokenizer: Any, text: str) -> list[int]:
    """Match the DeBERTa max-length patch used by the training scripts."""
    return tokenizer.encode(
        text.strip(),
        add_special_tokens=True,
        max_length=512,
        truncation=True,
    )


_bsu.sent_encode = _safe_sent_encode


# ─────────────────────────────────────────────
# SECTION 3: TOKEN METRICS
# ─────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase, strip EN/ES/CA articles and remove punctuation.

    Args:
        text: Raw text string to normalize.

    Returns:
        Normalized text matching ``research/training/train_*.py``.
    """
    text = str(text).lower()
    text = re.sub(
        r"\b(a|an|the|el|la|los|las|un|una|unos|unas|les|els|uns|unes)\b",
        " ",
        text,
    )
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute SQuAD-style token-level F1.

    Args:
        prediction: Generated answer.
        ground_truth: Reference answer.

    Returns:
        Token F1 in the 0.0-1.0 range.
    """
    pred_tok = normalize_text(prediction).split()
    truth_tok = normalize_text(ground_truth).split()
    if not pred_tok or not truth_tok:
        return 1.0 if pred_tok == truth_tok else 0.0
    common = Counter(pred_tok) & Counter(truth_tok)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tok)
    recall = n_common / len(truth_tok)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Return token-level longest common subsequence length."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(prediction: str, ground_truth: str) -> float:
    """Compute token-level ROUGE-L F1 using the training normalization.

    Args:
        prediction: Generated answer.
        ground_truth: Reference answer.

    Returns:
        ROUGE-L F1 in the 0.0-1.0 range.
    """
    pred_tok = normalize_text(prediction).split()
    truth_tok = normalize_text(ground_truth).split()
    if not pred_tok or not truth_tok:
        return 1.0 if pred_tok == truth_tok else 0.0
    lcs = _lcs_length(pred_tok, truth_tok)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tok)
    recall = lcs / len(truth_tok)
    return 2 * precision * recall / (precision + recall)


def percent_mean(values: list[float]) -> float:
    """Return a training-style percent mean rounded to two decimals."""
    return round((sum(values) / len(values)) * 100, 2) if values else 0.0


def percent_sd(values: list[float]) -> float:
    """Return sample standard deviation in percent units."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return round(math.sqrt(variance) * 100, 2)


# ─────────────────────────────────────────────
# SECTION 4: CHECKPOINT EVALUATION
# ─────────────────────────────────────────────

def load_generation(checkpoint_path: Path) -> dict[str, Any]:
    """Load a checkpoint and resolve its dataset-backed generation payload.

    Args:
        checkpoint_path: Path to a checkpoint JSON created by ``infer.py``.

    Returns:
        Generation payload with questions, answers and ground truths aligned.

    Raises:
        ValueError: If the checkpoint cannot be parsed as a RAG checkpoint.
    """
    with checkpoint_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return generation_from_checkpoint(payload, checkpoint_path)


def evaluate_checkpoint(checkpoint_path: Path, out_dir: Path, overwrite: bool) -> dict[str, Any]:
    """Score one checkpoint and write its per-question CSV.

    Args:
        checkpoint_path: Checkpoint JSON path.
        out_dir: Destination directory for ``<variant>.csv``.
        overwrite: Whether to overwrite an existing per-checkpoint CSV.

    Returns:
        Summary row for the comparison CSV.
    """
    generation = load_generation(checkpoint_path)
    questions = [str(value) for value in generation["questions"]]
    ground_truths = [str(value) for value in generation["ground_truths"]]
    answers = [str(value) for value in generation["answers"]]
    statuses = generation.get("question_statuses") or []

    if not any(ref.strip() for ref in ground_truths):
        raise ValueError(
            f"Checkpoint has no ground_truth values after dataset resolution: {checkpoint_path}"
        )

    out_path = out_dir / f"{checkpoint_path.stem}.csv"
    if out_path.exists() and not overwrite:
        print(f"  [skip] exists: {out_path}")
        return summarize_existing_csv(checkpoint_path, out_path, generation)

    print(
        f"\nBERTScore | {checkpoint_path.parent.parent.name}/"
        f"{checkpoint_path.stem} ({len(answers)} samples)"
    )
    p_tensor, r_tensor, f1_tensor = bert_score_fn(
        answers,
        ground_truths,
        model_type=BERTSCORE_MODEL,
        lang=BERTSCORE_LANG,
        rescale_with_baseline=BERTSCORE_RESCALE_WITH_BASELINE,
        batch_size=BERTSCORE_BATCH_SIZE,
        verbose=False,
    )
    bertscore_p = p_tensor.tolist()
    bertscore_r = r_tensor.tolist()
    bertscore_f1 = f1_tensor.tolist()

    token_f1 = [
        compute_f1(answer, ground_truth)
        for answer, ground_truth in zip(answers, ground_truths)
    ]
    rouge_l = [
        compute_rouge_l(answer, ground_truth)
        for answer, ground_truth in zip(answers, ground_truths)
    ]

    rows: list[dict[str, Any]] = []
    for i, (question, ground_truth, answer) in enumerate(
        zip(questions, ground_truths, answers)
    ):
        status = statuses[i] if i < len(statuses) and isinstance(statuses[i], dict) else {}
        rows.append({
            "question_number": i + 1,
            "variant": checkpoint_path.stem,
            "checkpoint_path": str(checkpoint_path),
            "dataset_path": generation["dataset_path"],
            "eval_corpus": generation.get("eval_corpus", ""),
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "status": status.get("status", ""),
            "reason": status.get("reason", ""),
            "token_f1": round(token_f1[i], 4),
            "rouge_l": round(rouge_l[i], 4),
            "bertscore_p": round(bertscore_p[i], 4),
            "bertscore_r": round(bertscore_r[i], 4),
            "bertscore_f1": round(bertscore_f1[i], 4),
            "bertscore_model": BERTSCORE_MODEL,
            "bertscore_lang": BERTSCORE_LANG,
            "bertscore_rescale_with_baseline": BERTSCORE_RESCALE_WITH_BASELINE,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    return summarize_metric_lists(
        checkpoint_path=checkpoint_path,
        output_path=out_path,
        generation=generation,
        token_f1=token_f1,
        rouge_l=rouge_l,
        bertscore_p=bertscore_p,
        bertscore_r=bertscore_r,
        bertscore_f1=bertscore_f1,
    )


def summarize_existing_csv(
    checkpoint_path: Path,
    output_path: Path,
    generation: dict[str, Any],
) -> dict[str, Any]:
    """Build a summary row from an existing per-checkpoint CSV."""
    df = pd.read_csv(output_path)
    return summarize_metric_lists(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        generation=generation,
        token_f1=[float(value) for value in df["token_f1"].tolist()],
        rouge_l=[float(value) for value in df["rouge_l"].tolist()],
        bertscore_p=[float(value) for value in df["bertscore_p"].tolist()],
        bertscore_r=[float(value) for value in df["bertscore_r"].tolist()],
        bertscore_f1=[float(value) for value in df["bertscore_f1"].tolist()],
    )


def summarize_metric_lists(
    checkpoint_path: Path,
    output_path: Path,
    generation: dict[str, Any],
    token_f1: list[float],
    rouge_l: list[float],
    bertscore_p: list[float],
    bertscore_r: list[float],
    bertscore_f1: list[float],
) -> dict[str, Any]:
    """Build the training-style aggregate row for one variant."""
    return {
        "comparison": checkpoint_path.parent.parent.name,
        "variant": checkpoint_path.stem,
        "checkpoint_path": str(checkpoint_path),
        "output_path": str(output_path),
        "dataset_path": generation["dataset_path"],
        "eval_corpus": generation.get("eval_corpus", ""),
        "n_samples": len(token_f1),
        "Token_F1": percent_mean(token_f1),
        "Token_F1_sd": percent_sd(token_f1),
        "ROUGE_L_F1": percent_mean(rouge_l),
        "ROUGE_L_F1_sd": percent_sd(rouge_l),
        "BERTScore_P": percent_mean(bertscore_p),
        "BERTScore_R": percent_mean(bertscore_r),
        "BERTScore_F1": percent_mean(bertscore_f1),
        "BERTScore_F1_sd": percent_sd(bertscore_f1),
        "bertscore_model": BERTSCORE_MODEL,
        "bertscore_batch_size": BERTSCORE_BATCH_SIZE,
        "bertscore_lang": BERTSCORE_LANG,
        "bertscore_rescale_with_baseline": BERTSCORE_RESCALE_WITH_BASELINE,
    }


# ─────────────────────────────────────────────
# SECTION 5: CLI
# ─────────────────────────────────────────────

def checkpoint_paths_from_arg(path: Path) -> list[Path]:
    """Resolve one CLI input into sorted checkpoint JSON paths."""
    resolved = path.resolve()
    if resolved.is_dir():
        return sorted(child for child in resolved.glob("*.json") if child.is_file())
    if resolved.is_file():
        return [resolved]
    raise FileNotFoundError(resolved)


def default_global_summary_path(checkpoint_dirs: list[Path]) -> Path:
    """Choose a shared summary path for multiple comparison directories."""
    parents = [path.resolve().parent for path in checkpoint_dirs]
    common = Path(os.path.commonpath([str(parent) for parent in parents]))
    return common / DEFAULT_GLOBAL_SUMMARY_FILENAME


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compute Token F1, ROUGE-L and BERTScore for RAG checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        action="append",
        required=True,
        type=Path,
        help="Checkpoint directory or individual checkpoint JSON. Repeatable.",
    )
    parser.add_argument(
        "--output-dir-name",
        default=DEFAULT_OUTPUT_DIR_NAME,
        help="Sibling output folder name next to checkpoints.",
    )
    parser.add_argument(
        "--comparison-filename",
        default=DEFAULT_COMPARISON_FILENAME,
        help="Per-directory aggregate CSV filename.",
    )
    parser.add_argument(
        "--global-summary",
        type=Path,
        default=None,
        help="Optional aggregate CSV across every processed input.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite existing per-checkpoint CSV files.",
    )
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    input_paths = [path.resolve() for path in args.checkpoint_dir]

    all_summary_rows: list[dict[str, Any]] = []
    for input_path in input_paths:
        checkpoints = checkpoint_paths_from_arg(input_path)
        if not checkpoints:
            print(f"[warn] no checkpoint JSON files found: {input_path}")
            continue

        checkpoint_parent = input_path if input_path.is_dir() else input_path.parent
        out_dir = checkpoint_parent.parent / args.output_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_rows = [
            evaluate_checkpoint(checkpoint_path, out_dir, overwrite=args.overwrite)
            for checkpoint_path in checkpoints
        ]
        summary_path = out_dir / args.comparison_filename
        pd.DataFrame(summary_rows).sort_values(["comparison", "variant"]).to_csv(
            summary_path,
            index=False,
            encoding="utf-8",
        )
        all_summary_rows.extend(summary_rows)
        print(f"Resumen: {summary_path}")

    if all_summary_rows:
        global_summary = (
            args.global_summary.resolve()
            if args.global_summary
            else default_global_summary_path(input_paths)
        )
        global_summary.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_summary_rows).sort_values(["comparison", "variant"]).to_csv(
            global_summary,
            index=False,
            encoding="utf-8",
        )
        print(f"\nResumen global: {global_summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
