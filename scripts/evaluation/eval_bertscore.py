"""
BERTScore evaluation for RAG models.
==================================================================

Evaluates all 3 models (or a specific one) -- both the base version and the
fine-tuned version -- using BERTScore as the primary semantic similarity
metric against the ground truth.

BERTScore measures contextual similarity between prediction and reference
using BERT embeddings, capturing paraphrases and semantic equivalences
that standard Token F1 (lexical overlap) cannot detect.

Metrics reported per sample and aggregated:
  - BERTScore Precision (P)  -- semantic precision of the generated text.
  - BERTScore Recall    (R)  -- semantic coverage of the reference.
  - BERTScore F1        (F1) -- harmonic mean of P and R.

Evaluated datasets (same frozen test sets as the training scripts):
  - Neural-Bridge RAG   (200 samples)
  - Dolly QA            (200 samples)
  - Aina RAG            (200 samples)

Pipeline:
  [1] Load datasets (same splits and filters as train-*.py).
  [2] For each selected model:
      a) Load base model -> generate predictions -> BERTScore.
      b) Load base model + LoRA adapter -> generate predictions -> BERTScore.
  [3] Save results in JSON and summary CSV.
  [4] Generate comparative plots.

Usage:
    python eval_bertscore.py                       # Evaluate all 3 models
    python eval_bertscore.py --model gemma-3       # Only gemma-3
    python eval_bertscore.py --model llama-3       # Only llama-3

Output (all under bertscore/):
    bertscore_results_{model}.json     -- detailed results per model
    bertscore_summary.csv              -- global summary
    plots/eval/bertscore_comparison.png
    bertscore_per_sample_{model}_{dataset}_{variant}.csv  -- P/R/F1 per sample (n=200)
    metrics_per_sample_{model}_{dataset}_{variant}.csv    -- Token F1 + Faithfulness per sample

    Use compute_std.py to compute mean +/- std over per-sample CSVs.

Dependencies:
    torch, transformers, peft, datasets, bert_score, tqdm, matplotlib
"""

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
# CRITICAL: disable triton/torch.compile BEFORE any torch or transformers
# import to avoid the torchao crash on the cluster (Python.h).

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"]  = "1"
os.environ["TRITON_DISABLE"]        = "1"
os.environ["TORCHDYNAMO_DISABLE"]   = "1"

import argparse
import gc
import json
import re
import sys
from collections import Counter

import torch
from bert_score import score as bert_score_fn
import bert_score.utils as _bsu

# ---------------------------------------------------------------------------
# MONKEY-PATCH: bert_score.utils.sent_encode
# ---------------------------------------------------------------------------
# The internal version of sent_encode calls tokenizer.encode() using
# tokenizer.model_max_length as max_length.  For DeBERTa that value can
# be sys.maxsize (~2^63), which causes the Rust backend of ``tokenizers``
# to raise an OverflowError when trying to store it as a 32-bit int.
# This patch forces max_length=512 (the physical limit of DeBERTa)
# and truncation=True.
# ---------------------------------------------------------------------------
def _safe_sent_encode(tokenizer, a):
    return tokenizer.encode(
        a.strip(),
        add_special_tokens=True,
        max_length=512,
        truncation=True,
    )

_bsu.sent_encode = _safe_sent_encode
# ---------------------------------------------------------------------------
from datasets import load_dataset
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS           = 2048
MAX_CONTEXT_TOKENS       = 2048
BERTSCORE_MODEL          = "microsoft/deberta-xlarge-mnli"

DOLLY_RAG_CATEGORIES = {"closed_qa", "information_extraction", "summarization"}

SYSTEM_PROMPT = (
    "You are a professional document analysis assistant. Your role is to answer "
    "questions accurately based on the provided document context.\n\n"
    "Guidelines:\n"
    "- Base your answers strictly on the information within the <context> tags.\n"
    "- Do not add information beyond what the context provides.\n"
    "- Formulate clear, well-structured responses in complete sentences.\n"
    "- For factual questions, be direct and precise.\n"
    "- For analytical or complex questions, provide detailed explanations "
    "referencing specific information from the context.\n"
    "- Always respond in the same language as the context "
    "(English, Spanish/Castellano, or Catalan/Català).\n"
    "- Synthesize information naturally rather than copying text verbatim."
)

MODEL_CONFIGS = {
    "qwen-3": {
        "hf_name":   "Qwen/Qwen3-14B",
        "display":   "Qwen3-FineTuned",
    },
    "llama-3": {
        "hf_name":   "meta-llama/Llama-3.1-8B-Instruct",
        "display":   "Llama3-FineTuned",
    },
    "gemma-3": {
        "hf_name":   "google/gemma-3-12b-it",
        "display":   "Gemma3-FineTuned",
    },
}

VALID_MODELS = list(MODEL_CONFIGS.keys())


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="BERTScore evaluation of RAG models (base vs. fine-tuned).",
)
parser.add_argument(
    "--model", choices=VALID_MODELS, default=None,
    help="Model to evaluate. If omitted, evaluates all 3.",
)
parser.add_argument(
    "--output-dir", default=None,
    help="Root output directory for training output (default: cwd).",
)
parser.add_argument(
    "--bertscore-model", default=BERTSCORE_MODEL,
    help=f"BERT model for BERTScore (default: {BERTSCORE_MODEL}).",
)
parser.add_argument(
    "--batch-size", type=int, default=32,
    help="Batch size for BERTScore (default: 32).",
)
args = parser.parse_args()

OUTPUT_ROOT = args.output_dir or os.getcwd()
BERTSCORE_MODEL = args.bertscore_model
models_to_eval = [args.model] if args.model else VALID_MODELS


# ─────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────
# Reuses the same filtering and normalization logic as train-*.py to
# ensure that the frozen test set is identical.

def _normalize_nb(example):
    """Normalize Neural-Bridge RAG fields to the common schema."""
    return {
        "instruction": (example.get("question") or "").strip(),
        "context":     (example.get("context")  or "").strip(),
        "response":    (example.get("answer")   or "").strip(),
    }


def _normalize_dolly(example):
    """Normalize Dolly QA fields to the common schema."""
    return {
        "instruction": (example.get("instruction") or "").strip(),
        "context":     (example.get("context")     or "").strip(),
        "response":    (example.get("response")    or "").strip(),
        "category":    (example.get("category")    or "").strip(),
    }


def _normalize_aina(example):
    """Normalize Aina RAG Multilingual fields, preserving ``lang`` for splitting."""
    return {
        "instruction": (example.get("instruction") or "").strip(),
        "context":     (example.get("context")     or "").strip(),
        "response":    (example.get("response")    or "").strip(),
        "lang":        (example.get("lang")        or "").strip(),
    }


def _filter_valid(ex):
    """Return True if all required fields are non-empty."""
    return (
        bool(ex["instruction"].strip())
        and bool(ex["context"].strip())
        and bool(ex["response"].strip())
    )


def _filter_long_response(ex, min_words=15):
    """Return True if the response has at least ``min_words`` words."""
    return len(ex["response"].split()) >= min_words


def _filter_dolly_rag(ex):
    """Return True if the example belongs to a RAG-relevant Dolly category."""
    return (
        ex["category"] in DOLLY_RAG_CATEGORIES
        and bool(ex["context"].strip())
    )


def load_eval_datasets() -> dict:
    """Load the 5 frozen test sets (identical splits to train-*.py v8).

    Aina RAG is split into 3 per-language subsets (EN, ES, CA).
    Full test partitions are used (no sample cap).

    Returns:
        Dictionary mapping dataset names to HuggingFace Dataset objects.
    """
    print("\n--> Loading evaluation datasets...")

    # Neural-Bridge RAG (test split)
    print("  Neural-Bridge RAG...")
    nb_test = (
        load_dataset("neural-bridge/rag-dataset-12000", split="test")
        .map(_normalize_nb, remove_columns=["context", "question", "answer"])
        .filter(_filter_valid)
        .shuffle(seed=42)
    )

    # Dolly QA (manual split: last 10%)
    print("  Dolly QA...")
    _dolly_all = (
        load_dataset("databricks/databricks-dolly-15k", split="train")
        .map(_normalize_dolly, remove_columns=["instruction", "context", "response", "category"])
        .filter(_filter_dolly_rag)
        .filter(_filter_valid)
        .filter(_filter_long_response)
        .shuffle(seed=42)
        .remove_columns(["category"])
    )
    nd = len(_dolly_all)
    nd_train = int(nd * 0.80)
    nd_val   = int(nd * 0.10)
    dolly_test = _dolly_all.select(range(nd_train + nd_val, nd))

    # Aina RAG Multilingual (test split, per language)
    print("  Aina RAG Multilingual (split by language)...")
    _AINA_EXTRA_COLS = ["id", "category", "extractive"]
    ds = load_dataset("projecte-aina/RAG_Multilingual", split="test")
    cols_to_remove = [c for c in _AINA_EXTRA_COLS if c in ds.column_names]
    aina_all = (
        ds
        .map(_normalize_aina, remove_columns=cols_to_remove)
        .filter(_filter_valid)
        .filter(_filter_long_response)
    )

    eval_datasets = {
        "Neural-Bridge RAG": nb_test,
        "Dolly QA":          dolly_test,
    }
    for lang_code, lang_label in [("en", "Aina-EN"), ("es", "Aina-ES"), ("ca", "Aina-CA")]:
        sub = aina_all.filter(lambda ex, lc=lang_code: ex["lang"] == lc)
        eval_datasets[lang_label] = sub.remove_columns(["lang"]).shuffle(seed=42)

    for name, ds in eval_datasets.items():
        print(f"    {name}: {len(ds)} samples (FROZEN)")

    return eval_datasets


# ─────────────────────────────────────────────
# RESPONSE GENERATION
# ─────────────────────────────────────────────
# Each model uses its own chat template and stop tokens.

def _build_prompt(tokenizer, instruction, context, model_key):
    """Build the prompt using the model's native chat template.

    Args:
        tokenizer: The HuggingFace tokenizer for the model.
        instruction: The user instruction / question.
        context: The document context to base the answer on.
        model_key: Model identifier (``"qwen-3"``, ``"llama-3"``, or ``"gemma-3"``).

    Returns:
        Formatted prompt string ready for tokenization.
    """
    ctx = (context or "").strip()
    if ctx:
        ctx_ids = tokenizer(ctx, add_special_tokens=False)["input_ids"]
        if len(ctx_ids) > MAX_CONTEXT_TOKENS:
            ctx = tokenizer.decode(ctx_ids[:MAX_CONTEXT_TOKENS], skip_special_tokens=True)
        user_msg = f"{instruction}\n\n<context>{ctx}</context>"
    else:
        user_msg = instruction

    if model_key == "qwen-3":
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    elif model_key == "llama-3":
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    elif model_key == "gemma-3":
        messages = [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_msg}"},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def _get_eos_ids(tokenizer, model_key):
    """Return the EOS token IDs specific to each model.

    Args:
        tokenizer: The HuggingFace tokenizer for the model.
        model_key: Model identifier.

    Returns:
        List of EOS token IDs used as stopping criteria.
    """
    if model_key == "qwen-3":
        return [tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]]
    elif model_key == "llama-3":
        eot_id = tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]
        return list({tokenizer.eos_token_id, eot_id})
    elif model_key == "gemma-3":
        eot_id = tokenizer.encode("<end_of_turn>", add_special_tokens=False)[-1]
        return list({tokenizer.eos_token_id, eot_id})


def generate_response(model, tokenizer, instruction, context, model_key):
    """Generate a response using the model's native format.

    Args:
        model: The loaded causal language model.
        tokenizer: The corresponding tokenizer.
        instruction: The user question.
        context: The document context.
        model_key: Model identifier.

    Returns:
        Generated response string, stripped of special tokens.
    """
    prompt = _build_prompt(tokenizer, instruction, context, model_key)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    eos_ids = _get_eos_ids(tokenizer, model_key)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def generate_all_predictions(model, tokenizer, eval_datasets, model_key, label):
    """Generate predictions for all datasets.

    Args:
        model: The loaded causal language model.
        tokenizer: The corresponding tokenizer.
        eval_datasets: Dictionary of dataset name to HuggingFace Dataset.
        model_key: Model identifier.
        label: Display label for progress bars (e.g. ``"BASE"``).

    Returns:
        Tuple of (predictions, ground_truths) where each is a
        dict mapping dataset names to lists of strings.
    """
    predictions = {}
    ground_truths = {}
    model.eval()

    for ds_name, ds in eval_datasets.items():
        preds = []
        gts = []
        for example in tqdm(ds, desc=f"{label} | {ds_name}"):
            pred = generate_response(
                model, tokenizer,
                example["instruction"], example["context"],
                model_key,
            )
            preds.append(pred)
            gts.append(example["response"])
        predictions[ds_name] = preds
        ground_truths[ds_name] = gts

    return predictions, ground_truths


# ─────────────────────────────────────────────
# BERTSCORE COMPUTATION
# ─────────────────────────────────────────────

def compute_bertscore(predictions, ground_truths, batch_size=32):
    """Compute BERTScore (P, R, F1) for each dataset.

    Args:
        predictions: Dict mapping dataset names to lists of predicted strings.
        ground_truths: Dict mapping dataset names to lists of reference strings.
        batch_size: Batch size for BERTScore computation.

    Returns:
        Dict mapping dataset names to score dictionaries containing
        per-sample lists (``P``, ``R``, ``F1``) and averages
        (``avg_P``, ``avg_R``, ``avg_F1``) as percentages.
    """
    results = {}
    for ds_name in predictions:
        preds = predictions[ds_name]
        refs  = ground_truths[ds_name]

        print(f"  Computing BERTScore for {ds_name} ({len(preds)} samples)...")
        P, R, F1 = bert_score_fn(
            preds, refs,
            model_type=BERTSCORE_MODEL,
            lang="en",
            rescale_with_baseline=True,
            batch_size=batch_size,
            verbose=False,
        )
        p_list  = P.tolist()
        r_list  = R.tolist()
        f1_list = F1.tolist()

        results[ds_name] = {
            "P":      [round(x, 4) for x in p_list],
            "R":      [round(x, 4) for x in r_list],
            "F1":     [round(x, 4) for x in f1_list],
            "avg_P":  round(sum(p_list)  / len(p_list)  * 100, 2),
            "avg_R":  round(sum(r_list)  / len(r_list)  * 100, 2),
            "avg_F1": round(sum(f1_list) / len(f1_list) * 100, 2),
        }
        print(f"    P={results[ds_name]['avg_P']:.2f}%  "
              f"R={results[ds_name]['avg_R']:.2f}%  "
              f"F1={results[ds_name]['avg_F1']:.2f}%")

    return results


# ─────────────────────────────────────────────
# AUXILIARY METRICS
# ─────────────────────────────────────────────
# Included for direct comparison with the results from train-*.py.

def normalize_text(text):
    """Normalize text by lowercasing, removing articles and punctuation.

    Args:
        text: Raw text string.

    Returns:
        Cleaned, whitespace-normalized string.
    """
    text = str(text).lower()
    text = re.sub(
        r'\b(a|an|the|el|la|los|las|un|una|unos|unas|les|els|uns|unes)\b',
        ' ', text
    )
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())


def compute_f1(prediction, ground_truth):
    """Compute token-level F1 between prediction and ground truth.

    Args:
        prediction: Predicted text string.
        ground_truth: Reference text string.

    Returns:
        Token F1 score as a float in [0, 1].
    """
    pred_tok  = normalize_text(prediction).split()
    truth_tok = normalize_text(ground_truth).split()
    if not pred_tok or not truth_tok:
        return 1.0 if pred_tok == truth_tok else 0.0
    common = Counter(pred_tok) & Counter(truth_tok)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec = n / len(pred_tok)
    rec  = n / len(truth_tok)
    return 2 * prec * rec / (prec + rec)


def compute_context_faithfulness(prediction, context):
    """Compute faithfulness as the fraction of prediction tokens found in context.

    Args:
        prediction: Predicted text string.
        context: Source context string.

    Returns:
        Faithfulness score as a float in [0, 1].
    """
    pred_types = set(normalize_text(prediction).split())
    ctx_types  = set(normalize_text(context).split())
    if not pred_types:
        return 0.0
    return len(pred_types & ctx_types) / len(pred_types)


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def plot_bertscore_comparison(all_results, output_path):
    """Generate a comparative BERTScore F1 chart (base vs. adapted) per model and dataset.

    Args:
        all_results: Dict mapping model keys to their evaluation result dicts.
        output_path: File path for the saved PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available -- skipping plots.")
        return

    models_in = list(all_results.keys())
    datasets  = ["Neural-Bridge RAG", "Dolly QA", "Aina-EN", "Aina-ES", "Aina-CA"]
    ds_short  = ["Neural-Bridge", "Dolly QA", "Aina-EN", "Aina-ES", "Aina-CA"]

    BASE_COLOR  = "#64748b"
    ADAPT_COLOR = "#2563eb"
    DELTA_COLOR = "#16a34a"
    NEG_COLOR   = "#dc2626"

    n_models = len(models_in)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 5), squeeze=False)

    for idx, model_key in enumerate(models_in):
        ax = axes[0, idx]
        data = all_results[model_key]
        display = MODEL_CONFIGS[model_key]["display"]

        base_vals = [data["base_bertscore"][ds]["avg_F1"] for ds in datasets]
        adapt_vals = [data["adapted_bertscore"][ds]["avg_F1"] for ds in datasets]

        x = np.arange(len(datasets))
        w = 0.35
        bars_base  = ax.bar(x - w/2, base_vals,  w, label="Base",       color=BASE_COLOR, zorder=3)
        bars_adapt = ax.bar(x + w/2, adapt_vals, w, label="Fine-tuned", color=ADAPT_COLOR, zorder=3)

        for i, (b, a) in enumerate(zip(base_vals, adapt_vals)):
            delta = a - b
            color = DELTA_COLOR if delta >= 0 else NEG_COLOR
            ax.text(x[i] + w/2, a + 0.5, f"\u0394{delta:+.1f}",
                    ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")
            ax.text(x[i] - w/2, b + 0.5, f"{b:.1f}",
                    ha="center", va="bottom", fontsize=7, color="gray")
            ax.text(x[i] + w/2, a + 0.5 + 2.0, f"{a:.1f}",
                    ha="center", va="bottom", fontsize=7, color=ADAPT_COLOR)

        ax.set_title(display, fontsize=12, fontweight="bold")
        ax.set_ylabel("BERTScore F1 (%)" if idx == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(ds_short, fontsize=9)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle("BERTScore F1 — Base vs. Fine-tuned", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


def plot_bertscore_aggregate(all_results, output_path):
    """Generate an aggregated BERTScore P/R/F1 chart per model.

    Args:
        all_results: Dict mapping model keys to their evaluation result dicts.
        output_path: File path for the saved PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    models_in = list(all_results.keys())
    datasets  = ["Neural-Bridge RAG", "Dolly QA", "Aina-EN", "Aina-ES", "Aina-CA"]

    BASE_COLOR  = "#64748b"
    ADAPT_COLOR = "#2563eb"
    metrics = ["avg_P", "avg_R", "avg_F1"]
    metric_labels = ["Precision", "Recall", "F1"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)

    for m_idx, (metric, m_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[0, m_idx]
        model_labels = [MODEL_CONFIGS[m]["display"] for m in models_in]

        base_vals = []
        adapt_vals = []
        for model_key in models_in:
            data = all_results[model_key]
            b = sum(data["base_bertscore"][ds][metric] for ds in datasets) / len(datasets)
            a = sum(data["adapted_bertscore"][ds][metric] for ds in datasets) / len(datasets)
            base_vals.append(b)
            adapt_vals.append(a)

        x = np.arange(len(models_in))
        w = 0.35
        ax.bar(x - w/2, base_vals,  w, label="Base",       color=BASE_COLOR, zorder=3)
        ax.bar(x + w/2, adapt_vals, w, label="Fine-tuned", color=ADAPT_COLOR, zorder=3)

        for i, (b, a) in enumerate(zip(base_vals, adapt_vals)):
            ax.text(x[i] - w/2, b + 0.5, f"{b:.1f}", ha="center", fontsize=8, color="gray")
            ax.text(x[i] + w/2, a + 0.5, f"{a:.1f}", ha="center", fontsize=8, color=ADAPT_COLOR)

        ax.set_title(f"BERTScore {m_label}", fontsize=11, fontweight="bold")
        ax.set_ylabel("%" if m_idx == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=9)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle("Aggregated BERTScore — Base vs. Fine-tuned per model",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def get_adapter_path(model_key):
    """Resolve the LoRA adapter path.

    Compatible with the cluster directory structure
    (``training-output-<model>``) and the local structure
    (``training-output/<model>/``).

    Args:
        model_key: Model identifier.

    Returns:
        Path to the directory containing ``adapter_config.json``.

    Raises:
        FileNotFoundError: If the adapter cannot be found in any
            expected location.
    """
    # Cluster structure: training-output-gemma, training-output-llama, etc.
    cluster_names = {
        "qwen-3":  "training-output-qwen",
        "llama-3": "training-output-llama",
        "gemma-3": "training-output-gemma",
    }
    cluster_path = os.path.join(OUTPUT_ROOT, cluster_names[model_key])
    if os.path.exists(os.path.join(cluster_path, "adapter_config.json")):
        return cluster_path

    # Local structure: training-output/<model>/
    local_path = os.path.join(OUTPUT_ROOT, "training-output", model_key)
    if os.path.exists(os.path.join(local_path, "adapter_config.json")):
        return local_path

    raise FileNotFoundError(
        f"LoRA adapter not found for {model_key}.\n"
        f"  Searched in: {cluster_path}\n"
        f"           and: {local_path}"
    )


def get_results_dir(model_key):
    """Resolve the output directory for results. Everything is saved under bertscore/.

    Args:
        model_key: Model identifier.

    Returns:
        Path to the results directory.
    """
    return os.path.join(OUTPUT_ROOT, "bertscore")


def evaluate_model(model_key, eval_datasets, batch_size=32):
    """Evaluate a complete model (base + adapted) with BERTScore.

    Args:
        model_key: Model identifier.
        eval_datasets: Dictionary of dataset name to HuggingFace Dataset.
        batch_size: Batch size for BERTScore computation.

    Returns:
        Result dictionary containing BERTScore and auxiliary metrics
        for both base and adapted variants.
    """
    cfg = MODEL_CONFIGS[model_key]
    hf_name = cfg["hf_name"]
    adapter_path = get_adapter_path(model_key)
    results_dir  = get_results_dir(model_key)

    print("\n" + "=" * 70)
    print(f"  MODEL: {cfg['display']}  ({hf_name})")
    print(f"  Adapter: {adapter_path}")
    print("=" * 70)

    # --- Load base model ---
    print(f"\n--> Loading base model: {hf_name}")
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # --- BASE predictions ---
    print("\n--> [BASE] Generating predictions...")
    base_preds, ground_truths = generate_all_predictions(
        model, tokenizer, eval_datasets, model_key, label="BASE"
    )
    print("\n--> [BASE] Computing BERTScore...")
    base_bertscore = compute_bertscore(base_preds, ground_truths, batch_size)

    # Auxiliary metrics (Token F1, Faithfulness) for comparison
    base_aux = {}
    base_aux_per_sample = {}
    for ds_name, ds in eval_datasets.items():
        f1s, faiths = [], []
        for pred, ex in zip(base_preds[ds_name], ds):
            f1s.append(compute_f1(pred, ex["response"]))
            faiths.append(compute_context_faithfulness(pred, ex["context"]))
        base_aux[ds_name] = {
            "Token_F1": round(sum(f1s) / len(f1s) * 100, 2),
            "Faithfulness": round(sum(faiths) / len(faiths) * 100, 2),
        }
        base_aux_per_sample[ds_name] = {"token_f1": f1s, "faithfulness": faiths}

    # --- Load LoRA adapter ---
    print(f"\n--> Applying LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    print("--> Adapter merged.")

    # --- ADAPTED predictions ---
    print("\n--> [ADAPTED] Generating predictions...")
    adapted_preds, _ = generate_all_predictions(
        model, tokenizer, eval_datasets, model_key, label="ADAPTED"
    )
    print("\n--> [ADAPTED] Computing BERTScore...")
    adapted_bertscore = compute_bertscore(adapted_preds, ground_truths, batch_size)

    # Adapted auxiliary metrics
    adapted_aux = {}
    adapted_aux_per_sample = {}
    for ds_name, ds in eval_datasets.items():
        f1s, faiths = [], []
        for pred, ex in zip(adapted_preds[ds_name], ds):
            f1s.append(compute_f1(pred, ex["response"]))
            faiths.append(compute_context_faithfulness(pred, ex["context"]))
        adapted_aux[ds_name] = {
            "Token_F1": round(sum(f1s) / len(f1s) * 100, 2),
            "Faithfulness": round(sum(faiths) / len(faiths) * 100, 2),
        }
        adapted_aux_per_sample[ds_name] = {"token_f1": f1s, "faithfulness": faiths}

    # --- Free GPU memory ---
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- Assemble result ---
    datasets_list = list(eval_datasets.keys())
    result = {
        "model":       model_key,
        "hf_name":     hf_name,
        "bertscore_model": BERTSCORE_MODEL,
        "base_bertscore":    base_bertscore,
        "adapted_bertscore": adapted_bertscore,
        "base_aux":          base_aux,
        "adapted_aux":       adapted_aux,
        "per_dataset": {},
    }

    for ds_name in datasets_list:
        b = base_bertscore[ds_name]
        a = adapted_bertscore[ds_name]
        result["per_dataset"][ds_name] = {
            "n_samples": len(eval_datasets[ds_name]),
            "base": {
                "BERTScore_P":  b["avg_P"],
                "BERTScore_R":  b["avg_R"],
                "BERTScore_F1": b["avg_F1"],
                "Token_F1":     base_aux[ds_name]["Token_F1"],
                "Faithfulness": base_aux[ds_name]["Faithfulness"],
            },
            "adapted": {
                "BERTScore_P":  a["avg_P"],
                "BERTScore_R":  a["avg_R"],
                "BERTScore_F1": a["avg_F1"],
                "Token_F1":     adapted_aux[ds_name]["Token_F1"],
                "Faithfulness": adapted_aux[ds_name]["Faithfulness"],
            },
            "deltas": {
                "BERTScore_P":  round(a["avg_P"]  - b["avg_P"],  2),
                "BERTScore_R":  round(a["avg_R"]  - b["avg_R"],  2),
                "BERTScore_F1": round(a["avg_F1"] - b["avg_F1"], 2),
                "Token_F1":     round(adapted_aux[ds_name]["Token_F1"] - base_aux[ds_name]["Token_F1"], 2),
                "Faithfulness": round(adapted_aux[ds_name]["Faithfulness"] - base_aux[ds_name]["Faithfulness"], 2),
            },
            "sample_scores": [],
        }
        # Save first 10 samples with detail for inspection
        for i in range(min(10, len(eval_datasets[ds_name]))):
            ex = eval_datasets[ds_name][i]
            result["per_dataset"][ds_name]["sample_scores"].append({
                "instruction":      ex["instruction"],
                "ground_truth":     ex["response"][:200],
                "base_prediction":  base_preds[ds_name][i][:200],
                "adapt_prediction": adapted_preds[ds_name][i][:200],
                "base_bertscore_f1":  base_bertscore[ds_name]["F1"][i],
                "adapt_bertscore_f1": adapted_bertscore[ds_name]["F1"][i],
                "base_token_f1":      compute_f1(base_preds[ds_name][i], ex["response"]),
                "adapt_token_f1":     compute_f1(adapted_preds[ds_name][i], ex["response"]),
            })

    # --- Save per-sample CSVs ---
    bertscore_per_sample_dir = os.path.join(OUTPUT_ROOT, "bertscore")
    os.makedirs(bertscore_per_sample_dir, exist_ok=True)
    for ds_name in list(eval_datasets.keys()):
        ds_slug = re.sub(r'[^a-z0-9]+', '_', ds_name.lower()).strip('_')
        for variant, bs_data, aux_data in [
            ("base",    base_bertscore,    base_aux_per_sample),
            ("adapted", adapted_bertscore, adapted_aux_per_sample),
        ]:
            # BERTScore per sample
            bs_csv = os.path.join(
                bertscore_per_sample_dir,
                f"bertscore_per_sample_{model_key}_{ds_slug}_{variant}.csv",
            )
            with open(bs_csv, "w", encoding="utf-8") as f:
                f.write("sample_idx,bs_precision,bs_recall,bs_f1\n")
                p_list = bs_data[ds_name]["P"]
                r_list = bs_data[ds_name]["R"]
                f1_list = bs_data[ds_name]["F1"]
                for i, (p, r, f1) in enumerate(zip(p_list, r_list, f1_list)):
                    f.write(f"{i},{p:.4f},{r:.4f},{f1:.4f}\n")

            # Token F1 + Faithfulness per sample
            aux_csv = os.path.join(
                bertscore_per_sample_dir,
                f"metrics_per_sample_{model_key}_{ds_slug}_{variant}.csv",
            )
            with open(aux_csv, "w", encoding="utf-8") as f:
                f.write("sample_idx,token_f1,faithfulness\n")
                tf1s  = aux_data[ds_name]["token_f1"]
                faiths = aux_data[ds_name]["faithfulness"]
                for i, (tf1, faith) in enumerate(zip(tf1s, faiths)):
                    f.write(f"{i},{tf1:.4f},{faith:.4f}\n")

    print(f"\n--> Per-sample CSVs saved to: {bertscore_per_sample_dir}")

    # --- Save JSON ---
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, f"bertscore_results_{model_key}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"\n--> Results saved: {json_path}")

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  BERTSCORE SUMMARY — {cfg['display']}")
    print(f"{'='*70}")
    for ds_name in datasets_list:
        d = result["per_dataset"][ds_name]
        print(f"\n  {ds_name} ({d['n_samples']} samples):")
        print(f"    Base     -> BERTScore F1: {d['base']['BERTScore_F1']:.2f}%  |  Token F1: {d['base']['Token_F1']:.2f}%")
        print(f"    Adapted  -> BERTScore F1: {d['adapted']['BERTScore_F1']:.2f}%  |  Token F1: {d['adapted']['Token_F1']:.2f}%")
        print(f"    Delta      BERTScore F1: {d['deltas']['BERTScore_F1']:+.2f}   |  Token F1: {d['deltas']['Token_F1']:+.2f}")

    return result


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    """Run the full BERTScore evaluation pipeline for all selected models."""
    eval_datasets = load_eval_datasets()

    all_results = {}
    for model_key in models_to_eval:
        result = evaluate_model(model_key, eval_datasets, batch_size=args.batch_size)
        all_results[model_key] = result

    # --- Plots (only if at least 1 model was evaluated) ---
    if all_results:
        # Per-model and per-dataset plot
        first_key = list(all_results.keys())[0]
        results_dir = get_results_dir(first_key)
        plots_dir = os.path.join(results_dir, "plots", "eval")
        os.makedirs(plots_dir, exist_ok=True)

        plot_bertscore_comparison(all_results, os.path.join(plots_dir, "bertscore_comparison.png"))

        if len(all_results) > 1:
            plot_bertscore_aggregate(all_results, os.path.join(plots_dir, "bertscore_aggregate.png"))

    # --- Global summary CSV ---
    csv_lines = ["model,dataset,variant,BERTScore_P,BERTScore_R,BERTScore_F1,Token_F1,Faithfulness"]
    for model_key, result in all_results.items():
        for ds_name in result["per_dataset"]:
            d = result["per_dataset"][ds_name]
            for variant in ["base", "adapted"]:
                v = d[variant]
                csv_lines.append(
                    f"{model_key},{ds_name},{variant},"
                    f"{v['BERTScore_P']},{v['BERTScore_R']},{v['BERTScore_F1']},"
                    f"{v['Token_F1']},{v['Faithfulness']}"
                )

    csv_path = os.path.join(OUTPUT_ROOT, "bertscore", "bertscore_summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"\n--> Global summary CSV: {csv_path}")

    print("\n" + "=" * 70)
    print("  BERTSCORE EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
