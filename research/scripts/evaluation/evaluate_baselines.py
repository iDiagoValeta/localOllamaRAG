"""
Baseline model evaluation benchmark for RAG.

Comparative benchmark of 7 base models (without fine-tuning) across 5 RAG
datasets (3 sources × language split), evaluated in two modes (with context /
without context) and two splits (dev / test). Each model is loaded and
unloaded sequentially to avoid GPU OOM errors.

Models evaluated:
  1. meta-llama/Llama-3.1-8B-Instruct      -- standard instruction-tuned
  2. mistralai/Mistral-7B-Instruct-v0.3    -- standard instruction-tuned
  3. Qwen/Qwen3-14B                        -- reasoner, thinking disabled
  4. Qwen/Qwen3.5-9B                       -- reasoner, thinking disabled
  5. Qwen/Qwen2.5-14B-Instruct             -- standard instruction-tuned
  6. google/gemma-3-12b-it                  -- standard instruction-tuned
  7. microsoft/phi-4                        -- standard instruction-tuned

Datasets:
  - neural-bridge/rag-dataset-12000       (EN, professional QA)
  - databricks/databricks-dolly-15k       (EN, RAG categories)
  - projecte-aina/RAG_Multilingual        (EN/ES/CA, multilingual)

Metrics (per dataset + weighted aggregate):
  - Token F1 (SQuAD-standard)
  - Context Faithfulness (%)
  - BERTScore P/R/F1 (DeBERTa-xlarge-mnli)
  - Avg Response Length (words)
  - Sentence Completeness (%)

Output:
  baseline-evaluation-output/baseline_evaluation.json
  baseline-evaluation-output/baseline_evaluation_samples.json

Usage:
    python evaluate_baselines.py
Dependencies:
    - torch
    - transformers (AutoModelForCausalLM, AutoTokenizer)
    - datasets (load_dataset)
    - tqdm
"""

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  CONFIGURATION
#  `-- 1. Environment and constants
#           CUDA env, output_dir, SEED=42 (torch + CUDA seeded here),
#           MODELS (order tweakable for parallel jobs),
#           Neural-Bridge dev carve, token caps, BERTScore model/batch,
#           SYSTEM_PROMPT / SYSTEM_PROMPT_NO_CONTEXT (aligned with train-qwen3),
#           DOLLY_RAG_CATEGORIES
#
#  EVALUATION AND METRICS
#  `-- 2. Metrics and inference
#           2.0 Monkey-patch bert_score sent_encode (DeBERTa max_length 512)
#           2.1 normalize_text, compute_f1, compute_context_faithfulness
#           2.2 generate_response — context truncation; Qwen3/3.5 enable_thinking=False;
#               EOS families: Qwen+Phi-4 ChatML; Gemma eos + end_of_turn; else native eos
#           2.3 evaluate_on_datasets — per-dataset metrics; faithfulness always
#               vs dataset context even when generation omits context; macro-style
#               per-sample means then ×100 for percentages; weighted aggregates
#               across datasets by n_samples
#           2.4 compute_bertscore_all, _extract_preds_gts
#
#  DATA
#  `-- 3. Dataset loading (once; shared by all models)
#           3.1 Normalizers _normalize_nb / _dolly / _aina
#           3.2 Filters _filter_valid, _filter_dolly_rag
#           3.3 Neural-Bridge: dev = last SAMPLES_NEURAL_BRIDGE_DEV of train; test = HF test
#           3.4 Dolly: 80/10/10 on RAG-filtered shuffle
#           3.5 Aina: validation/test per language Aina-EN/ES/CA
#           3.6 eval_datasets_dev, eval_datasets_test via _build_eval_dict
#
#  PIPELINE
#  `-- 4. Main loop (per model: per-dataset predictions checkpoint, load LLM
#           only for missing combos, both modes x dev, unload LLM, BERTScore,
#           rebuild metrics from cached samples, merge into all_results)
#           predictions_{slug}.json: updated after each (mode, dataset) pair —
#           per-dataset crash recovery; indent=2; stores instructions,
#           token_f1, rouge_l, faithfulness, words per sample.
#           baseline_checkpoint.json: saved per-model after BERTScore, atomic.
#  `-- 5. Summary table, baseline_evaluation.json, baseline_evaluation_samples.json
#
# ─────────────────────────────────────────────

import gc
import os
import re
import json
import torch
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from bert_score import score as bert_score_fn
import bert_score.utils as _bsu


# ─────────────────────────────────────────────
# SECTION 1: ENVIRONMENT AND CONSTANTS
# ─────────────────────────────────────────────

if not os.environ.get("HF_TOKEN") and not os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
    print("WARNING: HF_TOKEN is not set in the environment.")

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["TRITON_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

output_dir = os.path.join(os.getcwd(), "baseline-evaluation-output")
os.makedirs(output_dir, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen2.5-14B-Instruct",
    "google/gemma-3-12b-it",
    "microsoft/phi-4",
]

SAMPLES_NEURAL_BRIDGE_DEV = 1000
EVAL_MAX_NEW_TOKENS = 2048
MAX_CONTEXT_TOKENS  = 2048

# Dev caps — aligned with train-qwen3.py EVAL_CAP_DEV_* for cross-experiment comparability.
# Equal sizes across all datasets (balanced, per tutor's guidance).
# 5 × 200 = 1 000 samples per mode → ~18 h for 7 models × 2 modes on the cluster.
EVAL_CAP_NB_DEV    = 200   # Neural-Bridge
EVAL_CAP_DOLLY_DEV = 200   # Dolly
EVAL_CAP_AINA_DEV  = 200   # per-language Aina (EN / ES / CA)

BERTSCORE_MODEL      = "microsoft/deberta-xlarge-mnli"
BERTSCORE_BATCH_SIZE = 32

SYSTEM_PROMPT = (
    "You are a professional document analysis assistant. Your role is to answer "
    "questions accurately based on the provided document context.\n\n"
    "Guidelines:\n"
    "- Base your answers strictly on the information within the <context> tags.\n"
    "- Do not add information beyond what the context provides.\n"
    "- Preserve technical terms, notation, formulas, and numbers exactly as they appear.\n"
    "- Formulate clear, well-structured responses in complete sentences.\n"
    "- For factual questions, be direct and precise.\n"
    "- For analytical or complex questions, provide detailed explanations "
    "referencing specific information from the context.\n"
    "- Always respond in the same language as the context "
    "(English, Spanish/Castellano, or Catalan/Català)."
)

SYSTEM_PROMPT_NO_CONTEXT = (
    "You are a professional knowledge assistant. Your role is to answer "
    "questions accurately using your general knowledge.\n\n"
    "Guidelines:\n"
    "- Formulate clear, well-structured responses in complete sentences.\n"
    "- For factual questions, be direct and precise.\n"
    "- For analytical or complex questions, provide detailed explanations.\n"
    "- Respond in the same language as the question."
)

DOLLY_RAG_CATEGORIES = {"closed_qa", "information_extraction", "summarization"}


# ─────────────────────────────────────────────
# SECTION 2: METRICS AND INFERENCE
# ─────────────────────────────────────────────

def _safe_sent_encode(tokenizer, a):
    return tokenizer.encode(
        a.strip(), add_special_tokens=True, max_length=512, truncation=True,
    )


_bsu.sent_encode = _safe_sent_encode

def normalize_text(text: str) -> str:
    """Lowercase, strip articles (EN/ES/CA) and punctuation."""
    text = str(text).lower()
    text = re.sub(
        r'\b(a|an|the|el|la|los|las|un|una|unos|unas|les|els|uns|unes)\b',
        ' ', text
    )
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 (SQuAD-standard)."""
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


def _lcs_length(x: list, y: list) -> int:
    """Dynamic programming LCS length (token lists)."""
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
    """ROUGE-L F1 based on Longest Common Subsequence (token-level).

    Uses the same normalization as compute_f1 (EN/ES/CA articles + punctuation).

    Args:
        prediction:   The model's generated response text.
        ground_truth: The reference answer text.

    Returns:
        ROUGE-L F1 score between 0.0 and 1.0.
    """
    pred_tok  = normalize_text(prediction).split()
    truth_tok = normalize_text(ground_truth).split()
    if not pred_tok or not truth_tok:
        return 1.0 if pred_tok == truth_tok else 0.0
    lcs = _lcs_length(pred_tok, truth_tok)
    if lcs == 0:
        return 0.0
    prec = lcs / len(pred_tok)
    rec  = lcs / len(truth_tok)
    return 2 * prec * rec / (prec + rec)


def compute_context_faithfulness(prediction: str, context: str) -> float:
    """Compute Context Faithfulness: fraction of unique prediction word-types
    that also appear in the context.

    Args:
        prediction: The model's generated response text.
        context: The reference context document.

    Returns:
        A float between 0.0 and 1.0 representing the faithfulness ratio.
    """
    pred_types = set(normalize_text(prediction).split())
    ctx_types  = set(normalize_text(context).split())
    if not pred_types:
        return 0.0
    return len(pred_types & ctx_types) / len(pred_types)


def generate_response(
    model, tokenizer, instruction: str, context: str,
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS,
    model_name: str = "",
) -> str:
    """Run inference with the appropriate chat template for each model family.

    Handles model-specific differences:
      - Qwen3-14B / Qwen3.5-9B: uses enable_thinking=False to disable
        reasoning mode (reasoners unsuitable for direct RAG).
        EOS token: <|im_end|>.
      - Qwen2.5-14B-Instruct: standard apply_chat_template.
        EOS token: <|im_end|>.
      - Phi-4: standard apply_chat_template.
        EOS token: <|im_end|> (same convention as Qwen).
      - Gemma-3-12b-it: standard apply_chat_template.
        EOS token: <end_of_turn> (+ native eos as fallback).
      - Llama-3.1-8B-Instruct: standard apply_chat_template.
        EOS token: native tokenizer.eos_token_id.

    When with_context is False (called from evaluate_on_datasets with
    context=""), SYSTEM_PROMPT_NO_CONTEXT is used, which omits <context>
    tag references to avoid confusing the model.

    Args:
        model: The loaded causal language model.
        tokenizer: The corresponding tokenizer.
        instruction: The user question or instruction.
        context: The document context (empty string for no-context mode).
        max_new_tokens: Maximum number of tokens to generate.
        model_name: Full model name used for family detection.

    Returns:
        The generated response text, stripped of whitespace.
    """
    is_qwen3 = "Qwen3" in model_name

    ctx = (context or "").strip()
    if ctx:
        ctx_ids = tokenizer(ctx, add_special_tokens=False)["input_ids"]
        if len(ctx_ids) > MAX_CONTEXT_TOKENS:
            ctx = tokenizer.decode(ctx_ids[:MAX_CONTEXT_TOKENS], skip_special_tokens=True)
        user_msg = f"{instruction}\n\n<context>{ctx}</context>"
        system_prompt = SYSTEM_PROMPT
    else:
        user_msg = instruction
        system_prompt = SYSTEM_PROMPT_NO_CONTEXT

    chat_template_kwargs = {}
    if is_qwen3:
        chat_template_kwargs["enable_thinking"] = False

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user_msg},
        ],
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if "Qwen" in model_name or "phi-4" in model_name.lower():
        eos_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    elif "gemma" in model_name.lower():
        eot_ids = tokenizer.encode("<end_of_turn>", add_special_tokens=False)
        eos_id = ([tokenizer.eos_token_id] + eot_ids) if eot_ids else tokenizer.eos_token_id
    else:
        eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def evaluate_on_datasets(
    model, tokenizer, eval_datasets: dict,
    label: str = "MODEL",
    with_context: bool = True,
    model_name: str = "",
) -> tuple:
    """Evaluate a model on the provided eval_datasets dict.

    Args:
        model: The loaded causal language model.
        tokenizer: The corresponding tokenizer.
        eval_datasets: Dict mapping dataset names to HuggingFace datasets.
        label: Display label for progress bars.
        with_context: If True, passes context to the model. If False,
                      passes context="" (knowledge-only mode).
        model_name: Full model name for family-specific handling.

    Returns:
        A tuple of (all_metrics, all_results) where all_metrics is
        dict[ds_name -> metric dict] and all_results is
        dict[ds_name -> list of per-sample dicts].
    """
    all_metrics = {}
    all_results = {}
    model.eval()
    torch.cuda.empty_cache()

    mode_str = "CTX" if with_context else "NO-CTX"

    for ds_name, ds in eval_datasets.items():
        total_f1         = 0.0
        total_rouge_l    = 0.0
        total_faith      = 0.0
        total_words      = 0
        n_complete       = 0
        results          = []
        n = len(ds)

        for example in tqdm(ds, desc=f"{label} | {mode_str} | {ds_name}"):
            instruction  = example["instruction"]
            context      = example["context"] if with_context else ""
            ground_truth = example["response"]

            pred = generate_response(
                model, tokenizer, instruction, context,
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
                model_name=model_name,
            )

            f1      = compute_f1(pred, ground_truth)
            rouge_l = compute_rouge_l(pred, ground_truth)
            original_ctx = example["context"]
            faith = compute_context_faithfulness(pred, original_ctx)
            words = len(pred.split())
            is_complete = bool(pred.rstrip()) and pred.rstrip()[-1] in ".!?"

            total_f1      += f1
            total_rouge_l += rouge_l
            total_faith   += faith
            total_words   += words
            if is_complete:
                n_complete += 1

            results.append({
                "instruction":  instruction,
                "context":      original_ctx[:200] + "..." if len(original_ctx) > 200 else original_ctx,
                "ground_truth": ground_truth,
                "prediction":   pred,
                "f1":           round(f1,      4),
                "rouge_l":      round(rouge_l, 4),
                "faithfulness": round(faith,   4),
                "words":        words,
            })

        metrics = {
            "n_samples":                   n,
            "Token_F1":                    round((total_f1      / n) * 100, 2) if n else 0.0,
            "ROUGE_L_F1":                  round((total_rouge_l / n) * 100, 2) if n else 0.0,
            "Context_Faithfulness_Pct":    round((total_faith   / n) * 100, 2) if n else 0.0,
            "Avg_Response_Length_Words":   round( total_words   / n,         1) if n else 0.0,
            "Sentence_Completeness_Pct":   round((n_complete    / n) * 100, 1) if n else 0.0,
        }
        if not with_context:
            metrics["Context_Faithfulness_Note"] = "Computed vs unseen context (baseline overlap)"
        all_metrics[ds_name] = metrics
        all_results[ds_name] = results

        faith_note = "" if with_context else "  [N/A -- no context]"
        print(f"\n  {ds_name} ({n} samples):")
        print(f"    Token F1:               {metrics['Token_F1']:.2f}%")
        print(f"    ROUGE-L F1:             {metrics['ROUGE_L_F1']:.2f}%")
        print(f"    Context Faithfulness:   {metrics['Context_Faithfulness_Pct']:.2f}%{faith_note}")
        print(f"    Avg Response Length:    {metrics['Avg_Response_Length_Words']:.1f} words")
        print(f"    Sentence Completeness:  {metrics['Sentence_Completeness_Pct']:.1f}%")

    return all_metrics, all_results


def compute_bertscore_all(predictions: dict, ground_truths: dict) -> dict:
    """Compute BERTScore (P, R, F1) for every dataset in the predictions dict.

    Args:
        predictions:   {ds_name: [pred_str, ...]}.
        ground_truths: {ds_name: [gt_str, ...]}.

    Returns:
        {ds_name: {P, R, F1 (per-sample lists), avg_P, avg_R, avg_F1 (%)}}.
    """
    results = {}
    for ds_name in predictions:
        preds = predictions[ds_name]
        refs  = ground_truths[ds_name]
        print(f"    BERTScore | {ds_name} ({len(preds)} samples)...")
        P, R, F1 = bert_score_fn(
            preds, refs,
            model_type=BERTSCORE_MODEL,
            lang="en",
            rescale_with_baseline=True,
            batch_size=BERTSCORE_BATCH_SIZE,
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
        print(f"      P={results[ds_name]['avg_P']:.2f}%  "
              f"R={results[ds_name]['avg_R']:.2f}%  "
              f"F1={results[ds_name]['avg_F1']:.2f}%")
    return results


def _extract_preds_gts(eval_results: dict) -> tuple:
    """Extract parallel pred/gt lists from evaluate_on_datasets output."""
    preds = {ds: [r["prediction"]   for r in rs] for ds, rs in eval_results.items()}
    gts   = {ds: [r["ground_truth"] for r in rs] for ds, rs in eval_results.items()}
    return preds, gts


# ─────────────────────────────────────────────
# SECTION 3: DATASET LOADING (once, before loading models)
# ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("SECTION 3: Loading and freezing datasets")
print("=" * 70)

def _normalize_nb(example):
    """Neural-Bridge: question->instruction, context, answer->response."""
    return {
        "instruction": (example.get("question") or "").strip(),
        "context":     (example.get("context")  or "").strip(),
        "response":    (example.get("answer")   or "").strip(),
    }


def _normalize_dolly(example):
    """Dolly: instruction, context, response + category for filtering."""
    return {
        "instruction": (example.get("instruction") or "").strip(),
        "context":     (example.get("context")     or "").strip(),
        "response":    (example.get("response")    or "").strip(),
        "category":    (example.get("category")    or "").strip(),
    }


def _normalize_aina(example):
    """Aina RAG Multilingual: instruction/context/response + lang for splitting."""
    return {
        "instruction": (example.get("instruction") or "").strip(),
        "context":     (example.get("context")     or "").strip(),
        "response":    (example.get("response")    or "").strip(),
        "lang":        (example.get("lang")        or "").strip(),
    }


def _filter_valid(ex):
    return (
        bool(ex["instruction"].strip())
        and bool(ex["context"].strip())
        and bool(ex["response"].strip())
    )


def _filter_dolly_rag(ex):
    """Keep only Dolly rows that are RAG-relevant (non-empty context + correct category)."""
    return (
        ex["category"] in DOLLY_RAG_CATEGORIES
        and bool(ex["context"].strip())
    )

print("\n  Neural-Bridge RAG...")

_nb_train_full = (
    load_dataset("neural-bridge/rag-dataset-12000", split="train")
    .map(_normalize_nb, remove_columns=["context", "question", "answer"])
    .filter(_filter_valid)
    .shuffle(seed=42)
)
nb_n = len(_nb_train_full)
nb_dev_start = nb_n - SAMPLES_NEURAL_BRIDGE_DEV
nb_dev = _nb_train_full.select(range(nb_dev_start, min(nb_dev_start + SAMPLES_NEURAL_BRIDGE_DEV, nb_n)))

print(f"    dev={len(nb_dev)}  (test frozen for train-qwen3.py)")

print("  Dolly QA (RAG-relevant categories only, manual 80/10/10)...")
_dolly_all = (
    load_dataset("databricks/databricks-dolly-15k", split="train")
    .map(_normalize_dolly, remove_columns=["instruction", "context", "response", "category"])
    .filter(_filter_dolly_rag)
    .filter(_filter_valid)
    .shuffle(seed=42)
    .remove_columns(["category"])
)
nd = len(_dolly_all)
nd_train = int(nd * 0.80)
nd_val   = int(nd * 0.10)
dolly_dev  = _dolly_all.select(range(nd_train, nd_train + nd_val))
print(f"    RAG rows total: {nd}")
print(f"    dev={len(dolly_dev)}  (test frozen for train-qwen3.py)")

print("  Aina RAG Multilingual...")

_AINA_EXTRA_COLS = ["id", "category", "extractive"]
_AINA_LANGS = [("en", "Aina-EN"), ("es", "Aina-ES"), ("ca", "Aina-CA")]


def _load_aina_by_lang(split):
    """Load Aina RAG split and return ``{lang_label: Dataset}`` per language."""
    ds = load_dataset("projecte-aina/RAG_Multilingual", split=split)
    cols_to_remove = [c for c in _AINA_EXTRA_COLS if c in ds.column_names]
    ds = (
        ds
        .map(_normalize_aina, remove_columns=cols_to_remove)
        .filter(_filter_valid)
    )
    result = {}
    for lang_code, lang_label in _AINA_LANGS:
        sub = ds.filter(lambda ex, lc=lang_code: ex["lang"] == lc)
        result[lang_label] = sub.remove_columns(["lang"]).shuffle(seed=42)
    return result


_aina_dev_by_lang = _load_aina_by_lang("validation")
for lang_label in ["Aina-EN", "Aina-ES", "Aina-CA"]:
    print(f"    {lang_label}: dev={len(_aina_dev_by_lang[lang_label])}  (test frozen for train-qwen3.py)")

def _build_eval_dict(nb_ds, dolly_ds, aina_by_lang, split_label: str) -> dict:
    """Build a frozen eval dict using full partitions (no sample cap)."""
    out = {}
    for name, ds in [("Neural-Bridge RAG", nb_ds), ("Dolly QA", dolly_ds)]:
        out[name] = ds
        print(f"    {split_label} | {name}: {len(ds)} samples (FROZEN)")
    for lang_label in ["Aina-EN", "Aina-ES", "Aina-CA"]:
        out[lang_label] = aina_by_lang[lang_label]
        print(f"    {split_label} | {lang_label}: {len(aina_by_lang[lang_label])} samples (FROZEN)")
    return out


print("\n  Building frozen evaluation splits:")
eval_datasets_dev = _build_eval_dict(nb_dev, dolly_dev, _aina_dev_by_lang, "dev")

_CAPS_DEV = {
    "Neural-Bridge RAG": EVAL_CAP_NB_DEV,
    "Dolly QA":          EVAL_CAP_DOLLY_DEV,
    "Aina-EN":           EVAL_CAP_AINA_DEV,
    "Aina-ES":           EVAL_CAP_AINA_DEV,
    "Aina-CA":           EVAL_CAP_AINA_DEV,
}
for ds_name in list(eval_datasets_dev.keys()):
    cap = _CAPS_DEV.get(ds_name, len(eval_datasets_dev[ds_name]))
    eval_datasets_dev[ds_name] = eval_datasets_dev[ds_name].select(
        range(min(cap, len(eval_datasets_dev[ds_name])))
    )
    print(f"    dev | {ds_name}: capped to {len(eval_datasets_dev[ds_name])} samples")

total_dev = sum(len(d) for d in eval_datasets_dev.values())
print(f"\n--> Total dev samples: {total_dev}")
print("--> Test split: frozen (evaluated in train-qwen3.py only)")


# ─────────────────────────────────────────────
# SECTION 4: MAIN LOOP -- 7 Models x 2 Modes x dev only
# ─────────────────────────────────────────────

all_results = {}
CHECKPOINT_PATH = os.path.join(output_dir, "baseline_checkpoint.json")

if os.path.exists(CHECKPOINT_PATH):
    print(f"\n--> Loading existing checkpoint: {CHECKPOINT_PATH}")
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        all_results = {}
else:
    all_results = {}


def _atomic_json_save(path: str, data: dict, indent: int = 2):
    """Write data to path atomically via a .tmp swap file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    os.replace(tmp, path)


def _samples_to_pred_entry(samples: list) -> dict:
    """Convert a per-sample result list to a predictions checkpoint entry.

    Stores all raw values needed to recompute any aggregate metric without
    re-running inference.

    Args:
        samples: List of per-sample dicts with prediction, ground_truth,
                 instruction, f1, rouge_l, faithfulness, words keys.

    Returns:
        Dict with parallel lists for each field.
    """
    return {
        "predictions":   [s["prediction"]          for s in samples],
        "ground_truths": [s["ground_truth"]         for s in samples],
        "instructions":  [s.get("instruction", "")  for s in samples],
        "token_f1":      [s["f1"]                   for s in samples],
        "rouge_l":       [s["rouge_l"]              for s in samples],
        "faithfulness":  [s["faithfulness"]         for s in samples],
        "words":         [s["words"]                for s in samples],
    }


def _pred_entry_to_samples(entry: dict) -> list:
    """Reconstruct per-sample result list from a predictions checkpoint entry.

    Args:
        entry: Checkpoint entry dict produced by _samples_to_pred_entry.

    Returns:
        List of per-sample dicts with all fields needed for metric recomputation.
    """
    instrs = entry.get("instructions", [""] * len(entry["predictions"]))
    return [
        {
            "prediction":   pred,
            "ground_truth": gt,
            "instruction":  instr,
            "context":      "",
            "f1":           f1,
            "rouge_l":      rl,
            "faithfulness": fa,
            "words":        w,
        }
        for pred, gt, instr, f1, rl, fa, w in zip(
            entry["predictions"], entry["ground_truths"], instrs,
            entry["token_f1"], entry["rouge_l"],
            entry["faithfulness"], entry["words"],
        )
    ]


def _recompute_metrics(samples: list, with_context: bool) -> dict:
    """Recompute aggregate metrics from a list of per-sample dicts.

    Used when loading cached results so metrics are consistent whether
    produced by fresh inference or checkpoint reload.

    Args:
        samples: Per-sample list with f1, rouge_l, faithfulness, words,
                 and prediction fields.
        with_context: Whether the context was provided to the model.

    Returns:
        Metric dict matching the format produced by evaluate_on_datasets.
    """
    n = len(samples)
    if not n:
        return {"n_samples": 0}
    total_f1      = sum(s["f1"]          for s in samples)
    total_rouge_l = sum(s["rouge_l"]     for s in samples)
    total_faith   = sum(s["faithfulness"] for s in samples)
    total_words   = sum(s["words"]       for s in samples)
    n_complete    = sum(
        1 for s in samples
        if s["prediction"].rstrip() and s["prediction"].rstrip()[-1] in ".!?"
    )
    m = {
        "n_samples":                   n,
        "Token_F1":                    round(total_f1      / n * 100, 2),
        "ROUGE_L_F1":                  round(total_rouge_l / n * 100, 2),
        "Context_Faithfulness_Pct":    round(total_faith   / n * 100, 2),
        "Avg_Response_Length_Words":   round(total_words   / n,       1),
        "Sentence_Completeness_Pct":   round(n_complete    / n * 100, 1),
    }
    if not with_context:
        m["Context_Faithfulness_Note"] = "Computed vs unseen context (baseline overlap)"
    return m


for model_idx, model_name in enumerate(MODELS, 1):
    print("\n" + "=" * 70)
    print(f"[{model_idx}/{len(MODELS)}] Evaluating model: {model_name}")
    print("=" * 70)

    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as _f:
                _disk_ckpt = json.load(_f)
            all_results.update({k: v for k, v in _disk_ckpt.items() if k not in all_results})
        except Exception:
            pass
    if model_name in all_results:
        print(f"--> Model {model_name} already evaluated (skipping).")
        continue

    model_slug = model_name.split("/")[-1]
    _pred_ckpt_path = os.path.join(output_dir, f"predictions_{model_slug}.json")

    _pred_ckpt: dict = {}
    if os.path.exists(_pred_ckpt_path):
        try:
            with open(_pred_ckpt_path, "r", encoding="utf-8") as f:
                _pred_ckpt = json.load(f)
            _n_cached = sum(
                len(_pred_ckpt.get(m, {}).get("dev", {}))
                for m in ["with_context", "without_context"]
            )
            print(f"--> Loaded predictions checkpoint: {_pred_ckpt_path} "
                  f"({_n_cached} dataset-mode combos cached)")
        except Exception as e:
            print(f"    Warning: could not load predictions checkpoint: {e}")
            _pred_ckpt = {}

    _ds_names = list(eval_datasets_dev.keys())
    _combos_needed = [
        (mode, ds)
        for mode in ["with_context", "without_context"]
        for ds in _ds_names
        if _pred_ckpt.get(mode, {}).get("dev", {}).get(ds) is None
    ]

    model = None
    tokenizer = None

    if _combos_needed:
        print(f"\n--> Loading {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
        print(f"--> Model loaded: {model_name}")
    else:
        print(f"--> All inference cached. Proceeding to BERTScore for {model_slug}.")

    _raw_results: dict = {}

    for with_context in [True, False]:
        mode_key   = "with_context" if with_context else "without_context"
        mode_label = "WITH CONTEXT" if with_context else "WITHOUT CONTEXT"

        print(f"\n{'─' * 60}")
        print(f"  Mode: {mode_label}")
        print(f"{'─' * 60}")

        _raw_results[mode_key] = {"dev": {}}

        for ds_name, ds in eval_datasets_dev.items():
            cached = _pred_ckpt.get(mode_key, {}).get("dev", {}).get(ds_name)
            if cached is not None:
                print(f"  -- {ds_name}: loaded from checkpoint "
                      f"({len(cached['predictions'])} samples) --")
                _raw_results[mode_key]["dev"][ds_name] = _pred_entry_to_samples(cached)
                continue

            print(f"\n  -- {mode_label} | dev | {ds_name} --")
            _, results_single = evaluate_on_datasets(
                model, tokenizer, {ds_name: ds},
                label=model_slug,
                with_context=with_context,
                model_name=model_name,
            )
            _raw_results[mode_key]["dev"][ds_name] = results_single[ds_name]

            if mode_key not in _pred_ckpt:
                _pred_ckpt[mode_key] = {}
            if "dev" not in _pred_ckpt[mode_key]:
                _pred_ckpt[mode_key]["dev"] = {}
            _pred_ckpt[mode_key]["dev"][ds_name] = _samples_to_pred_entry(
                results_single[ds_name]
            )
            try:
                _atomic_json_save(_pred_ckpt_path, _pred_ckpt, indent=2)
                print(f"    --> Predictions checkpoint updated: {_pred_ckpt_path}")
            except Exception as e:
                print(f"    Warning: could not update predictions checkpoint: {e}")

    if model is not None:
        print(f"\n--> Unloading model {model_name} for BERTScore computation...")
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    model_results = {"with_context": {}, "without_context": {}}

    for with_context in [True, False]:
        mode_key   = "with_context" if with_context else "without_context"
        mode_label = "WITH CONTEXT" if with_context else "WITHOUT CONTEXT"

        raw_ds = _raw_results[mode_key]["dev"]
        per_ds_metrics = {
            ds: _recompute_metrics(samples, with_context)
            for ds, samples in raw_ds.items()
        }

        agg_f1 = agg_rouge_l = agg_faith = agg_words = agg_complete = 0.0
        agg_n = 0
        for ds_name, m in per_ds_metrics.items():
            n = m["n_samples"]
            agg_n        += n
            agg_f1       += m["Token_F1"]                  * n
            agg_rouge_l  += m["ROUGE_L_F1"]                * n
            agg_faith    += m["Context_Faithfulness_Pct"]  * n
            agg_words    += m["Avg_Response_Length_Words"]  * n
            agg_complete += m["Sentence_Completeness_Pct"] * n

        aggregate = {
            "n_samples":                   agg_n,
            "Token_F1":                    round(agg_f1      / agg_n, 2) if agg_n else 0.0,
            "ROUGE_L_F1":                  round(agg_rouge_l / agg_n, 2) if agg_n else 0.0,
            "Context_Faithfulness_Pct":    round(agg_faith   / agg_n, 2) if agg_n else 0.0,
            "Avg_Response_Length_Words":   round(agg_words   / agg_n, 1) if agg_n else 0.0,
            "Sentence_Completeness_Pct":   round(agg_complete / agg_n, 1) if agg_n else 0.0,
        }
        if not with_context:
            aggregate["Context_Faithfulness_Note"] = "N/A (no context provided to model)"

        split_data = dict(per_ds_metrics)
        split_data["aggregate"] = aggregate
        model_results[mode_key]["dev"] = split_data

        print(f"\n  Aggregate (dev, {mode_label}):")
        print(f"    Token F1:             {aggregate['Token_F1']:.2f}%")
        print(f"    ROUGE-L F1:           {aggregate['ROUGE_L_F1']:.2f}%")
        faith_note = "  [N/A -- no context]" if not with_context else ""
        print(f"    Context Faithfulness: {aggregate['Context_Faithfulness_Pct']:.2f}%{faith_note}")
        print(f"    Avg Response Length:  {aggregate['Avg_Response_Length_Words']:.1f} words")

    print(f"\n--> Computing BERTScore for {model_slug}...")

    for mode_key in ["with_context", "without_context"]:
        raw_ds = _raw_results[mode_key]["dev"]
        preds, gts = _extract_preds_gts(raw_ds)

        print(f"\n  [{mode_key} | dev]")
        bs = compute_bertscore_all(preds, gts)

        for ds_name in bs:
            model_results[mode_key]["dev"][ds_name]["BERTScore_P"]  = bs[ds_name]["avg_P"]
            model_results[mode_key]["dev"][ds_name]["BERTScore_R"]  = bs[ds_name]["avg_R"]
            model_results[mode_key]["dev"][ds_name]["BERTScore_F1"] = bs[ds_name]["avg_F1"]

        agg = model_results[mode_key]["dev"]["aggregate"]
        agg_bs_f1 = 0.0
        for ds_name in bs:
            n = model_results[mode_key]["dev"][ds_name]["n_samples"]
            agg_bs_f1 += bs[ds_name]["avg_F1"] * n
        agg["BERTScore_F1"] = round(agg_bs_f1 / agg["n_samples"], 2) if agg["n_samples"] else 0.0
        print(f"      Aggregate BERTScore F1: {agg['BERTScore_F1']:.2f}%")

    gc.collect()
    torch.cuda.empty_cache()
    print(f"--> BERTScore complete for {model_slug}. GPU cache cleared.")

    all_results[model_name] = model_results

    try:
        _disk = {}
        if os.path.exists(CHECKPOINT_PATH):
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                _disk = json.load(f)
        _disk.update(all_results)
        _atomic_json_save(CHECKPOINT_PATH, _disk, indent=4)
        print(f"--> Checkpoint saved: {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"--> Warning: could not save checkpoint: {e}")


# ─────────────────────────────────────────────
# SECTION 5: SUMMARY TABLE AND SAVE
# ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("BASELINE EVALUATION SUMMARY TABLE")
print("=" * 70)

header = (f"{'Model':<35s} | {'Split':<5s} | {'Mode':<12s} | "
          f"{'Token_F1':>9s} | {'ROUGE-L':>9s} | {'BS_F1':>9s} | {'Ctx_Faith':>10s} | {'Avg_Len':>8s}")
print(header)
print("─" * len(header))

for model_name in MODELS:
    short_name = model_name.split("/")[-1]
    if model_name not in all_results:
        continue
    for mode_key, mode_label in [("with_context", "with context"), ("without_context", "no context")]:
        for split_name in ["dev"]:
            agg = all_results[model_name][mode_key].get(split_name, {}).get("aggregate", {})
            f1      = agg.get("Token_F1",                  0.0)
            rl      = agg.get("ROUGE_L_F1",                0.0)
            bs_f1   = agg.get("BERTScore_F1",              0.0)
            faith   = agg.get("Context_Faithfulness_Pct",  0.0)
            avg_len = agg.get("Avg_Response_Length_Words",  0.0)

            faith_str = f"{faith:>9.2f}%" if mode_key == "with_context" else "   N/A    "

            print(f"{short_name:<35s} | {split_name:<5s} | {mode_label:<12s} | "
                  f"{f1:>8.2f}% | {rl:>8.2f}% | {bs_f1:>8.2f}% | {faith_str} | {avg_len:>7.1f}w")

print("─" * len(header))

output_path = os.path.join(output_dir, "baseline_evaluation.json")
try:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"\n--> Results saved: {output_path}")
except Exception as e:
    print(f"\n    Warning: could not save file: {e}")

samples_path = os.path.join(output_dir, "baseline_evaluation_samples.json")
try:
    compact = {}
    for model_name in MODELS:
        compact[model_name] = {}
        for mode_key in ["with_context", "without_context"]:
            compact[model_name][mode_key] = {}
            compact[model_name][mode_key]["dev"] = {
                    "aggregate": all_results[model_name][mode_key]["dev"].get("aggregate", {})
                }
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=4, ensure_ascii=False)
    print(f"--> Compact summary saved: {samples_path}")
except Exception as e:
    print(f"    Warning: could not save compact summary: {e}")

print("\n" + "=" * 70)
print("--> BASELINE EVALUATION COMPLETED")
print(f"    Full results:     {output_path}")
print(f"    Compact summary:  {samples_path}")
print("=" * 70)
