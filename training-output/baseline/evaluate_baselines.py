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
#  `-- 1. Environment and constants    CUDA env, models, caps, system prompts
#
#  EVALUATION AND METRICS
#  `-- 2. Metrics and inference
#           2.1 normalize_text()          EN/ES/CA normalization
#           2.2 compute_f1()              Token F1 (SQuAD-standard)
#           2.3 compute_context_faithfulness()  primary thesis metric
#           2.4 generate_response()       inference with chat template
#           2.5 evaluate_on_datasets()    loop over eval_datasets
#           2.6 compute_bertscore_all()   DeBERTa-xlarge-mnli
#           2.7 _extract_preds_gts()     helper for BERTScore input
#
#  DATA
#  `-- 3. Dataset loading (ONCE, before loading models)
#           3.1 Normalizers: _normalize_nb, _normalize_dolly, _normalize_aina
#           3.2 Filters: _filter_valid, _filter_long_response, _filter_dolly_rag
#           3.3 Neural-Bridge RAG      dev (train tail) + test (natural split)
#           3.4 Dolly QA               dev/test (manual 80/10/10)
#           3.5 Aina RAG Multilingual  dev (validation) + test (natural split)
#           3.6 Frozen dictionaries: eval_datasets_dev, eval_datasets_test
#
#  PIPELINE
#  |-- 4. Main loop   7 models x 2 modes x 2 splits + BERTScore
#  `-- 5. Summary table + save baseline_evaluation.json
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


# ---------------------------------------------------------------------------
# MONKEY-PATCH: bert_score.utils.sent_encode
# ---------------------------------------------------------------------------
# DeBERTa reports sys.maxsize as model_max_length, causing OverflowError in
# the Rust tokenizer backend.  Force max_length=512 and truncation=True.
# ---------------------------------------------------------------------------
def _safe_sent_encode(tokenizer, a):
    return tokenizer.encode(
        a.strip(), add_special_tokens=True, max_length=512, truncation=True,
    )

_bsu.sent_encode = _safe_sent_encode


# ─────────────────────────────────────────────
# SECTION 1: ENVIRONMENT AND CONSTANTS
# ─────────────────────────────────────────────
# CUDA environment variables, model list, evaluation caps, max tokens,
# and system prompts (with/without context).

# ─────────────────────────────────────────────
# Hugging Face Token Verification
# ─────────────────────────────────────────────
import os
if not os.environ.get("HF_TOKEN") and not os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
    print("WARNING: HF_TOKEN is not set in the environment.")

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["TRITON_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

output_dir = os.path.join(os.getcwd(), "baseline-evaluation-output")
os.makedirs(output_dir, exist_ok=True)

# -- Models to evaluate (sequentially, one at a time to avoid OOM) ----------
# Qwen3 and Qwen3.5 are reasoners: thinking is disabled for RAG.
# Qwen2.5-Instruct, Llama-3.1-Instruct, Gemma-3-12b-it and Phi-4 use
# standard chat template (none have active reasoning mode).
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen2.5-14B-Instruct",
    "google/gemma-3-12b-it",
    "microsoft/phi-4",
]

# NB has no natural validation split; carve from train (same as train.py)
SAMPLES_NEURAL_BRIDGE_DEV = 1000

# -- Tokens -----------------------------------------------------------------
EVAL_MAX_NEW_TOKENS = 2048
MAX_CONTEXT_TOKENS  = 2048

# -- BERTScore --------------------------------------------------------------
BERTSCORE_MODEL      = "microsoft/deberta-xlarge-mnli"
BERTSCORE_BATCH_SIZE = 32

# -- System prompt (identical to train.py v7.2) -----------------------------
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

# System prompt for no-context mode (no reference to <context> tags)
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
# Four RAG metrics with no external dependencies.
#
# Primary (main thesis evidence):
#   Context Faithfulness -- % of response token types that also appear
#     in the context. The difference between with_context and without_context
#     demonstrates how much the model relies on the provided document.
#
# Secondary:
#   Token F1              -- overlap with gold answer, SQuAD standard.
#   Avg Response Length   -- detects verbosity changes across models.
#   Sentence Completeness -- detects fragmented responses.

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
    is_qwen3 = "Qwen3" in model_name  # covers Qwen3-14B and Qwen3.5-9B

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

    # Determine EOS token(s):
    #   Qwen + Phi-4  -> <|im_end|>  (ChatML convention)
    #   Gemma-3       -> [<eos>, <end_of_turn>]  (both needed for clean stop)
    #   Llama, Mistral + rest  -> tokenizer.eos_token_id native
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

            f1    = compute_f1(pred, ground_truth)
            # Faithfulness: compute against original context even if not passed
            original_ctx = example["context"]
            faith = compute_context_faithfulness(pred, original_ctx)
            words = len(pred.split())
            is_complete = bool(pred.rstrip()) and pred.rstrip()[-1] in ".!?"

            total_f1    += f1
            total_faith += faith
            total_words += words
            if is_complete:
                n_complete += 1

            results.append({
                "instruction":  instruction,
                "context":      original_ctx[:200] + "..." if len(original_ctx) > 200 else original_ctx,
                "ground_truth": ground_truth,
                "prediction":   pred,
                "f1":           round(f1,    4),
                "faithfulness": round(faith, 4),
                "words":        words,
            })

        metrics = {
            "n_samples":                   n,
            "Token_F1":                    round((total_f1    / n) * 100, 2) if n else 0.0,
            "Context_Faithfulness_Pct":    round((total_faith / n) * 100, 2) if n else 0.0,
            "Avg_Response_Length_Words":   round( total_words / n,         1) if n else 0.0,
            "Sentence_Completeness_Pct":   round((n_complete  / n) * 100, 1) if n else 0.0,
        }
        if not with_context:
            metrics["Context_Faithfulness_Note"] = "Computed vs unseen context (baseline overlap)"
        all_metrics[ds_name] = metrics
        all_results[ds_name] = results

        faith_note = "" if with_context else "  [N/A -- no context]"
        print(f"\n  {ds_name} ({n} samples):")
        print(f"    Token F1:               {metrics['Token_F1']:.2f}%")
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
# Splits are loaded and frozen here. They are reused for ALL models.

print("\n" + "=" * 70)
print("SECTION 3: Loading and freezing datasets")
print("=" * 70)


# -- Normalizers ------------------------------------------------------------

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


# -- Filters ----------------------------------------------------------------

def _filter_valid(ex):
    return (
        bool(ex["instruction"].strip())
        and bool(ex["context"].strip())
        and bool(ex["response"].strip())
    )


def _filter_long_response(ex, min_words: int = 15):
    """Keep only responses with at least min_words words."""
    return len(ex["response"].split()) >= min_words


def _filter_dolly_rag(ex):
    """Keep only Dolly rows that are RAG-relevant (non-empty context + correct category)."""
    return (
        ex["category"] in DOLLY_RAG_CATEGORIES
        and bool(ex["context"].strip())
    )


# -- Neural-Bridge RAG -----------------------------------------------------

print("\n  Neural-Bridge RAG...")

# Dev split: carved from the main train split (same logic as train.py)
_nb_train_full = (
    load_dataset("neural-bridge/rag-dataset-12000", split="train")
    .map(_normalize_nb, remove_columns=["context", "question", "answer"])
    .filter(_filter_valid)
    .filter(_filter_long_response)
    .shuffle(seed=42)
)
nb_n = len(_nb_train_full)
# Take last N from train as dev (mirroring train.py val split)
nb_dev_start = nb_n - SAMPLES_NEURAL_BRIDGE_DEV
nb_dev = _nb_train_full.select(range(nb_dev_start, min(nb_dev_start + SAMPLES_NEURAL_BRIDGE_DEV, nb_n)))

# Test split: natural test split from HuggingFace
nb_test = (
    load_dataset("neural-bridge/rag-dataset-12000", split="test")
    .map(_normalize_nb, remove_columns=["context", "question", "answer"])
    .filter(_filter_valid)
    .filter(_filter_long_response)
    .shuffle(seed=42)
)
print(f"    dev={len(nb_dev)}, test={len(nb_test)}")


# -- Dolly QA ---------------------------------------------------------------

print("  Dolly QA (RAG-relevant categories only, manual 80/10/10)...")
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
dolly_dev  = _dolly_all.select(range(nd_train, nd_train + nd_val))
dolly_test = _dolly_all.select(range(nd_train + nd_val, nd))
print(f"    RAG rows total: {nd}")
print(f"    dev={len(dolly_dev)}, test={len(dolly_test)}")


# -- Aina RAG Multilingual -------------------------------------------------

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
        .filter(_filter_long_response)
    )
    result = {}
    for lang_code, lang_label in _AINA_LANGS:
        sub = ds.filter(lambda ex, lc=lang_code: ex["lang"] == lc)
        result[lang_label] = sub.remove_columns(["lang"]).shuffle(seed=42)
    return result


_aina_dev_by_lang  = _load_aina_by_lang("validation")
_aina_test_by_lang = _load_aina_by_lang("test")
for lang_label in ["Aina-EN", "Aina-ES", "Aina-CA"]:
    print(f"    {lang_label}: dev={len(_aina_dev_by_lang[lang_label])}, "
          f"test={len(_aina_test_by_lang[lang_label])}")


# -- Build frozen evaluation dictionaries ----------------------------------

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
eval_datasets_dev  = _build_eval_dict(nb_dev, dolly_dev, _aina_dev_by_lang, "dev")
eval_datasets_test = _build_eval_dict(nb_test, dolly_test, _aina_test_by_lang, "test")

total_dev  = sum(len(d) for d in eval_datasets_dev.values())
total_test = sum(len(d) for d in eval_datasets_test.values())
print(f"\n--> Total dev samples:  {total_dev}")
print(f"--> Total test samples: {total_test}")


# ─────────────────────────────────────────────
# SECTION 4: MAIN LOOP -- 7 Models x 2 Modes x 2 Splits
# ─────────────────────────────────────────────
# Order: defined by MODELS list (modifiable for parallel job distribution)
# Each model is loaded, evaluated across 4 combinations (mode x split),
# and unloaded with gc.collect() + torch.cuda.empty_cache() before the next.

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

for model_idx, model_name in enumerate(MODELS, 1):
    print("\n" + "=" * 70)
    print(f"[{model_idx}/{len(MODELS)}] Evaluating model: {model_name}")
    print("=" * 70)

    # Re-read checkpoint from disk: another concurrent job may have completed
    # this model since this job started.
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

    # -- Load model ---------------------------------------------------------
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

    model_results = {"with_context": {}, "without_context": {}}
    model_slug = model_name.split("/")[-1]

    # Collect raw eval_results (per-sample) for BERTScore later.
    # Structure: {mode_key: {split_name: all_results_dict}}.
    _raw_results = {}

    # -- Evaluate in both modes ---------------------------------------------
    for with_context in [True, False]:
        mode_key = "with_context" if with_context else "without_context"
        mode_label = "WITH CONTEXT" if with_context else "WITHOUT CONTEXT"

        print(f"\n{'─' * 60}")
        print(f"  Mode: {mode_label}")
        print(f"{'─' * 60}")

        mode_results = {}
        _raw_results[mode_key] = {}

        # -- Evaluate in both splits ----------------------------------------
        for split_name, eval_ds in [("dev", eval_datasets_dev), ("test", eval_datasets_test)]:
            print(f"\n  -- Split: {split_name.upper()} --")

            metrics, results = evaluate_on_datasets(
                model, tokenizer, eval_ds,
                label=f"{model_slug}",
                with_context=with_context,
                model_name=model_name,
            )
            _raw_results[mode_key][split_name] = results

            # Compute weighted aggregate
            agg_f1 = agg_faith = agg_words = agg_complete = 0.0
            agg_n = 0
            for ds_name, m in metrics.items():
                n = m["n_samples"]
                agg_n        += n
                agg_f1       += m["Token_F1"] * n
                agg_faith    += m["Context_Faithfulness_Pct"] * n
                agg_words    += m["Avg_Response_Length_Words"] * n
                agg_complete += m["Sentence_Completeness_Pct"] * n

            aggregate = {
                "n_samples":                   agg_n,
                "Token_F1":                    round(agg_f1 / agg_n, 2) if agg_n else 0.0,
                "Context_Faithfulness_Pct":    round(agg_faith / agg_n, 2) if agg_n else 0.0,
                "Avg_Response_Length_Words":   round(agg_words / agg_n, 1) if agg_n else 0.0,
                "Sentence_Completeness_Pct":   round(agg_complete / agg_n, 1) if agg_n else 0.0,
            }
            if not with_context:
                aggregate["Context_Faithfulness_Note"] = "N/A (no context provided to model)"

            split_data = dict(metrics)
            split_data["aggregate"] = aggregate
            mode_results[split_name] = split_data

            print(f"\n  Aggregate ({split_name}, {mode_label}):")
            print(f"    Token F1:             {aggregate['Token_F1']:.2f}%")
            faith_note = "  [N/A -- no context]" if not with_context else ""
            print(f"    Context Faithfulness: {aggregate['Context_Faithfulness_Pct']:.2f}%{faith_note}")
            print(f"    Avg Response Length:  {aggregate['Avg_Response_Length_Words']:.1f} words")

        model_results[mode_key] = mode_results

    # -- Save predictions checkpoint (crash recovery) -----------------------
    _pred_ckpt_path = os.path.join(output_dir, f"predictions_{model_slug}.json")
    try:
        _pred_ckpt = {}
        for mode_key in _raw_results:
            _pred_ckpt[mode_key] = {}
            for split_name, res in _raw_results[mode_key].items():
                _pred_ckpt[mode_key][split_name] = {
                    ds: {
                        "predictions":  [r["prediction"]   for r in samples],
                        "ground_truths": [r["ground_truth"] for r in samples],
                    }
                    for ds, samples in res.items()
                }
        with open(_pred_ckpt_path, "w", encoding="utf-8") as f:
            json.dump(_pred_ckpt, f, ensure_ascii=False)
        print(f"--> Predictions checkpoint saved: {_pred_ckpt_path}")
    except Exception as e:
        print(f"    Warning: could not save predictions checkpoint: {e}")

    # -- Unload LLM and free GPU memory before BERTScore --------------------
    print(f"\n--> Unloading model {model_name} for BERTScore computation...")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # -- BERTScore computation (DeBERTa-xlarge-mnli, ~1.3 GB) ---------------
    print(f"\n--> Computing BERTScore for {model_slug}...")

    for mode_key in ["with_context", "without_context"]:
        for split_name in ["dev", "test"]:
            raw_res = _raw_results[mode_key][split_name]
            preds, gts = _extract_preds_gts(raw_res)

            print(f"\n  [{mode_key} | {split_name}]")
            bs = compute_bertscore_all(preds, gts)

            # Merge BERTScore averages into the metrics dict
            for ds_name in bs:
                model_results[mode_key][split_name][ds_name]["BERTScore_P"]  = bs[ds_name]["avg_P"]
                model_results[mode_key][split_name][ds_name]["BERTScore_R"]  = bs[ds_name]["avg_R"]
                model_results[mode_key][split_name][ds_name]["BERTScore_F1"] = bs[ds_name]["avg_F1"]

            # Recompute aggregate including BERTScore
            agg = model_results[mode_key][split_name]["aggregate"]
            agg_bs_f1 = 0.0
            for ds_name in bs:
                n = model_results[mode_key][split_name][ds_name]["n_samples"]
                agg_bs_f1 += bs[ds_name]["avg_F1"] * n
            agg["BERTScore_F1"] = round(agg_bs_f1 / agg["n_samples"], 2) if agg["n_samples"] else 0.0

    # Free BERTScore GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    print(f"--> BERTScore complete for {model_slug}. GPU cache cleared.")

    all_results[model_name] = model_results

    # -- Save checkpoint (with BERTScore) -----------------------------------
    # Safe for concurrent jobs: read from disk, merge, atomic rename.
    # os.replace() is atomic on POSIX and Windows (same filesystem).
    try:
        _disk = {}
        if os.path.exists(CHECKPOINT_PATH):
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                _disk = json.load(f)
        _disk.update(all_results)
        _tmp_path = CHECKPOINT_PATH + ".tmp"
        with open(_tmp_path, "w", encoding="utf-8") as f:
            json.dump(_disk, f, indent=4, ensure_ascii=False)
        os.replace(_tmp_path, CHECKPOINT_PATH)
        print(f"--> Checkpoint saved: {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"--> Warning: could not save checkpoint: {e}")


# ─────────────────────────────────────────────
# SECTION 5: SUMMARY TABLE AND SAVE
# ─────────────────────────────────────────────
# Print table with weighted aggregates per model/split/mode.
# Save baseline_evaluation.json (full) and _samples.json (aggregates only).

print("\n" + "=" * 70)
print("BASELINE EVALUATION SUMMARY TABLE")
print("=" * 70)

# Table header
header = (f"{'Model':<35s} | {'Split':<5s} | {'Mode':<12s} | "
          f"{'Token_F1':>9s} | {'BS_F1':>9s} | {'Ctx_Faith':>10s} | {'Avg_Len':>8s}")
print(header)
print("─" * len(header))

for model_name in MODELS:
    short_name = model_name.split("/")[-1]
    if model_name not in all_results:
        continue
    for mode_key, mode_label in [("with_context", "with context"), ("without_context", "no context")]:
        for split_name in ["dev", "test"]:
            agg = all_results[model_name][mode_key][split_name].get("aggregate", {})
            f1      = agg.get("Token_F1", 0.0)
            bs_f1   = agg.get("BERTScore_F1", 0.0)
            faith   = agg.get("Context_Faithfulness_Pct", 0.0)
            avg_len = agg.get("Avg_Response_Length_Words", 0.0)

            faith_str = f"{faith:>9.2f}%" if mode_key == "with_context" else "   N/A    "

            print(f"{short_name:<35s} | {split_name:<5s} | {mode_label:<12s} | "
                  f"{f1:>8.2f}% | {bs_f1:>8.2f}% | {faith_str} | {avg_len:>7.1f}w")

print("─" * len(header))

# -- Save JSON --------------------------------------------------------------

output_path = os.path.join(output_dir, "baseline_evaluation.json")
try:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"\n--> Results saved: {output_path}")
except Exception as e:
    print(f"\n    Warning: could not save file: {e}")

# Also save compact samples for manual inspection
samples_path = os.path.join(output_dir, "baseline_evaluation_samples.json")
try:
    compact = {}
    for model_name in MODELS:
        compact[model_name] = {}
        for mode_key in ["with_context", "without_context"]:
            compact[model_name][mode_key] = {}
            for split_name in ["dev", "test"]:
                compact[model_name][mode_key][split_name] = {
                    "aggregate": all_results[model_name][mode_key][split_name].get("aggregate", {})
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
