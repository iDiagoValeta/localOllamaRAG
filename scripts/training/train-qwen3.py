"""
LoRA fine-tuning of Qwen3-14B for RAG (v9).

Full training and evaluation pipeline for a RAQ Q&A model that responds
strictly from the retrieved context in EN, ES or CA.

Phases:
  1-2.  Load base Qwen/Qwen3-14B and evaluate on frozen dev+test sets.
  3-4.  Load datasets: Neural-Bridge RAG, Dolly QA, Aina-EN/ES/CA.
        5-source proportional round-robin interleaving.
  5-10. Apply LoRA (r=32), tokenize, train, export best checkpoint.
  11.   Re-evaluate adapted model on same frozen dev+test sets.
  12.   BERTScore computation (DeBERTa-xlarge-mnli) after unloading main model.
  13.   Comparative summary: Token F1, BERTScore F1, CF-lexica per dataset x split.
  14.   Export training_stats.json and evaluation_comparison.json.

Usage:
    python train-qwen3.py
Dependencies:
    - torch, transformers, peft, datasets, tqdm, bert-score
"""

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  CONFIGURATION
#  +-- 1. Environment and constants   CUDA env vars, token limits
#  |        1.1 Dataset sizes            train/val caps per source
#  |        1.2 System prompt            aligned with SYSTEM_PROMPT_RAG from chat_pdfs
#  +-- 2. Base model loading          AutoModelForCausalLM + Qwen3-14B tokenizer
#
#  EVALUATION AND METRICS
#  +-- 3. Metrics and inference
#           3.1 Text normalization       (EN/ES/CA, strip articles and punctuation)
#           3.2 Token F1                 overlap with gold answer (SQuAD-standard)
#           3.3 Context Faithfulness     primary metric for the thesis
#           3.4 Generation              apply_chat_template + enable_thinking=False
#           3.5 Evaluation loop         loop over frozen eval_datasets
#           3.6 BERTScore batch          DeBERTa-xlarge-mnli, per-dataset
#           3.7 Checkpoint helpers       save/load predictions for crash recovery
#
#  DATA
#  +-- 4. Dataset loading
#           4.1 Normalizers              schema mapping -> instruction/context/response
#           4.2 Shared filters           valid, long_response, dolly_rag
#           4.3 Neural-Bridge RAG        9 600 train / 2 400 test
#           4.4 Dolly QA                 15 000 -> RAG filtered -> split 80/10/10
#           4.5 Aina RAG (EN/ES/CA)      split by lang, per-language partitions
#           4.6 Training set             5-source proportional round-robin
#           4.7 Validation set           Trainer (loss monitoring / early stopping)
#           4.8 Frozen dev+test sets     FROZEN -- same for BASE and ADAPTED
#
#  PIPELINE
#  +-- 5. Base model evaluation     pre-training baseline
#  +-- 6. LoRA adapter              r=32, alpha=64 (exploration), 7 target modules
#  +-- 7. Tokenization              ChatML + prompt loss masking
#  +-- 8. Training configuration    hyperparameters, early stopping, checkpoints
#  +-- 9. Training loop             Trainer.train()
#  +--10. Model export              save_pretrained (best checkpoint)
#  +--11. Adapted model evaluation  same frozen test set as Section 5
#  +--12. BERTScore computation     unload main model, DeBERTa, merge metrics
#  +--13. Comparative summary       deltas per dataset×split + weighted aggregate
#  +--14. Artifact export           training_stats.json, evaluation_comparison.json
#
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# VERSION HISTORY
# ─────────────────────────────────────────────
#
# v9 vs v8
#   - BASE vs ADAPTED comparison now evaluates on BOTH dev and test splits
#     (previously only test). Aligns with tutor's criterion: dev for decisions.
#   - Removed dead constants: SAMPLES_DOLLY_TRAIN (cap never reached),
#     SAMPLES_DOLLY_VAL (declared but never used).
#   - Dolly train size is now the real 80% of RAG-filtered rows (no cap).
#   - evaluation_comparison.json restructured: per_split.{dev,test}.{dataset}
#     with deltas and deltas_rel_pct (relative improvement %).
#   - BERTScore computation runs over "DatasetName|split" keys.
#
# v8 vs v7.2
#   - Aina RAG split into 3 per-language subsets (Aina-EN, Aina-ES, Aina-CA)
#     with independent train/val/test partitions.
#   - 5-source round-robin interleaving (NB, Dolly, Aina-EN, Aina-ES, Aina-CA).
#   - Full dev and test partitions (removed EVAL_SAMPLES_PER_DATASET cap).
#   - BERTScore integrated (DeBERTa-xlarge-mnli), with memory-safe pipeline.
#   - Evaluation reports per-language results for Aina.
#
# v7.2 vs v7.1
#   - Removed QASPER (too few samples after filtering).
#   - Fixed column mappings: Neural-Bridge and Aina use question/answer
#     on HuggingFace (not instruction/response); previously produced empty fields.
#   - Dolly filtered to RAG-relevant categories with non-empty context
#     (closed_qa, information_extraction, summarization).
#   - Added Context Faithfulness metric (primary evidence for the thesis):
#     % of unique tokens in the response that appear in the context.
#   - Test set loaded ONCE and shared by BASE and ADAPTED.
#   - Simplified to 3 datasets with balanced interleaving.
#
# v7.1 vs v7
#   - Fixed actual split sizes on HuggingFace.
#   - Each dataset uses its natural split (train/val/test). No data leakage.
#   - Interleaving with probabilities proportional to size.
#
# v7 vs v6
#   - NarrativeQA replaced by QASPER (scientific paper contexts).
#   - LR: 1e-4 -> 5e-5 (v6 showed aggressive collapse in the first 75 steps).
#   - warmup_ratio: 0.10 -> 0.05. Early stopping patience: 3 -> 5.
#   - save_strategy: "no" -> "steps" with load_best_model_at_end=True.
#   - EVAL_MAX_NEW_TOKENS: 1024 -> 2048 (avoided truncation in evaluation).
#   - System prompt: respond in the language of the CONTEXT (EN/ES/CA).
#
# v6 vs v5
#   - Integrated evaluation: BASE before training, ADAPTED after.
#   - Removed dependency on evaluate_lora.py.
#   - Metrics redesigned for RAG: Token F1, response length,
#     sentence completeness (removed Exact Match and abstention metrics).
#   - Added tqdm progress bars.
#
# v5 vs v4
#   - Removed DROP, SQuAD v1 and v2 (span-copying behavior).
#   - Added Dolly closed QA (instruction-following with reference text).
#   - 100% data in RAG format. System prompt redesigned for always-answer.
#
# v4 vs v3
#   - Added neural-bridge/rag-dataset-12000 (professional EN RAG QA).
#
# v3 vs v2
#   - Removed Natural Questions and OpenbookQA.
#
# v2 vs v1
#   - 3 epochs, EarlyStoppingCallback, interleave_datasets,
#     token-based truncation (previously character-based).
#
# ─────────────────────────────────────────────

import gc
import os
import re
import json
import math
import torch
from collections import Counter
from datasets import load_dataset, interleave_datasets, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)

from bert_score import score as bert_score_fn
import bert_score.utils as _bsu


# ---------------------------------------------------------------------------
# MONKEY-PATCH: bert_score.utils.sent_encode
# ---------------------------------------------------------------------------
# DeBERTa reports sys.maxsize as model_max_length, which the Rust tokenizer
# backend cannot store as a 32-bit int.  Force max_length=512 (physical
# limit of DeBERTa) and truncation=True.
# ---------------------------------------------------------------------------
def _safe_sent_encode(tokenizer, a):
    return tokenizer.encode(
        a.strip(), add_special_tokens=True, max_length=512, truncation=True,
    )

_bsu.sent_encode = _safe_sent_encode


# ─────────────────────────────────────────────
# SECTION 1: ENVIRONMENT AND CONSTANTS
# ─────────────────────────────────────────────
# CUDA environment variables, output paths, dataset caps, and system prompt.
# ─────────────────────────────────────────────

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["TRITON_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

output_dir = os.path.join(os.getcwd(), "training-output")
os.makedirs(output_dir, exist_ok=True)

model_name = "Qwen/Qwen3-14B"

SAMPLES_NEURAL_BRIDGE_TRAIN = 8000
SAMPLES_AINA_EN_TRAIN       = 3333
SAMPLES_AINA_ES_TRAIN       = 3333
SAMPLES_AINA_CA_TRAIN       = 3334

SAMPLES_NEURAL_BRIDGE_VAL   = 1000

MAX_NEW_TOKENS      = 2048
EVAL_MAX_NEW_TOKENS = 2048

MAX_LENGTH         = 4096
MAX_CONTEXT_TOKENS = 2048

BERTSCORE_MODEL      = "microsoft/deberta-xlarge-mnli"
BERTSCORE_BATCH_SIZE = 32

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


# ─────────────────────────────────────────────
# SECTION 2: BASE MODEL LOADING
# ─────────────────────────────────────────────
# AutoModelForCausalLM in bfloat16 with device_map="auto" + SDPA attention.
# ─────────────────────────────────────────────

print(f"\n--> [2] Loading base model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"
print("--> Base model loaded.")


# ─────────────────────────────────────────────
# SECTION 3: METRICS AND INFERENCE
# ─────────────────────────────────────────────
# Four RAG metrics with no external dependencies.
#
# Primary (main evidence for the thesis):
#   Context Faithfulness — % of response word-types that also appear
#     in the context. Post-fine-tuning increase demonstrates the model
#     synthesizes from the document rather than using prior knowledge.
#
# Secondary:
#   Token F1              — overlap with gold answer, SQuAD-standard.
#   Avg Response Length   — detects verbosity changes.
#   Sentence Completeness — detects fragmented responses (span-copying).
# ─────────────────────────────────────────────

DOLLY_RAG_CATEGORIES = {"closed_qa", "information_extraction", "summarization"}


def normalize_text(text: str) -> str:
    """Lowercase, strip articles (EN/ES/CA) and punctuation.

    Args:
        text: Raw text string to normalize.

    Returns:
        Cleaned, lowercased text with articles and punctuation removed.
    """
    text = str(text).lower()
    text = re.sub(
        r'\b(a|an|the|el|la|los|las|un|una|unos|unas|les|els|uns|unes)\b',
        ' ', text
    )
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score (SQuAD-standard).

    Args:
        prediction: Model-generated answer.
        ground_truth: Reference answer.

    Returns:
        F1 score between 0.0 and 1.0.
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


def compute_context_faithfulness(prediction: str, context: str) -> float:
    """Compute Context Faithfulness: fraction of unique prediction word-types
    that also appear in the context.

    Interpretation:
        > 0.70  Strongly grounded -- almost all content comes from context.
        0.50-0.70  Adequately grounded.
        < 0.50  Possible hallucination or excessive prior-knowledge use.

    Expected behavior:
        BASE model:    lower score (uses world knowledge freely).
        ADAPTED model: higher score (learned to stay in context).

    Args:
        prediction: Model-generated answer.
        context: Source context provided to the model.

    Returns:
        Faithfulness ratio between 0.0 and 1.0.
    """
    pred_types = set(normalize_text(prediction).split())
    ctx_types  = set(normalize_text(context).split())
    if not pred_types:
        return 0.0
    return len(pred_types & ctx_types) / len(pred_types)


def generate_response(
    model, tokenizer, instruction: str, context: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Run inference using the same prompt format as training.

    Context is truncated to MAX_CONTEXT_TOKENS (mirrors format_and_tokenize).

    Args:
        model: The causal LM (base or adapted).
        tokenizer: Corresponding tokenizer.
        instruction: User question.
        context: Document context for grounding.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated response string.
    """
    ctx = (context or "").strip()
    if ctx:
        ctx_ids = tokenizer(ctx, add_special_tokens=False)["input_ids"]
        if len(ctx_ids) > MAX_CONTEXT_TOKENS:
            ctx = tokenizer.decode(ctx_ids[:MAX_CONTEXT_TOKENS], skip_special_tokens=True)
        user_msg = f"{instruction}\n\n<context>{ctx}</context>"
    else:
        user_msg = instruction

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False)[0],
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def evaluate_on_datasets(model, tokenizer, eval_datasets: dict, label: str = "MODEL") -> tuple:
    """Evaluate a model on the provided eval_datasets dict.

    IMPORTANT: eval_datasets must be the same frozen object for BASE and
    ADAPTED evaluations so that both models answer identical questions on
    identical contexts. This is enforced by the caller -- see Section 5 and 11.

    Args:
        model: The causal LM to evaluate.
        tokenizer: Corresponding tokenizer.
        eval_datasets: Dict mapping dataset name to HF Dataset.
        label: Display label for progress bars (e.g. "BASE", "ADAPTED").

    Returns:
        Tuple of (all_metrics, all_results) where:
            all_metrics: dict[ds_name -> metric dict]
            all_results: dict[ds_name -> list of per-sample dicts]

    Metrics per dataset:
        Token_F1                    answer relevance vs gold (%)
        Context_Faithfulness_Pct    grounding in provided context (%)
        Avg_Response_Length_Words   mean word count
        Sentence_Completeness_Pct  % responses ending with . ! ?
    """
    all_metrics = {}
    all_results = {}
    model.eval()
    torch.cuda.empty_cache()

    for ds_name, ds in eval_datasets.items():
        total_f1         = 0.0
        total_faith      = 0.0
        total_words      = 0
        n_complete       = 0
        results          = []
        n = len(ds)

        for example in tqdm(ds, desc=f"{label} | {ds_name}"):
            instruction  = example["instruction"]
            context      = example["context"]
            ground_truth = example["response"]

            pred = generate_response(
                model, tokenizer, instruction, context,
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
            )

            f1    = compute_f1(pred, ground_truth)
            faith = compute_context_faithfulness(pred, context)
            words = len(pred.split())
            is_complete = bool(pred.rstrip()) and pred.rstrip()[-1] in ".!?"

            total_f1    += f1
            total_faith += faith
            total_words += words
            if is_complete:
                n_complete += 1

            results.append({
                "instruction":  instruction,
                "context":      context[:200] + "..." if len(context) > 200 else context,
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
        all_metrics[ds_name] = metrics
        all_results[ds_name] = results

        print(f"\n  {ds_name} ({n} samples):")
        print(f"    Token F1:               {metrics['Token_F1']:.2f}%")
        print(f"    Context Faithfulness:   {metrics['Context_Faithfulness_Pct']:.2f}%")
        print(f"    Avg Response Length:    {metrics['Avg_Response_Length_Words']:.1f} words")
        print(f"    Sentence Completeness:  {metrics['Sentence_Completeness_Pct']:.1f}%")

    return all_metrics, all_results


# --- 3.6 BERTScore batch computation ---

def compute_bertscore_all(predictions: dict, ground_truths: dict) -> dict:
    """Compute BERTScore (P, R, F1) for every dataset in the predictions dict.

    Args:
        predictions:   {ds_name: [pred_str, ...]}.
        ground_truths: {ds_name: [gt_str, ...]}.

    Returns:
        {ds_name: {P: [...], R: [...], F1: [...], avg_P, avg_R, avg_F1}}.
        Averages are percentages (0-100).  Per-sample values are raw (0-1).
    """
    results = {}
    for ds_name in predictions:
        preds = predictions[ds_name]
        refs  = ground_truths[ds_name]
        print(f"  BERTScore | {ds_name} ({len(preds)} samples)...")
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
        print(f"    P={results[ds_name]['avg_P']:.2f}%  "
              f"R={results[ds_name]['avg_R']:.2f}%  "
              f"F1={results[ds_name]['avg_F1']:.2f}%")
    return results


# --- 3.7 Checkpoint helpers ---

def _save_predictions_checkpoint(eval_results: dict, path: str):
    """Persist predictions + per-sample metrics so BERTScore can be computed
    even if the script is restarted after a crash."""
    data = {}
    for ds_name, samples in eval_results.items():
        data[ds_name] = {
            "predictions":   [s["prediction"]   for s in samples],
            "ground_truths": [s["ground_truth"]  for s in samples],
            "token_f1":      [s["f1"]            for s in samples],
            "faithfulness":  [s["faithfulness"]  for s in samples],
            "words":         [s["words"]         for s in samples],
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"--> Predictions checkpoint saved: {path}")


def _extract_preds_gts(eval_results: dict) -> tuple:
    """Extract parallel pred/gt lists from evaluate_on_datasets output.

    Returns:
        (predictions, ground_truths) — both ``{ds_name: [str, ...]}``.
    """
    preds = {ds: [r["prediction"]   for r in rs] for ds, rs in eval_results.items()}
    gts   = {ds: [r["ground_truth"] for r in rs] for ds, rs in eval_results.items()}
    return preds, gts


# ─────────────────────────────────────────────
# SECTION 4: DATASET LOADING
# ─────────────────────────────────────────────
# Split usage:
#   train -> SFT (proportional round-robin interleaving)
#   val   -> Trainer eval_dataset (loss monitoring / early stopping)
#   test  -> frozen set, loaded ONCE, shared by BASE and ADAPTED
#
# Schemas normalized to: instruction | context | response
#   Neural-Bridge: context | question | answer
#   Dolly:         instruction | context | response | category
#   Aina:          instruction | context | response
# ─────────────────────────────────────────────

print("\n--> [4] Loading datasets...")


def _normalize_nb(example):
    """Normalize Neural-Bridge schema: question->instruction, answer->response.

    Args:
        example: Raw dataset row with question/context/answer fields.

    Returns:
        Dict with instruction/context/response keys.
    """
    return {
        "instruction": (example.get("question") or "").strip(),
        "context":     (example.get("context")  or "").strip(),
        "response":    (example.get("answer")   or "").strip(),
    }


def _normalize_dolly(example):
    """Normalize Dolly schema: instruction, context, response.

    Only RAG-relevant categories with non-empty context are kept.
    Filtering happens AFTER normalization via _filter_dolly_rag.

    Args:
        example: Raw dataset row with instruction/context/response/category fields.

    Returns:
        Dict with instruction/context/response/category keys.
    """
    return {
        "instruction": (example.get("instruction") or "").strip(),
        "context":     (example.get("context")     or "").strip(),
        "response":    (example.get("response")    or "").strip(),
        "category":    (example.get("category")    or "").strip(),
    }


def _normalize_aina(example):
    """Normalize Aina RAG Multilingual schema.

    Actual columns (verified from HuggingFace):
      id | instruction | context | response | category | lang | extractive
    Maps directly to instruction/context/response (same schema as Dolly).

    Args:
        example: Raw dataset row.

    Returns:
        Dict with instruction/context/response keys.
    """
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


def _filter_long_response(ex, min_words: int = 15):
    """Keep only responses with at least min_words words.

    Args:
        ex: Dataset example with a 'response' field.
        min_words: Minimum word count threshold.

    Returns:
        True if the response meets the minimum length.
    """
    return len(ex["response"].split()) >= min_words


def _filter_dolly_rag(ex):
    """Keep only Dolly rows that are RAG-relevant (non-empty context + correct category).

    Args:
        ex: Dataset example with 'category' and 'context' fields.

    Returns:
        True if the row belongs to a RAG-relevant category with context.
    """
    return (
        ex["category"] in DOLLY_RAG_CATEGORIES
        and bool(ex["context"].strip())
    )


print("  Neural-Bridge RAG...")
_nb_train_full = (
    load_dataset("neural-bridge/rag-dataset-12000", split="train")
    .map(_normalize_nb, remove_columns=["context", "question", "answer"])
    .filter(_filter_valid)
    .filter(_filter_long_response)
    .shuffle(seed=42)
)
nb_n = len(_nb_train_full)
nb_val_start = nb_n - SAMPLES_NEURAL_BRIDGE_VAL
nb_train = _nb_train_full.select(range(min(SAMPLES_NEURAL_BRIDGE_TRAIN, nb_val_start)))
nb_val   = _nb_train_full.select(range(nb_val_start, nb_n))
nb_test  = (
    load_dataset("neural-bridge/rag-dataset-12000", split="test")
    .map(_normalize_nb, remove_columns=["context", "question", "answer"])
    .filter(_filter_valid)
    .shuffle(seed=42)
)
print(f"    train={len(nb_train)}, val={len(nb_val)}, test={len(nb_test)}")


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
dolly_train = _dolly_all.select(range(nd_train))
dolly_val   = _dolly_all.select(range(nd_train, nd_train + nd_val))
dolly_test  = _dolly_all.select(range(nd_train + nd_val, nd))
print(f"    RAG rows total: {nd}")
print(f"    train={len(dolly_train)}, val={len(dolly_val)}, test={len(dolly_test)}")


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


_aina_train_by_lang = _load_aina_by_lang("train")
_aina_val_by_lang   = _load_aina_by_lang("validation")
_aina_test_by_lang  = _load_aina_by_lang("test")

_AINA_TRAIN_CAPS = {
    "Aina-EN": SAMPLES_AINA_EN_TRAIN,
    "Aina-ES": SAMPLES_AINA_ES_TRAIN,
    "Aina-CA": SAMPLES_AINA_CA_TRAIN,
}
aina_trains = {}
for lang_label in ["Aina-EN", "Aina-ES", "Aina-CA"]:
    cap = _AINA_TRAIN_CAPS[lang_label]
    tr = _aina_train_by_lang[lang_label]
    aina_trains[lang_label] = tr.select(range(min(cap, len(tr))))
    vl = _aina_val_by_lang[lang_label]
    ts = _aina_test_by_lang[lang_label]
    print(f"    {lang_label}: train={len(aina_trains[lang_label])}, "
          f"val={len(vl)}, test={len(ts)}")

_train_list  = [nb_train, dolly_train,
                aina_trains["Aina-EN"], aina_trains["Aina-ES"], aina_trains["Aina-CA"]]
_train_names = ["Neural-Bridge", "Dolly", "Aina-EN", "Aina-ES", "Aina-CA"]
_train_sizes = [len(d) for d in _train_list]
_total       = sum(_train_sizes)
_probs       = [s / _total for s in _train_sizes]

print("\n  Interleaving training datasets (round-robin, proportional probs):")
for name, n, p in zip(_train_names, _train_sizes, _probs):
    print(f"    {name:16s}: {n:6d} samples  p={p:.3f}")

dataset = interleave_datasets(
    _train_list,
    probabilities=_probs,
    seed=42,
    stopping_strategy="all_exhausted",
)
print(f"--> Combined train dataset: {len(dataset)} samples")

_val_raw = concatenate_datasets([
    nb_val, dolly_val,
    _aina_val_by_lang["Aina-EN"], _aina_val_by_lang["Aina-ES"], _aina_val_by_lang["Aina-CA"],
]).shuffle(seed=42)
eval_dataset_raw = _val_raw
print(f"--> Validation (Trainer): {len(eval_dataset_raw)} samples")
print(f"    NB={len(nb_val)}, Dolly={len(dolly_val)}, "
      f"Aina-EN={len(_aina_val_by_lang['Aina-EN'])}, "
      f"Aina-ES={len(_aina_val_by_lang['Aina-ES'])}, "
      f"Aina-CA={len(_aina_val_by_lang['Aina-CA'])}")

print("\n  Fixed Frozen Evaluation Sets (Base vs Adapted):")

eval_datasets_dev = {
    "Neural-Bridge RAG": nb_val,
    "Dolly QA":          dolly_val,
    "Aina-EN":           _aina_val_by_lang["Aina-EN"],
    "Aina-ES":           _aina_val_by_lang["Aina-ES"],
    "Aina-CA":           _aina_val_by_lang["Aina-CA"],
}
eval_datasets_test = {
    "Neural-Bridge RAG": nb_test,
    "Dolly QA":          dolly_test,
    "Aina-EN":           _aina_test_by_lang["Aina-EN"],
    "Aina-ES":           _aina_test_by_lang["Aina-ES"],
    "Aina-CA":           _aina_test_by_lang["Aina-CA"],
}

for split_label, eds in [("dev", eval_datasets_dev), ("test", eval_datasets_test)]:
    for name, ds in eds.items():
        print(f"    {split_label} | {name}: {len(ds)} samples (FROZEN)")

total_dev  = sum(len(d) for d in eval_datasets_dev.values())
total_test = sum(len(d) for d in eval_datasets_test.values())
print(f"--> Total dev samples:  {total_dev}")
print(f"--> Total test samples: {total_test}")

train_loaders = [
    ("Neural-Bridge RAG", None, SAMPLES_NEURAL_BRIDGE_TRAIN),
    ("Dolly QA",          None, nd_train),
    ("Aina-EN",           None, SAMPLES_AINA_EN_TRAIN),
    ("Aina-ES",           None, SAMPLES_AINA_ES_TRAIN),
    ("Aina-CA",           None, SAMPLES_AINA_CA_TRAIN),
]


# ─────────────────────────────────────────────
# SECTION 5: BASE MODEL EVALUATION
# ─────────────────────────────────────────────
# Evaluates on both dev and test splits. Reused as-is in Section 11.
# ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("--> [5] Evaluating BASE model (pre-training baseline)")
print("    (Same frozen dev+test sets that will be used in Section 11)")
print("=" * 70)

base_metrics  = {}
base_results  = {}

for split_label, eval_ds in [("dev", eval_datasets_dev), ("test", eval_datasets_test)]:
    print(f"\n  -- Split: {split_label.upper()} --")
    m, r = evaluate_on_datasets(model, tokenizer, eval_ds, label=f"BASE-{split_label}")
    for ds_name in m:
        key = f"{ds_name}|{split_label}"
        base_metrics[key] = m[ds_name]
        base_results[key] = r[ds_name]

base_eval_path = os.path.join(output_dir, "eval_base.json")
try:
    with open(base_eval_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metrics": base_metrics,
             "samples": {k: v[:10] for k, v in base_results.items()}},
            f, indent=4, ensure_ascii=False,
        )
    print(f"--> Base evaluation saved: {base_eval_path}")
except Exception as e:
    print(f"    Warning: could not save base eval: {e}")

_save_predictions_checkpoint(
    base_results, os.path.join(output_dir, "predictions_base.json"),
)

gc.collect()
torch.cuda.empty_cache()
print("--> GPU cache cleared after base evaluation.")


# ─────────────────────────────────────────────
# SECTION 6: LoRA ADAPTER
# ─────────────────────────────────────────────
# r=32, alpha=64 (exploration: half of r=64, alpha=128). dropout=0.05.
# Target: q/k/v/o_proj + gate/up/down_proj.
# ─────────────────────────────────────────────

print("\n--> [6] Applying LoRA adapter...")
model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_config)
print("--> LoRA adapter applied.")
model.print_trainable_parameters()


# ─────────────────────────────────────────────
# SECTION 7: TOKENIZATION AND FORMATTING
# ─────────────────────────────────────────────
# Qwen3 ChatML format via apply_chat_template(enable_thinking=False).
# User message: f"{instruction}\n\n<context>{ctx}</context>"
#   - identical to generate_response() in chat_pdfs.py.
# Loss masked on prompt tokens (-100): only the response is learned.
# ─────────────────────────────────────────────

def format_and_tokenize(examples):
    """Tokenize examples into ChatML format with prompt loss masking.

    Args:
        examples: Batch dict with instruction/context/response lists.

    Returns:
        Dict with input_ids, labels, and attention_mask lists.
    """
    all_input_ids    = []
    all_labels       = []
    all_attention    = []

    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    for instruction, context, response in zip(
        examples["instruction"],
        examples["context"],
        examples["response"],
    ):
        ctx = (context or "").strip()
        if not ctx:
            all_input_ids.append([])
            all_labels.append([])
            all_attention.append([])
            continue

        ctx_ids = tokenizer(ctx, add_special_tokens=False)["input_ids"]
        if len(ctx_ids) > MAX_CONTEXT_TOKENS:
            ctx = tokenizer.decode(ctx_ids[:MAX_CONTEXT_TOKENS], skip_special_tokens=True)

        user_msg = f"{instruction}\n\n<context>{ctx}</context>"
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        prompt_ids   = tokenizer(prompt_text,           add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(f"{response}<|im_end|>", add_special_tokens=False)["input_ids"]
        full_ids     = prompt_ids + response_ids

        if len(full_ids) > MAX_LENGTH:
            max_resp_len = MAX_LENGTH - len(prompt_ids)
            if max_resp_len < 50:
                all_input_ids.append([])
                all_labels.append([])
                all_attention.append([])
                continue
            response_ids = response_ids[:max_resp_len - 1] + im_end_id
            full_ids     = prompt_ids + response_ids

        labels       = [-100] * len(prompt_ids) + response_ids
        attention    = [1]    * len(full_ids)

        all_input_ids.append(full_ids)
        all_labels.append(labels)
        all_attention.append(attention)

    return {
        "input_ids":      all_input_ids,
        "labels":         all_labels,
        "attention_mask": all_attention,
    }


print("\n--> [7] Tokenizing datasets...")
tokenized_train = dataset.map(
    format_and_tokenize, batched=True, batch_size=1000,
    remove_columns=dataset.column_names, desc="Tokenising train",
).filter(lambda x: len(x["input_ids"]) > 0)

tokenized_eval = eval_dataset_raw.map(
    format_and_tokenize, batched=True, batch_size=1000,
    remove_columns=eval_dataset_raw.column_names, desc="Tokenising val",
).filter(lambda x: len(x["input_ids"]) > 0)

print(f"--> Train: {len(tokenized_train)} | Val: {len(tokenized_eval)} tokenised samples")


# ─────────────────────────────────────────────
# SECTION 8: TRAINING CONFIGURATION
# ─────────────────────────────────────────────
# LR 5e-5 cosine (1e-4 caused collapse in v6). Effective batch 16.
# load_best_model_at_end=True: exports the checkpoint with lowest eval_loss.
# ─────────────────────────────────────────────

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, padding=True, pad_to_multiple_of=8,
)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True,
    tf32=True,
    optim="adamw_bnb_8bit",
    max_grad_norm=1.0,
    logging_steps=25,
    logging_first_step=True,
    save_strategy="steps",
    save_steps=300,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=150,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
)

print("\n--> [8] Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)


# ─────────────────────────────────────────────
# SECTION 9: TRAINING LOOP
# ─────────────────────────────────────────────

print("\n--> [9] Starting training...")
print(f"    Epochs:            {training_args.num_train_epochs}")
print(f"    Effective batch:   {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"    LR:                {training_args.learning_rate}  (cosine, patience=5)")
print(f"    Best checkpoint:   load_best_model_at_end=True")
trainer.train()


# ─────────────────────────────────────────────
# SECTION 10: MODEL EXPORT
# ─────────────────────────────────────────────
# Saves the checkpoint with lowest eval_loss (not the last step).
# ─────────────────────────────────────────────

print(f"\n--> [10] Saving adapted model to {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("--> Adapter + tokenizer saved.")


# ─────────────────────────────────────────────
# SECTION 11: ADAPTED MODEL EVALUATION
# ─────────────────────────────────────────────
# Same frozen dev+test dicts as Section 5 -> objective BASE vs ADAPTED comparison.
# ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("--> [11] Evaluating ADAPTED model")
print("    (Same frozen dev+test sets as Section 5)")
print("=" * 70)

model.gradient_checkpointing_disable()

adapted_metrics  = {}
adapted_results  = {}

for split_label, eval_ds in [("dev", eval_datasets_dev), ("test", eval_datasets_test)]:
    print(f"\n  -- Split: {split_label.upper()} --")
    m, r = evaluate_on_datasets(model, tokenizer, eval_ds, label=f"ADAPTED-{split_label}")
    for ds_name in m:
        key = f"{ds_name}|{split_label}"
        adapted_metrics[key] = m[ds_name]
        adapted_results[key] = r[ds_name]

_save_predictions_checkpoint(
    adapted_results, os.path.join(output_dir, "predictions_adapted.json"),
)

# --- Final eval loss (requires model + trainer alive) ---
print("\n--> Computing final eval loss...")
_eval_loss = None
_perplexity = None
try:
    ev = trainer.evaluate()
    if "eval_loss" in ev:
        _eval_loss  = ev["eval_loss"]
        _perplexity = math.exp(ev["eval_loss"])
        print(f"    Eval Loss: {_eval_loss:.4f}  Perplexity: {_perplexity:.2f}")
except Exception as e:
    print(f"    Warning: final eval failed: {e}")

# --- Qualitative production samples (requires model alive) ---
print("\n--> Generating qualitative production samples...")
test_prompts = [
    {
        "instruction": "Why is the scaling factor (1/√dk) applied in Scaled Dot-Product Attention?",
        "context": (
            "The two most commonly used attention functions are additive attention and "
            "dot-product (multiplicative) attention. Dot-product attention is identical "
            "to our algorithm, except for the scaling factor of 1/√dk. While the two are "
            "similar in theoretical complexity, dot-product attention is much faster and "
            "more space-efficient in practice, since it can be implemented using highly "
            "optimized matrix multiplication code. For large values of dk, the dot products "
            "grow large in magnitude, pushing the softmax function into regions where it has "
            "extremely small gradients. To counteract this effect, we scale the dot products "
            "by 1/√dk."
        ),
        "description": "[EN] Technical RAG -- analytical answer expected",
    },
    {
        "instruction": "¿Cuáles fueron las consecuencias territoriales del Tratado de Utrecht para España?",
        "context": (
            "El Tratado de Utrecht, firmado en 1713, puso fin a la Guerra de Sucesión Española. "
            "España cedió Gibraltar y Menorca a Gran Bretaña, y los Países Bajos españoles "
            "y territorios italianos pasaron a Austria. Felipe V fue reconocido como rey de "
            "España, pero renunció a sus derechos al trono francés. El tratado también otorgó "
            "a Gran Bretaña el asiento, un contrato monopolístico para abastecer de esclavos "
            "africanos a las colonias españolas en América."
        ),
        "description": "[ES] Historical QA -- detailed answer expected",
    },
    {
        "instruction": "Quines eren les característiques principals del sistema de reg de l'Albufera?",
        "context": (
            "L'Albufera de València és un parc natural situat a uns 10 km al sud de la ciutat. "
            "El sistema de reg de la marjal es basa en una xarxa de canals i sèquies que distribueixen "
            "l'aigua provinent del riu Xúquer cap als arrossars. Les comportes de la Devesa regulen "
            "el nivell de l'aigua de la llacuna, essencial per al cultiu de l'arròs. Durant "
            "l'estiu, els arrossars s'inunden de manera controlada per afavorir la germinació, "
            "i a la tardor s'asseca el terreny per permetre la collita. Aquesta gestió hídrica "
            "és fonamental per mantenir tanto la producció agrícola com l'ecosistema natural."
        ),
        "description": "[CA] Technical QA -- detailed answer in Catalan expected",
    },
]

gen_samples = []
for i, t in enumerate(test_prompts, 1):
    print(f"\n[Sample {i}] {t['description']}")
    print(f"  Q: {t['instruction'][:90]}...")
    try:
        r = generate_response(model, tokenizer, t["instruction"], t["context"])
        print(f"  A: {r[:300]}{'...' if len(r) > 300 else ''}")
        gen_samples.append({"instruction": t["instruction"],
                            "context": t["context"], "response": r})
    except Exception as e:
        print(f"  Error: {e}")


# ─────────────────────────────────────────────
# SECTION 12: BERTSCORE COMPUTATION
# ─────────────────────────────────────────────
# BERTScore uses DeBERTa-xlarge-mnli (~1.3 GB).  The main model (14B) is
# unloaded first to free GPU memory and avoid OOM.  Predictions were saved
# to disk in Sections 5 and 11, so the script can also resume here after a
# crash without re-generating predictions.
# ─────────────────────────────────────────────

# Capture trainer state before deletion (needed in Section 14)
_trainer_global_step   = trainer.state.global_step
_trainer_dataset_size  = len(tokenized_train)
_trainer_effective_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
_trainer_log_history   = trainer.state.log_history

print("\n" + "=" * 70)
print("--> [12] Computing BERTScore (semantic similarity vs ground truth)")
print("    Unloading main model to free GPU memory...")
print("=" * 70)

model.gradient_checkpointing_disable()
del model, tokenizer, trainer
gc.collect()
torch.cuda.empty_cache()
print("--> GPU memory freed.")

base_preds, base_gts       = _extract_preds_gts(base_results)
adapted_preds, adapted_gts = _extract_preds_gts(adapted_results)

print("\n--> BERTScore for BASE predictions...")
base_bertscore = compute_bertscore_all(base_preds, base_gts)

print("\n--> BERTScore for ADAPTED predictions...")
adapted_bertscore = compute_bertscore_all(adapted_preds, adapted_gts)

# Merge BERTScore averages into the metrics dicts so the comparison
# section can treat them like any other metric.
# Keys in base_results/adapted_results are "DatasetName|split".
for combo_key in base_bertscore:
    for label, bs_data, m_dict in [
        ("base",    base_bertscore,    base_metrics),
        ("adapted", adapted_bertscore, adapted_metrics),
    ]:
        m_dict[combo_key]["BERTScore_P"]  = bs_data[combo_key]["avg_P"]
        m_dict[combo_key]["BERTScore_R"]  = bs_data[combo_key]["avg_R"]
        m_dict[combo_key]["BERTScore_F1"] = bs_data[combo_key]["avg_F1"]

# Save BERTScore checkpoint
_bs_ckpt = {
    "bertscore_model": BERTSCORE_MODEL,
    "base":    {ds: {k: v for k, v in d.items() if k.startswith("avg_")} for ds, d in base_bertscore.items()},
    "adapted": {ds: {k: v for k, v in d.items() if k.startswith("avg_")} for ds, d in adapted_bertscore.items()},
}
with open(os.path.join(output_dir, "bertscore_checkpoint.json"), "w", encoding="utf-8") as f:
    json.dump(_bs_ckpt, f, indent=4, ensure_ascii=False)

# Free BERTScore GPU memory before the comparison section
gc.collect()
torch.cuda.empty_cache()


# ─────────────────────────────────────────────
# SECTION 13: COMPARATIVE SUMMARY
# ─────────────────────────────────────────────
# Deltas per dataset + weighted aggregate by sample count.
# ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("COMPARATIVE SUMMARY: BASE vs ADAPTED")
print("=" * 70)

METRIC_KEYS = [
    ("Token_F1",               "Token F1",               "%"),
    ("BERTScore_F1",           "BERTScore F1",           "%"),
    ("Context_Faithfulness_Pct","Context Faithfulness",   "%"),
    ("Avg_Response_Length_Words","Avg Response Length",   "words"),
    ("Sentence_Completeness_Pct","Sentence Completeness", "%"),
]

AGG_KEYS = ["Token_F1", "BERTScore_F1", "Context_Faithfulness_Pct"]

comparison = {"per_split": {}, "aggregate": {}}

_ds_names = list(eval_datasets_dev.keys())

for split_label in ["dev", "test"]:
    print(f"\n{'─' * 70}")
    print(f"  Split: {split_label.upper()}")
    print(f"{'─' * 70}")

    split_comp = {}
    agg_base  = {k: 0.0 for k in AGG_KEYS}
    agg_adapt = {k: 0.0 for k in AGG_KEYS}
    agg_n = 0

    for ds_name in _ds_names:
        key = f"{ds_name}|{split_label}"
        b = base_metrics[key]
        a = adapted_metrics[key]
        n = b["n_samples"]
        agg_n += n
        for k in AGG_KEYS:
            agg_base[k]  += b[k] * n
            agg_adapt[k] += a[k] * n

        print(f"\n  {ds_name} ({n} samples):")
        for mkey, label, unit in METRIC_KEYS:
            delta     = a[mkey] - b[mkey]
            delta_rel = (delta / b[mkey] * 100) if b[mkey] != 0 else float("nan")
            sign      = "+" if delta >= 0 else ""
            sign_rel  = "+" if delta_rel >= 0 else ""
            rel_str   = f"{sign_rel}{delta_rel:.1f}%" if not (delta_rel != delta_rel) else "n/a"
            print(f"    {label:26s}  Base={b[mkey]:.2f}{unit}  "
                  f"Adapted={a[mkey]:.2f}{unit}  Δ={sign}{delta:.2f}pp  "
                  f"Δrel={rel_str}")

        deltas     = {mkey: round(a[mkey] - b[mkey], 2) for mkey, *_ in METRIC_KEYS}
        deltas_rel = {
            mkey: round((a[mkey] - b[mkey]) / b[mkey] * 100, 2) if b[mkey] != 0 else None
            for mkey, *_ in METRIC_KEYS
        }
        split_comp[ds_name] = {
            "base": b, "adapted": a, "deltas": deltas, "deltas_rel_pct": deltas_rel,
            "sample_pairs": [
                {
                    "instruction":          br["instruction"],
                    "ground_truth":         br["ground_truth"],
                    "base_prediction":      br["prediction"],
                    "adapted_prediction":   ar["prediction"],
                    "base_f1":              br["f1"],
                    "adapted_f1":           ar["f1"],
                    "base_faithfulness":    br["faithfulness"],
                    "adapted_faithfulness": ar["faithfulness"],
                }
                for br, ar in zip(base_results[key][:5], adapted_results[key][:5])
            ],
        }

    comparison["per_split"][split_label] = split_comp

    if agg_n > 0:
        agg_b = {k: agg_base[k]  / agg_n for k in AGG_KEYS}
        agg_a = {k: agg_adapt[k] / agg_n for k in AGG_KEYS}

        agg_entry = {"Total_Samples": agg_n}
        for k in AGG_KEYS:
            d_abs = round(agg_a[k] - agg_b[k], 2)
            d_rel = round((agg_a[k] - agg_b[k]) / agg_b[k] * 100, 2) if agg_b[k] != 0 else None
            agg_entry[f"Base_{k}"]     = round(agg_b[k], 2)
            agg_entry[f"Adapted_{k}"]  = round(agg_a[k], 2)
            agg_entry[f"Delta_{k}"]    = d_abs
            agg_entry[f"DeltaRel_{k}"] = d_rel

        comparison["aggregate"][split_label] = agg_entry

        print(f"\n{'=' * 70}")
        print(f"WEIGHTED AGGREGATE {split_label.upper()} ({agg_n} samples)")
        for k in AGG_KEYS:
            d_abs = agg_a[k] - agg_b[k]
            d_rel = (d_abs / agg_b[k] * 100) if agg_b[k] != 0 else float("nan")
            rel_str = f"{d_rel:+.1f}%" if not (d_rel != d_rel) else "n/a"
            print(f"  {k:30s}  Base={agg_b[k]:.2f}%  -> Adapted={agg_a[k]:.2f}%"
                  f"   Δ={d_abs:+.2f}pp  Δrel={rel_str}")
        print(f"{'=' * 70}")


# ─────────────────────────────────────────────
# SECTION 14: ARTIFACT EXPORT
# ─────────────────────────────────────────────
# training_stats.json        -- full summary + log_history + samples
# evaluation_comparison.json -- deltas per dataset + base/adapted pairs
# ─────────────────────────────────────────────

training_summary = {
    "model_name":       model_name,
    "version":          "v9",
    "total_steps":      _trainer_global_step,
    "dataset_size":     _trainer_dataset_size,
    "effective_batch":  _trainer_effective_batch,
    "datasets":         [name for name, _, _ in train_loaders],
    "eval_loss":        _eval_loss,
    "perplexity":       _perplexity,
    "comparison":       comparison,
    "log_history":      _trainer_log_history,
    "generated_samples": gen_samples,
}

for path, obj in [
    (os.path.join(output_dir, "training_stats.json"),       training_summary),
    (os.path.join(output_dir, "evaluation_comparison.json"), comparison),
]:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4, ensure_ascii=False)
        print(f"--> Saved: {path}")
    except Exception as e:
        print(f"    Warning: could not save {path}: {e}")

print("\n" + "=" * 70)
print("--> PROCESS COMPLETED")
print(f"    Adapted model:         {output_dir}")
print(f"    Training stats:        {os.path.join(output_dir, 'training_stats.json')}")
print(f"    Evaluation comparison: {os.path.join(output_dir, 'evaluation_comparison.json')}")
print(f"    Base eval backup:      {base_eval_path}")
print("=" * 70)
