"""
LoRA Training Script for Qwen3-14B (TFG) — v6
==============================================

This script implements an end-to-end fine-tuning AND evaluation pipeline:
1) Loading the base model Qwen/Qwen3-14B.
2) Evaluating the base model on held-out test splits (pre-training baseline).
3) Applying LoRA adapter and fine-tuning on 4 RAG-native sources
   (NarrativeQA, Neural-Bridge RAG, Dolly closed QA, and Aina RAG).
4) Evaluating the adapted model on the same test splits (post-training).
5) Producing a comparative summary: Base vs Adapted with per-dataset metrics.

Target behavior: NotebookLM-style document Q&A — strict context adherence,
professional well-structured answers adapted to question complexity.

All training samples include document context wrapped in <context> tags.
The pipeline always provides context at inference time (embedding threshold
gates the response protocol), so context-free samples are excluded.

Design principle: The RAG retriever guarantees relevant context by filtering
with a similarity threshold. The model should ALWAYS answer from the provided
context and never abstain. Abstention training was removed because it caused
false negatives (refusing to answer despite having the correct fragment).

Metrics (RAG-relevant, zero additional dependencies):
- Token F1 (Counter-based, SQuAD-standard): content overlap with gold answer.
- Avg Response Length (in words): detects conciseness/verbosity bias.
- Sentence Completeness %: detects span-copying (incomplete fragment answers).

Changes vs v5:
- Integrated evaluation: base model evaluated BEFORE training, adapted model
  AFTER training, producing a single comparative JSON report
- Removed dependency on separate evaluate_lora.py script
- Added test split loaders for all 4 training datasets
- Metrics redesigned for RAG relevance: Token F1, response length analysis,
  sentence completeness (removed Exact Match and abstention metrics)
- Added tqdm progress bars for evaluation loops

Changes vs v4:
- Removed DROP, SQuAD v1, SQuAD v2 (span-copying)
- Added Dolly closed QA (instruction-following with reference text)
- 100% RAG-format training data
- System prompt redesigned for always-answer behavior

Changes vs v3:
- Added neural-bridge/rag-dataset-12000 (EN professional RAG QA)

Changes vs v2:
- Removed Natural Questions, OpenbookQA

Changes vs v1:
- 3 epochs, EarlyStoppingCallback, interleave_datasets, token-based truncation
"""

import os
import re
import json
import math
import torch
import bitsandbytes as bnb
from collections import Counter
from datasets import load_dataset, Dataset, interleave_datasets
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForSeq2Seq,
)


# =============================================================================
# SECTION 1: ENVIRONMENT, CONSTANTS & SYSTEM PROMPT
# =============================================================================
# Global configuration used across training and evaluation.
# The system prompt is defined here because it must be identical in:
#   - format_and_tokenize (training)
#   - generate_response (evaluation & inference)
# =============================================================================

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["TRITON_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

output_dir = os.path.join(os.getcwd(), "training-output")
os.makedirs(output_dir, exist_ok=True)

# --- Model ---
model_name = "Qwen/Qwen3-14B"

# --- Training dataset sizes ---
SAMPLES_NARRATIVEQA = 8000
SAMPLES_NEURAL_BRIDGE = 10000
SAMPLES_DOLLY = 5000
SAMPLES_AINA = 14000

# --- Evaluation ---
EVAL_SAMPLES_PER_DATASET = 200
MAX_NEW_TOKENS = 2048  # ~1500 words; allows fully detailed NotebookLM-style answers
EVAL_MAX_NEW_TOKENS = 512  # reduced for eval to avoid OOM; sufficient to measure F1/completeness

# --- Tokenization ---
# MAX_LENGTH: total sequence length in training (prompt + context + response).
# Budget breakdown (based on real RAG pipeline debug logs):
#   - System prompt + ChatML overhead:   ~210 tokens
#   - User question:                     ~30-80 tokens
#   - RAG context (4-15 fragments w/     ~1,500-4,000 tokens
#     Contextual Retrieval metadata):
#   - Detailed response:                 ~500-1,500 tokens
#   - Worst-case total:                  ~5,800 tokens
# 8192 provides comfortable headroom for the worst case.
# Qwen3-14B supports up to 128k; 8192 is well within its capacity.
MAX_LENGTH = 8192
MAX_CONTEXT_TOKENS = 4096  # fits 15 RAG fragments with Contextual Retrieval headers

# --- System prompt (shared between training and evaluation) ---
system_prompt = (
    "You are a professional document analysis assistant. Your role is to answer "
    "questions accurately based on the provided document context.\n\n"
    "Guidelines:\n"
    "- Base your answers strictly on the information within the <context> tags.\n"
    "- Do not add information beyond what the context provides.\n"
    "- Formulate clear, well-structured responses in complete sentences.\n"
    "- For factual questions, be direct and precise.\n"
    "- For analytical or complex questions, provide detailed explanations "
    "referencing specific information from the context.\n"
    "- Always respond in the same language as the question.\n"
    "- Synthesize information naturally rather than copying text verbatim."
)


# =============================================================================
# SECTION 2: BASE MODEL LOADING
# =============================================================================
# Loads the base model and tokenizer WITHOUT LoRA adapter.
# The base model is evaluated first (Section 5) to establish a pre-training
# baseline before the adapter is applied for fine-tuning.
# =============================================================================

print(f"\n--> [SECTION 2] Loading base model: {model_name}")
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

print("--> Base model loaded (no adapter applied).")


# =============================================================================
# SECTION 3: EVALUATION METRICS & INFERENCE
# =============================================================================
# Defines metrics relevant for RAG pipeline evaluation:
#
# - Token F1: Counter-based token overlap (SQuAD-standard). Measures how
#   much of the gold answer content appears in the prediction.
#
# - Avg Response Length: Mean word count of generated responses. Detects
#   conciseness bias (too terse) or verbosity bias (too verbose).
#
# - Sentence Completeness: Percentage of responses that end with sentence-
#   ending punctuation (. ! ?). Low values indicate span-copying behavior
#   (e.g. answering "30" instead of "There are 30 boards.").
#
# These metrics require zero additional dependencies.
# =============================================================================

def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, articles (EN/ES/CA) and extra whitespace."""
    text = str(text).lower()
    text = re.sub(r'\b(a|an|the|el|la|los|las|un|una|unos|unas|les|els)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score using Counter (SQuAD-standard)."""
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 1.0 if pred_tokens == truth_tokens else 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    prec = num_common / len(pred_tokens)
    rec = num_common / len(truth_tokens)
    return 2 * (prec * rec) / (prec + rec)


def generate_response(model, tokenizer, instruction, context, max_new_tokens=MAX_NEW_TOKENS):
    """
    Generates an inference response using the same prompt format as training.
    Uses deterministic (greedy) decoding with thinking disabled.
    Context is truncated to MAX_CONTEXT_TOKENS to mirror training and avoid OOM.

    Args:
        model: The model (base or adapted).
        tokenizer: The tokenizer.
        instruction: User question.
        context: Retrieved context from the RAG pipeline.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Response text string.
    """
    ctx = (context or "").strip()

    # Truncate context to MAX_CONTEXT_TOKENS (mirrors format_and_tokenize)
    if ctx:
        ctx_ids = tokenizer(ctx, add_special_tokens=False, truncation=True, max_length=MAX_CONTEXT_TOKENS)["input_ids"]
        ctx = tokenizer.decode(ctx_ids, skip_special_tokens=True)
        user_msg = f"{instruction}\n\n<context>{ctx}</context>"
    else:
        user_msg = instruction

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
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

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()


def evaluate_on_datasets(model, tokenizer, eval_datasets, label="MODEL"):
    """
    Evaluates a model on multiple datasets and returns per-dataset metrics.

    Metrics computed per dataset:
    - Token F1 (average)
    - Average response length (words)
    - Sentence completeness (% of responses ending with . ! or ?)

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer.
        eval_datasets: Dict of {dataset_name: HuggingFace Dataset}.
        label: Label for display (e.g., "BASE", "ADAPTED").

    Returns:
        Tuple of (all_metrics: dict, all_results: dict with per-sample details).
    """
    all_metrics = {}
    all_results = {}

    model.eval()
    torch.cuda.empty_cache()

    for ds_name, ds in eval_datasets.items():
        results = []
        total_f1 = 0.0
        total_response_words = 0
        complete_sentences = 0
        n = len(ds)

        for example in tqdm(ds, desc=f"{label} | {ds_name}"):
            instruction = example["instruction"]
            context = example["context"]
            ground_truth = example["response"]

            pred = generate_response(model, tokenizer, instruction, context, max_new_tokens=EVAL_MAX_NEW_TOKENS)

            f1 = compute_f1(pred, ground_truth)
            total_f1 += f1

            pred_words = len(pred.split())
            total_response_words += pred_words

            # Sentence completeness: response ends with sentence-ending punctuation
            stripped = pred.rstrip()
            if stripped and stripped[-1] in ".!?":
                complete_sentences += 1

            results.append({
                "instruction": instruction,
                "context": context[:200] + "..." if len(context) > 200 else context,
                "ground_truth": ground_truth,
                "prediction": pred,
                "f1": round(f1, 4),
                "response_words": pred_words,
            })

        metrics = {
            "n_samples": n,
            "Token_F1": round((total_f1 / n) * 100, 2) if n > 0 else 0,
            "Avg_Response_Length_Words": round(total_response_words / n, 1) if n > 0 else 0,
            "Sentence_Completeness_Pct": round((complete_sentences / n) * 100, 1) if n > 0 else 0,
        }

        all_metrics[ds_name] = metrics
        all_results[ds_name] = results

        print(f"\n  {ds_name} ({n} samples):")
        print(f"    Token F1:              {metrics['Token_F1']:.2f}%")
        print(f"    Avg Response Length:    {metrics['Avg_Response_Length_Words']:.1f} words")
        print(f"    Sentence Completeness: {metrics['Sentence_Completeness_Pct']:.1f}%")

    return all_metrics, all_results


# =============================================================================
# SECTION 4: DATASET LOADING (TRAIN + EVAL)
# =============================================================================
# Loads 4 RAG-native datasets with separate train and eval portions:
#
#   1. NarrativeQA     — single split, partitioned into non-overlapping portions
#   2. Neural-Bridge    — train split for training, test split for evaluation
#   3. Dolly closed QA  — single split, partitioned into non-overlapping portions
#   4. Aina RAG         — train split for training, test split for evaluation
#
# For datasets without a dedicated test split (NarrativeQA, Dolly), the
# train loader reserves EVAL_SAMPLES_PER_DATASET samples and the eval
# loader takes those reserved samples via index-offset with the same
# shuffle seed (42), guaranteeing zero overlap.
#
# Each dataset is normalized to (instruction, context, response) format.
# =============================================================================

print("\n--> [SECTION 4] Loading datasets (train + eval splits)...")


# --- Shared normalizer functions ---

def _normalize_narrativeqa(example):
    """Normalize NarrativeQA example to (instruction, context, response)."""
    context = ""
    for key in ["summary", "document", "context", "text"]:
        if key in example and example[key]:
            context = str(example[key]).strip()
            if context:
                break

    question = ""
    for key in ["question", "query"]:
        if key in example and example[key]:
            question = str(example[key]).strip()
            if question:
                break

    answer = ""
    for key in ["answer", "answer1", "response"]:
        if key in example and example[key]:
            answer = str(example[key]).strip()
            if answer:
                break

    return {"instruction": question, "context": context, "response": answer}


def _normalize_neural_bridge(example):
    """Normalize Neural-Bridge RAG example."""
    return {
        "instruction": (example.get("question") or "").strip(),
        "context": (example.get("context") or "").strip(),
        "response": (example.get("answer") or "").strip(),
    }


def _normalize_dolly(example):
    """Normalize Dolly example."""
    return {
        "instruction": (example.get("instruction") or "").strip(),
        "context": (example.get("context") or "").strip(),
        "response": (example.get("response") or "").strip(),
    }


def _filter_valid(example):
    """Keep only samples with non-empty instruction, context, and response."""
    return (
        bool(example["instruction"].strip())
        and bool(example["context"].strip())
        and bool(example["response"].strip())
    )


# --- Train dataset loaders ---

def load_narrativeqa_train(n_samples):
    """NarrativeQA train portion — reserves EVAL_SAMPLES_PER_DATASET for eval."""
    print(f"    Loading NarrativeQA train, target: {n_samples}...")
    ds = load_dataset("meithnav/narrativeqa", split="train")
    ds = ds.map(_normalize_narrativeqa, remove_columns=ds.column_names)
    ds = ds.filter(_filter_valid)
    ds = ds.shuffle(seed=42)
    max_train = max(0, len(ds) - EVAL_SAMPLES_PER_DATASET)
    actual = min(n_samples, max_train)
    ds = ds.select(range(actual))
    print(f"    NarrativeQA train: {len(ds)} samples (reserved {EVAL_SAMPLES_PER_DATASET} for eval)")
    return ds


def load_neural_bridge_train(n_samples):
    """Neural-Bridge train portion (first N from shuffled train split)."""
    print(f"    Loading Neural-Bridge RAG train, target: {n_samples}...")
    ds = load_dataset("neural-bridge/rag-dataset-12000", split="train")
    ds = ds.map(_normalize_neural_bridge, remove_columns=ds.column_names)
    ds = ds.filter(_filter_valid)
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))
    print(f"    Neural-Bridge train: {len(ds)} samples")
    return ds


def load_dolly_train(n_samples):
    """Dolly train portion — reserves EVAL_SAMPLES_PER_DATASET for eval."""
    print(f"    Loading Dolly QA train, target: {n_samples}...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    ds = ds.map(_normalize_dolly, remove_columns=ds.column_names)
    ds = ds.filter(_filter_valid)
    ds = ds.shuffle(seed=42)
    max_train = max(0, len(ds) - EVAL_SAMPLES_PER_DATASET)
    actual = min(n_samples, max_train)
    ds = ds.select(range(actual))
    print(f"    Dolly QA train: {len(ds)} samples (reserved {EVAL_SAMPLES_PER_DATASET} for eval)")
    return ds


def load_aina_train(n_samples):
    """Aina RAG Multilingual train split."""
    print(f"    Loading Aina RAG train, target: {n_samples}...")
    ds = load_dataset("projecte-aina/RAG_Multilingual", split="train")
    cols_to_keep = ["instruction", "context", "response"]
    cols_to_remove = [c for c in ds.column_names if c not in cols_to_keep]
    ds = ds.remove_columns(cols_to_remove)
    ds = ds.filter(lambda x: (x["instruction"] or "").strip() and (x["response"] or "").strip())
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))
    print(f"    Aina RAG train: {len(ds)} samples")
    return ds


# --- Eval dataset loaders ---

def load_narrativeqa_eval(n_samples, train_size=SAMPLES_NARRATIVEQA):
    """
    NarrativeQA eval portion — indices AFTER the training portion.
    meithnav/narrativeqa has no test split, so eval samples are taken
    from indices [split_point, split_point + n_samples) of the shuffled
    train split, guaranteeing zero overlap with training data.
    """
    print(f"    Loading NarrativeQA eval, target: {n_samples}...")
    ds = load_dataset("meithnav/narrativeqa", split="train")
    ds = ds.map(_normalize_narrativeqa, remove_columns=ds.column_names)
    ds = ds.filter(_filter_valid)
    ds = ds.shuffle(seed=42)
    split_point = min(train_size, max(0, len(ds) - n_samples))
    end = min(split_point + n_samples, len(ds))
    ds = ds.select(range(split_point, end))
    print(f"    NarrativeQA eval: {len(ds)} samples (indices {split_point}–{end - 1})")
    return ds


def load_neural_bridge_eval(n_samples):
    """Neural-Bridge RAG test split for evaluation."""
    print(f"    Loading Neural-Bridge RAG test, target: {n_samples}...")
    ds = load_dataset("neural-bridge/rag-dataset-12000", split="test")
    ds = ds.map(_normalize_neural_bridge, remove_columns=ds.column_names)
    ds = ds.filter(_filter_valid)
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))
    print(f"    Neural-Bridge test: {len(ds)} samples")
    return ds


def load_dolly_eval(n_samples, train_size=SAMPLES_DOLLY):
    """
    Dolly eval portion — indices AFTER the training portion.
    Dolly has no dedicated test split, so eval samples are taken from
    indices [split_point, split_point + n_samples) of the shuffled train
    split, guaranteeing zero overlap with training data.
    split_point is capped to always leave room for n_samples eval samples.
    """
    print(f"    Loading Dolly QA eval, target: {n_samples}...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    ds = ds.map(_normalize_dolly, remove_columns=ds.column_names)
    ds = ds.filter(_filter_valid)
    ds = ds.shuffle(seed=42)
    split_point = min(train_size, max(0, len(ds) - n_samples))
    end = min(split_point + n_samples, len(ds))
    ds = ds.select(range(split_point, end))
    print(f"    Dolly QA eval: {len(ds)} samples (indices {split_point}–{end - 1})")
    return ds


def load_aina_eval(n_samples):
    """Aina RAG Multilingual test split for evaluation."""
    print(f"    Loading Aina RAG test, target: {n_samples}...")
    ds = load_dataset("projecte-aina/RAG_Multilingual", split="test")
    cols_to_keep = ["instruction", "context", "response"]
    cols_to_remove = [c for c in ds.column_names if c not in cols_to_keep]
    ds = ds.remove_columns(cols_to_remove)
    ds = ds.filter(lambda x: (x["instruction"] or "").strip() and (x["response"] or "").strip())
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))
    print(f"    Aina RAG test: {len(ds)} samples")
    return ds


# --- Load all training datasets ---

print("\n  --- Training Datasets ---")
all_datasets = []
train_loaders = [
    ("NarrativeQA", load_narrativeqa_train, SAMPLES_NARRATIVEQA),
    ("Neural-Bridge RAG", load_neural_bridge_train, SAMPLES_NEURAL_BRIDGE),
    ("Dolly QA", load_dolly_train, SAMPLES_DOLLY),
    ("Aina RAG", load_aina_train, SAMPLES_AINA),
]

for name, loader_fn, n in train_loaders:
    try:
        ds = loader_fn(n)
        all_datasets.append(ds)
    except Exception as e:
        print(f"    WARNING: Failed to load {name}: {e}")
        print(f"    Continuing without {name}...")

# Interleave for balanced gradient updates across sources
dataset = interleave_datasets(all_datasets, seed=42, stopping_strategy="all_exhausted")
print(f"--> Combined train dataset: {len(dataset)} samples (interleaved, all_exhausted)")

# Validation split for loss monitoring during training (NOT the eval datasets)
eval_size = min(1000, int(len(dataset) * 0.03))
splits = dataset.train_test_split(test_size=eval_size, seed=42)
dataset = splits["train"]
eval_dataset_raw = splits["test"]
print(f"--> Train: {len(dataset)}, Validation (loss monitoring): {len(eval_dataset_raw)}")

# --- Load all evaluation datasets ---

print("\n  --- Evaluation Datasets (for Base vs Adapted comparison) ---")
eval_datasets = {}
eval_loaders = [
    ("NarrativeQA", load_narrativeqa_eval),
    ("Neural-Bridge RAG", load_neural_bridge_eval),
    ("Dolly QA", load_dolly_eval),
    ("Aina RAG", load_aina_eval),
]

for name, loader_fn in eval_loaders:
    try:
        eval_datasets[name] = loader_fn(EVAL_SAMPLES_PER_DATASET)
    except Exception as e:
        print(f"    WARNING: Failed to load eval {name}: {e}")

total_eval = sum(len(ds) for ds in eval_datasets.values())
print(f"--> Total evaluation samples: {total_eval} across {len(eval_datasets)} datasets")


# =============================================================================
# SECTION 5: BASE MODEL EVALUATION
# =============================================================================
# Evaluates the base Qwen3-14B model (without LoRA) on the test splits
# to establish pre-training baselines for comparison after fine-tuning.
# Results are saved immediately in case training crashes later.
# =============================================================================

print("\n" + "=" * 70)
print("--> [SECTION 5] Evaluating BASE model (pre-training baseline)")
print("=" * 70)

base_metrics, base_results = evaluate_on_datasets(
    model, tokenizer, eval_datasets, label="BASE"
)

# Save base results immediately (crash-safe)
base_eval_path = os.path.join(output_dir, "eval_base.json")
try:
    with open(base_eval_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metrics": base_metrics, "samples": {k: v[:10] for k, v in base_results.items()}},
            f, indent=4, ensure_ascii=False,
        )
    print(f"--> Base evaluation saved to: {base_eval_path}")
except Exception as e:
    print(f"    Warning: Could not save base eval: {e}")


# =============================================================================
# SECTION 6: LoRA ADAPTER APPLICATION
# =============================================================================
# Applies the LoRA adapter to the base model for fine-tuning.
# Gradient checkpointing is enabled here (needed for training, not for
# the base evaluation above).
# =============================================================================

print(f"\n--> [SECTION 6] Applying LoRA adapter...")

model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, peft_config)
print("--> LoRA adapter applied.")
model.print_trainable_parameters()


# =============================================================================
# SECTION 7: TOKENIZATION & PROMPT FORMATTING
# =============================================================================
# Prepares the combined dataset for supervised fine-tuning (SFT) with Qwen3's
# ChatML format. Key design decisions:
#
# - Uses tokenizer.apply_chat_template(enable_thinking=False) to generate
#   prompts WITHOUT <think> reasoning blocks.
#
# - Loss is masked on the prompt tokens (-100) so the model only learns to
#   generate the assistant's response.
#
# - Context is wrapped in <context>...</context> tags within the user message.
# =============================================================================

def format_and_tokenize(examples):
    """
    Converts a batch into Qwen3 causal format for SFT-RAG training.
    Masks prompt tokens with -100 so loss is computed only on the response.
    """
    all_input_ids = []
    all_labels = []
    all_attention_mask = []

    for instruction, context, response in zip(
        examples["instruction"],
        examples["context"],
        examples["response"]
    ):
        ctx = (context or "").strip()

        # Safeguard: skip samples without context
        if not ctx:
            all_input_ids.append([])
            all_labels.append([])
            all_attention_mask.append([])
            continue

        # Truncate context by tokens (more precise than by characters)
        ctx_ids = tokenizer(ctx, add_special_tokens=False, truncation=True, max_length=MAX_CONTEXT_TOKENS)["input_ids"]
        ctx = tokenizer.decode(ctx_ids, skip_special_tokens=True)
        user_msg = f"{instruction}\n\n<context>{ctx}</context>"

        # Build prompt using apply_chat_template with thinking disabled
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        response_text = f"{response}<|im_end|>"

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=False
        )["input_ids"]

        response_ids = tokenizer(
            response_text,
            add_special_tokens=False,
            truncation=False
        )["input_ids"]

        full_ids = prompt_ids + response_ids

        if len(full_ids) > MAX_LENGTH:
            max_response_len = MAX_LENGTH - len(prompt_ids)

            if max_response_len < 50:
                all_input_ids.append([])
                all_labels.append([])
                all_attention_mask.append([])
                continue

            response_ids = response_ids[:max_response_len - 1]
            im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)
            if len(response_ids) > 0 and response_ids[-1:] != im_end_id:
                response_ids = response_ids[:-1] + im_end_id

            full_ids = prompt_ids + response_ids

        labels = [-100] * len(prompt_ids) + response_ids

        if len(full_ids) != len(labels):
            all_input_ids.append([])
            all_labels.append([])
            all_attention_mask.append([])
            continue

        attention_mask = [1] * len(full_ids)

        all_input_ids.append(full_ids)
        all_labels.append(labels)
        all_attention_mask.append(attention_mask)

    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attention_mask,
    }


print("\n--> [SECTION 7] Tokenizing training dataset...")
tokenized_dataset = dataset.map(
    format_and_tokenize,
    batched=True,
    batch_size=1000,
    remove_columns=dataset.column_names,
    desc="Tokenizing train",
)

original_len = len(tokenized_dataset)
tokenized_dataset = tokenized_dataset.filter(
    lambda x: len(x["input_ids"]) > 0,
    desc="Filtering valid examples"
)
print(f"--> Train: {len(tokenized_dataset)} valid from {original_len} original")

print("--> Tokenizing validation dataset...")
tokenized_eval = eval_dataset_raw.map(
    format_and_tokenize,
    batched=True,
    batch_size=1000,
    remove_columns=eval_dataset_raw.column_names,
    desc="Tokenizing eval",
)
tokenized_eval = tokenized_eval.filter(
    lambda x: len(x["input_ids"]) > 0,
    desc="Filtering valid (val)"
)
print(f"--> Validation: {len(tokenized_eval)} valid examples")


# =============================================================================
# SECTION 8: TRAINING CONFIGURATION
# =============================================================================
# Hyperparameters for LoRA fine-tuning with early stopping.
# =============================================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
)


class LightEarlyStoppingCallback(TrainerCallback):
    """
    Lightweight early stopping (no disk checkpoints).
    Monitors eval_loss, stops after `patience` evals without improvement.
    """
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = float("inf")
        self.no_improve_count = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_loss = metrics.get("eval_loss")
        if current_loss is None:
            return

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.no_improve_count = 0
            print(f"    [EarlyStopping] New best eval_loss: {current_loss:.4f}")
        else:
            self.no_improve_count += 1
            print(
                f"    [EarlyStopping] No improvement for {self.no_improve_count}/{self.patience} "
                f"evals (best: {self.best_loss:.4f}, current: {current_loss:.4f})"
            )

        if self.no_improve_count >= self.patience:
            print(f"    [EarlyStopping] Stopping: no improvement for {self.patience} evals.")
            control.should_training_stop = True


training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.10,
    weight_decay=0.01,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    gradient_checkpointing=True,
    bf16=True,
    tf32=True,
    optim="adamw_bnb_8bit",
    max_grad_norm=1.0,
    logging_steps=25,
    logging_first_step=True,
    save_strategy="no",
    eval_strategy="steps",
    eval_steps=100,
    report_to="none",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
)

print("\n--> [SECTION 8] Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    processing_class=tokenizer,
    callbacks=[LightEarlyStoppingCallback(patience=3)],
)


# =============================================================================
# SECTION 9: TRAINING LOOP
# =============================================================================

print("\n--> [SECTION 9] Starting training...")
print(f"    - Epochs: {training_args.num_train_epochs}")
print(f"    - Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"    - Learning rate: {training_args.learning_rate}")

trainer.train()


# =============================================================================
# SECTION 10: MODEL EXPORT
# =============================================================================
# Saves the trained LoRA adapter and tokenizer.
# =============================================================================

print(f"\n--> [SECTION 10] Saving adapted model to {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("--> Adapter + tokenizer saved.")


# =============================================================================
# SECTION 11: ADAPTED MODEL EVALUATION & COMPARATIVE SUMMARY
# =============================================================================
# Evaluates the fine-tuned model on the same test splits used for the base
# model (Section 5). Produces a per-dataset comparison table and saves a
# comprehensive JSON report with:
#   - Per-dataset metrics (Base vs Adapted)
#   - Delta analysis
#   - Qualitative sample pairs
#   - Training summary (loss, steps, perplexity)
# =============================================================================

print("\n" + "=" * 70)
print("--> [SECTION 11] Evaluating ADAPTED model (post-training)")
print("=" * 70)

# Disable gradient checkpointing for clean inference
model.gradient_checkpointing_disable()

adapted_metrics, adapted_results = evaluate_on_datasets(
    model, tokenizer, eval_datasets, label="ADAPTED"
)


# --- Comparative Summary ---

print("\n" + "=" * 70)
print("COMPARATIVE SUMMARY: BASE vs ADAPTED")
print("=" * 70)

comparison = {"per_dataset": {}, "aggregate": {}}
agg_base_f1 = agg_adapted_f1 = 0.0
agg_n = 0

for ds_name in eval_datasets:
    b = base_metrics[ds_name]
    a = adapted_metrics[ds_name]
    delta_f1 = a["Token_F1"] - b["Token_F1"]
    n = b["n_samples"]

    agg_base_f1 += b["Token_F1"] * n
    agg_adapted_f1 += a["Token_F1"] * n
    agg_n += n

    print(f"\n  {ds_name} ({n} samples):")
    print(f"    Token F1:       Base={b['Token_F1']:.2f}%  Adapted={a['Token_F1']:.2f}%  Δ={delta_f1:+.2f}pp")
    print(f"    Avg Length:     Base={b['Avg_Response_Length_Words']:.1f}  Adapted={a['Avg_Response_Length_Words']:.1f} words")
    print(f"    Completeness:   Base={b['Sentence_Completeness_Pct']:.1f}%  Adapted={a['Sentence_Completeness_Pct']:.1f}%")

    comparison["per_dataset"][ds_name] = {
        "base": b,
        "adapted": a,
        "deltas": {
            "Token_F1": round(delta_f1, 2),
            "Avg_Response_Length_Words": round(
                a["Avg_Response_Length_Words"] - b["Avg_Response_Length_Words"], 1
            ),
            "Sentence_Completeness_Pct": round(
                a["Sentence_Completeness_Pct"] - b["Sentence_Completeness_Pct"], 1
            ),
        },
    }

    # Save first 5 sample pairs per dataset for qualitative review
    sample_pairs = []
    for b_res, a_res in zip(base_results[ds_name][:5], adapted_results[ds_name][:5]):
        sample_pairs.append({
            "instruction": b_res["instruction"],
            "ground_truth": b_res["ground_truth"],
            "base_prediction": b_res["prediction"],
            "adapted_prediction": a_res["prediction"],
            "base_f1": b_res["f1"],
            "adapted_f1": a_res["f1"],
        })
    comparison["per_dataset"][ds_name]["sample_pairs"] = sample_pairs


# Weighted aggregate
if agg_n > 0:
    agg_base = agg_base_f1 / agg_n
    agg_adapted = agg_adapted_f1 / agg_n
    comparison["aggregate"] = {
        "Base_F1": round(agg_base, 2),
        "Adapted_F1": round(agg_adapted, 2),
        "Delta_F1": round(agg_adapted - agg_base, 2),
        "Total_Samples": agg_n,
    }

    print(f"\n{'=' * 70}")
    print(f"WEIGHTED AGGREGATE ({agg_n} samples)")
    print(f"  Token F1: Base={agg_base:.2f}% → Adapted={agg_adapted:.2f}%  Δ={agg_adapted - agg_base:+.2f}pp")
    print(f"{'=' * 70}")


# --- Training summary ---

training_summary = {
    "model_name": model_name,
    "total_steps": trainer.state.global_step,
    "final_loss": trainer.state.log_history[-1].get("loss") if trainer.state.log_history else None,
    "dataset_size": len(tokenized_dataset),
    "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
    "datasets_used": [name for name, _, _ in train_loaders],
    "eval_loss": None,
    "perplexity": None,
    "comparison": comparison,
    "log_history": trainer.state.log_history,
}

# Final eval loss + perplexity from the Trainer's validation set
print("\n--> Computing final eval loss on validation set...")
eval_results = trainer.evaluate()
if "eval_loss" in eval_results:
    training_summary["eval_loss"] = eval_results["eval_loss"]
    training_summary["perplexity"] = math.exp(eval_results["eval_loss"])
    print(f"    Eval Loss: {eval_results['eval_loss']:.4f}")
    print(f"    Perplexity: {training_summary['perplexity']:.2f}")


# --- Qualitative generation samples ---

print("\n--> Generating qualitative test samples...")
print("-" * 70)

test_prompts = [
    {
        "instruction": "What is the main function of the mitochondria in a cell?",
        "context": (
            "The mitochondria are membrane-bound organelles found in the cytoplasm "
            "of eukaryotic cells. They are often referred to as the powerhouse of "
            "the cell because they generate most of the cell's supply of adenosine "
            "triphosphate (ATP), which is used as a source of chemical energy. "
            "Mitochondria also play a role in cell signaling, cellular "
            "differentiation, cell death, and the control of the cell cycle and "
            "cell growth."
        ),
        "description": "[EN] Science fact QA: biology context"
    },
    {
        "instruction": "¿Cuál fue el resultado principal del Tratado de Utrecht?",
        "context": (
            "El Tratado de Utrecht, firmado en 1713, puso fin a la Guerra de "
            "Sucesión Española. España cedió Gibraltar y Menorca a Gran Bretaña, "
            "y los Países Bajos españoles y territorios italianos pasaron a "
            "Austria. Felipe V fue reconocido como rey de España, pero renunció "
            "a sus derechos al trono francés."
        ),
        "description": "[ES] QA histórica: Tratado de Utrecht"
    },
    {
        "instruction": "Quines són les principals característiques de l'Albufera de València?",
        "context": (
            "L'Albufera de València és un parc natural situat a uns 10 km al sud "
            "de la ciutat. Es tracta d'una llacuna d'aigua dolça separada del mar "
            "per una estreta franja d'arena coneguda com la Devesa. És un dels "
            "ecosistemes més importants de la península Ibèrica, amb més de 300 "
            "espècies d'aus. A més, els arrossars que l'envolten són fonamentals "
            "per al cultiu de l'arròs utilitzat en la paella valenciana."
        ),
        "description": "[CA/VAL] QA geogràfica: l'Albufera de València"
    },
    {
        "instruction": "Explain how photosynthesis works and why it is important for life on Earth.",
        "context": (
            "Photosynthesis is a biological process used by plants, algae, and "
            "certain bacteria to convert light energy into chemical energy stored "
            "in glucose. The process occurs primarily in the chloroplasts of plant "
            "cells and involves two main stages: the light-dependent reactions, "
            "which take place in the thylakoid membranes and produce ATP and NADPH, "
            "and the Calvin cycle, which occurs in the stroma and uses these energy "
            "carriers to fix carbon dioxide into organic molecules. Photosynthesis "
            "is responsible for producing the oxygen in Earth's atmosphere and forms "
            "the base of virtually all food chains. Without photosynthesis, most "
            "life forms on Earth could not exist, as it provides both the oxygen "
            "needed for aerobic respiration and the organic compounds that serve "
            "as food for heterotrophs."
        ),
        "description": "[EN] Analytical QA: detailed explanation expected"
    },
]

generated_samples = []
for i, test in enumerate(test_prompts, 1):
    print(f"\n[Sample {i}] {test['description']}")
    print(f"  Q: {test['instruction'][:80]}{'...' if len(test['instruction']) > 80 else ''}")

    try:
        resp = generate_response(model, tokenizer, test["instruction"], test["context"])
        print(f"  A: {resp[:300]}{'...' if len(resp) > 300 else ''}")
        generated_samples.append({
            "instruction": test["instruction"],
            "context": test["context"],
            "response": resp,
        })
    except Exception as e:
        print(f"  Error: {e}")

training_summary["generated_samples"] = generated_samples


# --- Save all artifacts ---

log_history_path = os.path.join(output_dir, "training_stats.json")
comparison_path = os.path.join(output_dir, "evaluation_comparison.json")

try:
    with open(log_history_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=4, ensure_ascii=False)
    print(f"\n--> Training stats saved to: {log_history_path}")
except Exception as e:
    print(f"Error saving stats: {e}")

try:
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=4, ensure_ascii=False)
    print(f"--> Evaluation comparison saved to: {comparison_path}")
except Exception as e:
    print(f"Error saving comparison: {e}")

print("\n" + "=" * 70)
print("--> PROCESS COMPLETE")
print(f"    - Adapted model:          {output_dir}")
print(f"    - Training stats:         {log_history_path}")
print(f"    - Evaluation comparison:  {comparison_path}")
print(f"    - Base eval backup:       {base_eval_path}")
print("=" * 70)
