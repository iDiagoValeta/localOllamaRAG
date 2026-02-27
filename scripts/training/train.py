"""
LoRA Training Script for Qwen3-14B (TFG) — v2
==============================================

This script implements an end-to-end fine-tuning pipeline for:
1) Loading and adapting the base model `Qwen/Qwen3-14B` with LoRA.
2) Preparing a multi-source QA dataset from 7 sources (DROP, SQuAD 1.1,
   SQuAD 2.0, NarrativeQA, OpenbookQA, Natural Questions, and Aina RAG).
3) Training with loss masking on the prompt (learns only the response).
4) Saving artifacts, metrics, and qualitative generation samples.

The model is trained in non-thinking mode (no <think> reasoning blocks),
optimized for RAG pipelines where direct, context-grounded answers are preferred.

Changes vs v1:
- 3 epochs with LR 1e-4 (was 1 epoch, 2e-4) + EarlyStoppingCallback(patience=3)
- SQuAD 2.0: 20% unanswerable kept to teach abstention (reduces hallucinations)
- interleave_datasets for balanced gradient updates across sources
- Token-based context truncation (was character-based)
- Concise system prompt to save tokens for context/response
- Deterministic (greedy) test generation for reproducibility
- warmup_ratio=0.10, max_grad_norm=1.0 for stability
"""

import os
import json
import math
import torch
import bitsandbytes as bnb
from datasets import load_dataset, concatenate_datasets, Dataset, interleave_datasets
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForSeq2Seq,
)

# =============================================================================
# SECTION 1: ENVIRONMENT & HARDWARE CONFIGURATION
# =============================================================================
# Stability settings for GPU/HPC execution:
# - Disables dynamic compilation paths that can introduce instability.
# - Adjusts CUDA memory reservation policy to reduce fragmentation.
# =============================================================================

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["TRITON_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

output_dir = os.path.join(os.getcwd(), "training-output")
os.makedirs(output_dir, exist_ok=True)


# =============================================================================
# SECTION 2: BASE MODEL LOADING (QWEN3-14B)
# =============================================================================
# Loads the base model and tokenizer for efficient LoRA fine-tuning.
# Uses BF16 + automatic device mapping + gradient checkpointing to optimize
# memory usage when training a 14B parameter model.
#
# The model is Qwen3-14B (base, not Instruct) so the SFT process teaches
# the desired RAG behavior from scratch without conflicting instruction tuning.
# =============================================================================

model_name = "Qwen/Qwen3-14B"

print(f"--> [SECTION 2] Loading base model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa",
)

model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, peft_config)
print("--> Model ready.")
model.print_trainable_parameters()


# =============================================================================
# SECTION 3: MULTI-SOURCE DATASET LOADING
# =============================================================================
# Loads and normalizes 7 QA datasets into a unified (instruction, context,
# response) format suitable for RAG-style supervised fine-tuning:
#
#   1. DROP          — Discrete reasoning over paragraphs
#   2. SQuAD 1.1     — Extractive QA from Wikipedia passages
#   3. SQuAD 2.0     — Same as 1.1 but includes unanswerable (filtered out)
#   4. NarrativeQA   — Reading comprehension over long documents (summaries)
#   5. OpenbookQA    — Common-sense reasoning with science facts
#   6. Natural Qs    — Real Google Search questions (no context available)
#   7. Aina RAG      — Multilingual RAG dataset (ES/CA/EN)
#
# Each loader returns a HuggingFace Dataset with columns:
#   instruction (str), context (str), response (str)
# =============================================================================

# Number of samples to take from each dataset (configurable)
# Sizes are balanced to reduce oversampling with interleave_datasets(all_exhausted).
# Aina stays largest for multilingual coverage (ES/CA/EN).
SAMPLES_DROP = 7000
SAMPLES_SQUAD_V1 = 7000
SAMPLES_SQUAD_V2 = 7000
SAMPLES_NARRATIVEQA = 7000
SAMPLES_OPENBOOKQA = 5000
SAMPLES_NQ = 5000
SAMPLES_AINA = 10000

print("--> [SECTION 3] Loading and normalizing datasets...")


def load_drop(n_samples):
    """
    Loads DROP dataset (discrete reasoning over paragraphs).
    Maps: passage -> context, question -> instruction, answers.spans[0] -> response.
    """
    print(f"    Loading DROP (ucinlp/drop), target: {n_samples} samples...")
    ds = load_dataset("ucinlp/drop", split="train")

    def normalize(example):
        spans = example.get("answers_spans", {}).get("spans", [])
        answer = spans[0] if spans else ""
        return {
            "instruction": example["question"],
            "context": example["passage"],
            "response": answer,
        }

    ds = ds.map(normalize, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["instruction"].strip() and x["response"].strip())
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))
    print(f"    DROP: {len(ds)} samples loaded")
    return ds


def load_squad_v1(n_samples):
    """
    Loads SQuAD 1.1 (extractive QA from Wikipedia).
    Maps: context -> context, question -> instruction, answers.text[0] -> response.
    """
    print(f"    Loading SQuAD 1.1 (rajpurkar/squad), target: {n_samples} samples...")
    ds = load_dataset("rajpurkar/squad", split="train")

    def normalize(example):
        texts = example.get("answers", {}).get("text", [])
        answer = texts[0] if texts else ""
        return {
            "instruction": example["question"],
            "context": example["context"],
            "response": answer,
        }

    ds = ds.map(normalize, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["instruction"].strip() and x["response"].strip())
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))
    print(f"    SQuAD 1.1: {len(ds)} samples loaded")
    return ds


# Standard response for unanswerable questions (teaches the model to abstain)
UNANSWERABLE_RESPONSE = (
    "The provided context does not contain enough information "
    "to answer this question."
)


def load_squad_v2(n_samples):
    """
    Loads SQuAD 2.0 (extractive QA with unanswerable questions).
    Keeps ~20% unanswerable questions so the model learns to abstain
    when the context doesn't contain the answer — critical for RAG.
    """
    print(f"    Loading SQuAD 2.0 (rajpurkar/squad_v2), target: {n_samples} samples...")
    ds = load_dataset("rajpurkar/squad_v2", split="train")

    def normalize(example):
        texts = example.get("answers", {}).get("text", [])
        answer = texts[0] if texts else ""
        is_unanswerable = not answer.strip()
        return {
            "instruction": example["question"],
            "context": example["context"],
            "response": UNANSWERABLE_RESPONSE if is_unanswerable else answer,
            "is_unanswerable": is_unanswerable,
        }

    ds = ds.map(normalize, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["instruction"].strip())

    # Separate answerable and unanswerable, then mix 80/20
    answerable = ds.filter(lambda x: not x["is_unanswerable"])
    unanswerable = ds.filter(lambda x: x["is_unanswerable"])

    n_answerable = int(n_samples * 0.80)
    n_unanswerable = n_samples - n_answerable

    answerable = answerable.shuffle(seed=42).select(range(min(n_answerable, len(answerable))))
    unanswerable = unanswerable.shuffle(seed=42).select(range(min(n_unanswerable, len(unanswerable))))

    ds = concatenate_datasets([answerable, unanswerable]).shuffle(seed=42)
    ds = ds.remove_columns(["is_unanswerable"])
    print(f"    SQuAD 2.0: {len(ds)} samples loaded ({len(unanswerable)} unanswerable)")
    return ds


def load_narrativeqa(n_samples):
    """
    Loads NarrativeQA (reading comprehension over long documents).
    Uses the meithnav/narrativeqa text-only adaptation with document summaries
    as context, which is practical for training vs. the original full-book format.
    """
    print(f"    Loading NarrativeQA (meithnav/narrativeqa), target: {n_samples} samples...")
    ds = load_dataset("meithnav/narrativeqa", split="train")

    col_names = ds.column_names

    def normalize(example):
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

        return {
            "instruction": question,
            "context": context,
            "response": answer,
        }

    ds = ds.map(normalize, remove_columns=col_names)
    ds = ds.filter(lambda x: x["instruction"].strip() and x["response"].strip())
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))
    print(f"    NarrativeQA: {len(ds)} samples loaded")
    return ds


def load_openbookqa(n_samples):
    """
    Loads OpenbookQA (common-sense reasoning with open-book science facts).
    Uses the 'additional' config to get fact1 as context.
    Maps: fact1 -> context, question_stem -> instruction, correct choice -> response.
    """
    print(f"    Loading OpenbookQA (allenai/openbookqa), target: {n_samples} samples...")
    ds = load_dataset("allenai/openbookqa", "additional", split="train")

    def normalize(example):
        fact = example.get("fact1", "")
        question = example.get("question_stem", "")
        answer_key = example.get("answerKey", "")
        choices = example.get("choices", {})

        answer = ""
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        for label, text in zip(labels, texts):
            if label == answer_key:
                answer = text
                break

        return {
            "instruction": question,
            "context": fact,
            "response": answer,
        }

    ds = ds.map(normalize, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["instruction"].strip() and x["response"].strip())
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))
    print(f"    OpenbookQA: {len(ds)} samples loaded")
    return ds


def load_natural_questions(n_samples):
    """
    Loads Natural Questions (real Google Search queries).
    Uses the sentence-transformers adaptation with clean query/answer columns.
    These samples have no context passage, teaching the model to handle
    context-free situations gracefully.
    """
    print(f"    Loading Natural Questions (sentence-transformers/natural-questions), target: {n_samples} samples...")
    ds = load_dataset("sentence-transformers/natural-questions", split="train")

    def normalize(example):
        query = example.get("query", example.get("question", ""))
        answer = example.get("answer", example.get("text", ""))
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        return {
            "instruction": str(query).strip(),
            "context": "",
            "response": str(answer).strip(),
        }

    ds = ds.map(normalize, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["instruction"].strip() and x["response"].strip())
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))
    print(f"    Natural Questions: {len(ds)} samples loaded")
    return ds


def load_aina_rag(n_samples):
    """
    Loads the Aina RAG Multilingual dataset (ES/CA/EN).
    Maps: instruction -> instruction, context -> context, response -> response.
    """
    print(f"    Loading Aina RAG (projecte-aina/RAG_Multilingual), target: {n_samples} samples...")
    ds = load_dataset("projecte-aina/RAG_Multilingual", split="train")

    cols_to_keep = ["instruction", "context", "response"]
    cols_to_remove = [c for c in ds.column_names if c not in cols_to_keep]
    ds = ds.remove_columns(cols_to_remove)
    ds = ds.filter(lambda x: (x["instruction"] or "").strip() and (x["response"] or "").strip())
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))
    print(f"    Aina RAG: {len(ds)} samples loaded")
    return ds


# Load all datasets and merge
all_datasets = []
loaders = [
    ("DROP", load_drop, SAMPLES_DROP),
    ("SQuAD 1.1", load_squad_v1, SAMPLES_SQUAD_V1),
    ("SQuAD 2.0", load_squad_v2, SAMPLES_SQUAD_V2),
    ("NarrativeQA", load_narrativeqa, SAMPLES_NARRATIVEQA),
    ("OpenbookQA", load_openbookqa, SAMPLES_OPENBOOKQA),
    ("Natural Questions", load_natural_questions, SAMPLES_NQ),
    ("Aina RAG", load_aina_rag, SAMPLES_AINA),
]

for name, loader_fn, n in loaders:
    try:
        ds = loader_fn(n)
        all_datasets.append(ds)
    except Exception as e:
        print(f"    WARNING: Failed to load {name}: {e}")
        print(f"    Continuing without {name}...")

# Use interleave to alternate samples from each dataset in round-robin.
# stopping_strategy="all_exhausted" cycles smaller datasets until the largest
# is fully consumed, ensuring no data is wasted (smaller sets get mild oversampling).
dataset = interleave_datasets(all_datasets, seed=42, stopping_strategy="all_exhausted")
print(f"--> Combined dataset: {len(dataset)} total training samples (interleaved, all_exhausted)")
print(f"--> Columns: {dataset.column_names}")

# Create a small eval set from a held-out portion
eval_size = min(1000, int(len(dataset) * 0.03))
splits = dataset.train_test_split(test_size=eval_size, seed=42)
dataset = splits["train"]
eval_dataset_raw = splits["test"]
print(f"--> Train: {len(dataset)}, Eval: {len(eval_dataset_raw)}")


# =============================================================================
# SECTION 4: TOKENIZATION & PROMPT FORMATTING
# =============================================================================
# Prepares the combined dataset for supervised fine-tuning (SFT) with Qwen3's
# ChatML format. Key design decisions:
#
# - Uses tokenizer.apply_chat_template(enable_thinking=False) to generate
#   prompts WITHOUT <think> reasoning blocks. This trains the model to produce
#   direct answers, which is more effective for RAG pipelines.
#
# - Loss is masked on the prompt tokens (-100) so the model only learns to
#   generate the assistant's response.
#
# - Context is wrapped in <context>...</context> tags within the user message.
# =============================================================================

system_prompt = (
    "You are a helpful assistant. Answer questions based EXCLUSIVELY "
    "on the provided context.\n"
    "Rules: Use ONLY the <context> content. Do NOT fabricate information. "
    "If the answer is not in the context, state it clearly. "
    "Be clear, concise, and well-structured."
)

MAX_LENGTH = 2048
MAX_CONTEXT_TOKENS = 1500

def format_and_tokenize(examples):
    """
    Converts a batch into Qwen3 causal format for SFT-RAG training.

    The function:
    1) Builds a structured prompt with system/user/context using apply_chat_template.
    2) Tokenizes prompt and response independently.
    3) Controls maximum length and discards invalid cases.
    4) Masks the prompt in `labels` with -100 so loss is computed only on the response.

    Returns:
        dict with `input_ids`, `labels`, and `attention_mask` for the Trainer.
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

        if ctx:
            # Truncate context by tokens (more precise than by characters)
            ctx_ids = tokenizer(ctx, add_special_tokens=False, truncation=False)["input_ids"]
            if len(ctx_ids) > MAX_CONTEXT_TOKENS:
                ctx_ids = ctx_ids[:MAX_CONTEXT_TOKENS]
                ctx = tokenizer.decode(ctx_ids, skip_special_tokens=True)
            user_msg = f"{instruction}\n\n<context>{ctx}</context>"
        else:
            user_msg = instruction

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


print("--> Tokenizing training dataset...")
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


print(f"--> Train dataset: {len(tokenized_dataset)} valid examples from {original_len} original")

print("--> Tokenizing evaluation dataset...")
tokenized_eval = eval_dataset_raw.map(
    format_and_tokenize,
    batched=True,
    batch_size=1000,
    remove_columns=eval_dataset_raw.column_names,
    desc="Tokenizing eval",
)
tokenized_eval = tokenized_eval.filter(
    lambda x: len(x["input_ids"]) > 0,
    desc="Filtering valid examples (eval)"
)


print(f"--> Eval dataset: {len(tokenized_eval)} valid examples")


# =============================================================================
# SECTION 5: TRAINING CONFIGURATION
# =============================================================================
# Final hyperparameters used for training. Execution is controlled by
# `num_train_epochs=3` with early stopping (patience=3 evals) over the
# interleaved dataset, with gradient accumulation for effective batch=32.
# =============================================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
)


class LightEarlyStoppingCallback(TrainerCallback):
    """
    Lightweight early stopping that does NOT require checkpoints on disk.
    Monitors eval_loss and stops training if it doesn't improve for
    `patience` consecutive evaluations. Uses zero disk space.
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
            print(f"    [EarlyStopping] Stopping training: no improvement for {self.patience} evals.")
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

print("--> [SECTION 5] Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[LightEarlyStoppingCallback(patience=3)],
)


# =============================================================================
# SECTION 6: TRAINING LOOP
# =============================================================================
# Executes supervised training with periodic evaluation.
# =============================================================================

print("--> [SECTION 6] Starting training...")
print(f"    - Total steps: {training_args.max_steps}")
print(f"    - Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"    - Learning rate: {training_args.learning_rate}")

trainer.train()


# =============================================================================
# SECTION 7: FINAL EXPORT & METRICS
# =============================================================================
# Persists artifacts and metrics for experimental reproducibility:
# - Trained LoRA adapter.
# - Tokenizer used.
# - Training history in JSON for post-analysis.
# =============================================================================

print(f"--> [SECTION 7] Saving final model to {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

log_history_path = os.path.join(output_dir, "training_stats.json")
print(f"--> Saving statistics to: {log_history_path}")

training_summary = {
    "model_name": model_name,
    "total_steps": trainer.state.global_step,
    "final_loss": trainer.state.log_history[-1].get("loss") if trainer.state.log_history else None,
    "dataset_size": len(tokenized_dataset),
    "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
    "datasets_used": [name for name, _, _ in loaders],
    "log_history": trainer.state.log_history,
}

try:
    with open(log_history_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=4, ensure_ascii=False)
except Exception as e:
    print(f"Could not save log: {e}")

print("--> Training completed!")
print(f"    - Steps completed: {trainer.state.global_step}")
if training_summary["final_loss"]:
    print(f"    - Final loss: {training_summary['final_loss']:.4f}")


# =============================================================================
# SECTION 8: FINAL EVALUATION WITH SAMPLES
# =============================================================================
# Final evaluation at two levels:
# - Quantitative: eval_loss and perplexity on validation set.
# - Qualitative: generation of test samples with explicit context.
# =============================================================================

print("\n" + "="*70)
print("--> [SECTION 8] Final evaluation with sample generation")
print("="*70)

print("\n--> Computing metrics on evaluation set...")
eval_results = trainer.evaluate()
perplexity = math.exp(eval_results["eval_loss"]) if "eval_loss" in eval_results else None

print(f"    - Eval Loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
if perplexity:
    print(f"    - Perplexity: {perplexity:.2f}")

training_summary["eval_loss"] = eval_results.get("eval_loss")
training_summary["perplexity"] = perplexity

test_prompts = [
    {
        "instruction": "What is the main function of the mitochondria in a cell?",
        "context": "The mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells. They are often referred to as the powerhouse of the cell because they generate most of the cell's supply of adenosine triphosphate (ATP), which is used as a source of chemical energy.",
        "description": "[EN] Science fact QA: biology context"
    },
    {
        "instruction": "¿Cuál fue el resultado principal del Tratado de Utrecht?",
        "context": "El Tratado de Utrecht, firmado en 1713, puso fin a la Guerra de Sucesión Española. España cedió Gibraltar y Menorca a Gran Bretaña, y los Países Bajos españoles y territorios italianos pasaron a Austria. Felipe V fue reconocido como rey de España, pero renunció a sus derechos al trono francés.",
        "description": "[ES] QA histórica: Tratado de Utrecht"
    },
    {
        "instruction": "Quines són les principals característiques de l'Albufera de València?",
        "context": "L'Albufera de València és un parc natural situat a uns 10 km al sud de la ciutat. Es tracta d'una llacuna d'aigua dolça separada del mar per una estreta franja d'arena coneguda com la Devesa. És un dels ecosistemes més importants de la península Ibèrica, amb més de 300 espècies d'aus. A més, els arrossars que l'envolten són fonamentals per al cultiu de l'arròs utilitzat en la paella valenciana.",
        "description": "[CA/VAL] QA geogràfica: l'Albufera de València"
    },
    {
        "instruction": "What causes volcanic eruptions?",
        "context": "The Mediterranean diet emphasizes the consumption of fruits, vegetables, whole grains, legumes, and olive oil. Fish and poultry are preferred over red meat. Studies have shown that this dietary pattern is associated with reduced risk of cardiovascular disease and improved longevity.",
        "description": "[ABSTENTION] Irrelevant context: should decline to answer"
    },
]

print("\n--> Generating test responses...")
print("-" * 70)

model.eval()


def generate_response(instruction, context=None, max_new_tokens=256):
    """
    Generates an inference response using the same prompt format as training,
    ensuring train/inference consistency. Uses non-thinking mode.

    Args:
        instruction: User question.
        context: Retrieved context (optional, recommended for RAG mode).
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        Response text string.
    """
    if context:
        user_msg = f"{instruction}\n\n<context>{context}</context>"
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

    torch.manual_seed(42)  # Reproducible generation across runs
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for comparability
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False)[0],
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


generated_samples = []
for i, test in enumerate(test_prompts, 1):
    print(f"\n[Sample {i}] {test['description']}")
    print(f"  Question: {test['instruction'][:80]}{'...' if len(test['instruction']) > 80 else ''}")
    if test["context"]:
        print(f"  Context: {test['context'][:60]}...")

    try:
        response = generate_response(test["instruction"], test["context"])
        print(f"  Response: {response[:300]}{'...' if len(response) > 300 else ''}")

        generated_samples.append({
            "instruction": test["instruction"],
            "context": test["context"],
            "response": response,
        })
    except Exception as e:
        print(f"  Error generating response: {e}")

print("\n" + "-" * 70)

samples_path = os.path.join(output_dir, "generated_samples.json")
try:
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(generated_samples, f, indent=4, ensure_ascii=False)
    print(f"--> Samples saved to: {samples_path}")
except Exception as e:
    print(f"Error saving samples: {e}")

try:
    with open(log_history_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=4, ensure_ascii=False)
except Exception as e:
    print(f"Error updating statistics: {e}")

print("\n" + "="*70)
print("--> Process complete!")
print(f"    - Model saved to: {output_dir}")
print(f"    - Statistics: {log_history_path}")
print(f"    - Samples: {samples_path}")
print("="*70)
