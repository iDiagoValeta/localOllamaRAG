"""
train.py — Fine-tuning LoRA de Qwen3-14B para RAG (TFG) — v7.2
===============================================================

Pipeline de entrenamiento y evaluación comparativa en 5 etapas:
  1. Cargar modelo base Qwen/Qwen3-14B.
  2. Evaluar el modelo base sobre un test set congelado (baseline).
  3. Aplicar adaptador LoRA y fine-tunear sobre 3 datasets RAG nativos.
  4. Evaluar el modelo adaptado sobre el MISMO test set.
  5. Generar resumen comparativo con métricas específicas de RAG.

Objetivo: asistente de Q&A documental con adherencia estricta al contexto
recuperado. Responde en el idioma del contexto (EN, ES o CA).

Uso:
    python train.py
"""

# =============================================================================
# MAPA DEL MÓDULO — Índice de secciones
# =============================================================================
#
#  CONFIGURACIÓN
#  |-- 1. Entorno y constantes   variables de entorno CUDA, limites de tokens
#  |        1.1 Tamaños de dataset   caps de train/val por fuente
#  |        1.2 System prompt        alineado con SYSTEM_PROMPT_RAG de chat_pdfs
#  `-- 2. Carga del modelo base  AutoModelForCausalLM + tokenizador Qwen3-14B
#
#  EVALUACIÓN Y MÉTRICAS
#  `-- 3. Métricas e inferencia
#           3.1 Normalización de texto   (EN/ES/CA, sin artículos ni puntuación)
#           3.2 Token F1                 overlap con gold answer (SQuAD-estandar)
#           3.3 Context Faithfulness     métrica primaria del TFG
#           3.4 Generación              apply_chat_template + enable_thinking=False
#           3.5 Bucle de evaluación     loop sobre eval_datasets congelado
#
#  DATOS
#  `-- 4. Carga de datasets
#           4.1 Normalizadores          mapeo de esquemas -> instruction/context/response
#           4.2 Filtros compartidos     valid, long_response, dolly_rag
#           4.3 Neural-Bridge RAG       9 600 train / 2 400 test
#           4.4 Dolly QA                15 000 -> filtrado RAG -> split 80/10/10
#           4.5 Aina RAG Multilingual   42 300 train / 8 460 val / 5 640 test
#           4.6 Conjunto de train       interleaving round-robin proporcional
#           4.7 Conjunto de validación  Trainer (loss monitoring / early stopping)
#           4.8 Test set congelado      FROZEN — mismo para BASE y ADAPTADO
#
#  PIPELINE
#  |-- 5. Evaluación modelo base    baseline pre-entrenamiento
#  |-- 6. Adaptador LoRA            r=64, alpha=128, 7 módulos target
#  |-- 7. Tokenización              ChatML + máscara de pérdida en prompt
#  |-- 8. Configuración Trainer     hiperparámetros, early stopping, checkpoints
#  |-- 9. Bucle de entrenamiento    Trainer.train()
#  |--10. Exportación del modelo    save_pretrained (mejor checkpoint)
#  |--11. Evaluación adaptado       mismo test set congelado que sección 5
#  |--12. Resumen comparativo       deltas por dataset + agregado ponderado
#  `--13. Guardado de artefactos    training_stats.json, evaluation_comparison.json
#
# =============================================================================

# =============================================================================
# HISTORIAL DE VERSIONES
# =============================================================================
#
# v7.2 vs v7.1
#   - Eliminado QASPER (muy pocas muestras tras filtrado).
#   - Corregidos mapeos de columnas: Neural-Bridge y Aina usan question/answer
#     en HuggingFace (no instruction/response); antes producían campos vacíos.
#   - Dolly filtrado a categorías RAG relevantes con contexto no vacío
#     (closed_qa, information_extraction, summarization).
#   - Añadida métrica Context Faithfulness (evidencia principal del TFG):
#     % de tokens únicos de la respuesta que aparecen en el contexto.
#   - Test set cargado UNA vez y compartido por BASE y ADAPTADO.
#   - Simplificado a 3 datasets con interleaving balanceado.
#
# v7.1 vs v7
#   - Corregidos tamaños reales de splits en HuggingFace.
#   - Cada dataset usa su split natural (train/val/test). Sin data leakage.
#   - Interleaving con probabilidades proporcionales al tamaño.
#
# v7 vs v6
#   - NarrativeQA reemplazado por QASPER (contextos de papers científicos).
#   - LR: 1e-4 → 5e-5 (v6 mostraba colapso agresivo en los primeros 75 pasos).
#   - warmup_ratio: 0.10 → 0.05. Early stopping patience: 3 → 5.
#   - save_strategy: "no" → "steps" con load_best_model_at_end=True.
#   - EVAL_MAX_NEW_TOKENS: 1024 → 2048 (evitaba truncación en evaluación).
#   - System prompt: responder en el idioma del CONTEXTO (EN/ES/CA).
#
# v6 vs v5
#   - Evaluación integrada: BASE antes del entrenamiento, ADAPTADO después.
#   - Eliminada dependencia de evaluate_lora.py.
#   - Métricas rediseñadas para RAG: Token F1, longitud de respuesta,
#     completitud de frases (eliminados Exact Match y métricas de abstención).
#   - Añadidas barras de progreso tqdm.
#
# v5 vs v4
#   - Eliminados DROP, SQuAD v1 y v2 (comportamiento de span-copying).
#   - Añadido Dolly closed QA (instruction-following con texto de referencia).
#   - 100% datos en formato RAG. System prompt rediseñado para always-answer.
#
# v4 vs v3
#   - Añadido neural-bridge/rag-dataset-12000 (QA RAG profesional EN).
#
# v3 vs v2
#   - Eliminados Natural Questions y OpenbookQA.
#
# v2 vs v1
#   - 3 épocas, EarlyStoppingCallback, interleave_datasets,
#     truncación basada en tokens (antes por caracteres).
#
# =============================================================================

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


# =============================================================================
# SECCIÓN 1: ENTORNO Y CONSTANTES
# =============================================================================
# Variables de entorno CUDA, rutas de salida, caps de dataset y system prompt.
# =============================================================================

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["TRITON_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

output_dir = os.path.join(os.getcwd(), "training-output")
os.makedirs(output_dir, exist_ok=True)

model_name = "Qwen/Qwen3-14B"

SAMPLES_NEURAL_BRIDGE_TRAIN = 8000
SAMPLES_DOLLY_TRAIN         = 10000
SAMPLES_AINA_TRAIN          = 10000

SAMPLES_NEURAL_BRIDGE_VAL   = 300
SAMPLES_DOLLY_VAL           = 1000
SAMPLES_AINA_VAL            = 500

EVAL_SAMPLES_PER_DATASET = 200

MAX_NEW_TOKENS      = 2048
EVAL_MAX_NEW_TOKENS = 2048

MAX_LENGTH         = 4096
MAX_CONTEXT_TOKENS = 2048

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


# =============================================================================
# SECCIÓN 2: CARGA DEL MODELO BASE
# =============================================================================
# AutoModelForCausalLM en bfloat16 con device_map="auto" + SDPA attention.
# =============================================================================

print(f"\n--> [2] Cargando modelo base: {model_name}")
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


# =============================================================================
# SECCIÓN 3: MÉTRICAS E INFERENCIA
# =============================================================================
# Cuatro métricas RAG sin dependencias externas.
#
# Primaria (evidencia principal del TFG):
#   Context Faithfulness — % de tipos de token de la respuesta que aparecen
#     en el contexto. Incremento post-fine-tuning demuestra que el modelo
#     sintetiza desde el documento en lugar de usar conocimiento previo.
#
# Secundarias:
#   Token F1              — overlap con gold answer, estandar SQuAD.
#   Avg Response Length   — detecta cambios de verbosidad.
#   Sentence Completeness — detecta respuestas fragmentadas (span-copying).
# =============================================================================

DOLLY_RAG_CATEGORIES = {"closed_qa", "information_extraction", "summarization"}


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
    """
    Context Faithfulness: fraction of unique prediction word-types that
    also appear in the context.

    Interpretation:
        > 0.70  Strongly grounded — almost all content comes from context.
        0.50-0.70  Adequately grounded.
        < 0.50  Possible hallucination or excessive prior-knowledge use.

    Expected behaviour:
        BASE model:    lower score (uses world knowledge freely).
        ADAPTED model: higher score (learned to stay in context).
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
    """
    Inference using the same prompt format as training.
    Context is truncated to MAX_CONTEXT_TOKENS (mirrors format_and_tokenize).
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
    """
    Evaluate a model on the provided eval_datasets dict.

    IMPORTANT: eval_datasets must be the same frozen object for BASE and
    ADAPTED evaluations so that both models answer identical questions on
    identical contexts. This is enforced by the caller — see Section 5 and 11.

    Returns:
        all_metrics  dict[ds_name → metric dict]
        all_results  dict[ds_name → list of per-sample dicts]

    Metrics (per dataset):
        Token_F1                    answer relevance vs gold (%)
        Context_Faithfulness_Pct    grounding in provided context (%)
        Avg_Response_Length_Words   mean word count
        Sentence_Completeness_Pct   % responses ending with . ! ?
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


# =============================================================================
# SECCIÓN 4: CARGA DE DATASETS
# =============================================================================
# Uso de splits:
#   train -> SFT (interleaving round-robin proporcional)
#   val   -> eval_dataset del Trainer (loss monitoring / early stopping)
#   test  -> set congelado, cargado UNA vez, compartido por BASE y ADAPTADO
#
# Esquemas normalizados a: instruction | context | response
#   Neural-Bridge: context | question | answer
#   Dolly:         instruction | context | response | category
#   Aina:          instruction | context | response
# =============================================================================

print("\n--> [4] Cargando datasets...")


def _normalize_nb(example):
    """Neural-Bridge: question→instruction, context, answer→response."""
    return {
        "instruction": (example.get("question") or "").strip(),
        "context":     (example.get("context")  or "").strip(),
        "response":    (example.get("answer")   or "").strip(),
    }


def _normalize_dolly(example):
    """
    Dolly: instruction, context, response.
    Only RAG-relevant categories with non-empty context are kept.
    Filtering happens AFTER normalisation via _filter_dolly_rag.
    """
    return {
        "instruction": (example.get("instruction") or "").strip(),
        "context":     (example.get("context")     or "").strip(),
        "response":    (example.get("response")    or "").strip(),
        "category":    (example.get("category")    or "").strip(),
    }


def _normalize_aina(example):
    """
    Aina RAG Multilingual actual columns (verified from HuggingFace):
      id | instruction | context | response | category | lang | extractive
    Maps directly to instruction/context/response (same schema as Dolly).
    """
    return {
        "instruction": (example.get("instruction") or "").strip(),
        "context":     (example.get("context")     or "").strip(),
        "response":    (example.get("response")    or "").strip(),
    }


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
dolly_train = _dolly_all.select(range(min(SAMPLES_DOLLY_TRAIN, nd_train)))
dolly_val   = _dolly_all.select(range(nd_train, nd_train + nd_val))
dolly_test  = _dolly_all.select(range(nd_train + nd_val, nd))
print(f"    RAG rows total: {nd}")
print(f"    train={len(dolly_train)}, val={len(dolly_val)}, test={len(dolly_test)}")


print("  Aina RAG Multilingual...")


_AINA_EXTRA_COLS = ["id", "category", "lang", "extractive"]


def _load_aina(split):
    ds = load_dataset("projecte-aina/RAG_Multilingual", split=split)
    cols_to_remove = [c for c in _AINA_EXTRA_COLS if c in ds.column_names]
    return (
        ds
        .map(_normalize_aina, remove_columns=cols_to_remove)
        .filter(_filter_valid)
        .filter(_filter_long_response)
        .shuffle(seed=42)
    )


_aina_train_full = _load_aina("train")
aina_train = _aina_train_full.select(range(min(SAMPLES_AINA_TRAIN, len(_aina_train_full))))
_aina_val_full = _load_aina("validation")
aina_val   = _aina_val_full.select(range(min(SAMPLES_AINA_VAL, len(_aina_val_full))))
aina_test  = _load_aina("test")
print(f"    train={len(aina_train)}, val={len(aina_val)}, test={len(aina_test)}")

_train_list  = [nb_train, dolly_train, aina_train]
_train_sizes = [len(d) for d in _train_list]
_total       = sum(_train_sizes)
_probs       = [s / _total for s in _train_sizes]

print("\n  Interleaving training datasets (round-robin, proportional probs):")
for name, n, p in zip(["Neural-Bridge", "Dolly", "Aina"], _train_sizes, _probs):
    print(f"    {name:16s}: {n:6d} samples  p={p:.3f}")

dataset = interleave_datasets(
    _train_list,
    probabilities=_probs,
    seed=42,
    stopping_strategy="all_exhausted",
)
print(f"--> Combined train dataset: {len(dataset)} samples")

_val_raw       = concatenate_datasets([nb_val, dolly_val, aina_val]).shuffle(seed=42)
eval_dataset_raw = _val_raw
print(f"--> Validation (Trainer): {len(eval_dataset_raw)} samples")
print(f"    NB={len(nb_val)}, Dolly={len(dolly_val)}, Aina={len(aina_val)}")

print("\n  Fixed Frozen Test Sets (Base vs Adapted):")
eval_datasets = {}
for name, test_ds in [
    ("Neural-Bridge RAG", nb_test),
    ("Dolly QA",          dolly_test),
    ("Aina RAG",          aina_test),
]:
    n = min(EVAL_SAMPLES_PER_DATASET, len(test_ds))
    eval_datasets[name] = test_ds.select(range(n))
    print(f"    {name}: {n} samples (FROZEN — same for BASE and ADAPTED)")

print(f"--> Total evaluation samples: {sum(len(d) for d in eval_datasets.values())}")

train_loaders = [
    ("Neural-Bridge RAG", None, SAMPLES_NEURAL_BRIDGE_TRAIN),
    ("Dolly QA",          None, SAMPLES_DOLLY_TRAIN),
    ("Aina RAG",          None, SAMPLES_AINA_TRAIN),
]


# =============================================================================
# SECCIÓN 5: EVALUACIÓN DEL MODELO BASE (baseline pre-entrenamiento)
# =============================================================================
# eval_datasets NO se modifica aqui. Se reutiliza tal cual en la sección 11.
# =============================================================================

print("\n" + "=" * 70)
print("--> [5] Evaluando modelo BASE (baseline pre-entrenamiento)")
print("    (Mismo test set congelado que se usara en la seccion 11)")
print("=" * 70)

base_metrics, base_results = evaluate_on_datasets(
    model, tokenizer, eval_datasets, label="BASE"
)

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

import gc
gc.collect()
torch.cuda.empty_cache()
print("--> GPU cache cleared after base evaluation.")


# =============================================================================
# SECCIÓN 6: ADAPTADOR LoRA
# =============================================================================
# r=64, alpha=128, dropout=0.05. Target: q/k/v/o_proj + gate/up/down_proj.
# =============================================================================

print("\n--> [6] Aplicando adaptador LoRA...")
model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_config)
print("--> LoRA adapter applied.")
model.print_trainable_parameters()


# =============================================================================
# SECCIÓN 7: TOKENIZACIÓN Y FORMATO DE PROMPT
# =============================================================================
# Formato ChatML de Qwen3 via apply_chat_template(enable_thinking=False).
# Mensaje usuario: f"{instruction}\n\n<context>{ctx}</context>"
#   - identico al de generar_respuesta() en chat_pdfs.py.
# Perdida enmascarada en tokens del prompt (-100): solo aprende la respuesta.
# =============================================================================

def format_and_tokenize(examples):
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


print("\n--> [7] Tokenizando datasets...")
tokenized_train = dataset.map(
    format_and_tokenize, batched=True, batch_size=1000,
    remove_columns=dataset.column_names, desc="Tokenising train",
).filter(lambda x: len(x["input_ids"]) > 0)

tokenized_eval = eval_dataset_raw.map(
    format_and_tokenize, batched=True, batch_size=1000,
    remove_columns=eval_dataset_raw.column_names, desc="Tokenising val",
).filter(lambda x: len(x["input_ids"]) > 0)

print(f"--> Train: {len(tokenized_train)} | Val: {len(tokenized_eval)} tokenised samples")


# =============================================================================
# SECCIÓN 8: CONFIGURACIÓN DEL ENTRENAMIENTO
# =============================================================================
# LR 5e-5 cosine (1e-4 causo colapso en v6). Batch efectivo 16.
# load_best_model_at_end=True: exporta el checkpoint con menor eval_loss.
# =============================================================================

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

print("\n--> [8] Inicializando Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)


# =============================================================================
# SECCIÓN 9: BUCLE DE ENTRENAMIENTO
# =============================================================================

print("\n--> [9] Iniciando entrenamiento...")
print(f"    Epochs:            {training_args.num_train_epochs}")
print(f"    Batch efectivo:    {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"    LR:                {training_args.learning_rate}  (cosine, patience=5)")
print(f"    Mejor checkpoint: load_best_model_at_end=True")
trainer.train()


# =============================================================================
# SECCIÓN 10: EXPORTACIÓN DEL MODELO
# =============================================================================
# Se guarda el checkpoint de menor eval_loss (no el ultimo paso).
# =============================================================================

print(f"\n--> [10] Guardando modelo adaptado en {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("--> Adapter + tokenizer saved.")


# =============================================================================
# SECCIÓN 11: EVALUACIÓN DEL MODELO ADAPTADO
# =============================================================================
# Mismo dict congelado que la sección 5 -> comparacion objetiva BASE vs ADAPTADO.
# =============================================================================

print("\n" + "=" * 70)
print("--> [11] Evaluando modelo ADAPTADO")
print("    (Mismo test set congelado que la seccion 5)")
print("=" * 70)

model.gradient_checkpointing_disable()

adapted_metrics, adapted_results = evaluate_on_datasets(
    model, tokenizer, eval_datasets, label="ADAPTED"
)


# =============================================================================
# SECCIÓN 12: RESUMEN COMPARATIVO BASE vs ADAPTADO
# =============================================================================
# Deltas por dataset + agregado ponderado por numero de muestras.
# =============================================================================

print("\n" + "=" * 70)
print("RESUMEN COMPARATIVO: BASE vs ADAPTADO")
print("=" * 70)

METRIC_KEYS = [
    ("Token_F1",               "Token F1",               "%"),
    ("Context_Faithfulness_Pct","Context Faithfulness",   "%"),
    ("Avg_Response_Length_Words","Avg Response Length",   "words"),
    ("Sentence_Completeness_Pct","Sentence Completeness", "%"),
]

comparison  = {"per_dataset": {}, "aggregate": {}}
agg_base_f1 = agg_adapted_f1 = agg_base_faith = agg_adapted_faith = 0.0
agg_n = 0

for ds_name in eval_datasets:
    b = base_metrics[ds_name]
    a = adapted_metrics[ds_name]
    n = b["n_samples"]
    agg_n += n

    agg_base_f1      += b["Token_F1"]               * n
    agg_adapted_f1   += a["Token_F1"]               * n
    agg_base_faith   += b["Context_Faithfulness_Pct"] * n
    agg_adapted_faith += a["Context_Faithfulness_Pct"] * n

    print(f"\n  {ds_name} ({n} samples):")
    for key, label, unit in METRIC_KEYS:
        delta = a[key] - b[key]
        sign  = "+" if delta >= 0 else ""
        print(f"    {label:26s}  Base={b[key]:.2f}{unit}  "
              f"Adapted={a[key]:.2f}{unit}  Δ={sign}{delta:.2f}{unit}")

    deltas = {key: round(a[key] - b[key], 2) for key, *_ in METRIC_KEYS}
    comparison["per_dataset"][ds_name] = {
        "base": b, "adapted": a, "deltas": deltas,
        "sample_pairs": [
            {
                "instruction":       br["instruction"],
                "ground_truth":      br["ground_truth"],
                "base_prediction":   br["prediction"],
                "adapted_prediction": ar["prediction"],
                "base_f1":           br["f1"],
                "adapted_f1":        ar["f1"],
                "base_faithfulness": br["faithfulness"],
                "adapted_faithfulness": ar["faithfulness"],
            }
            for br, ar in zip(base_results[ds_name][:5], adapted_results[ds_name][:5])
        ],
    }

if agg_n > 0:
    agg_b_f1    = agg_base_f1    / agg_n
    agg_a_f1    = agg_adapted_f1 / agg_n
    agg_b_faith = agg_base_faith / agg_n
    agg_a_faith = agg_adapted_faith / agg_n

    comparison["aggregate"] = {
        "Total_Samples":          agg_n,
        "Base_Token_F1":          round(agg_b_f1,    2),
        "Adapted_Token_F1":       round(agg_a_f1,    2),
        "Delta_Token_F1":         round(agg_a_f1  - agg_b_f1,    2),
        "Base_Faithfulness":      round(agg_b_faith, 2),
        "Adapted_Faithfulness":   round(agg_a_faith, 2),
        "Delta_Faithfulness":     round(agg_a_faith - agg_b_faith, 2),
    }

    print(f"\n{'=' * 70}")
    print(f"AGREGADO PONDERADO ({agg_n} muestras — mismas preguntas para ambos modelos)")
    print(f"  Token F1:            Base={agg_b_f1:.2f}%  → Adapted={agg_a_f1:.2f}%"
          f"   Δ={agg_a_f1 - agg_b_f1:+.2f}pp")
    print(f"  Context Faithfulness: Base={agg_b_faith:.2f}%  → Adapted={agg_a_faith:.2f}%"
          f"   Δ={agg_a_faith - agg_b_faith:+.2f}pp")
    print(f"{'=' * 70}")


# =============================================================================
# SECCIÓN 13: GUARDADO DE ARTEFACTOS
# =============================================================================
# training_stats.json        — resumen completo + log_history + muestras
# evaluation_comparison.json — deltas por dataset + pares base/adaptado
# =============================================================================

training_summary = {
    "model_name":       model_name,
    "version":          "v7.2",
    "total_steps":      trainer.state.global_step,
    "dataset_size":     len(tokenized_train),
    "effective_batch":  training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
    "datasets":         [name for name, _, _ in train_loaders],
    "eval_loss":        None,
    "perplexity":       None,
    "comparison":       comparison,
    "log_history":      trainer.state.log_history,
}

print("\n--> Calculando eval loss final...")
try:
    ev = trainer.evaluate()
    if "eval_loss" in ev:
        training_summary["eval_loss"]    = ev["eval_loss"]
        training_summary["perplexity"]   = math.exp(ev["eval_loss"])
        print(f"    Eval Loss: {ev['eval_loss']:.4f}  "
              f"Perplexity: {training_summary['perplexity']:.2f}")
except Exception as e:
    print(f"    Warning: final eval failed: {e}")


print("\n--> Generando muestras cualitativas de produccion...")
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
        "description": "[EN] Technical RAG — analytical answer expected",
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
        "description": "[ES] QA histórica — respuesta elaborada esperada",
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
        "description": "[CA] QA tècnica — resposta elaborada en català esperada",
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

training_summary["generated_samples"] = gen_samples

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
print("--> PROCESO COMPLETADO")
print(f"    Adapted model:         {output_dir}")
print(f"    Training stats:        {os.path.join(output_dir, 'training_stats.json')}")
print(f"    Evaluation comparison: {os.path.join(output_dir, 'evaluation_comparison.json')}")
print(f"    Base eval backup:      {base_eval_path}")
print("=" * 70)
