"""
evaluate_baselines.py — Evaluación de modelos base para RAG (TFG) — v1.1
=========================================================================

Benchmark comparativo de 4 modelos base (sin fine-tuning) sobre 3 datasets
RAG, evaluados en dos modos (con contexto / sin contexto) y dos splits
(dev / test).  Cada modelo se carga y descarga secuencialmente para evitar
OOM en GPU.

Modelos evaluados:
  1. Qwen/Qwen3-14B              — razonador, thinking desactivado
  2. Qwen/Qwen3.5-9B             — razonador, thinking desactivado
  3. Qwen/Qwen2.5-14B-Instruct   — instruction-tuned estándar
  4. meta-llama/Llama-3.1-8B-Instruct — instruction-tuned estándar

Datasets:
  - neural-bridge/rag-dataset-12000       (EN, QA profesional)
  - databricks/databricks-dolly-15k       (EN, categorías RAG)
  - projecte-aina/RAG_Multilingual        (EN/ES/CA, multilingüe)

Métricas (por dataset + agregado ponderado):
  - Token F1 (SQuAD-standard)
  - Context Faithfulness (%)
  - Avg Response Length (words)
  - Sentence Completeness (%)

Salida:
  baseline-evaluation-output/baseline_evaluation.json
  baseline-evaluation-output/baseline_evaluation_samples.json

Uso:
    python evaluate_baselines.py
"""

# =============================================================================
# MAPA DEL MÓDULO — Índice de secciones
# =============================================================================
#
#  CONFIGURACIÓN
#  `-- 1. Entorno y constantes       CUDA env, modelos, caps, system prompts
#
#  EVALUACIÓN Y MÉTRICAS
#  `-- 2. Métricas e inferencia
#           2.1 normalize_text()          normalización EN/ES/CA
#           2.2 compute_f1()              Token F1 (SQuAD-standard)
#           2.3 compute_context_faithfulness()  métrica primaria del TFG
#           2.4 generate_response()       inferencia con chat template
#                                         (Qwen3/3.5: enable_thinking=False;
#                                          Qwen2.5/Llama: template estándar)
#           2.5 evaluate_on_datasets()    bucle sobre eval_datasets
#
#  DATOS
#  `-- 3. Carga de datasets (UNA sola vez, antes de cargar modelos)
#           3.1 Normalizadores: _normalize_nb, _normalize_dolly, _normalize_aina
#           3.2 Filtros: _filter_valid, _filter_long_response, _filter_dolly_rag
#           3.3 Neural-Bridge RAG      dev (train tail) + test (natural split)
#           3.4 Dolly QA               dev/test (manual 80/10/10)
#           3.5 Aina RAG Multilingual  dev (validation) + test (natural split)
#           3.6 Diccionarios congelados: eval_datasets_dev, eval_datasets_test
#
#  PIPELINE
#  |-- 4. Bucle principal   4 modelos × 2 modos × 2 splits (secuencial)
#  `-- 5. Tabla resumen + guardado de baseline_evaluation.json
#
# =============================================================================

import gc
import os
import re
import json
import torch
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# SECCIÓN 1: ENTORNO Y CONSTANTES
# =============================================================================
# Variables de entorno CUDA, lista de modelos, caps de evaluación, tokens
# máximos y system prompts (con/sin contexto).
# =============================================================================

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["TRITON_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

output_dir = os.path.join(os.getcwd(), "baseline-evaluation-output")
os.makedirs(output_dir, exist_ok=True)

# ── Modelos a evaluar (secuencialmente, uno a la vez para evitar OOM) ────────
# Qwen3 y Qwen3.5 son razonadores: se desactiva thinking para RAG.
# Qwen2.5-Instruct y Llama-3.2-Instruct usan chat template estándar.
MODELS = [
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen2.5-14B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]

# ── Caps por split ───────────────────────────────────────────────────────────
EVAL_SAMPLES_DEV  = 150   # por dataset, split de validación
EVAL_SAMPLES_TEST = 200   # por dataset, split de test (congelado)

# ── Tokens ───────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS      = 2048
EVAL_MAX_NEW_TOKENS = 2048
MAX_CONTEXT_TOKENS  = 2048

# ── System prompt (idéntico al de train.py v7.2) ────────────────────────────
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

# System prompt para modo sin contexto (sin referencia a <context> tags)
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


# =============================================================================
# SECCIÓN 2: MÉTRICAS E INFERENCIA
# =============================================================================
# Cuatro métricas RAG sin dependencias externas.
#
# Primaria (evidencia principal del TFG):
#   Context Faithfulness — % de tipos de token de la respuesta que aparecen
#     en el contexto. Diferencia entre with_context y without_context
#     demuestra cuánto se apoya el modelo en el documento proporcionado.
#
# Secundarias:
#   Token F1              — overlap con gold answer, estándar SQuAD.
#   Avg Response Length   — detecta cambios de verbosidad entre modelos.
#   Sentence Completeness — detecta respuestas fragmentadas.
# =============================================================================

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
    """
    pred_types = set(normalize_text(prediction).split())
    ctx_types  = set(normalize_text(context).split())
    if not pred_types:
        return 0.0
    return len(pred_types & ctx_types) / len(pred_types)


def generate_response(
    model, tokenizer, instruction: str, context: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    model_name: str = "",
) -> str:
    """
    Inference with chat template. Handles model-specific differences:
      - Qwen3-14B / Qwen3.5-9B: uses enable_thinking=False to disable
        reasoning mode (razonadores inapropiados para RAG directo).
        EOS token: <|im_end|>.
      - Qwen2.5-14B-Instruct: standard apply_chat_template.
        EOS token: <|im_end|>.
      - Llama-3.2-11B-Instruct: standard apply_chat_template.
        EOS token: tokenizer.eos_token_id nativo.

    Cuando with_context es False (llamado desde evaluate_on_datasets con
    context=""), se usa SYSTEM_PROMPT_NO_CONTEXT que no menciona <context>
    tags para no confundir al modelo.
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

    # Determine EOS token — Qwen uses <|im_end|>, Llama uses tokenizer.eos_token_id
    if is_qwen3 or "Qwen" in model_name:
        eos_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
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
    """
    Evaluate a model on the provided eval_datasets dict.

    Args:
        with_context: If True, passes context to the model. If False,
                      passes context="" (knowledge-only mode).

    Returns:
        all_metrics  dict[ds_name → metric dict]
        all_results  dict[ds_name → list of per-sample dicts]
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
        all_metrics[ds_name] = metrics
        all_results[ds_name] = results

        faith_note = "" if with_context else "  [N/A — no context]"
        print(f"\n  {ds_name} ({n} samples):")
        print(f"    Token F1:               {metrics['Token_F1']:.2f}%")
        print(f"    Context Faithfulness:   {metrics['Context_Faithfulness_Pct']:.2f}%{faith_note}")
        print(f"    Avg Response Length:    {metrics['Avg_Response_Length_Words']:.1f} words")
        print(f"    Sentence Completeness:  {metrics['Sentence_Completeness_Pct']:.1f}%")

    return all_metrics, all_results


# =============================================================================
# SECCIÓN 3: CARGA DE DATASETS (una sola vez, antes de cargar modelos)
# =============================================================================
# Los splits se cargan y congelan aquí. Se reutilizan para TODOS los modelos.
# =============================================================================

print("\n" + "=" * 70)
print("SECCIÓN 3: Cargando y congelando datasets")
print("=" * 70)


# ── Normalizadores ───────────────────────────────────────────────────────────

def _normalize_nb(example):
    """Neural-Bridge: question→instruction, context, answer→response."""
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
    """Aina RAG Multilingual: instruction/context/response (direct mapping)."""
    return {
        "instruction": (example.get("instruction") or "").strip(),
        "context":     (example.get("context")     or "").strip(),
        "response":    (example.get("response")    or "").strip(),
    }


# ── Filtros ──────────────────────────────────────────────────────────────────

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


# ── Neural-Bridge RAG ────────────────────────────────────────────────────────

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
# Take last EVAL_SAMPLES_DEV from train as dev (mirroring train.py val split)
nb_dev_start = nb_n - EVAL_SAMPLES_DEV
nb_dev = _nb_train_full.select(range(nb_dev_start, min(nb_dev_start + EVAL_SAMPLES_DEV, nb_n)))

# Test split: natural test split from HuggingFace
nb_test = (
    load_dataset("neural-bridge/rag-dataset-12000", split="test")
    .map(_normalize_nb, remove_columns=["context", "question", "answer"])
    .filter(_filter_valid)
    .filter(_filter_long_response)
    .shuffle(seed=42)
)
print(f"    dev={len(nb_dev)}, test={len(nb_test)}")


# ── Dolly QA ─────────────────────────────────────────────────────────────────

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


# ── Aina RAG Multilingual ────────────────────────────────────────────────────

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


_aina_val_full = _load_aina("validation")
aina_dev  = _aina_val_full.select(range(min(EVAL_SAMPLES_DEV, len(_aina_val_full))))
aina_test = _load_aina("test")
print(f"    dev={len(aina_dev)}, test={len(aina_test)}")


# ── Construir diccionarios de evaluación congelados ──────────────────────────

def _build_eval_dict(nb_ds, dolly_ds, aina_ds, cap: int, split_label: str) -> dict:
    """Build a frozen eval dict capping each dataset to `cap` samples."""
    out = {}
    for name, ds in [
        ("Neural-Bridge RAG", nb_ds),
        ("Dolly QA",          dolly_ds),
        ("Aina RAG",          aina_ds),
    ]:
        n = min(cap, len(ds))
        out[name] = ds.select(range(n))
        print(f"    {split_label} | {name}: {n} samples (FROZEN)")
    return out


print("\n  Building frozen evaluation splits:")
eval_datasets_dev  = _build_eval_dict(nb_dev, dolly_dev, aina_dev, EVAL_SAMPLES_DEV, "dev")
eval_datasets_test = _build_eval_dict(nb_test, dolly_test, aina_test, EVAL_SAMPLES_TEST, "test")

total_dev  = sum(len(d) for d in eval_datasets_dev.values())
total_test = sum(len(d) for d in eval_datasets_test.values())
print(f"\n--> Total dev samples:  {total_dev}")
print(f"--> Total test samples: {total_test}")


# =============================================================================
# SECCIÓN 4: BUCLE PRINCIPAL — 4 Modelos × 2 Modos × 2 Splits
# =============================================================================
# Orden: Qwen3-14B → Qwen3.5-9B → Qwen2.5-14B-Instruct → Llama-3.1-8B-Instruct
# Cada modelo se carga, evalúa en 4 combinaciones (modo × split), y se
# descarga con gc.collect() + torch.cuda.empty_cache() antes del siguiente.
# =============================================================================

all_results = {}

for model_idx, model_name in enumerate(MODELS, 1):
    print("\n" + "=" * 70)
    print(f"[{model_idx}/{len(MODELS)}] Evaluando modelo: {model_name}")
    print("=" * 70)

    # ── Cargar modelo ────────────────────────────────────────────────────────
    print(f"\n--> Cargando {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    print(f"--> Modelo cargado: {model_name}")

    model_results = {"with_context": {}, "without_context": {}}

    # ── Evaluar en ambos modos ───────────────────────────────────────────────
    for with_context in [True, False]:
        mode_key = "with_context" if with_context else "without_context"
        mode_label = "CON CONTEXTO" if with_context else "SIN CONTEXTO"

        print(f"\n{'─' * 60}")
        print(f"  Modo: {mode_label}")
        print(f"{'─' * 60}")

        mode_results = {}

        # ── Evaluar en ambos splits ──────────────────────────────────────────
        for split_name, eval_ds in [("dev", eval_datasets_dev), ("test", eval_datasets_test)]:
            print(f"\n  ── Split: {split_name.upper()} ──")

            metrics, results = evaluate_on_datasets(
                model, tokenizer, eval_ds,
                label=f"{model_name.split('/')[-1]}",
                with_context=with_context,
                model_name=model_name,
            )

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
            faith_note = "  [N/A — no context]" if not with_context else ""
            print(f"    Context Faithfulness: {aggregate['Context_Faithfulness_Pct']:.2f}%{faith_note}")
            print(f"    Avg Response Length:  {aggregate['Avg_Response_Length_Words']:.1f} words")

        model_results[mode_key] = mode_results

    all_results[model_name] = model_results

    # ── Descargar modelo y liberar memoria ───────────────────────────────────
    print(f"\n--> Descargando modelo {model_name} y liberando memoria GPU...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("--> GPU cache cleared.\n")


# =============================================================================
# SECCIÓN 5: TABLA RESUMEN Y GUARDADO
# =============================================================================
# Imprime tabla con agregados ponderados por modelo/split/modo.
# Guarda baseline_evaluation.json (completo) y _samples.json (solo agregados).
# =============================================================================

print("\n" + "=" * 70)
print("TABLA RESUMEN DE EVALUACIÓN DE BASELINES")
print("=" * 70)

# Table header
header = f"{'Modelo':<35s} | {'Split':<5s} | {'Modo':<12s} | {'Token_F1':>9s} | {'Ctx_Faith':>10s} | {'Avg_Len':>8s}"
print(header)
print("─" * len(header))

for model_name in MODELS:
    short_name = model_name.split("/")[-1]
    for mode_key, mode_label in [("with_context", "con contexto"), ("without_context", "sin contexto")]:
        for split_name in ["dev", "test"]:
            agg = all_results[model_name][mode_key][split_name].get("aggregate", {})
            f1      = agg.get("Token_F1", 0.0)
            faith   = agg.get("Context_Faithfulness_Pct", 0.0)
            avg_len = agg.get("Avg_Response_Length_Words", 0.0)

            faith_str = f"{faith:>9.2f}%" if mode_key == "with_context" else "   N/A    "

            print(f"{short_name:<35s} | {split_name:<5s} | {mode_label:<12s} | {f1:>8.2f}% | {faith_str} | {avg_len:>7.1f}w")

print("─" * len(header))

# ── Guardar JSON ─────────────────────────────────────────────────────────────

output_path = os.path.join(output_dir, "baseline_evaluation.json")
try:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"\n--> Resultados guardados: {output_path}")
except Exception as e:
    print(f"\n    Warning: no se pudo guardar el fichero: {e}")

# También guardar muestras compactas para inspección manual
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
    print(f"--> Resumen compacto guardado: {samples_path}")
except Exception as e:
    print(f"    Warning: no se pudo guardar el resumen compacto: {e}")

print("\n" + "=" * 70)
print("--> EVALUACIÓN DE BASELINES COMPLETADA")
print(f"    Resultados completos: {output_path}")
print(f"    Resumen compacto:     {samples_path}")
print("=" * 70)
