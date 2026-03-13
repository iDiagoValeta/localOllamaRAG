"""
eval_bertscore.py — Evaluación con BERTScore de modelos RAG (TFG)
==================================================================

Evalúa los 3 modelos (o uno concreto) — tanto la versión base como la
versión fine-tuneada — empleando BERTScore como métrica principal de
similitud semántica frente al ground truth.

BERTScore mide la similitud contextual entre predicción y referencia
utilizando embeddings BERT, capturando paráfrasis y equivalencias
semánticas que el Token F1 estándar (overlap léxico) no detecta.

Métricas reportadas por muestra y agregadas:
  - BERTScore Precision (P)  — precisión semántica del texto generado.
  - BERTScore Recall    (R)  — cobertura semántica de la referencia.
  - BERTScore F1        (F1) — media armónica de P y R.

Datasets evaluados (mismos test sets congelados que los scripts de training):
  - Neural-Bridge RAG   (200 muestras)
  - Dolly QA            (200 muestras)
  - Aina RAG            (200 muestras)

Pipeline:
  [1] Cargar datasets (mismos splits y filtros que train-*.py).
  [2] Para cada modelo seleccionado:
      a) Cargar modelo base → generar predicciones → BERTScore.
      b) Cargar modelo base + adaptador LoRA → generar predicciones → BERTScore.
  [3] Guardar resultados en JSON y CSV resumen.
  [4] Generar gráficas comparativas.

Uso:
    python eval_bertscore.py                       # Evalúa los 3 modelos
    python eval_bertscore.py --model gemma-3       # Solo gemma-3
    python eval_bertscore.py --model llama-3       # Solo llama-3

Salida (en training-output/<model>/):
    bertscore_results.json    — resultados detallados por muestra
    plots/eval/bertscore_comparison.png
"""

# =============================================================================
# CRÍTICO: deshabilitar triton/torch.compile ANTES de cualquier import de torch
# o transformers para evitar el crash de torchao en el clúster (Python.h)
# =============================================================================
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
# La versión interna de sent_encode llama a tokenizer.encode() usando
# tokenizer.model_max_length como max_length. Para DeBERTa ese valor puede
# ser sys.maxsize (≈2^63), lo que hace que el backend Rust de `tokenizers`
# lance OverflowError al intentar almacenarlo en un int de 32 bits.
# El parche fuerza max_length=512 (límite físico de DeBERTa) y truncation=True.
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

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN
# =============================================================================

EVAL_SAMPLES_PER_DATASET = 200
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


# =============================================================================
# SECCIÓN 2: CLI
# =============================================================================

parser = argparse.ArgumentParser(
    description="Evaluación BERTScore de modelos RAG (base vs. fine-tuned).",
)
parser.add_argument(
    "--model", choices=VALID_MODELS, default=None,
    help="Modelo a evaluar. Si se omite, evalúa los 3.",
)
parser.add_argument(
    "--output-dir", default=None,
    help="Directorio raíz de salida de entrenamiento (por defecto: cwd).",
)
parser.add_argument(
    "--bertscore-model", default=BERTSCORE_MODEL,
    help=f"Modelo BERT para BERTScore (default: {BERTSCORE_MODEL}).",
)
parser.add_argument(
    "--batch-size", type=int, default=32,
    help="Batch size para BERTScore (default: 32).",
)
args = parser.parse_args()

OUTPUT_ROOT = args.output_dir or os.getcwd()
BERTSCORE_MODEL = args.bertscore_model
models_to_eval = [args.model] if args.model else VALID_MODELS


# =============================================================================
# SECCIÓN 3: CARGA DE DATASETS (MISMOS SPLITS QUE TRAINING)
# =============================================================================
# Reutiliza la misma lógica de filtrado y normalización que train-*.py
# para garantizar que el test set congelado sea idéntico.
# =============================================================================

def _normalize_nb(example):
    return {
        "instruction": (example.get("question") or "").strip(),
        "context":     (example.get("context")  or "").strip(),
        "response":    (example.get("answer")   or "").strip(),
    }


def _normalize_dolly(example):
    return {
        "instruction": (example.get("instruction") or "").strip(),
        "context":     (example.get("context")     or "").strip(),
        "response":    (example.get("response")    or "").strip(),
        "category":    (example.get("category")    or "").strip(),
    }


def _normalize_aina(example):
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


def _filter_long_response(ex, min_words=15):
    return len(ex["response"].split()) >= min_words


def _filter_dolly_rag(ex):
    return (
        ex["category"] in DOLLY_RAG_CATEGORIES
        and bool(ex["context"].strip())
    )


def load_eval_datasets() -> dict:
    """Carga los 3 test sets congelados (idénticos a los de train-*.py)."""
    print("\n--> Cargando datasets de evaluación...")

    # Neural-Bridge RAG (test split)
    print("  Neural-Bridge RAG...")
    nb_test = (
        load_dataset("neural-bridge/rag-dataset-12000", split="test")
        .map(_normalize_nb, remove_columns=["context", "question", "answer"])
        .filter(_filter_valid)
        .shuffle(seed=42)
    )

    # Dolly QA (manual split: último 10%)
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

    # Aina RAG Multilingual (test split)
    print("  Aina RAG Multilingual...")
    _AINA_EXTRA_COLS = ["id", "category", "lang", "extractive"]
    ds = load_dataset("projecte-aina/RAG_Multilingual", split="test")
    cols_to_remove = [c for c in _AINA_EXTRA_COLS if c in ds.column_names]
    aina_test = (
        ds
        .map(_normalize_aina, remove_columns=cols_to_remove)
        .filter(_filter_valid)
        .filter(_filter_long_response)
        .shuffle(seed=42)
    )

    eval_datasets = {}
    for name, test_ds in [
        ("Neural-Bridge RAG", nb_test),
        ("Dolly QA",          dolly_test),
        ("Aina RAG",          aina_test),
    ]:
        n = min(EVAL_SAMPLES_PER_DATASET, len(test_ds))
        eval_datasets[name] = test_ds.select(range(n))
        print(f"    {name}: {n} muestras (FROZEN)")

    return eval_datasets


# =============================================================================
# SECCIÓN 4: GENERACIÓN DE RESPUESTAS (por modelo)
# =============================================================================
# Cada modelo usa su propio chat template y tokens de parada.
# =============================================================================

def _build_prompt(tokenizer, instruction, context, model_key):
    """Construye el prompt usando el chat template nativo del modelo."""
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
    """Devuelve los EOS token IDs específicos de cada modelo."""
    if model_key == "qwen-3":
        return [tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]]
    elif model_key == "llama-3":
        eot_id = tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]
        return list({tokenizer.eos_token_id, eot_id})
    elif model_key == "gemma-3":
        eot_id = tokenizer.encode("<end_of_turn>", add_special_tokens=False)[-1]
        return list({tokenizer.eos_token_id, eot_id})


def generate_response(model, tokenizer, instruction, context, model_key):
    """Genera una respuesta usando el formato nativo del modelo."""
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
    """Genera predicciones para todos los datasets. Retorna dict[ds → list[str]]."""
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


# =============================================================================
# SECCIÓN 5: CÁLCULO DE BERTSCORE
# =============================================================================

def compute_bertscore(predictions, ground_truths, batch_size=32):
    """
    Calcula BERTScore (P, R, F1) para cada dataset.

    Retorna:
        dict[ds_name → {
            "P": [float], "R": [float], "F1": [float],
            "avg_P": float, "avg_R": float, "avg_F1": float
        }]
    """
    results = {}
    for ds_name in predictions:
        preds = predictions[ds_name]
        refs  = ground_truths[ds_name]

        print(f"  Calculando BERTScore para {ds_name} ({len(preds)} muestras)...")
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


# =============================================================================
# SECCIÓN 6: MÉTRICAS AUXILIARES (Token F1, Faithfulness)
# =============================================================================
# Se incluyen para comparación directa con los resultados de train-*.py.
# =============================================================================

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(
        r'\b(a|an|the|el|la|los|las|un|una|unos|unas|les|els|uns|unes)\b',
        ' ', text
    )
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())


def compute_f1(prediction, ground_truth):
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
    pred_types = set(normalize_text(prediction).split())
    ctx_types  = set(normalize_text(context).split())
    if not pred_types:
        return 0.0
    return len(pred_types & ctx_types) / len(pred_types)


# =============================================================================
# SECCIÓN 7: GRÁFICAS
# =============================================================================

def plot_bertscore_comparison(all_results, output_path):
    """Genera gráfica comparativa de BERTScore F1 (base vs adapted) por modelo y dataset."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib no disponible — se omiten las gráficas.")
        return

    models_in = list(all_results.keys())
    datasets  = ["Neural-Bridge RAG", "Dolly QA", "Aina RAG"]
    ds_short  = ["Neural-Bridge", "Dolly QA", "Aina RAG"]

    BASE_COLOR  = "#64748b"
    ADAPT_COLOR = "#2563eb"
    DELTA_COLOR = "#16a34a"
    NEG_COLOR   = "#dc2626"

    n_models = len(models_in)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)

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
    print(f"  Gráfica guardada: {output_path}")


def plot_bertscore_aggregate(all_results, output_path):
    """Gráfica agregada de BERTScore P/R/F1 por modelo."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    models_in = list(all_results.keys())
    datasets  = ["Neural-Bridge RAG", "Dolly QA", "Aina RAG"]

    BASE_COLOR  = "#64748b"
    ADAPT_COLOR = "#2563eb"
    metrics = ["avg_P", "avg_R", "avg_F1"]
    metric_labels = ["Precision", "Recall", "F1"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)

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

    fig.suptitle("BERTScore agregado — Base vs. Fine-tuned por modelo",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfica guardada: {output_path}")


# =============================================================================
# SECCIÓN 8: PIPELINE PRINCIPAL
# =============================================================================

def get_adapter_path(model_key):
    """
    Resuelve la ruta del adaptador LoRA.
    Compatible con la estructura del clúster (training-output-<modelo>)
    y con la estructura local (training-output/<modelo>/).
    """
    # Estructura del clúster: training-output-gemma, training-output-llama, etc.
    cluster_names = {
        "qwen-3":  "training-output-qwen",
        "llama-3": "training-output-llama",
        "gemma-3": "training-output-gemma",
    }
    cluster_path = os.path.join(OUTPUT_ROOT, cluster_names[model_key])
    if os.path.exists(os.path.join(cluster_path, "adapter_config.json")):
        return cluster_path

    # Estructura local: training-output/<modelo>/
    local_path = os.path.join(OUTPUT_ROOT, "training-output", model_key)
    if os.path.exists(os.path.join(local_path, "adapter_config.json")):
        return local_path

    raise FileNotFoundError(
        f"No se encontró el adaptador LoRA para {model_key}.\n"
        f"  Buscado en: {cluster_path}\n"
        f"           y: {local_path}"
    )


def get_results_dir(model_key):
    """Resuelve el directorio de salida para resultados."""
    # Misma lógica de detección que get_adapter_path
    cluster_names = {
        "qwen-3":  "training-output-qwen",
        "llama-3": "training-output-llama",
        "gemma-3": "training-output-gemma",
    }
    cluster_path = os.path.join(OUTPUT_ROOT, cluster_names[model_key])
    if os.path.isdir(cluster_path):
        return cluster_path
    return os.path.join(OUTPUT_ROOT, "training-output", model_key)


def evaluate_model(model_key, eval_datasets, batch_size=32):
    """Evalúa un modelo completo (base + adapted) con BERTScore."""
    cfg = MODEL_CONFIGS[model_key]
    hf_name = cfg["hf_name"]
    adapter_path = get_adapter_path(model_key)
    results_dir  = get_results_dir(model_key)

    print("\n" + "=" * 70)
    print(f"  MODELO: {cfg['display']}  ({hf_name})")
    print(f"  Adaptador: {adapter_path}")
    print("=" * 70)

    # --- Cargar modelo base ---
    print(f"\n--> Cargando modelo base: {hf_name}")
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

    # --- Predicciones BASE ---
    print("\n--> [BASE] Generando predicciones...")
    base_preds, ground_truths = generate_all_predictions(
        model, tokenizer, eval_datasets, model_key, label="BASE"
    )
    print("\n--> [BASE] Calculando BERTScore...")
    base_bertscore = compute_bertscore(base_preds, ground_truths, batch_size)

    # Métricas auxiliares (Token F1, Faithfulness) para comparación
    base_aux = {}
    for ds_name, ds in eval_datasets.items():
        f1s, faiths = [], []
        for pred, ex in zip(base_preds[ds_name], ds):
            f1s.append(compute_f1(pred, ex["response"]))
            faiths.append(compute_context_faithfulness(pred, ex["context"]))
        base_aux[ds_name] = {
            "Token_F1": round(sum(f1s) / len(f1s) * 100, 2),
            "Faithfulness": round(sum(faiths) / len(faiths) * 100, 2),
        }

    # --- Cargar adaptador LoRA ---
    print(f"\n--> Aplicando adaptador LoRA desde: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    print("--> Adaptador fusionado.")

    # --- Predicciones ADAPTED ---
    print("\n--> [ADAPTED] Generando predicciones...")
    adapted_preds, _ = generate_all_predictions(
        model, tokenizer, eval_datasets, model_key, label="ADAPTED"
    )
    print("\n--> [ADAPTED] Calculando BERTScore...")
    adapted_bertscore = compute_bertscore(adapted_preds, ground_truths, batch_size)

    # Métricas auxiliares adaptadas
    adapted_aux = {}
    for ds_name, ds in eval_datasets.items():
        f1s, faiths = [], []
        for pred, ex in zip(adapted_preds[ds_name], ds):
            f1s.append(compute_f1(pred, ex["response"]))
            faiths.append(compute_context_faithfulness(pred, ex["context"]))
        adapted_aux[ds_name] = {
            "Token_F1": round(sum(f1s) / len(f1s) * 100, 2),
            "Faithfulness": round(sum(faiths) / len(faiths) * 100, 2),
        }

    # --- Liberar GPU ---
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- Montar resultado ---
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
        # Guardar primeras 10 muestras con detalle para inspección
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

    # --- Guardar JSON ---
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "bertscore_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"\n--> Resultados guardados: {json_path}")

    # --- Resumen ---
    print(f"\n{'='*70}")
    print(f"  RESUMEN BERTScore — {cfg['display']}")
    print(f"{'='*70}")
    for ds_name in datasets_list:
        d = result["per_dataset"][ds_name]
        print(f"\n  {ds_name} ({d['n_samples']} muestras):")
        print(f"    Base     → BERTScore F1: {d['base']['BERTScore_F1']:.2f}%  |  Token F1: {d['base']['Token_F1']:.2f}%")
        print(f"    Adapted  → BERTScore F1: {d['adapted']['BERTScore_F1']:.2f}%  |  Token F1: {d['adapted']['Token_F1']:.2f}%")
        print(f"    Δ          BERTScore F1: {d['deltas']['BERTScore_F1']:+.2f}   |  Token F1: {d['deltas']['Token_F1']:+.2f}")

    return result


# =============================================================================
# SECCIÓN 9: MAIN
# =============================================================================

def main():
    eval_datasets = load_eval_datasets()

    all_results = {}
    for model_key in models_to_eval:
        result = evaluate_model(model_key, eval_datasets, batch_size=args.batch_size)
        all_results[model_key] = result

    # --- Gráficas (solo si hay al menos 1 modelo evaluado) ---
    if all_results:
        # Gráfica por modelo y dataset
        first_key = list(all_results.keys())[0]
        results_dir = get_results_dir(first_key)
        plots_dir = os.path.join(results_dir, "plots", "eval")
        os.makedirs(plots_dir, exist_ok=True)

        plot_bertscore_comparison(all_results, os.path.join(plots_dir, "bertscore_comparison.png"))

        if len(all_results) > 1:
            plot_bertscore_aggregate(all_results, os.path.join(plots_dir, "bertscore_aggregate.png"))

    # --- CSV resumen global ---
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

    csv_path = os.path.join(OUTPUT_ROOT, "bertscore_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"\n--> CSV resumen global: {csv_path}")

    print("\n" + "=" * 70)
    print("  EVALUACIÓN BERTSCORE COMPLETADA")
    print("=" * 70)


if __name__ == "__main__":
    main()
