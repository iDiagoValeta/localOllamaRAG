"""
Script de entrenamiento LoRA para Qwen 2.5 14B (TFG)
=====================================================

Este archivo implementa un flujo end-to-end para:
1) Cargar y adaptar el modelo base `Qwen/Qwen2.5-14B-Instruct`.
2) Preparar el dataset multilingüe de RAG (`projecte-aina/RAG_Multilingual`).
3) Entrenar con enmascarado de pérdida sobre el prompt (solo aprende la respuesta).
4) Guardar artefactos, métricas y muestras cualitativas de generación.

El objetivo metodológico del TFG es entrenar un asistente académico que responda
de forma fundamentada en contexto, minimizando alucinaciones mediante un formato
de prompt estricto con etiqueta `<contexto>...</contexto>`.
"""

import os
import json
import torch
import bitsandbytes as bnb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN DEL ENTORNO Y HARDWARE
# =============================================================================
# Configuración de estabilidad para ejecución en GPU/HPC:
# - Desactiva rutas de compilación dinámica que pueden introducir inestabilidad.
# - Ajusta la política de reserva de memoria CUDA para reducir fragmentación.
# =============================================================================

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["TRITON_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

output_dir = os.path.join(os.getcwd(), "training-output")
os.makedirs(output_dir, exist_ok=True)


# =============================================================================
# SECCIÓN 2: CARGA DEL MODELO BASE (QWEN 2.5)
# =============================================================================
# Carga del modelo base y del tokenizador para fine-tuning eficiente con LoRA.
# Se utiliza BF16 + device_map automático + gradient checkpointing para optimizar
# el uso de memoria en entrenamiento de un modelo de 14B parámetros.
# =============================================================================

model_name = "Qwen/Qwen2.5-14B-Instruct"

print(f"--> [SECCIÓN 2] Cargando modelo base...")
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
print("--> Modelo listo.")
model.print_trainable_parameters()


# =============================================================================
# SECCIÓN 3: PROCESAMIENTO DE DATOS
# =============================================================================
# Preparación del dataset multilingüe para aprendizaje supervisado (SFT) con
# formato RAG: instrucción + contexto + respuesta. Se conservan solo campos
# esenciales, se filtran ejemplos inválidos y se tokeniza con enmascarado de loss
# para aprender únicamente la respuesta del asistente.
# =============================================================================

print("--> [SECCIÓN 3] Preparando dataset RAG_Multilingual...")

ds = load_dataset("projecte-aina/RAG_Multilingual")

COLS_TO_KEEP = ["instruction", "context", "response"]
cols_to_remove = [c for c in ds["train"].column_names if c not in COLS_TO_KEEP]
dataset = ds["train"].remove_columns(cols_to_remove)
eval_dataset_raw = ds["validation"].remove_columns(cols_to_remove)
dataset = dataset.filter(lambda x: (x["instruction"] or "").strip() and (x["response"] or "").strip())
eval_dataset_raw = eval_dataset_raw.filter(lambda x: (x["instruction"] or "").strip() and (x["response"] or "").strip())
print(f"--> Dataset: {len(dataset)} train, {len(eval_dataset_raw)} eval (ES/CA/EN)")
print(f"--> Columnas: {dataset.column_names}")

system_prompt = """Eres un asistente que responde preguntas basándote EXCLUSIVAMENTE en el contexto proporcionado.

REGLAS ESTRICTAS:
1. Responde SOLO con información que esté contenida en el contexto dentro de <contexto>...</contexto>.
2. No inventes, no uses conocimiento externo. Si la respuesta no está en el contexto, indícalo claramente.
3. Cita o parafrasea el material del contexto cuando sea útil.
4. Responde en el mismo idioma que use el usuario (español, catalán o inglés).
5. Sé claro, conciso y estructurado."""

MAX_LENGTH = 2048
MAX_CONTEXT_CHARS = 4000


def format_and_tokenize(examples):
    """
    Convierte un batch al formato causal de Qwen 2.5 para entrenamiento SFT-RAG.

    La función:
    1) Construye prompt estructurado con system/user/contexto usando tokens <|im_start|>/<|im_end|>.
    2) Tokeniza prompt y respuesta de forma independiente.
    3) Controla longitud máxima y descarta casos no válidos.
    4) Enmascara el prompt en `labels` con -100 para calcular pérdida solo en respuesta.

    Returns:
        dict con `input_ids`, `labels` y `attention_mask` para el Trainer.
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
        if len(ctx) > MAX_CONTEXT_CHARS:
            ctx = ctx[:MAX_CONTEXT_CHARS] + "..."

        user_msg = f"{instruction}\n\n<contexto>{ctx}</contexto>" if ctx else instruction

        prompt_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
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


print("--> Tokenizando dataset...")
tokenized_dataset = dataset.map(
    format_and_tokenize,
    batched=True,
    batch_size=1000,
    remove_columns=dataset.column_names,
    desc="Tokenizando",
)

original_len = len(tokenized_dataset)
tokenized_dataset = tokenized_dataset.filter(
    lambda x: len(x["input_ids"]) > 0,
    desc="Filtrando ejemplos válidos"
)
print(f"--> Dataset train: {len(tokenized_dataset)} ejemplos válidos de {original_len} originales")

print("--> Tokenizando dataset de evaluación (subconjunto de 500)...")
eval_dataset_raw = eval_dataset_raw.select(range(min(500, len(eval_dataset_raw))))
tokenized_eval = eval_dataset_raw.map(
    format_and_tokenize,
    batched=True,
    batch_size=1000,
    remove_columns=eval_dataset_raw.column_names,
    desc="Tokenizando eval",
)
tokenized_eval = tokenized_eval.filter(
    lambda x: len(x["input_ids"]) > 0,
    desc="Filtrando ejemplos válidos (eval)"
)
print(f"--> Dataset eval: {len(tokenized_eval)} ejemplos válidos")


# =============================================================================
# SECCIÓN 4: CONFIGURACIÓN DE ENTRENAMIENTO
# =============================================================================
# Definición de hiperparámetros finales utilizados en el entrenamiento del TFG.
# La ejecución se controla por `num_train_epochs=1` sobre el conjunto completo,
# con acumulación de gradiente para mantener batch efectivo alto sin exceder VRAM.
# =============================================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    gradient_checkpointing=True,
    bf16=True,
    tf32=True,
    optim="adamw_bnb_8bit",
    max_grad_norm=0.3,
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

print("--> [SECCIÓN 4] Iniciando Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    tokenizer=tokenizer,
)


# =============================================================================
# SECCIÓN 5: BUCLE DE ENTRENAMIENTO
# =============================================================================
# Ejecución del entrenamiento supervisado con evaluación periódica y checkpoints.
# =============================================================================

print("--> [SECCIÓN 5] Iniciando entrenamiento...")
print(f"    - Steps totales: {training_args.max_steps}")
print(f"    - Batch efectivo: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"    - Learning rate: {training_args.learning_rate}")

trainer.train()


# =============================================================================
# SECCIÓN 6: EXPORTACIÓN FINAL Y MÉTRICAS
# =============================================================================
# Persistencia de artefactos y métricas para reproducibilidad experimental:
# - Adaptador LoRA entrenado.
# - Tokenizador utilizado.
# - Historial de entrenamiento en JSON para análisis posterior.
# =============================================================================

print(f"--> [SECCIÓN 6] Guardando modelo final en {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

log_history_path = os.path.join(output_dir, "training_stats.json")
print(f"--> Guardando estadísticas en: {log_history_path}")

training_summary = {
    "total_steps": trainer.state.global_step,
    "final_loss": trainer.state.log_history[-1].get("loss") if trainer.state.log_history else None,
    "dataset_size": len(tokenized_dataset),
    "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
    "log_history": trainer.state.log_history,
}

try:
    with open(log_history_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=4, ensure_ascii=False)
except Exception as e:
    print(f"No se pudo guardar el log: {e}")

print("--> ¡Entrenamiento completado!")
print(f"    - Steps completados: {trainer.state.global_step}")
if training_summary["final_loss"]:
    print(f"    - Loss final: {training_summary['final_loss']:.4f}")


# =============================================================================
# SECCIÓN 7: EVALUACIÓN FINAL CON MUESTRAS
# =============================================================================
# Evaluación final en dos niveles:
# - Cuantitativo: `eval_loss` y perplejidad en validación.
# - Cualitativo: generación de muestras multilingües con contexto explícito.
# =============================================================================

print("\n" + "="*70)
print("--> [SECCIÓN 7] Evaluación final con generación de muestras")
print("="*70)

print("\n--> Calculando métricas en conjunto de evaluación...")
import math
eval_results = trainer.evaluate()
perplexity = math.exp(eval_results["eval_loss"]) if "eval_loss" in eval_results else None

print(f"    - Eval Loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
if perplexity:
    print(f"    - Perplexity: {perplexity:.2f}")

training_summary["eval_loss"] = eval_results.get("eval_loss")
training_summary["perplexity"] = perplexity

test_prompts = [
    {
        "instruction": "What aircraft will be available for use by the US Air Force in 2017?",
        "context": "The USAF's KC-135 and KC-10 aerial refueling aircraft are based on civilian jets. The KC-46A Pegasus is undergoing testing and is projected to be delivered to USAF units starting in 2017.",
        "description": "RAG EN: pregunta sobre texto recuperado"
    },
    {
        "instruction": "¿Qué avión estará disponible para la USAF en 2017?",
        "context": "Los aviones KC-135 y KC-10 de reabastecimiento aéreo de la USAF están basados en jets civiles. El KC-46A Pegasus está en pruebas y se prevé entregarlo a unidades de la USAF a partir de 2017.",
        "description": "RAG ES: pregunta con contexto en español"
    },
    {
        "instruction": "Quin avió estarà disponible per a la USAF el 2017?",
        "context": "Els avions KC-135 i KC-10 de reabastiment aeri de la USAF es basen en jets civils. El KC-46A Pegasus està en proves i es preveu lliurar-lo a unitats de la USAF a partir de 2017.",
        "description": "RAG CA: pregunta con contexto en catalán"
    },
]

print("\n--> Generando respuestas de prueba...")
print("-" * 70)

model.eval()

def generate_response(instruction: str, context: str = None, max_new_tokens: int = 256):
    """
    Genera respuesta de inferencia con el mismo formato de prompt usado
    durante entrenamiento, garantizando coherencia train/inference.

    Args:
        instruction: Pregunta del usuario.
        context: Contexto recuperado (opcional, recomendado en modo RAG).
        max_new_tokens: Límite máximo de tokens a generar.

    Returns:
        Texto de respuesta sin espacios extremos.
    """
    if context:
        user_msg = f"{instruction}\n\n<contexto>{context}</contexto>"
    else:
        user_msg = instruction

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False)[0],
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

generated_samples = []
for i, test in enumerate(test_prompts, 1):
    print(f"\n[Muestra {i}] {test['description']}")
    print(f"  Pregunta: {test['instruction'][:80]}{'...' if len(test['instruction']) > 80 else ''}")
    if test["context"]:
        print(f"  Contexto: {test['context'][:60]}...")

    try:
        response = generate_response(test["instruction"], test["context"])
        print(f"  Respuesta: {response[:300]}{'...' if len(response) > 300 else ''}")

        generated_samples.append({
            "instruction": test["instruction"],
            "context": test["context"],
            "response": response,
        })
    except Exception as e:
        print(f"  Error generando respuesta: {e}")

print("\n" + "-" * 70)

samples_path = os.path.join(output_dir, "generated_samples.json")
try:
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(generated_samples, f, indent=4, ensure_ascii=False)
    print(f"--> Muestras guardadas en: {samples_path}")
except Exception as e:
    print(f"Error guardando muestras: {e}")

try:
    with open(log_history_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=4, ensure_ascii=False)
except Exception as e:
    print(f"Error actualizando estadísticas: {e}")

print("\n" + "="*70)
print("--> ¡Proceso completo terminado!")
print(f"    - Modelo guardado en: {output_dir}")
print(f"    - Estadísticas: {log_history_path}")
print(f"    - Muestras: {samples_path}")
print("="*70)
