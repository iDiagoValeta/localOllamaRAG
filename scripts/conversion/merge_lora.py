"""
Script de fusión LoRA + modelo base (TFG)
==========================================

Este script carga el adaptador LoRA entrenado y lo fusiona con el modelo base
original para generar una versión consolidada lista para exportación a GGUF.

Resultado:
- Modelo fusionado en `models/merged-model`.
- Tokenizador guardado junto al modelo para mantener compatibilidad.
"""

import argparse
import os
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# SECCIÓN 1: RUTAS Y VALIDACIÓN DE ARTEFACTOS
# =============================================================================
# Define rutas del proyecto y valida que exista el adaptador LoRA entrenado
# antes de iniciar el proceso de fusión.
# =============================================================================

VALID_MODELS = ["qwen-3", "llama-3"]

parser = argparse.ArgumentParser(description="Fusiona adaptador LoRA con modelo base.")
parser.add_argument(
    "--model", choices=VALID_MODELS, default="qwen-3",
    help="Modelo a fusionar (default: qwen-3).",
)
args = parser.parse_args()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LORA_PATH   = os.path.join(PROJECT_ROOT, "training-output", args.model)
MERGED_PATH = os.path.join(PROJECT_ROOT, "models", "merged-model", args.model)

if not os.path.exists(os.path.join(LORA_PATH, "adapter_config.json")):
    raise FileNotFoundError(
        f"No se encontró el adaptador LoRA en {LORA_PATH}. "
        "Ejecuta primero el entrenamiento (scripts/training/train-{args.model}.py)."
    )

os.makedirs(MERGED_PATH, exist_ok=True)

# =============================================================================
# SECCIÓN 2: CARGA DE CONFIGURACIÓN DEL ADAPTADOR
# =============================================================================
# Recupera desde el adaptador LoRA el identificador del modelo base utilizado
# durante entrenamiento, garantizando trazabilidad de la fusión.
# =============================================================================

print("=" * 60)
print("[1/4] Cargando configuración del adaptador LoRA...")
print("=" * 60)
peft_config = PeftConfig.from_pretrained(LORA_PATH)
base_model_name = peft_config.base_model_name_or_path
print(f"  Modelo base: {base_model_name}")
print(f"  Ruta LoRA:   {LORA_PATH}")

# =============================================================================
# SECCIÓN 3: CARGA DEL MODELO BASE Y TOKENIZADOR
# =============================================================================
# Carga el modelo base en CPU para fusionar pesos de forma determinista y
# sin requerir aceleración GPU en esta etapa de postproceso.
# =============================================================================

print("\n" + "=" * 60)
print("[2/4] Descargando/cargando modelo base (puede tardar)...")
print("=" * 60)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
print("  Modelo base cargado.")

# =============================================================================
# SECCIÓN 4: FUSIÓN DE ADAPTADOR LORA
# =============================================================================
# Inserta el adaptador sobre el modelo base y materializa los pesos finales
# mediante `merge_and_unload`.
# =============================================================================

print("\n" + "=" * 60)
print("[3/4] Fusionando adaptador LoRA con modelo base...")
print("=" * 60)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
merged_model = model.merge_and_unload()
print("  Fusión completada.")

# =============================================================================
# SECCIÓN 5: EXPORTACIÓN DEL MODELO FUSIONADO
# =============================================================================
# Guarda modelo y tokenizador en disco para etapas posteriores de conversión
# a GGUF y cuantización.
# =============================================================================

print("\n" + "=" * 60)
print(f"[4/4] Guardando modelo fusionado en: {MERGED_PATH}")
print("=" * 60)
merged_model.save_pretrained(MERGED_PATH, safe_serialization=True)
tokenizer.save_pretrained(MERGED_PATH)

print("\n" + "=" * 60)
print("¡COMPLETADO!")
print(f"Modelo fusionado guardado en: {MERGED_PATH}")
print("=" * 60)
