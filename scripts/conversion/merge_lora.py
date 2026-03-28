"""
LoRA adapter merger for base model consolidation.

Loads a trained LoRA adapter and merges it with the original base model
to produce a single consolidated model ready for GGUF export. The merged
model and its tokenizer are saved together to preserve compatibility
for downstream conversion and quantization steps.

Usage:
    python scripts/conversion/merge_lora.py --model qwen-3
Dependencies:
    - torch
    - peft (PeftModel, PeftConfig)
    - transformers (AutoModelForCausalLM, AutoTokenizer)
"""


# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  CONFIGURATION
#  +-- 1. Imports and CLI args
#  +-- 2. Paths and artifact validation
#
#  PIPELINE
#  +-- 3. Adapter configuration loading
#  +-- 4. Base model and tokenizer loading
#  +-- 5. LoRA adapter merge
#  +-- 6. Merged model export
#
# ─────────────────────────────────────────────

import argparse
import os
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────
# SECTION 1: IMPORTS AND CLI ARGS
# ─────────────────────────────────────────────

# Accepts both HF_TOKEN and HUGGINGFACE_HUB_TOKEN for compatibility.
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None

# ─────────────────────────────────────────────
# SECTION 2: PATHS AND ARTIFACT VALIDATION
# ─────────────────────────────────────────────

VALID_MODELS = ["qwen-3", "llama-3", "gemma-3"]

parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model.")
parser.add_argument(
    "--model", choices=VALID_MODELS, default="qwen-3",
    help="Model to merge (default: qwen-3).",
)
args = parser.parse_args()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LORA_PATH   = os.path.join(PROJECT_ROOT, "training-output", args.model)
MERGED_PATH = os.path.join(PROJECT_ROOT, "models", "merged-model", args.model)

if not os.path.exists(os.path.join(LORA_PATH, "adapter_config.json")):
    raise FileNotFoundError(
        f"LoRA adapter not found at {LORA_PATH}. "
        "Run training first (scripts/training/train-{args.model}.py)."
    )

os.makedirs(MERGED_PATH, exist_ok=True)

# ─────────────────────────────────────────────
# SECTION 3: ADAPTER CONFIGURATION LOADING
# ─────────────────────────────────────────────

print("=" * 60)
print("[1/4] Loading LoRA adapter configuration...")
print("=" * 60)
peft_config = PeftConfig.from_pretrained(LORA_PATH)
base_model_name = peft_config.base_model_name_or_path
print(f"  Base model: {base_model_name}")
print(f"  LoRA path:  {LORA_PATH}")

# ─────────────────────────────────────────────
# SECTION 4: BASE MODEL AND TOKENIZER LOADING
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("[2/4] Downloading/loading base model (may take a while)...")
print("=" * 60)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
    token=HF_TOKEN,
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=HF_TOKEN)
print("  Base model loaded.")

# ─────────────────────────────────────────────
# SECTION 5: LORA ADAPTER MERGE
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("[3/4] Merging LoRA adapter with base model...")
print("=" * 60)
model = PeftModel.from_pretrained(base_model, LORA_PATH, token=HF_TOKEN)
merged_model = model.merge_and_unload()
print("  Merge completed.")

# ─────────────────────────────────────────────
# SECTION 6: MERGED MODEL EXPORT
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"[4/4] Saving merged model to: {MERGED_PATH}")
print("=" * 60)
merged_model.save_pretrained(MERGED_PATH, safe_serialization=True)
tokenizer.save_pretrained(MERGED_PATH)

print("\n" + "=" * 60)
print("COMPLETED!")
print(f"Merged model saved to: {MERGED_PATH}")
print("=" * 60)
