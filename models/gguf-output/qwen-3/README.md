---
language:
  - en
  - es
  - ca
license: mit
base_model:
  - Qwen/Qwen3-14B
tags:
  - rag
  - retrieval-augmented-generation
  - lora
  - qwen3
  - multilingual
  - ollama
  - gguf
pipeline_tag: text-generation
---

# Qwen3-14B RAG (LoRA fine-tuned) — Q4_K_M GGUF

Quantized **GGUF** build of **[Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)** with a **LoRA** adapter merged in, for **retrieval-augmented question answering**. The model answers **only from supplied document context** in **English, Spanish, or Catalan**, using the same RAG-oriented system prompt as **MonkeyGrab**, a local RAG stack developed for a **Bachelor's thesis (TFG) at the Universitat Politècnica de València (UPV)**.

## Source code, thesis, and contact

The **full MonkeyGrab application repository is not public yet** (defense / publication timeline). This Hugging Face model repo ships **inference assets** (when the GGUF file is present — see below), the **Ollama `Modelfile`**, and a **`reproduction/`** folder with frozen copies of the training script, merge utility, and **`evaluation_comparison.json`** so methodology and metrics remain auditable.

**Contact:** [nadiva1243@gmail.com](mailto:nadiva1243@gmail.com) for questions about training, evaluation, Ollama usage, or when the full repository will be released.

**Weights file:** the `Modelfile` expects **`Qwen3-14B-Q4_K_M.gguf`** next to it (`FROM ./Qwen3-14B-Q4_K_M.gguf`). The **Q4_K_M** build is published on this Hub repo (“Files” tab). For a local-only copy, build with `scripts/conversion/` (merge → `convert_hf_to_gguf.py` → `quantize_to_q4km.ps1`) or download from the Hub.

## Files in this repo

| File | Description |
|------|-------------|
| `Qwen3-14B-Q4_K_M.gguf` | Full weights after LoRA merge, **Q4_K_M** quantization (Ollama / llama.cpp). |
| `Modelfile` | Ollama recipe: Qwen3 chat template (`enable_thinking` off in template), RAG system prompt, generation parameters. |
| `README.md` | This model card (mirrored under `models/gguf-output/qwen-3/` when the codebase is public). |
| `LICENSE` | MIT — applies to model card, `Modelfile`, and files added here by nadiva1243. |
| `reproduction/train-qwen3.py` | Snapshot of `scripts/training/train-qwen3.py` (v10) used for this adapter. |
| `reproduction/merge_lora.py` | Snapshot of `scripts/conversion/merge_lora.py` (merge LoRA into dense weights before GGUF export). |
| `reproduction/evaluation_comparison.json` | Frozen eval export (base vs adapted, dev/test, per dataset + aggregate). |
| `reproduction/CONVERSION.md` | Short notes: merge → GGUF → quantization → Ollama. |

## Base model and method

- **Base:** [`Qwen/Qwen3-14B`](https://huggingface.co/Qwen/Qwen3-14B) (ChatML-style; thinking disabled in training and in the bundled Ollama template).
- **Adaptation:** PEFT **LoRA** → merge into dense weights → **GGUF** export and **Q4_K_M** quantization via **merge + llama.cpp** (see **`reproduction/`** on this Hub).

### LoRA configuration

| Setting | Value |
|--------|--------|
| `r` | 32 |
| `lora_alpha` | 64 |
| `lora_dropout` | 0.05 |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| `bias` | `none` |

### Training (`train-qwen3.py`, v10)

Balanced **5-way** interleaving (**3,200** train samples per source), frozen **dev** (**320** samples × 5 datasets) and **full test** splits, ChatML user messages with `<context>…</context>`, **loss only on the assistant completion**, **AdamW 8-bit**, cosine schedule with warmup, **bf16** + gradient checkpointing, early stopping on `eval_loss`. See **`reproduction/train-qwen3.py`** for exact hyperparameters and caps.

### Evaluation protocol

- **Metrics:** Token F1, ROUGE-L F1, BERTScore F1 (`microsoft/deberta-xlarge-mnli`), plus auxiliary context faithfulness / completeness where logged.
- **Artifacts:** **`reproduction/evaluation_comparison.json`** on this Hub.

## Evaluation results

Values are **percentage points** (0–100 scale). **Δ** = adapted − base; **Δ rel** = relative change vs base (%). Source: `reproduction/evaluation_comparison.json`.

### Weighted aggregate (all five sources)

| Split | *N* | Metric | Base | Adapted | Δ (pp) | Δ rel (%) |
|-------|-----|--------|------|---------|--------|-----------|
| **Dev** | 1,600 | Token F1 | 43.33 | 59.95 | +16.63 | +38.37 |
| **Dev** | 1,600 | ROUGE-L F1 | 36.62 | 49.81 | +13.19 | +36.02 |
| **Dev** | 1,600 | BERTScore F1 | 38.94 | 53.07 | +14.13 | +36.28 |
| **Test** | 8,490 | Token F1 | 42.99 | 62.46 | +19.47 | +45.29 |
| **Test** | 8,490 | ROUGE-L F1 | 36.13 | 51.88 | +15.75 | +43.58 |
| **Test** | 8,490 | BERTScore F1 | 38.83 | 55.61 | +16.78 | +43.20 |

### Per-dataset **dev** (320 samples each)

| Dataset | Token F1 (B → A) | ROUGE-L F1 (B → A) | BERTScore F1 (B → A) |
|---------|------------------|---------------------|----------------------|
| Neural-Bridge RAG | 60.38 → **79.55** | 54.65 → **75.85** | 56.04 → **77.60** |
| Dolly QA | 48.30 → **49.88** | 40.94 → **44.04** | 42.41 → **45.38** |
| Aina-EN | 36.53 → **56.91** | 30.15 → **43.48** | 35.62 → **51.86** |
| Aina-ES | 33.66 → **55.57** | 27.09 → **41.52** | 28.94 → **44.22** |
| Aina-CA | 37.77 → **57.86** | 30.26 → **44.14** | 31.70 → **46.29** |

*B* = base Qwen3-14B, *A* = adapted model (same harness).

### Relation to the baseline benchmark

Base dev numbers are aligned with the multi-model benchmark produced by `evaluate_baselines.py` (slug `Qwen3-14B`, `predictions_Qwen3-14B.json` in the future public `training-output/baseline/` tree). After LoRA, use the **adapted** columns above.

## Hardware compatibility (inference)

| Setup | Guidance |
|-------|----------|
| **GPU (recommended)** | **Q4_K_M** at ~14B class: plan for **~10–12 GB VRAM** in Ollama with moderate context; more if you raise `num_ctx`. |
| **Context length** | The bundled `Modelfile` sets **`num_ctx` 32768** — reduce if you hit OOM. |
| **CPU** | Supported by Ollama/llama.cpp-style runners, but much slower at this size. |
| **Training** | See **`reproduction/train-qwen3.py`** — **CUDA**, bf16, gradient checkpointing, 8-bit optimiser. |

## Ollama

Place **`Qwen3-14B-Q4_K_M.gguf`** next to **`Modelfile`** (or adjust `FROM`). Then:

```bash
ollama create qwen3-rag -f Modelfile
ollama run qwen3-rag
```

Defaults include `temperature` 0.15, `top_p` 0.9, `repeat_penalty` 1.15 (see `Modelfile`).

## Limitations

- Intended for **grounded** QA; do not treat as unconstrained world knowledge without retrieval.
- **Q4_K_M** is a size/speed trade-off vs higher bit-width.
- Quality depends on retrieval and on wrapping context in `<context>…</context>` as in training.

## License

- **MIT** — This model card, the `Modelfile`, and other metadata/scripts added here by **nadiva1243** are under the [MIT License](https://opensource.org/licenses/MIT) (see `LICENSE`).
- **Base weights** — The GGUF is a derivative of [`Qwen/Qwen3-14B`](https://huggingface.co/Qwen/Qwen3-14B). You must comply with the **license and terms** of the base model (see the upstream model card, typically **Apache-2.0**) and with any requirements of the **training datasets** when redistributing or using the weights.

## Citation

```bibtex
@misc{qwen3_rag_gguf_monkeygrab,
  title        = {Qwen3-14B RAG LoRA Fine-tune (Q4_K_M GGUF)},
  author       = {nadiva1243},
  year         = {2026},
  howpublished = {Hugging Face: \url{https://huggingface.co/nadiva1243/qwen3RAG}},
  note         = {Base: Qwen/Qwen3-14B; training: MonkeyGrab train-qwen3.py v10}
}
```
