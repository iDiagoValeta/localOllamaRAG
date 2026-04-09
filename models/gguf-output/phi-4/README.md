---
language:
  - en
  - es
  - ca
license: mit
base_model:
  - microsoft/phi-4
tags:
  - rag
  - retrieval-augmented-generation
  - lora
  - phi4
  - multilingual
  - ollama
  - gguf
pipeline_tag: text-generation
---

# Phi-4 RAG (LoRA fine-tuned) — Q4_K_M GGUF

Quantized **GGUF** build of **Microsoft Phi-4** with a **LoRA** adapter merged in, for **retrieval-augmented question answering**. The model answers **only from supplied document context** in **English, Spanish, or Catalan**, using the same RAG-oriented system prompt as the **MonkeyGrab** project (TFG, Universitat Politècnica de València).

## Files in this repo

| File | Description |
|------|-------------|
| `Phi4-Q4_K_M.gguf` | Full weights after LoRA merge, **Q4_K_M** quantization (local inference, e.g. Ollama). |
| `Modelfile` | Ollama recipe: ChatML template, system prompt, sampling parameters. |
| `README.md` | This model card (mirrored in the source tree under `models/gguf-output/phi-4/`). |

## Base model and method

- **Base:** [`microsoft/phi-4`](https://huggingface.co/microsoft/phi-4) (ChatML-style; end-of-turn `<|redacted_im_end|>`).
- **Adaptation:** PEFT **LoRA** → merge into dense weights → **GGUF** export and **Q4_K_M** quantization via the project toolchain (`scripts/conversion/`, llama.cpp binaries).

### LoRA configuration

| Setting | Value |
|--------|--------|
| `r` | 32 |
| `lora_alpha` | 64 |
| `lora_dropout` | 0.05 |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| `bias` | `none` |

### Training (`scripts/training/train-phi4.py`, v1)

- **Seed:** 42.
- **Task format:** System + user message with instruction and `<context>...</context>`; **loss only on the assistant completion** (prompt labels masked).
- **Data (balanced 5-way interleaving, 3,200 train samples per source):**
  - Neural-Bridge RAG,
  - Dolly (categories: `closed_qa`, `information_extraction`, `summarization`), 80/10/10 split after filter,
  - Aina RAG — **EN**, **ES**, **CA**.
- **Sequence limits:** `max_length` 4096, context truncated to **2048** tokens; eval generation up to **2048** new tokens.
- **Optimizer / schedule:** AdamW 8-bit, **lr** 5e-5, **cosine** with **warmup_ratio** 0.05, **weight_decay** 0.01, **max_grad_norm** 1.0.
- **Batching:** `per_device_train_batch_size` 1, **gradient_accumulation_steps** 16 → **effective batch 16**; **bf16** + **TF32**; gradient checkpointing on.
- **Epochs:** 3; checkpoints every **300** steps (keep 3); eval every **150** steps; **load_best_model_at_end** on `eval_loss`; **early stopping** patience **5**.

### Evaluation protocol

- **Frozen dev/test** splits: identical for **base** (`microsoft/phi-4`) and **adapted** (LoRA merged) runs.
- **Dev:** 320 samples per dataset × 5 sources = **1,600** examples (aligned with `scripts/evaluation/evaluate_baselines.py` / `training-output/baseline/` for cross-experiment comparability).
- **Test:** **full** held-out splits (**8,490** examples total across sources).
- **Metrics:** Token F1, ROUGE-L F1, BERTScore F1 (`microsoft/deberta-xlarge-mnli`); BERTScore computed after unloading the generative model.
- **Artifacts:** `training-output/phi-4/evaluation_comparison.json` in the code repo.

## Evaluation results

Values are **percentage points** (0–100 scale). **Δ** = adapted − base; **Δ rel** = relative change vs base (%).

### Weighted aggregate (all five sources)

| Split | *N* | Metric | Base | Adapted | Δ (pp) | Δ rel (%) |
|-------|-----|--------|------|---------|--------|-----------|
| **Dev** | 1,600 | Token F1 | 45.17 | 60.24 | +15.07 | +33.36 |
| **Dev** | 1,600 | ROUGE-L F1 | 37.18 | 50.49 | +13.31 | +35.79 |
| **Dev** | 1,600 | BERTScore F1 | 39.59 | 53.48 | +13.89 | +35.07 |
| **Test** | 8,490 | Token F1 | 45.42 | 63.20 | +17.78 | +39.14 |
| **Test** | 8,490 | ROUGE-L F1 | 37.21 | 52.97 | +15.76 | +42.35 |
| **Test** | 8,490 | BERTScore F1 | 39.90 | 56.42 | +16.52 | +41.41 |

### Per-dataset **dev** (320 samples each)

| Dataset | Token F1 (B → A) | ROUGE-L F1 (B → A) | BERTScore F1 (B → A) |
|---------|------------------|---------------------|----------------------|
| Neural-Bridge RAG | 50.46 → **81.17** | 45.46 → **77.46** | 46.79 → **79.34** |
| Dolly QA | 44.46 → **50.95** | 38.21 → **45.51** | 38.88 → **46.24** |
| Aina-EN | 44.67 → **56.15** | 35.32 → **43.16** | 41.61 → **50.42** |
| Aina-ES | 40.47 → **57.11** | 31.44 → **43.37** | 33.35 → **45.66** |
| Aina-CA | 45.80 → **55.82** | 35.48 → **42.95** | 37.32 → **45.72** |

*B* = base Phi-4, *A* = adapted model (same harness). Full test breakdowns and qualitative pairs are in `evaluation_comparison.json`.

### Relation to the baseline benchmark

The **base** dev numbers plug into the same evaluation design as the multi-model benchmark under `training-output/baseline/` (`evaluate_baselines.py`, `predictions_phi-4.json`), so Phi-4 **before** LoRA is comparable to other models in that suite; **after** LoRA, use the **adapted** columns above.

## Hardware compatibility (inference)

| Setup | Guidance |
|-------|----------|
| **GPU (recommended)** | **~10 GB VRAM** is a practical minimum for this **Q4_K_M** ~14B-class GGUF in Ollama at moderate batching; **8 GB** may work with shorter context or slower offloading. |
| **Context length** | The bundled `Modelfile` sets **`num_ctx` 16384** — raising context increases VRAM/RAM use roughly linearly; reduce `num_ctx` if you hit OOM. |
| **CPU** | Supported by Ollama/llama.cpp-style runners, but **much slower** than a discrete GPU for this model size. |
| **Training** | LoRA training used **bf16**, gradient checkpointing, and 8-bit optimiser on a **CUDA** GPU (see `train-phi4.py`); that is separate from these inference notes. |

## Ollama

Put `Phi4-Q4_K_M.gguf` next to `Modelfile` (or set `FROM` to your local path). Then:

```bash
ollama create phi4-rag -f Modelfile
ollama run phi4-rag
```

Bundled generation defaults include `num_ctx` 16384, `temperature` 0.15, `top_p` 0.9, `repeat_penalty` 1.15 (see `Modelfile`).

## Limitations

- Intended for **grounded** QA; do not treat as unconstrained world-knowledge without retrieval.
- **Q4_K_M** is a speed/size trade-off vs higher bit-width or FP16.
- Quality depends on retrieval and on wrapping context in `<context>...</context>` as in training.

## License

- **MIT** — This model card, the `Modelfile`, and other metadata added by **nadiva1243** are released under the [MIT License](https://opensource.org/licenses/MIT) (see the `LICENSE` file in this repository).
- **Base weights** — The **GGUF** is a derivative of [`microsoft/phi-4`](https://huggingface.co/microsoft/phi-4). You must also comply with the **license and terms** of the base model and with any requirements of the **training datasets** you rely on when redistributing or using the weights.

## Citation

```bibtex
@misc{phi4_rag_gguf_monkeygrab,
  title        = {Phi-4 RAG LoRA Fine-tune (Q4_K_M GGUF)},
  author       = {nadiva1243},
  year         = {2026},
  howpublished = {Hugging Face: \url{https://huggingface.co/nadiva1243/phi4RAG}},
  note         = {Base: microsoft/phi-4; training: MonkeyGrab train-phi4.py v1}
}
```
