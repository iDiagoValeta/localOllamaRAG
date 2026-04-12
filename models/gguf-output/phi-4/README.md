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

Quantized **GGUF** build of **[microsoft/phi-4](https://huggingface.co/microsoft/phi-4)** with a **LoRA** adapter merged in, fine-tuned for **retrieval-augmented question answering**. The model answers **only from supplied document context** in **English, Spanish, or Catalan**, using the same RAG-oriented system prompt as **MonkeyGrab**, a local, fully private RAG stack developed for a **Bachelor's thesis (TFG) at the Universitat Politècnica de València (UPV)**.

## Source code, thesis, and contact

The **full MonkeyGrab application repository is not public yet** (defense / publication timeline). This Hugging Face model repo ships **inference assets** (`Phi4-Q4_K_M.gguf`), the **Ollama `Modelfile`**, and a **`reproduction/`** folder with frozen copies of the training script, merge utility, and **`evaluation_comparison.json`** so methodology and metrics remain auditable without requiring access to the full codebase.

**Contact:** [nadiva1243@gmail.com](mailto:nadiva1243@gmail.com) for questions about training, evaluation, Ollama usage, or when the full repository will be released.

**GGUF pipeline (high level):** LoRA fine-tuning on the datasets below → merge with `merge_lora.py` (see `reproduction/`) → GGUF export via the llama.cpp toolchain → **Q4_K_M** quantization. The merge script documents expected paths and flags.

## Files in this repo

| File | Description |
|------|-------------|
| `Phi4-Q4_K_M.gguf` | Full weights after LoRA merge, **Q4_K_M** quantization. |
| `Modelfile` | Ollama recipe: ChatML template, RAG system prompt, sampling parameters. |
| `README.md` | This model card. |
| `LICENSE` | MIT — applies to the model card, `Modelfile`, and files added here by nadiva1243 (not to Microsoft's base terms). |
| `reproduction/train-phi4.py` | Snapshot of `scripts/training/train-phi4.py` (v1) used for this adapter. |
| `reproduction/merge_lora.py` | Snapshot of `scripts/conversion/merge_lora.py` used to merge the LoRA weights into a dense checkpoint before GGUF export. |
| `reproduction/evaluation_comparison.json` | Frozen evaluation export (base vs. adapted, dev/test splits, per dataset + weighted aggregate). |
| `reproduction/CONVERSION.md` | Step-by-step notes: merge → GGUF → Q4_K_M quantization → Ollama import. |

## Base model and method

- **Base:** [`microsoft/phi-4`](https://huggingface.co/microsoft/phi-4) — 14B-parameter transformer (ChatML-style; end-of-turn token `<|im_end|>`).
- **Adaptation:** PEFT **LoRA** fine-tuning on five RAG-focused datasets → LoRA adapter merged into dense weights → **GGUF** export → **Q4_K_M** quantization.

### LoRA configuration

| Setting | Value |
|---------|-------|
| `r` | 32 |
| `lora_alpha` | 64 |
| `lora_dropout` | 0.05 |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| `bias` | `none` |

### Training (`train-phi4.py`, v1)

- **Seed:** 42 (propagates to torch / NumPy / CUDA via `set_seed`).
- **Task format:** ChatML `<|im_start|>user … <|im_end|>` with the instruction and `<context>…</context>` on the user turn; **loss computed only on the assistant completion** (prompt labels masked with `–100`).
- **Data — balanced 5-way interleaving (3,200 train samples per source, 16,000 total):**
  - [`neural-bridge/rag-dataset-12000`](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000)
  - [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (categories: `closed_qa`, `information_extraction`, `summarization`) — 80/10/10 split after filter
  - [`projecte-aina/RAG_Multilingual`](https://huggingface.co/datasets/projecte-aina/RAG_Multilingual) — **EN**, **ES**, **CA** subsets
- **Sequence limits:** `max_length` 4,096 tokens; context truncated to **2,048** tokens; generation up to **2,048** new tokens.
- **Optimizer / schedule:** AdamW 8-bit, **lr** 5e-5, **cosine** decay with **warmup_ratio** 0.05, **weight_decay** 0.01, **max_grad_norm** 1.0.
- **Batching:** `per_device_train_batch_size` 1, **gradient_accumulation_steps** 16 → **effective batch 16**; **bf16** + **TF32**; gradient checkpointing enabled.
- **Epochs:** 3; checkpoints saved every **300** steps (keep last 3); eval every **150** steps; **load_best_model_at_end** on `eval_loss`; **early stopping** patience **3** evaluations.

### Evaluation protocol

- **Frozen dev/test splits:** identical for the **base** (`microsoft/phi-4`) and the **adapted** (LoRA merged) model — no data leakage.
- **Dev:** 320 samples × 5 sources = **1,600 examples** (aligned with `evaluate_baselines.py` for cross-experiment comparability).
- **Test:** full held-out splits — **8,490 examples** total across all five sources.
- **Metrics:** Token F1, ROUGE-L F1, BERTScore F1 (`microsoft/deberta-xlarge-mnli`); BERTScore is computed after unloading the generative model to fit in GPU memory.
- **Artifacts:** all metric values and sample pairs are in `reproduction/evaluation_comparison.json`.

## Evaluation results

Values are **percentage points** (0–100 scale). **Δ (pp)** = adapted − base; **Δ rel (%)** = relative change vs. base.

### Weighted aggregate (all five sources)

| Split | *N* | Metric | Base | Adapted | Δ (pp) | Δ rel (%) |
|-------|-----|--------|------|---------|--------|-----------|
| **Dev** | 1,600 | Token F1 | 45.17 | 60.24 | +15.07 | +33.36 |
| **Dev** | 1,600 | ROUGE-L F1 | 37.18 | 50.49 | +13.31 | +35.79 |
| **Dev** | 1,600 | BERTScore F1 | 39.59 | 53.48 | +13.89 | +35.07 |
| **Test** | 8,490 | Token F1 | 45.42 | 63.20 | +17.78 | +39.14 |
| **Test** | 8,490 | ROUGE-L F1 | 37.21 | 52.97 | +15.76 | +42.35 |
| **Test** | 8,490 | BERTScore F1 | 39.90 | 56.42 | +16.52 | +41.41 |

### Per-dataset dev (320 samples each)

| Dataset | Token F1 (Base → Adapted) | ROUGE-L F1 (Base → Adapted) | BERTScore F1 (Base → Adapted) |
|---------|--------------------------|------------------------------|-------------------------------|
| Neural-Bridge RAG | 50.46 → **81.17** | 45.46 → **77.46** | 46.79 → **79.34** |
| Dolly QA | 44.46 → **50.95** | 38.21 → **45.51** | 38.88 → **46.24** |
| Aina-EN | 44.67 → **56.15** | 35.32 → **43.16** | 41.61 → **50.42** |
| Aina-ES | 40.47 → **57.11** | 31.44 → **43.37** | 33.35 → **45.66** |
| Aina-CA | 45.80 → **55.82** | 35.48 → **42.95** | 37.32 → **45.72** |

Full test-split breakdowns and qualitative sample pairs are in `reproduction/evaluation_comparison.json`.

### Relation to the baseline benchmark

The **base** dev numbers are aligned with the multi-model benchmark (`evaluate_baselines.py`, `predictions_phi-4.json`), so Phi-4 **before** fine-tuning is directly comparable to the other models in that suite. For post-LoRA performance, use the **Adapted** columns above.

## Hardware compatibility (inference)

| Setup | Notes |
|-------|-------|
| **GPU (recommended)** | **~10 GB VRAM** is a practical minimum for this **Q4_K_M** ~14B-class GGUF in Ollama at moderate batching; **8 GB** may work with shorter context or with slower GPU offloading. |
| **Context length** | The bundled `Modelfile` sets **`num_ctx` 16384** — raising context increases VRAM/RAM use roughly linearly; reduce `num_ctx` if you hit OOM. |
| **CPU** | Supported by Ollama / llama.cpp runners, but significantly slower than a discrete GPU at this model size. |
| **Training hardware** | LoRA training used **bf16**, gradient checkpointing, and an 8-bit optimizer on a CUDA GPU (see `reproduction/train-phi4.py`); this is separate from these inference notes. |

## Ollama

Place `Phi4-Q4_K_M.gguf` next to `Modelfile` (or adjust the `FROM` path). Then:

```bash
ollama create phi4-rag -f Modelfile
ollama run phi4-rag
```

Generation defaults in the bundled `Modelfile`: `num_ctx` 16384, `temperature` 0.15, `top_p` 0.9, `repeat_penalty` 1.15.

## Limitations

- Intended for **grounded** QA over retrieved context; do not rely on it as an unconstrained world-knowledge model without retrieval.
- **Q4_K_M** is a speed/size trade-off versus higher bit-widths or FP16.
- Response quality depends on the quality of the retrieved context and on wrapping it in `<context>…</context>` tags as in training.

## License

- **MIT** — The model card, `Modelfile`, and other metadata added by **nadiva1243** are released under the [MIT License](https://opensource.org/licenses/MIT) (see the `LICENSE` file in this repository).
- **Base weights** — The GGUF is a derivative of [`microsoft/phi-4`](https://huggingface.co/microsoft/phi-4). You must also comply with the **license and terms** of the base model and with any requirements of the **training datasets** when redistributing or using the weights.

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
