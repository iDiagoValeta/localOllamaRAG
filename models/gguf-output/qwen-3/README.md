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

Quantized **GGUF** build of **[Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)** with a **LoRA** adapter merged in, fine-tuned for **retrieval-augmented question answering**. The model answers **only from supplied document context** in **English, Spanish, or Catalan**, using the same RAG-oriented system prompt as **MonkeyGrab**, a local, fully private RAG stack developed for a **Bachelor's thesis (TFG) at the Universitat Politècnica de València (UPV)**.

## Source code, thesis, and contact

The **full MonkeyGrab application repository is not public yet** (defense / publication timeline). This Hugging Face model repo ships **inference assets** (`Qwen3-14B-Q4_K_M.gguf`), the **Ollama `Modelfile`**, and a **`reproduction/`** folder with frozen copies of the training script, merge utility, and **`evaluation_comparison.json`** so methodology and metrics remain auditable without requiring access to the full codebase.

**Contact:** [nadiva1243@gmail.com](mailto:nadiva1243@gmail.com) for questions about training, evaluation, Ollama usage, or when the full repository will be released.

**GGUF pipeline (high level):** LoRA fine-tuning on the datasets below → merge with `merge_lora.py` (see `reproduction/`) → GGUF export via the llama.cpp toolchain → **Q4_K_M** quantization. The merge script documents expected paths and flags.

## Files in this repo

| File | Description |
|------|-------------|
| `Qwen3-14B-Q4_K_M.gguf` | Full weights after LoRA merge, **Q4_K_M** quantization. |
| `Modelfile` | Ollama recipe: Qwen3 chat template (thinking disabled), RAG system prompt, sampling parameters. |
| `README.md` | This model card. |
| `LICENSE` | MIT — applies to the model card, `Modelfile`, and files added here by nadiva1243 (not to Alibaba's base terms). |
| `reproduction/train-qwen3.py` | Snapshot of `scripts/training/train-qwen3.py` (v10) used for this adapter. |
| `reproduction/merge_lora.py` | Snapshot of `scripts/conversion/merge_lora.py` used to merge the LoRA weights into a dense checkpoint before GGUF export. |
| `reproduction/evaluation_comparison.json` | Frozen evaluation export (base vs. adapted, dev/test splits, per dataset + weighted aggregate). |
| `reproduction/CONVERSION.md` | Step-by-step notes: merge → GGUF → Q4_K_M quantization → Ollama import. |

## Base model and method

- **Base:** [`Qwen/Qwen3-14B`](https://huggingface.co/Qwen/Qwen3-14B) — 14B-parameter transformer (ChatML-style; end-of-turn token `<|im_end|>`).
- **Adaptation:** PEFT **LoRA** fine-tuning on five RAG-focused datasets → LoRA adapter merged into dense weights → **GGUF** export → **Q4_K_M** quantization.

> **Thinking mode:** Qwen3 supports explicit chain-of-thought via `<think>…</think>` tokens. In both training and the bundled `Modelfile`, **thinking is disabled** (`enable_thinking=False` / empty `<think></think>` block in the template) so the full token budget goes to the grounded answer rather than internal reasoning traces.

### LoRA configuration

| Setting | Value |
|---------|-------|
| `r` | 32 |
| `lora_alpha` | 64 |
| `lora_dropout` | 0.05 |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| `bias` | `none` |

### Training (`train-qwen3.py`, v10)

- **Seed:** 42 (propagates to torch / NumPy / CUDA via `set_seed`).
- **Task format:** ChatML `<|im_start|>user … <|im_end|>` with the instruction and `<context>…</context>` on the user turn; **loss computed only on the assistant completion** (prompt labels masked with `–100`).
- **Data — balanced 5-way interleaving (3,200 train samples per source, 16,000 total):**
  - [`neural-bridge/rag-dataset-12000`](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000)
  - [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (categories: `closed_qa`, `information_extraction`, `summarization`) — 80/10/10 split after filter
  - [`projecte-aina/RAG_Multilingual`](https://huggingface.co/datasets/projecte-aina/RAG_Multilingual) — **EN**, **ES**, **CA** subsets
- **Sequence limits:** `max_length` 4,096 tokens; context truncated to **2,048** tokens; generation up to **2,048** new tokens.
- **Optimizer / schedule:** AdamW 8-bit, **lr** 5e-5, **cosine** decay with **warmup_ratio** 0.05, **weight_decay** 0.01, **max_grad_norm** 1.0.
- **Batching:** `per_device_train_batch_size` 1, **gradient_accumulation_steps** 16 → **effective batch 16**; **bf16** + **TF32**; gradient checkpointing enabled.
- **Epochs:** 3; checkpoints saved every **300** steps (keep last 3); eval every **150** steps; **load_best_model_at_end** on `eval_loss`; **early stopping** with **patience 5** evaluations for **the training run that produced this checkpoint** (tables below).
- **Follow-up:** After that run, further experiments suggested **tighter early stopping**; `reproduction/train-qwen3.py` now uses **patience 3** as the default for new runs. The metrics in this card are unchanged and still refer to the **patience-5** run.

### Evaluation protocol

- **Frozen dev/test splits:** identical for the **base** (`Qwen/Qwen3-14B`) and the **adapted** (LoRA merged) model — no data leakage.
- **Dev:** 320 samples × 5 sources = **1,600 examples** (aligned with `evaluate_baselines.py` for cross-experiment comparability).
- **Test:** full held-out splits — **8,490 examples** total across all five sources.
- **Metrics:** Token F1, ROUGE-L F1, BERTScore F1 (`microsoft/deberta-xlarge-mnli`); BERTScore is computed after unloading the generative model to fit in GPU memory.
- **Artifacts:** all metric values and sample pairs are in `reproduction/evaluation_comparison.json`.

## Evaluation results

Values are **percentage points** (0–100 scale). **Δ (pp)** = adapted − base; **Δ rel (%)** = relative change vs. base.

### Weighted aggregate (all five sources)

| Split | *N* | Metric | Base | Adapted | Δ (pp) | Δ rel (%) |
|-------|-----|--------|------|---------|--------|-----------|
| **Dev** | 1,600 | Token F1 | 43.33 | 59.95 | +16.63 | +38.37 |
| **Dev** | 1,600 | ROUGE-L F1 | 36.62 | 49.81 | +13.19 | +36.02 |
| **Dev** | 1,600 | BERTScore F1 | 38.94 | 53.07 | +14.13 | +36.28 |
| **Test** | 8,490 | Token F1 | 42.99 | 62.46 | +19.47 | +45.29 |
| **Test** | 8,490 | ROUGE-L F1 | 36.13 | 51.88 | +15.75 | +43.58 |
| **Test** | 8,490 | BERTScore F1 | 38.83 | 55.61 | +16.78 | +43.20 |

### Per-dataset dev (320 samples each)

| Dataset | Token F1 (Base → Adapted) | ROUGE-L F1 (Base → Adapted) | BERTScore F1 (Base → Adapted) |
|---------|--------------------------|------------------------------|-------------------------------|
| Neural-Bridge RAG | 60.38 → **79.55** | 54.65 → **75.85** | 56.04 → **77.60** |
| Dolly QA | 48.30 → **49.88** | 40.94 → **44.04** | 42.41 → **45.38** |
| Aina-EN | 36.53 → **56.91** | 30.15 → **43.48** | 35.62 → **51.86** |
| Aina-ES | 33.66 → **55.57** | 27.09 → **41.52** | 28.94 → **44.22** |
| Aina-CA | 37.77 → **57.86** | 30.26 → **44.14** | 31.70 → **46.29** |

Full test-split breakdowns and qualitative sample pairs are in `reproduction/evaluation_comparison.json`.

### Relation to the baseline benchmark

The **base** dev numbers are aligned with the multi-model benchmark (`evaluate_baselines.py`, `predictions_Qwen3-14B.json`), so Qwen3-14B **before** fine-tuning is directly comparable to the other models in that suite. For post-LoRA performance, use the **Adapted** columns above.

## Hardware compatibility (inference)

| Setup | Notes |
|-------|-------|
| **GPU (recommended)** | **~10–12 GB VRAM** is a practical minimum for this **Q4_K_M** ~14B-class GGUF in Ollama at moderate batching; more VRAM is needed if you raise `num_ctx`. |
| **Context length** | The bundled `Modelfile` sets **`num_ctx` 32768** — reduce if you hit OOM. |
| **CPU** | Supported by Ollama / llama.cpp runners, but significantly slower than a discrete GPU at this model size. |
| **Training hardware** | LoRA training used **bf16**, gradient checkpointing, and an 8-bit optimizer on a CUDA GPU (see `reproduction/train-qwen3.py`); this is separate from these inference notes. |

## Ollama

Place `Qwen3-14B-Q4_K_M.gguf` next to `Modelfile` (or adjust the `FROM` path). Then:

```bash
ollama create qwen3-rag -f Modelfile
ollama run qwen3-rag
```

Generation defaults in the bundled `Modelfile`: `num_ctx` 32768, `temperature` 0.15, `top_p` 0.9, `repeat_penalty` 1.15.

## Limitations

- Intended for **grounded** QA over retrieved context; do not rely on it as an unconstrained world-knowledge model without retrieval.
- **Q4_K_M** is a speed/size trade-off versus higher bit-widths or FP16.
- Response quality depends on the quality of the retrieved context and on wrapping it in `<context>…</context>` tags as in training.

## License

- **MIT** — The model card, `Modelfile`, and other metadata added by **nadiva1243** are released under the [MIT License](https://opensource.org/licenses/MIT) (see the `LICENSE` file in this repository).
- **Base weights** — The GGUF is a derivative of [`Qwen/Qwen3-14B`](https://huggingface.co/Qwen/Qwen3-14B). You must also comply with the **license and terms** of the base model (typically **Apache-2.0**, see the upstream model card) and with any requirements of the **training datasets** when redistributing or using the weights.

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
