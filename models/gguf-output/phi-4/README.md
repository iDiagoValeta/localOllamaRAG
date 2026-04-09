---
language:
  - en
  - es
  - ca
license: mit
base_model: microsoft/phi-4
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

- **Base:** [`microsoft/phi-4`](https://huggingface.co/microsoft/phi-4) (ChatML-style; end-of-turn `<|im_end|>`).
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

### Evaluation (frozen splits)

- Same **dev/test** partitions for base vs adapted models.
- **Dev:** up to **320** samples per dataset (Neural-Bridge, Dolly, Aina-EN / ES / CA) for alignment with the baseline benchmark.
- **Test:** full held-out splits (no size cap beyond validity filters).
- **Metrics:** Token F1, ROUGE-L F1, BERTScore F1 (`microsoft/deberta-xlarge-mnli`), plus auxiliary context faithfulness; BERTScore after unloading the generative model.

Training artifacts (`training_stats.json`, `evaluation_comparison.json`) live under `training-output/phi-4/` in the code repository, not on this Hub model repo.

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
