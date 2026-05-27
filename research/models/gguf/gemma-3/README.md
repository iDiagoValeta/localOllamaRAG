# Gemma-3-12B RAG fine-tuned — status: **not available in Ollama**

LoRA rank 32 over [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it) using the same corpus as [Qwen3-14B](../qwen-3/README.md) and [Phi-4](../phi-4/README.md) (Neural-Bridge RAG + filtered Dolly + Aina EN/ES/CA, rank 32).

## Why it is not published

The adapter trained correctly and the merged weights exist in
`research/models/merged-model/gemma-3/`, but **none of the GGUF conversion
paths importable into Ollama** (base branch of llama.cpp, the
`gemma3-improvements` branch, manual tokenizer conversion) produce a binary
that preserves generation of non-ASCII characters (accents, ñ, ç). Every
sample is truncated right before the first multi-byte byte.

Technical details, attempts and diagnosis:
[`../../../docs/GEMMA3_CONVERSION_ISSUE.md`](../../../docs/GEMMA3_CONVERSION_ISSUE.md).

## What is kept in the repo

- `Modelfile` — reference only; **not importable** as is, because the `.gguf` referenced by `FROM` is not published.
- Raw LoRA adapter: `research/training-output/gemma-3/` (weights gitignored, metrics and `evaluation_comparison.json` versioned).

To reproduce the training (not the conversion):

```bash
python research/training/train_gemma3.py
python research/conversion/merge_lora.py --model gemma-3
```

From this point on, the GGUF conversion is **out of scope**.
