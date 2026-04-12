# Merge, GGUF export, and Ollama (reference)

1. **Merge LoRA** into the base dense weights (CUDA; set `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` if needed):

   `python merge_lora.py --model phi-4`

2. **Export to GGUF** using your local **`llama.cpp`** build (`convert_hf_to_gguf.py` or the workflow you used when producing `Phi4-Q4_K_M.gguf`).

3. **Quantize** to **Q4_K_M** (e.g. `llama-quantize` or the project’s `scripts/conversion/quantize_to_q4km.ps1` on Windows).

4. **Ollama:** put the resulting `.gguf` next to `Modelfile` (see `FROM` line) and run `ollama create <name> -f Modelfile`.

This file is a short checklist; exact revision flags depend on your `llama.cpp` version.
