# Gemma-3 fine-tuned → Ollama conversion: failure post-mortem

> [!CAUTION]
> **Final status (2026-05-14): abandoned.** The fine-tuned Gemma-3 model is **not
> included** in the final evaluation. The LoRA adapter is trained and the
> merged weights are kept, but no conversion path produced a GGUF importable into
> Ollama without breaking non-ASCII token generation (accents / ñ) or crashing
> the Ollama runtime. This document records **what went wrong** as a technical
> post-mortem; it intentionally does **not** propose ways to unblock the pipeline
> because the model was dropped from production. The model directory README
> (`research/models/gguf/gemma-3/README.md`) points here.

- Date of investigation: 2026-04-18
- Model: `google/gemma-3-12b-it` fine-tuned with LoRA (adapter in `research/training-output/gemma-3/`)
- Merged model: `research/models/merged-model/gemma-3/`
- GGUF target: `research/models/gguf/gemma-3/`
- Production generators that *do* work: `Qwen3-FineTuned:latest`, `phi4-finetuned:latest`

---

## 1. Original symptom: truncated non-ASCII generation

After converting the fine-tuned model and registering it in Ollama, generation
systematically truncated accents and `ñ`:

```
"elaboraci"       → should be "elaboración"
"tambi"           → should be "también"
"Espa"            → should be "España"
"gastronom espa"  → should be "gastronomía española"
```

The pattern was consistent: generation cut off exactly before each non-ASCII
character.

---

## 2. Root cause: wrong tokenizer model written to the GGUF

`llama.cpp/convert_hf_to_gguf.py` contains the `Gemma3Model` class with a
`set_vocab()` method. When the model directory has no `tokenizer.model` (the
SentencePiece binary), the code fell into an `else` branch calling
`_set_vocab_gpt2()`:

```python
# ORIGINAL CODE (wrong for Gemma-3 without tokenizer.model)
def set_vocab(self):
    if (self.dir_model / "tokenizer.model").is_file():
        self._set_vocab_sentencepiece()
        self.gguf_writer.add_add_space_prefix(False)
    else:
        self._set_vocab_gpt2()   # ← writes tokenizer_model="gpt2"
```

`_set_vocab_gpt2()` writes `tokenizer_model="gpt2"` into the GGUF, so llama.cpp
applies GPT-2 byte-to-unicode decoding when reading tokens:

- Gemma-3 uses BPE with tokens stored as direct Unicode strings (e.g. `"▁elaboración"`).
- GPT-2 byte-to-unicode expects bytes encoded in a specific 256-character mapping.
- Decoding `"ó"` (U+00F3) as GPT-2 → byte 0xF3, combined with `"n"` → 0x6E →
  invalid UTF-8 sequence → silently dropped.
- The `▁` character (U+2581, SentencePiece space marker) fell outside the GPT-2
  256-byte range → surfaced as `UNK_BYTE_0xe29681`.

---

## 3. Failed conversion attempts (documented for the record)

### Attempt 1 — Patch `tokenizer_model="llama"` in the `else`

Replacing the `else` to emit `tokenizer_model="llama"` via `get_vocab_base()`
produced an Ollama crash on every request:

```
panic: runtime error: slice bounds out of range [:5] with capacity 0
github.com/ollama/ollama/tokenizer.NewSentencePiece(...)
    tokenizer/sentencepiece.go:27 +0x5df
github.com/ollama/ollama/model/models/gemma3.New(...)
    model/models/gemma3/model.go:82 +0x639
```

Cause: Ollama 0.21.0 introduced a new Gemma3 engine
(`model/models/gemma3/model.go`) that, on `tokenizer_model="llama"`, calls
`NewSentencePiece()` expecting tokenizer data inside the GGUF. The data was not
there → crash.

### Attempt 2 — Copy `tokenizer.model` into the merged directory

The merged model had no `tokenizer.model` (Google's SentencePiece binary). It was
copied into the merged directory so `set_vocab()` would take the
`_set_vocab_sentencepiece()` branch (`sentencepiece==0.2.1` had to be installed
first). The F16 conversion succeeded (23.5 GB), but Ollama produced the **same**
`NewSentencePiece` crash at `sentencepiece.go:27`, because
`_set_vocab_sentencepiece()` also writes `tokenizer_model="llama"`, hitting the
same new-engine code path.

### Attempt 3 — Add `tokenizer.ggml.merges` to the GGUF

GGUF comparison against the official `gemma3:4b`:

| GGUF field | Gemma3-FineTuned (ours) | gemma3:4b (official) |
|---|---|---|
| `tokenizer.ggml.model` | `"llama"` | `"llama"` |
| `tokenizer.ggml.tokens` | array 262208 | array 262145 |
| `tokenizer.ggml.scores` | array 262208 | array 262145 |
| `tokenizer.ggml.merges` | **MISSING** | **array 514906** |

Injecting the 514,906 BPE merges from `tokenizer.json` into the GGUF made the F16
conversion succeed, but **Q4_K_M quantization failed**:

```
gguf_init_from_file_impl: key 'tokenizer.ggml.merges' has invalid GGUF type 9
gguf_init_from_file_impl: failed to read key-value pairs
llama_model_quantize: failed to quantize
```

Cause: the `llama-quantize` binary in use (build b7999) does not support
`GGUF type 9` (ARRAY) for `tokenizer.ggml.merges`; support was added in later
GGUF format versions. A build-path mismatch in `quantize_to_q4km.ps1`
(`llama-bin/llama-quantize.exe` vs the expected `llama-bin/bin/llama-quantize.exe`)
caused the stale b7999 binary to be re-downloaded instead of the newer one.

---

## 4. Final state

The conversion pipeline is blocked at Q4_K_M quantization due to a chain of
tokenizer/format incompatibilities between the fine-tuned Gemma-3 export, the
new Ollama Gemma3 engine, and the GGUF tooling versions. The decision for the
TFG was to **drop Gemma-3 from production**: `Qwen3-FineTuned:latest` and
`phi4-finetuned:latest` are operational and cover the evaluation. Gemma-3 is
reported in the thesis as "pending deployment due to tooling incompatibilities,
not representative of model quality".

### Code/state changes left from the investigation

| File | Change | State |
|---|---|---|
| `llama.cpp/convert_hf_to_gguf.py` | Patched `Gemma3Model.set_vocab()`: BPE merges + `else` with `"llama"` | Applied (investigation artifact) |
| `research/models/merged-model/gemma-3/tokenizer.model` | Copied from an external Gemma model dir | Copied |
| `rag/chat_pdfs.py` | `_detectar_idioma()` + `idioma_doc` in contextual retrieval and OCR | Applied (kept; unrelated to Gemma-3) |
