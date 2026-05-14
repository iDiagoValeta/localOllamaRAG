# Gemma-3-12B RAG fine-tuned — estado: **no disponible en Ollama**

LoRA rank 32 sobre [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it) usando el mismo corpus que [Qwen3-14B](../qwen-3/README.md) y [Phi-4](../phi-4/README.md) (Neural-Bridge RAG + Dolly filtrado + Aina EN/ES/CA, rank 32).

## Por qué no se publica

El adaptador entrenó correctamente y los pesos mergeados existen en
`research/models/merged-model/gemma-3/`, pero **ninguno de los flujos de
conversión a GGUF importable en Ollama** (rama base de llama.cpp, rama
`gemma3-improvements`, conversión manual del tokenizer) produce un binario
que mantenga la generación de caracteres no-ASCII (acentos, ñ, ç). Todas
las muestras se truncan justo antes del primer byte multibyte.

Detalles técnicos, intentos y diagnóstico:
[`../../../conversion/GEMMA3_CONVERSION_ISSUE.md`](../../../conversion/GEMMA3_CONVERSION_ISSUE.md).

## Qué se conserva en el repo

- `Modelfile` — referencia, **no es importable** tal cual: el `.gguf` referenciado por `FROM` no se publica.
- Adaptador LoRA crudo: `research/training-output/gemma-3/` (pesos gitignored, métricas y `evaluation_comparison.json` versionados).

Para reproducir el entrenamiento (no la conversión):

```bash
python research/training/train_gemma3.py
python research/conversion/merge_lora.py --model gemma-3
```

A partir de aquí, la conversión a GGUF queda **fuera del alcance del TFG**.
