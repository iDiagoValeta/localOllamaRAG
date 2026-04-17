# Problema de conversión Gemma-3 Fine-tuned → Ollama

Fecha: 2026-04-18  
Modelo: `google/gemma-3-12b-it` fine-tuneado con LoRA (adaptador en `training-output/gemma-3/`)  
Modelo merged: `models/merged-model/gemma-3/`  
GGUF destino: `models/gguf-output/gemma-3/`

---

## Síntomas originales

Al convertir el modelo fine-tuneado y registrarlo en Ollama, la generación producía texto
con los acentos y la ñ sistemáticamente truncados:

```
"elaboraci"       → debería ser "elaboración"
"tambi"           → debería ser "también"
"Espa"            → debería ser "España"
"gastronom espa"  → debería ser "gastronomía española"
```

El patrón era consistente: la generación se cortaba justo antes de cada carácter no-ASCII.

---

## Diagnóstico del problema de acentos

### Causa raíz

`convert_hf_to_gguf.py` contiene la clase `Gemma3Model` con el método `set_vocab()`.
Cuando el directorio del modelo no tiene `tokenizer.model` (el binario SentencePiece),
el código caía en un `else` que llamaba a `_set_vocab_gpt2()`.

Este método escribe `tokenizer_model="gpt2"` en el GGUF, lo que hace que llama.cpp
aplique la decodificación GPT-2 byte-to-unicode al leer los tokens. El problema:

- Gemma-3 usa BPE con tokens almacenados como strings Unicode directos (ej. `"▁elaboración"`)
- GPT-2 byte-to-unicode espera bytes encodificados en un mapping específico de 256 caracteres
- Al decodificar `"ó"` (U+00F3) como si fuera GPT-2 → byte 0xF3, combinado con "n" → 0x6E
  → secuencia UTF-8 inválida → descartada silenciosamente

El carácter `▁` (U+2581, marcador de espacio SentencePiece) caía fuera del rango de 256
bytes de GPT-2 → aparecía como `UNK_BYTE_0xe29681`.

### Código afectado

`llama.cpp/convert_hf_to_gguf.py`, clase `Gemma3Model`, método `set_vocab()`:

```python
# CÓDIGO ORIGINAL (erróneo para Gemma-3 sin tokenizer.model)
def set_vocab(self):
    if (self.dir_model / "tokenizer.model").is_file():
        self._set_vocab_sentencepiece()
        self.gguf_writer.add_add_space_prefix(False)
    else:
        self._set_vocab_gpt2()   # ← PROBLEMA: escribe tokenizer_model="gpt2"
```

---

## Intentos de solución

### Intento 1 — Patch `tokenizer_model="llama"` en el `else`

**Cambio:** Reemplazar el `else` para usar `tokenizer_model="llama"` en lugar de `"gpt2"`.

```python
else:
    tokens, toktypes, tokpre = self.get_vocab_base()
    self.gguf_writer.add_tokenizer_model("llama")
    self.gguf_writer.add_tokenizer_pre(tokpre)
    self.gguf_writer.add_token_list(tokens)
    self.gguf_writer.add_token_types(toktypes)
    self.gguf_writer.add_add_space_prefix(False)
    special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
    special_vocab.add_to_gguf(self.gguf_writer)
```

**Resultado:** Crash de Ollama en cada petición.

```
panic: runtime error: slice bounds out of range [:5] with capacity 0
github.com/ollama/ollama/tokenizer.NewSentencePiece(...)
    tokenizer/sentencepiece.go:27 +0x5df
github.com/ollama/ollama/model/models/gemma3.New(...)
    model/models/gemma3/model.go:82 +0x639
```

**Causa:** Ollama 0.21.0 introdujo un nuevo engine para Gemma3 (`model/models/gemma3/model.go`)
que, cuando detecta `tokenizer_model="llama"`, llama a `NewSentencePiece()` esperando
encontrar datos del tokenizer en el GGUF. Al no encontrarlos → crash.

---

### Intento 2 — Copiar `tokenizer.model` al directorio merged

**Hallazgo:** El merged model no tiene `tokenizer.model` (el binario SentencePiece de Google).
Este archivo sí existe en `C:/Users/nadiv/work/mi_modelo_gemma/tokenizer.model` (4.7 MB).

**Acción:** Copiar `tokenizer.model` al directorio merged para que `set_vocab()` tome
el path correcto (`_set_vocab_sentencepiece()` en lugar del `else`).

```python
# Con tokenizer.model presente, se activa la rama correcta:
if (self.dir_model / "tokenizer.model").is_file():
    self._set_vocab_sentencepiece()   # ← correcto
    self.gguf_writer.add_add_space_prefix(False)
```

**Problema adicional:** `sentencepiece` no estaba instalado → `ModuleNotFoundError`.
Solución: `pip install sentencepiece==0.2.1`.

**Resultado de conversión:** F16 generado correctamente (23.5 GB).

**Resultado en Ollama:** Mismo crash `NewSentencePiece` en `sentencepiece.go:27`.

**Causa:** `_set_vocab_sentencepiece()` también escribe `tokenizer_model="llama"`,
lo que activa el mismo code path del nuevo engine de Ollama.

---

### Intento 3 — Añadir `tokenizer.ggml.merges` al GGUF

**Diagnóstico mediante comparación de GGUFs:**

| Campo GGUF | Gemma3-FineTuned (nuestro) | gemma3:4b (oficial Ollama) |
|---|---|---|
| `tokenizer.ggml.model` | `"llama"` | `"llama"` |
| `tokenizer.ggml.tokens` | array 262208 | array 262145 |
| `tokenizer.ggml.scores` | array 262208 | array 262145 |
| `tokenizer.ggml.merges` | **AUSENTE** | **array 514906** |

El modelo oficial tiene 514,906 entradas en `tokenizer.ggml.merges` (las reglas BPE del
`tokenizer.json`). El nuevo engine de Ollama los necesita para inicializar `NewSentencePiece`.

**Cambio en `convert_hf_to_gguf.py`:**

```python
def set_vocab(self):
    if (self.dir_model / "tokenizer.model").is_file():
        self._set_vocab_sentencepiece()
        self.gguf_writer.add_add_space_prefix(False)
        # Añadir merges BPE del tokenizer.json para Ollama 0.21+
        tokenizer_json = self.dir_model / "tokenizer.json"
        if tokenizer_json.is_file():
            import json as _json
            with open(tokenizer_json, encoding="utf-8") as _f:
                _tj = _json.load(_f)
            _merges = _tj.get("model", {}).get("merges", [])
            if _merges:
                self.gguf_writer.add_token_merges(_merges)
```

`tokenizer.json` del merged model confirmado con 514,906 merges.

**Conversión F16:** Exitosa. El nuevo GGUF incluye `tokenizer.ggml.merges`.

**Cuantización a Q4_K_M:** FALLO.

```
gguf_init_from_file_impl: key 'tokenizer.ggml.merges' has invalid GGUF type 9
gguf_init_from_file_impl: failed to read key-value pairs
llama_model_quantize: failed to quantize
```

**Causa:** El binario `llama-quantize` descargado por el script de cuantización es del
build b7999 (descargado de `ggml-org/llama.cpp/releases`). Este build no soporta
`GGUF type 9` (ARRAY) para la clave `tokenizer.ggml.merges`. El soporte se añadió
en versiones posteriores del formato GGUF.

**Nota sobre el script:** `quantize_to_q4km.ps1` busca `llama-quantize.exe` en
`llama-bin/bin/llama-quantize.exe`. Se descargó el build b8833 a `llama-bin/`, pero
el binario quedó en `llama-bin/llama-quantize.exe` (sin subdirectorio `bin/`), por lo
que el script no lo encontró y descargó b7999 de nuevo.

---

## Estado final

El proceso de conversión queda bloqueado en este punto. Para desbloquear se necesita una
de las siguientes acciones:

### Opción A — Actualizar `llama-quantize` a un build reciente

Descargar el build b8833 o posterior y colocarlo en `llama-bin/bin/llama-quantize.exe`:

```powershell
# Descarga manual:
# https://github.com/ggml-org/llama.cpp/releases/download/b8833/llama-b8833-bin-win-cpu-x64.zip
# Extraer y copiar llama-quantize.exe a:
# C:\Users\nadiv\repos\localOllamaRAG\llama-bin\bin\llama-quantize.exe
```

Luego:
```powershell
.\scripts\conversion\quantize_to_q4km.ps1 `
    models\gguf-output\gemma-3\gemma3-finetuned-f16.gguf `
    models\gguf-output\gemma-3\Gemma3-12B-Q4_K_M.gguf
ollama create Gemma3-FineTuned -f models/gguf-output/gemma-3/Modelfile
```

### Opción B — Compilar `llama-quantize` desde el submódulo

Si hay CMake y MSVC instalados:

```powershell
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j 4
# Binario resultante: llama.cpp/build/bin/Release/llama-quantize.exe
# El script lo detecta automáticamente como primer candidato
```

### Opción C — Abandonar Gemma-3 para producción del TFG

Los modelos Qwen3-FineTuned y phi4-finetuned ya están operativos en Ollama y funcionan
correctamente. Gemma-3 puede mencionarse en la memoria como "pendiente de despliegue"
debido a incompatibilidades de tooling, que no son representativas de la calidad del modelo.

---

## Resumen de cambios aplicados en esta sesión

| Archivo | Cambio | Estado |
|---|---|---|
| `llama.cpp/convert_hf_to_gguf.py` | Patch `Gemma3Model.set_vocab()`: merges BPE + else con `"llama"` | Aplicado |
| `models/merged-model/gemma-3/tokenizer.model` | Copiado desde `C:/Users/nadiv/work/mi_modelo_gemma/` | Copiado |
| `rag/chat_pdfs.py` | `_detectar_idioma()` + `idioma_doc` en contextual retrieval y OCR | Aplicado |
| `llama-bin/llama-quantize.exe` | Build b8833 descargado (pero en ruta incorrecta para el script) | Parcial |
