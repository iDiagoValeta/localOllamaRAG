# CLAUDE.md — MonkeyGrab (localOllamaRAG)

## Descripción del proyecto

**MonkeyGrab** es un sistema RAG (Retrieval-Augmented Generation) completamente local desarrollado por Nacho como Trabajo de Fin de Grado (TFG) en la Universidad Politécnica de Valencia (UPV). El sistema indexa PDFs, recupera fragmentos relevantes mediante búsqueda híbrida y genera respuestas fundamentadas usando un LLM servido por Ollama. Ningún dato sale de la máquina del usuario en uso normal.

El proyecto combina dos dimensiones:
- **Capa de investigación experimental**: entrenamiento LoRA, evaluación con métricas múltiples, trazabilidad de experimentos
- **Núcleo RAG/servicio local**: pipeline de producción con CLI y Web

Roles de modelos (variables de entorno; constantes `MODELO_*` en `rag/chat_pdfs.py`):

| Rol | Variable | Descripción |
|-----|----------|-------------|
| Generador RAG | `OLLAMA_RAG_MODEL` | **Función:** generar la respuesta al usuario. **Condición:** modo `/rag`. **Detalle:** salida por streaming vía API de Ollama. |
| Chat y sub-consultas | `OLLAMA_CHAT_MODEL` | **Función:** conversación en `/chat` y, si aplica, hasta 3 sub-queries para la búsqueda híbrida (`generar_queries_con_llm`). **Condición:** `/chat` siempre; sub-queries solo si `USAR_LLM_QUERY_DECOMPOSITION`. **Detalle:** la descomposición no usa `OLLAMA_CONTEXTUAL_MODEL`. |
| Embeddings | `OLLAMA_EMBED_MODEL` | **Función:** vectorizar chunks al indexar y la pregunta al recuperar. **Condición:** fases de indexación y de recuperación. **Detalle:** el path de ChromaDB incluye un slug del modelo; con Nomic se aplican prefijos `search_query:` / `search_document:`. |
| Contextual retrieval | `OLLAMA_CONTEXTUAL_MODEL` | **Función:** enriquecer el texto de cada chunk antes de embeberlo. **Condición:** indexación con `USAR_CONTEXTUAL_RETRIEVAL`. **Detalle:** no interviene en `/chat` ni en la descomposición de la pregunta. |
| RECOMP | `OLLAMA_RECOMP_MODEL` | **Función:** sintetizar o comprimir fragmentos recuperados antes del generador. **Condición:** `USAR_RECOMP_SYNTHESIS`. **Detalle:** mismo esquema que el resto de `MODELO_*` (env + literal en `chat_pdfs.py`). |
| Visión / OCR | `OLLAMA_OCR_MODEL` | **Función:** producir descripción textual de figuras raster en PDFs. **Condición:** `USAR_EMBEDDINGS_IMAGEN`. **Detalle:** descarta salidas spam, eco de prompt y eco de caption cuando aplica. |
| Reranker | `RERANKER_QUALITY` | **Función:** re-puntuar y reordenar candidatos tras la búsqueda híbrida. **Condición:** `USAR_RERANKER` y `sentence-transformers` disponible. **Detalle:** Cross-encoder local (BGE o MiniLM según `quality`/`speed`); no es un modelo Ollama. |

Resolución de valores: cada `MODELO_*` toma primero la variable de entorno; si no está definida, el segundo argumento de `os.getenv` en `rag/chat_pdfs.py`. Esa tabla, el README y tu `.env` pueden mostrar literales distintos: **manda el entorno en el proceso que ejecutes**, no un «estado oficial» del repo.

---

## Estructura de archivos clave

```
localOllamaRAG/
├── generate_diagram.py           # Diagrama de arquitectura vía Kroki.io (salida habitual: docs/monkeygrab_architecture.*)
├── rag/
│   ├── chat_pdfs.py              # Motor RAG principal: indexación, recuperación, generación
│   ├── export_fragments.py       # Exporta chunks de ChromaDB a TXT/JSONL para debug e inspección
│   ├── requirements.txt          # Dependencias del núcleo RAG
│   ├── debug_context_issues.md   # Análisis de issues menores en la presentación del contexto al generador
│   ├── debug_rag/                # Dumps de debug de queries RAG (generados en tiempo de ejecución, no versionados)
│   ├── pdfs/                     # PDFs a indexar con el sistema RAG (no versionados)
│   ├── ragbench_pdfs/            # PDFs del benchmark RAGBench — uso exclusivo de run_eval_ragbench.py (gitignored)
│   ├── mi_vector_db/             # ChromaDB producción (PDFs en pdfs/; gitignored)
│   ├── ragbench_vector_db/       # ChromaDB RAGBench (evaluación; generado, no versionado)
│   ├── historial_chat.json       # Historial modo CHAT (gitignored)
│   └── cli/
│       ├── app.py                # Clase MonkeyGrabCLI: bucle interactivo y dispatch de comandos
│       ├── display.py            # Singleton `ui`: toda la salida visual Rich (panels, tables, spinners)
│       ├── renderer.py           # Renderizado ANSI de bajo nivel (legacy)
│       └── theme.py              # Paleta de colores MonkeyGrab
├── web/
│   ├── app.py                    # Backend Flask: REST + SSE, sirve React
│   ├── requirements.txt          # flask, flask-cors
│   └── zip/dist/                 # Build frontend React (assets estáticos; carpeta ignorada en git si solo build)
├── scripts/
│   ├── requirements.txt          # Stack torch, transformers, peft, datasets, bert-score, matplotlib, etc.
│   ├── training/
│   │   ├── train-qwen3.py        # Fine-tuning LoRA Qwen3-14B (v10) — modelo principal del TFG
│   │   ├── train-phi4.py         # Fine-tuning LoRA Phi-4 (v1) — mismo protocolo que Qwen3 v10
│   │   ├── train-llama3.1.py     # Fine-tuning LoRA Llama-3.1-8B
│   │   ├── train-gemma3.py       # Fine-tuning LoRA Gemma-3-12B-IT
│   │   └── run-general.sh        # Plantilla SLURM genérica (cluster)
│   ├── evaluation/
│   │   ├── evaluate_baselines.py # Benchmark de 7 modelos base (Token F1, ROUGE-L, BERTScore, CF-lexica)
│   │   ├── inspect_splits.py     # Audita tamaño de splits por dataset antes/después de filtros
│   │   └── run-baselines.sh      # Lanzamiento SLURM de evaluate_baselines.py
│   ├── conversion/
│   │   ├── merge_lora.py         # Fusiona adaptador LoRA con base para exportar a GGUF
│   │   ├── build_ollama.bat      # Automatiza creación del modelo en Ollama (Windows)
│   │   └── quantize_to_q4km.ps1  # Cuantiza modelo merged a Q4_K_M con llama-bin
│   └── tests/
│       ├── test_nothink.py       # Test supresión de <think> en Qwen3 vía Ollama
│       ├── test_ollama_stream_nothink.py
│       └── test_image_rag.py     # Tests de pipeline con imágenes en RAG
├── evaluation/
│   ├── run_eval.py               # Evaluación RAGAS del pipeline RAG en vivo
│   ├── run_eval_ragbench.py      # Evaluación RAGAS sobre PDFs de RAGBench (Vectara)
│   ├── requirements.txt          # ragas, langchain-google-genai, pandas, etc.
│   ├── chunks_*.txt              # Exportaciones de chunks (export_fragments.py; generado, gitignored)
├── training-output/
│   ├── qwen-3/                   # Adaptador LoRA + artefactos de eval (JSON, CSV, plots)
│   │   └── explore-rank/         # Artefactos de la exploración de rango LoRA (r=32 vs r=64)
│   ├── llama-3/                  # Adaptador LoRA Llama-3.1
│   ├── gemma-3/                  # Adaptador LoRA Gemma-3
│   ├── phi-4/                    # Adaptador LoRA Phi-4 (artefactos pesados gitignored; se conserva training_stats.json)
│   ├── bertscore/                # CSVs por muestra, resumen BERTScore (DATOS OBSOLETOS — re-ejecutar)
│   │   └── plots/                # Gráficas BERTScore generadas
│   └── baseline/                 # Resultados del benchmark de 7 modelos base
│       ├── baseline_evaluation.json          # 7 modelos x 5 datasets x dev/test x con/sin contexto
│       ├── baseline_evaluation_samples.json  # Predicciones cualitativas por muestra
│       ├── baseline_checkpoint.json          # Checkpoint incremental (permite reanudar)
│       ├── predictions_{modelo}.json         # Predicciones por modelo (7 archivos)
│       ├── generate_reports.py               # Genera tablas Markdown + CSVs desde baseline_evaluation.json
│       ├── reports/                          # Tablas comparativas + CSVs + figuras (generado, gitignored)
│       └── 200/                              # Versión anterior con cap de 200 muestras (referencia)
├── docs/
│   ├── monkeygrab_architecture.svg           # Diagrama de arquitectura (exportable también a PNG vía generate_diagram.py)
│   ├── investigacionMetricas.md              # Investigación sobre métricas de evaluación RAG
│   ├── palabras.md                           # Notas de vocabulario y terminología del TFG
│   └── splits.md                             # Análisis de splits de datasets
├── llama-bin/                    # Binarios llama.cpp compilados para Windows (llama-quantize, etc.; gitignored)
├── models/
│   ├── gguf-output/              # Modelos GGUF cuantizados (gitignored)
│   └── merged-model/             # Modelos merged antes de convertir (gitignored)
├── llama.cpp/                    # llama.cpp para conversión/cuantización GGUF (gitignored)
├── README.md
└── CLAUDE.md                     # Contexto del repositorio para asistentes (este archivo)
```

---

## Arquitectura del pipeline

### Pipeline de indexación (al arrancar o con /reindex)

Constantes y rutas en `rag/chat_pdfs.py` (`CARPETA_DOCS` / `DOCS_FOLDER`, `PATH_DB`, `COLLECTION_NAME`, etc.).

```
PDFs (CARPETA_DOCS)
  -> Extracción texto: pymupdf4llm (preferido) / pypdf (fallback)
  -> [Opt] USAR_EMBEDDINGS_IMAGEN: fitz + caption por posición (CAPTION_MARGIN_PX) + MODELO_OCR (OLLAMA_OCR_MODEL)
  -> Chunking: CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH
  -> [Opt] USAR_CONTEXTUAL_RETRIEVAL: enriquecimiento con MODELO_CONTEXTUAL (OLLAMA_CONTEXTUAL_MODEL)
  -> Embedding: MODELO_EMBEDDING (OLLAMA_EMBED_MODEL); prefijos EMBED_PREFIX_* si el modelo lo requiere
  -> Persistencia: ChromaDB en PATH_DB
```

### Pipeline de recuperación (por cada query en modo RAG)

```
Query usuario (MIN_LONGITUD_PREGUNTA_RAG)
  -> [Opt] USAR_LLM_QUERY_DECOMPOSITION: sub-consultas vía MODELO_CHAT (OLLAMA_CHAT_MODEL), hasta 3; activación según longitud de pregunta en realizar_busqueda_hibrida
  -> Búsqueda semántica: embeddings MODELO_EMBEDDING, N_RESULTADOS_SEMANTICOS
  -> [Opt] USAR_BUSQUEDA_HIBRIDA: búsqueda léxica, N_RESULTADOS_KEYWORD
  -> [Opt] USAR_BUSQUEDA_EXHAUSTIVA: términos críticos filtrados
  -> Fusión: score_semantic + score_keyword -> score_final (pesos en realizar_busqueda_hibrida)
  -> [Opt] USAR_RERANKER: CrossEncoder (RERANKER_QUALITY), TOP_K_RERANK_CANDIDATES, TOP_K_AFTER_RERANK
  -> Filtrado: UMBRAL_RELEVANCIA; con reranker activo, UMBRAL_SCORE_RERANKER
  -> Selección: TOP_K_FINAL
  -> [Opt] EXPANDIR_CONTEXTO: N_TOP_PARA_EXPANSION + expandir_con_chunks_adyacentes
  -> [Opt] USAR_OPTIMIZACION_CONTEXTO; recorte de contexto con MAX_CONTEXTO_CHARS
```

### Generación

```
Pregunta + <context> (construir_contexto_para_modelo u opcionalmente sintetizar_contexto_recomp si USAR_RECOMP_SYNTHESIS)
  -> MODELO_RAG (OLLAMA_RAG_MODEL): streaming vía Ollama; opciones numéricas en la llamada (temperature, top_p, repeat_penalty, num_ctx) en generar_respuesta / generar_respuesta_silenciosa
```

---

## Comandos importantes

### Arrancar el sistema

```bash
# CLI (desde la raíz del proyecto)
cd rag
python chat_pdfs.py

# Web (desde la raíz del proyecto)
python web/app.py
# Abre http://localhost:5000
```

### Comandos CLI en tiempo de ejecución

| Comando | Descripción |
|---------|-------------|
| `/rag` | Modo RAG (consulta de documentos) |
| `/chat` | Modo CHAT (conversación general) |
| `/docs` | Lista documentos indexados |
| `/temas` | Resumen de tópicos por documento |
| `/stats` | Estadísticas de la base de datos vectorial |
| `/reindex` | Borrar DB y reindexar todos los PDFs |
| `/limpiar` o `/clear` | Limpiar historial de chat |
| `/ayuda` o `/help` | Mostrar ayuda |
| `/salir` o `/exit` | Salir guardando historial |

### Entrenamiento LoRA

```bash
# Qwen3-14B (modelo de producción, requiere ~24GB VRAM)
python scripts/training/train-qwen3.py

# Phi-4 (14B, mismo protocolo que Qwen3 v10, pendiente de lanzar en cluster)
python scripts/training/train-phi4.py

# Llama-3.1-8B
python scripts/training/train-llama3.1.py

# Gemma-3-12B-IT (requiere HF_TOKEN, no compatible con Ollama vía GGUF)
python scripts/training/train-gemma3.py

# Fusionar adaptador para exportar a GGUF
python scripts/conversion/merge_lora.py --model qwen-3
# Opciones: qwen-3, llama-3, gemma-3

# Visualizar curvas de entrenamiento
python scripts/training/plot_training.py --model qwen-3
```

### Evaluación

```bash
# Benchmark de 7 modelos base — Token F1, ROUGE-L F1, BERTScore, CF-lexica (dev + test, con/sin contexto)
# 7 modelos: Llama-3.1-8B, Mistral-7B, Qwen2.5-14B, Qwen3-14B, Qwen3.5-9B, Gemma-3-12B, Phi-4
# Aina evaluado desglosado por idioma: Aina-EN, Aina-ES, Aina-CA
python scripts/evaluation/evaluate_baselines.py
# Salida en: training-output/baseline/  (JSONs + predicciones por modelo)

# Regenerar tablas Markdown + CSVs desde baseline_evaluation.json
python training-output/baseline/generate_reports.py
# Salida en: training-output/baseline/reports/

# Auditar tamaño de splits por dataset antes/después de filtros
python scripts/evaluation/inspect_splits.py

# Exportar chunks de ChromaDB a texto para inspección
python rag/export_fragments.py              # ambos stores (mi_vector_db + ragbench_vector_db)
python rag/export_fragments.py --mi-only    # solo PDFs propios
# Salida en: evaluation/chunks_mi_vector_db.txt / chunks_ragbench_vector_db.txt

# Estadísticas mean +/- std sobre los CSVs de training-output/bertscore/ (histórico)
python scripts/evaluation/compute_std.py

# RAGAS sobre el pipeline en vivo (necesita GOOGLE_API_KEY en .env)
python evaluation/run_eval.py
python evaluation/run_eval.py --dataset evaluation/mi_dataset.json --verbose

# RAGAS sobre PDFs de RAGBench (Vectara)
python evaluation/run_eval_ragbench.py

# NOTA: BERTScore ya NO requiere un script separado.
# Se computa dentro de train-qwen3.py (Section 12) y evaluate_baselines.py.
```

### Generación de diagrama de arquitectura

```bash
python generate_diagram.py
python generate_diagram.py --output docs/monkeygrab_architecture.png
python generate_diagram.py --format svg --output docs/monkeygrab_architecture.svg
```

### Variables de entorno relevantes

Referencias orientativas (ajusta todo vía entorno; no implican un único despliegue válido).

| Variable | Referencia | Descripción |
|----------|------------|-------------|
| `OLLAMA_RAG_MODEL` | `Qwen3-FineTuned` | Modelo generador RAG |
| `OLLAMA_CHAT_MODEL` | `gemma3:4b` | Modelo modo chat |
| `OLLAMA_EMBED_MODEL` | `embeddinggemma:latest` | Modelo de embeddings |
| `OLLAMA_CONTEXTUAL_MODEL` | `gemma3:4b` | Modelo contextual retrieval |
| `OLLAMA_OCR_MODEL` | `qwen3-vl:8b` | Modelo visión para descripción de imágenes en PDFs |
| `DOCS_FOLDER` | `rag/pdfs/` | Carpeta de PDFs a indexar |
| `RERANKER_QUALITY` | `quality` | `quality` (BAAI/bge) o `speed` (MiniLM) |
| `HF_TOKEN` | — | Token HuggingFace (necesario para Gemma-3) |
| `GOOGLE_API_KEY` | — | API key Gemini para evaluación RAGAS |

---

## Patrones de programación — seguir al escribir código nuevo

### 1. MODULE MAP al inicio de cada módulo (OBLIGATORIO)

Todo archivo Python no trivial abre con un MODULE MAP comentado que indexa sus secciones. **Todos los archivos del proyecto ya siguen este patrón** (verificado 2026-03-28). Al añadir un archivo nuevo, incluir el MODULE MAP desde el inicio.

```python
# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  CONFIGURATION
#  +-- 1. Imports
#  +-- 2. Global config
#  |      +-- 2.1 Models
#  |      +-- 2.2 Pipeline flags
#  |
#  BUSINESS LOGIC
#  +-- 3. Retrieval
#  +-- 4. Generation
#
#  ENTRY
#  +-- 5. main()
#
# ─────────────────────────────────────────────
```

### 2. Separadores de sección con nombre en mayúsculas

```python
# ─────────────────────────────────────────────
# SECTION 3: GLOBAL CONFIGURATION
# ─────────────────────────────────────────────
```

Sub-secciones con guiones simples:

```python
# --- 3.1 Pipeline flags ---
```

### 3. Constantes globales al inicio, antes de cualquier lógica

Orden: imports stdlib -> third-party -> local, luego constantes globales (modelos, rutas, flags, parámetros numéricos).

```python
MODELO_RAG = os.getenv("OLLAMA_RAG_MODEL", "Qwen3-FineTuned")
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 350
USAR_RERANKER = True
```

### 4. Inicialización defensiva del entorno antes de imports pesados

Variables CUDA/Triton se fijan ANTES de importar torch o transformers:

```python
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TRITON_DISABLE"] = "1"

import torch  # después
```

### 5. Dependencias opcionales con try/except + flag booleano

```python
try:
    import pymupdf4llm
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
```

La lógica posterior lee el flag, nunca relanza el ImportError.

### 6. Pipelines explícitamente por fases

Nombrar y separar visualmente cada fase: carga -> preparación -> inferencia -> evaluación -> exportación de artefactos.

### 7. Salida orientada a artefactos

Los scripts experimentales siempre exportan: JSON de métricas, CSV por muestra, plots. Nunca solo stdout.

### 11. Tablas de resultados — sin desviación típica, preferir mejora relativa

Por indicación del tutor: las tablas de resultados del TFG **no incluyen desviación típica**. El formato preferido para cuantificar el efecto del fine-tuning es:

- **Δ absoluto** (en pp): diferencia entre base y adaptado. Ej: `+3.2 pp`.
- **Δ relativo** (en %): `(adaptado - base) / base × 100`. Ej: `+7.4 %`.

Los artefactos JSON (`evaluation_comparison.json`) ya incluyen ambos campos (`delta_pp` y `delta_rel_pct`). No es necesario recomputarlos.

El script `compute_std.py` existe por razones históricas; **no usarlo para tablas definitivas** del TFG.

### 8. Naming mixto ES/EN (convención establecida del proyecto)

- Funciones de dominio RAG en español: `realizar_busqueda_hibrida`, `indexar_documentos`, `cargar_historial`
- Variables de configuración en inglés: `CHUNK_SIZE`, `TOP_K_FINAL`, `UMBRAL_RELEVANCIA`
- Docstrings y comentarios en inglés
- Al añadir código nuevo: seguir el patrón del módulo donde se trabaje (no mezclar dentro del mismo bloque)

### 9. Scripts de entrenamiento con VERSION HISTORY

Los scripts de entrenamiento incluyen un bloque `VERSION HISTORY` al inicio documentando cambios de versión con justificación técnica.

### 10. Script-first, no arquitectura enterprise

La lógica principal vive en módulos + main(), no en jerarquías de clases/dominios separados. La única excepción es `MonkeyGrabCLI` (rag/cli/app.py) que encapsula el bucle CLI.

---

## Patrones de documentación — seguir al escribir código nuevo

### Docstring de módulo

```python
"""
NombreModulo -- descripción corta en una línea.

Explicación extendida del propósito, las fases del pipeline que implementa,
y cualquier decisión de diseño relevante.

Usage:
    python nombre_modulo.py [--arg valor]

Dependencies:
    - paquete1, paquete2
    - modulo_interno (proyecto)
"""
```

### Docstring de función (estilo Google)

```python
def realizar_busqueda_hibrida(pregunta: str, collection) -> tuple:
    """Execute hybrid search combining semantic and keyword strategies.

    Runs semantic search via ChromaDB embeddings, keyword search via
    $contains filters, applies RRF score fusion, and optionally reranks
    with a CrossEncoder model.

    Args:
        pregunta: User question string.
        collection: ChromaDB collection to search against.

    Returns:
        Tuple of (ranked_fragments, best_score, metrics_dict).

    Raises:
        ValueError: If the collection is empty or None.
    """
```

### Comentarios inline

Solo para lógica no obvia. Nunca comentar lo que el código ya dice:

```python
# CUDA max_length overflow workaround: DeBERTa reports sys.maxsize as model_max_length,
# which the Rust tokenizer cannot store as a 32-bit int. Cap at 512 (physical limit).
max_length = min(tokenizer.model_max_length, 512)
```

### Flags de pipeline documentados en el mismo bloque

```python
# --- 3.3 Pipeline flags (per-stage toggle) ---
USAR_CONTEXTUAL_RETRIEVAL = True   # enrich chunks with LLM context before indexing
USAR_LLM_QUERY_DECOMPOSITION = True  # decompose query into 3 sub-queries
USAR_RERANKER = RERANKER_AVAILABLE   # disabled automatically if sentence-transformers missing
USAR_RECOMP_SYNTHESIS = False        # experimental; toggle in this file / env as needed
```

---

## Reglas de comportamiento para Claude

1. **NUNCA hacer commit ni push a GitHub sin preguntar al usuario primero.** Esta regla no tiene excepciones.
2. **Responder siempre en español** en este proyecto.
3. **Seguir los patrones de programación y documentación** descritos arriba al escribir código nuevo.
4. **No modificar requirements.txt** sin confirmar con el usuario: las versiones están fijadas intencionalmente (especialmente el stack de training).
5. **No cambiar `USAR_RECOMP_SYNTHESIS` (u otros flags de pipeline) sin acordarlo con el usuario**: son configuración explícita, con efectos en latencia y coste.
6. **No tocar llama.cpp/**: es un submódulo externo, no código del proyecto.
7. Al proponer cambios en `chat_pdfs.py`: tener en cuenta que `web/app.py` importa directamente constantes y funciones de ese módulo (`PATH_DB`, `COLLECTION_NAME`, `indexar_documentos`, `evaluar_pregunta_rag`, etc.). Un renombrado rompe el backend web.
8. **Los datos en `training-output/bertscore/` son obsoletos** (versión anterior del experimento). No citarlos como resultados definitivos. Los resultados definitivos vendrán de re-ejecutar `train-qwen3.py` (v10) y `evaluate_baselines.py` en cluster con el nuevo esquema (Aina por idioma, ROUGE-L + BERTScore integrado, 320 muestras dev).

---

## Dependencias y requisitos del entorno

### Prerrequisitos del sistema

- Python 3.10+
- [Ollama](https://ollama.ai/) instalado y en ejecución local
- GPU con CUDA recomendada para reranking y fine-tuning; CPU funciona para inferencia
- ~24GB VRAM para fine-tuning de Qwen3-14B

### Instalación

```bash
# Nucleo RAG (obligatorio)
pip install -r rag/requirements.txt

# Interfaz web (opcional)
pip install -r web/requirements.txt

# Evaluación RAGAS (opcional)
pip install -r evaluation/requirements.txt

# Fine-tuning (opcional, requiere GPU)
pip install -r scripts/requirements.txt
```

### Modelos Ollama necesarios

Los modelos se eligen vía variables de entorno (tabla en la sección homónima). Los valores de referencia son solo ejemplos; ajusta cada variable según hardware y despliegue.

```bash
ollama pull <OLLAMA_RAG_MODEL>          # generador RAG (obligatorio)
ollama pull <OLLAMA_EMBED_MODEL>        # embeddings (obligatorio)
ollama pull <OLLAMA_CONTEXTUAL_MODEL>   # contextual retrieval en indexación (si lo usas)
ollama pull <OLLAMA_OCR_MODEL>          # descripción de imágenes en PDFs (opcional)
```

### Stack de training (versiones fijadas)

```
torch==2.6.0
transformers==4.57.6
peft==0.18.1
datasets==4.3.0
accelerate==1.12.0
bitsandbytes==0.49.1
safetensors==0.7.0
bert-score>=0.3.13
```

---

## Deuda técnica conocida y consideraciones de diseño

### Deuda activa

- **Duplicación entre scripts de entrenamiento**: `train-qwen3.py` está en v10 y `train-phi4.py` en v1 (ambos con Aina por idioma, ROUGE-L, BERTScore integrado, dev+test separados, 320 muestras dev). `train-llama3.1.py` y `train-gemma3.py` están desactualizados respecto a ese esquema. Si se repiten esos experimentos, hay que alinearlos con la lógica de Qwen3 v10. `train-phi4.py` está listo pero pendiente de lanzar en cluster.
- **`training-output/bertscore/`**: los CSVs y JSON existentes corresponden a la versión anterior del experimento (Aina como bloque único, caps de muestras). Son datos obsoletos; serán reemplazados cuando finalicen los experimentos en cluster. No usar para tablas definitivas del TFG.
- **Mezcla de idioma en nombres**: funciones de dominio en español (`realizar_busqueda_hibrida`) junto a constantes en inglés (`CHUNK_SIZE`). Es una convención establecida, no un error, pero puede resultar sorprendente.
- **Lógica concentrada en scripts largos**: `chat_pdfs.py` supera las 1000 líneas. Toda la lógica RAG vive ahí. Refactorizar requeriría actualizar los imports en `web/app.py` y `evaluation/run_eval.py`.

### Limitaciones conocidas

- **Gemma-3-12B**: el adaptador LoRA no puede desplegarse vía Ollama por incompatibilidad GGUF. El adaptador existe y está publicado, pero no se usa en producción.
- **RECOMP synthesis** (`USAR_RECOMP_SYNTHESIS`): implementado; activación solo vía ese flag. Suele aumentar latencia; conviene medir en tu entorno si compensa.
- **Evaluación RAGAS**: requiere `GOOGLE_API_KEY` (Gemini 2.0 Flash como juez LLM). No es totalmente local.
- **Regeneración de datos de evaluación por muestra**: los artefactos de `training-output/bertscore/` se regeneran re-ejecutando `train-qwen3.py` completo (Section 12 genera BERTScore). No hay script separado de BERTScore.

### Decisiones de diseño relevantes

- **RRF fusion 55/45**: la ponderación semántica/léxica fue calibrada empíricamente. No cambiar sin evaluación.
- **`UMBRAL_RELEVANCIA = 0.50`**: umbral mínimo para que un resultado RAG se considere relevante. Bajarlo aumenta recall pero introduce ruido.
- **`TOP_K_FINAL = 6`**: número de fragmentos que llegan al LLM. El contexto máximo es `MAX_CONTEXTO_CHARS = 8192`.
- **ChromaDB persistente por combinación `(carpeta_docs, embedding_model)`**: el path de la DB incluye el slug del modelo de embedding. Cambiar el modelo de embedding invalida la DB existente y requiere reindexar.
- **`enable_thinking=False` en Qwen3**: el modelo fine-tuneado usa razonamiento interno. La inferencia en producción suprime el bloque `<think>` para reducir latencia y tokens de salida. Ver `scripts/tests/test_nothink.py`.

---
## Artefactos de evaluación

Los resultados de los experimentos están en `training-output/`. Los datos marcados como (OBSOLETOS) corresponden a la versión anterior del experimento (Aina como bloque único, caps de muestras); serán reemplazados al terminar el cluster.

| Artefacto | Ubicación | Descripción |
|-----------|-----------|-------------|
| Adaptador LoRA Qwen3 | `training-output/qwen-3/` | `adapter_config.json`, `adapter_model.safetensors` |
| Comparación base/adaptado | `training-output/qwen-3/evaluation_comparison.json` | Deltas per-dataset con Token F1, BERTScore F1, CF-lexica; separados por dev/test |
| Estadísticas de training | `training-output/qwen-3/training_stats.json` | Loss, pasos, tiempo (v10) |
| Predicciones checkpoint | `training-output/qwen-3/predictions_base.json` / `predictions_adapted.json` | Predicciones + Token F1 + CF-lexica por muestra; permite recomputar BERTScore sin regenerar |
| BERTScore checkpoint | `training-output/qwen-3/bertscore_checkpoint.json` | BERTScore P/R/F1 por dataset (BASE y ADAPTED) |
| Exploración rango LoRA | `training-output/qwen-3/explore-rank/` | Artefactos comparación r=32 vs r=64 |
| Baseline completo | `training-output/baseline/baseline_evaluation.json` | Token F1, ROUGE-L F1, BERTScore F1, CF-lexica para 7 modelos x 5 datasets x dev/test x con/sin contexto (incluye Phi-4) |
| Baseline checkpoint | `training-output/baseline/baseline_checkpoint.json` | Checkpoint incremental (permite reanudar si falla) |
| Predicciones baseline | `training-output/baseline/predictions_{modelo}.json` | Predicciones por modelo (7 archivos); permite recomputar métricas |
| Tablas de resultados | `training-output/baseline/reports/` | Markdown + CSVs + figuras; generado por `generate_reports.py` |
| Baseline 200-sample | `training-output/baseline/200/` | Versión anterior con cap de 200 muestras (referencia histórica) |
| Diagrama arquitectura | `docs/monkeygrab_architecture.png` / `.svg` | Generado por `generate_diagram.py` (raíz del repo) |

