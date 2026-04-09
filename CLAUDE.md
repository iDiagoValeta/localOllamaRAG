# CLAUDE.md — MonkeyGrab (localOllamaRAG)

## 1. Descripción del proyecto

**MonkeyGrab** es un sistema RAG (Retrieval-Augmented Generation) completamente local desarrollado como TFG (Máster MUITSS, UPV). Indexa PDFs, recupera fragmentos relevantes mediante búsqueda híbrida y genera respuestas fundamentadas usando un LLM servido por Ollama. Ningún dato sale de la máquina del usuario en uso normal.

El proyecto tiene dos dimensiones:
- **Capa experimental**: entrenamiento LoRA, evaluación con métricas múltiples, trazabilidad de experimentos.
- **Sistema RAG/producción**: pipeline completo con CLI y Web.

---

## 2. Reglas de comportamiento (leer antes de actuar)

1. **NUNCA hacer commit ni push a GitHub sin preguntar al usuario primero.** Sin excepciones.
2. **Responder siempre en español** en este proyecto.
3. **Seguir los patrones de código** de la Sección 8 al escribir código nuevo.
4. **No modificar `requirements.txt`** sin confirmar con el usuario: las versiones están fijadas intencionalmente (especialmente el stack de training).
5. **No cambiar flags de pipeline** (`USAR_RECOMP_SYNTHESIS`, etc.) sin acordarlo con el usuario: tienen efecto directo en latencia y coste.
6. **No tocar `llama.cpp/`**: es un submódulo externo.
7. Al proponer cambios en `chat_pdfs.py`: `web/app.py` importa directamente constantes y funciones de ese módulo (`PATH_DB`, `COLLECTION_NAME`, `indexar_documentos`, `evaluar_pregunta_rag`…). Un renombrado rompe el backend web.
8. **Tras cualquier modificación, revisar si hay que actualizar `CLAUDE.md` o `README.md`**: cambios en estructura de archivos, flags de pipeline, modelos, scripts, rutas, estado del experimento o convenciones de código deben reflejarse en la documentación del proyecto para mantenerla sincronizada.

---

## 3. Estado actual del experimento

| Tarea | Estado |
|-------|--------|
| Evaluación base (7 modelos, 320 muestras/dataset) | ✅ Completada — `training-output/baseline/` |
| Fine-tuning Qwen3-14B (v10) | 🔄 En ejecución en cluster |
| Fine-tuning Phi-4 (v1) | 🔄 En ejecución en cluster |
| Fine-tuning Gemma-3-12B (v2) | ⏳ Pendiente — script actualizado, listo para lanzar |
| Evaluación base/adaptado (test) | ⏳ Pendiente de resultados de training |

**Modelos viables para producción**: Qwen3-14B, Phi-4 y Gemma-3-12B (scripts actualizados y compatibles con GGUF/Ollama).

**Scripts de training** (alineados con v10): `train-qwen3.py`, `train-phi4.py`, `train-gemma3.py`.

---

## 4. Estructura de archivos clave

```
localOllamaRAG/
├── generate_diagram.py           # Diagrama de arquitectura vía Kroki.io
├── rag/
│   ├── chat_pdfs.py              # Motor RAG principal: indexación, recuperación, generación
│   ├── export_fragments.py       # Exporta chunks de ChromaDB a TXT/JSONL para debug
│   ├── requirements.txt
│   ├── debug_context_issues.md   # Análisis de issues en presentación del contexto
│   ├── debug_rag/                # Dumps de debug de queries (runtime, gitignored)
│   ├── pdfs/                     # PDFs a indexar (gitignored)
│   ├── ragbench_pdfs/            # PDFs RAGBench — uso exclusivo de run_eval_ragbench.py (gitignored)
│   ├── mi_vector_db/             # ChromaDB producción (gitignored)
│   ├── ragbench_vector_db/       # ChromaDB RAGBench (gitignored)
│   ├── historial_chat.json       # Historial modo CHAT (gitignored)
│   └── cli/
│       ├── app.py                # MonkeyGrabCLI: bucle interactivo y dispatch de comandos
│       ├── display.py            # Singleton `ui`: salida visual Rich
│       ├── renderer.py           # Renderizado ANSI de bajo nivel (legacy)
│       └── theme.py              # Paleta de colores
├── web/
│   ├── app.py                    # Backend Flask: REST + SSE, sirve React
│   ├── requirements.txt
│   └── zip/                      # Fuente React + build compilado
│       ├── src/                  # Código fuente TypeScript (App.tsx, main.tsx, index.css)
│       ├── public/               # Assets estáticos (logos)
│       ├── dist/                 # Build compilado (gitignored)
│       ├── package.json          # Dependencias npm
│       ├── tsconfig.json         # Configuración TypeScript
│       └── vite.config.ts        # Configuración Vite
├── scripts/
│   ├── requirements.txt          # torch, transformers, peft, datasets, bert-score…
│   ├── training/
│   │   ├── train-qwen3.py        # LoRA Qwen3-14B (v10) ✅ actualizado
│   │   ├── train-phi4.py         # LoRA Phi-4 (v1) ✅ actualizado
│   │   ├── train-gemma3.py       # LoRA Gemma-3-12B (v2) ✅ actualizado
│   │   └── run-general.sh        # Plantilla SLURM genérica (cluster)
│   ├── evaluation/
│   │   ├── evaluate_baselines.py # Benchmark 7 modelos base (Token F1, ROUGE-L, BERTScore, CF-léxica)
│   │   ├── inspect_splits.py     # Audita tamaño de splits antes/después de filtros
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
│   └── requirements.txt          # ragas, langchain-google-genai, pandas…
├── training-output/
│   ├── qwen-3/                   # Adaptador LoRA Qwen3 (artefactos pesados gitignored)
│   │   └── plots/                # Curvas de training/eval (gitignored)
│   ├── phi-4/                    # Adaptador LoRA Phi-4 (artefactos pesados gitignored)
│   │   └── plots/                # Curvas de training/eval (gitignored)
│   ├── gemma-3/                  # Adaptador LoRA Gemma-3 (artefactos pesados gitignored)
│   │   └── plots/                # Curvas de training/eval (gitignored)
│   └── baseline/                 # Resultados benchmark 7 modelos base (320 muestras)
│       ├── baseline_evaluation.json          # 7 modelos × 5 datasets × con/sin contexto
│       ├── baseline_evaluation_samples.json  # Predicciones cualitativas por muestra
│       ├── baseline_checkpoint.json          # Checkpoint incremental (permite reanudar)
│       ├── predictions_{modelo}.json         # Predicciones por modelo (7 archivos)
│       ├── generate_reports.py               # Genera tablas Markdown + CSVs
│       ├── reports/                          # Tablas + CSVs + figuras (gitignored)
│       └── 200/                              # Versión anterior cap=200 (referencia histórica)
├── docs/
│   ├── monkeygrab_architecture.png
│   ├── monkeygrab_architecture.svg
│   ├── investigacionMetricas.md
│   ├── palabras.md
│   └── splits.md                 # Análisis de splits de datasets
├── llama-bin/                    # Binarios llama.cpp compilados para Windows (gitignored)
├── models/
│   └── gguf-output/
│       ├── qwen-3/               # GGUF Qwen3-14B + Modelfile (solo Modelfile versionado)
│       ├── phi-4/                # GGUF Phi-4 + Modelfile (solo Modelfile versionado)
│       └── gemma-3/              # GGUF Gemma-3-12B + Modelfile (solo Modelfile versionado)
├── llama.cpp/                    # Submódulo externo — no modificar (gitignored)
├── README.md
└── CLAUDE.md
```

---

## 5. Roles de modelos

Cada rol se configura vía variable de entorno; si no está definida, usa el segundo argumento de `os.getenv` en `rag/chat_pdfs.py`. **Manda siempre el entorno del proceso en ejecución**, no ningún «estado oficial» del repo.

| Rol | Variable | Descripción |
|-----|----------|-------------|
| Generador RAG | `OLLAMA_RAG_MODEL` | Genera la respuesta al usuario en modo `/rag`. Salida por streaming. |
| Chat y sub-consultas | `OLLAMA_CHAT_MODEL` | Conversación en `/chat` y hasta 3 sub-queries para búsqueda híbrida. |
| Embeddings | `OLLAMA_EMBED_MODEL` | Vectoriza chunks al indexar y la pregunta al recuperar. El path de ChromaDB incluye slug del modelo. |
| Contextual retrieval | `OLLAMA_CONTEXTUAL_MODEL` | Enriquece el texto de cada chunk antes de embebedarlo. Solo en indexación con `USAR_CONTEXTUAL_RETRIEVAL`. |
| RECOMP | `OLLAMA_RECOMP_MODEL` | Sintetiza/comprime fragmentos recuperados antes del generador. Solo con `USAR_RECOMP_SYNTHESIS`. |
| Visión / OCR | `OLLAMA_OCR_MODEL` | Describe textualmente figuras raster en PDFs. Solo con `USAR_EMBEDDINGS_IMAGEN`. |
| Reranker | `RERANKER_QUALITY` | CrossEncoder local (BGE o MiniLM). No es modelo Ollama. Solo con `USAR_RERANKER`. |

Variables de entorno de referencia (ajustar según hardware y despliegue):

| Variable | Referencia | Descripción |
|----------|------------|-------------|
| `OLLAMA_RAG_MODEL` | `Qwen3-FineTuned` | Generador RAG |
| `OLLAMA_CHAT_MODEL` | `gemma3:4b` | Modo chat |
| `OLLAMA_EMBED_MODEL` | `embeddinggemma:latest` | Embeddings |
| `OLLAMA_CONTEXTUAL_MODEL` | `gemma3:4b` | Contextual retrieval |
| `OLLAMA_OCR_MODEL` | `qwen3-vl:8b` | Descripción de imágenes |
| `DOCS_FOLDER` | `rag/pdfs/` | Carpeta de PDFs a indexar |
| `RERANKER_QUALITY` | `quality` | `quality` (BAAI/bge) o `speed` (MiniLM) |
| `HF_TOKEN` | — | Token HuggingFace (necesario para Gemma-3) |
| `GOOGLE_API_KEY` | — | API key Gemini para evaluación RAGAS |

---

## 6. Arquitectura del pipeline

### Indexación (al arrancar o con `/reindex`)

```
PDFs (CARPETA_DOCS)
  -> Extracción texto: pymupdf4llm (preferido) / pypdf (fallback)
  -> [Opt] USAR_EMBEDDINGS_IMAGEN: fitz + caption + OLLAMA_OCR_MODEL
  -> Chunking: CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH
  -> [Opt] USAR_CONTEXTUAL_RETRIEVAL: enriquecimiento con OLLAMA_CONTEXTUAL_MODEL
  -> Embedding: OLLAMA_EMBED_MODEL; prefijos EMBED_PREFIX_* si el modelo lo requiere
  -> Persistencia: ChromaDB en PATH_DB
```

### Recuperación (por cada query en modo RAG)

```
Query usuario (MIN_LONGITUD_PREGUNTA_RAG)
  -> [Opt] USAR_LLM_QUERY_DECOMPOSITION: sub-consultas vía OLLAMA_CHAT_MODEL (hasta 3)
  -> Búsqueda semántica: OLLAMA_EMBED_MODEL, N_RESULTADOS_SEMANTICOS
  -> [Opt] USAR_BUSQUEDA_HIBRIDA: búsqueda léxica, N_RESULTADOS_KEYWORD
  -> [Opt] USAR_BUSQUEDA_EXHAUSTIVA: términos críticos filtrados
  -> Fusión RRF: score_semantic + score_keyword -> score_final (pesos 55/45, no cambiar sin evaluación)
  -> [Opt] USAR_RERANKER: CrossEncoder, TOP_K_RERANK_CANDIDATES, TOP_K_AFTER_RERANK
  -> Filtrado: UMBRAL_RELEVANCIA (0.50); con reranker: UMBRAL_SCORE_RERANKER
  -> Selección: TOP_K_FINAL (6 fragmentos)
  -> [Opt] EXPANDIR_CONTEXTO: N_TOP_PARA_EXPANSION + chunks adyacentes
  -> [Opt] USAR_OPTIMIZACION_CONTEXTO: recorte con MAX_CONTEXTO_CHARS (8192)
```

### Generación

```
Pregunta + <context>
  -> [Opt] USAR_RECOMP_SYNTHESIS: sintetizar_contexto_recomp vía OLLAMA_RECOMP_MODEL
  -> OLLAMA_RAG_MODEL: streaming vía Ollama (temperature, top_p, repeat_penalty, num_ctx)
```

---

## 7. Comandos importantes

### Arrancar el sistema

```bash
# CLI
cd rag && python chat_pdfs.py

# Web
python web/app.py   # http://localhost:5000
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
| `/limpiar` / `/clear` | Limpiar historial de chat |
| `/ayuda` / `/help` | Mostrar ayuda |
| `/salir` / `/exit` | Salir guardando historial |

### Entrenamiento LoRA

```bash
# Qwen3-14B (v10) — en ejecución en cluster
python scripts/training/train-qwen3.py

# Phi-4 (v1) — en ejecución en cluster
python scripts/training/train-phi4.py

# Gemma-3-12B (v2)
python scripts/training/train-gemma3.py

# Fusionar adaptador para exportar a GGUF
python scripts/conversion/merge_lora.py --model qwen-3   # opciones: qwen-3, gemma-3, phi-4
```

### Evaluación

```bash
# Benchmark 7 modelos base (Token F1, ROUGE-L, BERTScore, CF-léxica; 320 muestras/dataset)
python scripts/evaluation/evaluate_baselines.py
# Salida: training-output/baseline/

# Regenerar tablas Markdown + CSVs desde baseline_evaluation.json
python training-output/baseline/generate_reports.py
# Salida: training-output/baseline/reports/

# Auditar splits por dataset
python scripts/evaluation/inspect_splits.py

# Exportar chunks de ChromaDB
python rag/export_fragments.py              # ambos stores
python rag/export_fragments.py --mi-only    # solo PDFs propios

# RAGAS sobre el pipeline en vivo (requiere GOOGLE_API_KEY)
python evaluation/run_eval.py
python evaluation/run_eval_ragbench.py
```

### Diagrama de arquitectura

```bash
python generate_diagram.py
python generate_diagram.py --output docs/monkeygrab_architecture.png
python generate_diagram.py --format svg --output docs/monkeygrab_architecture.svg
```

---

## 8. Patrones de código

### 1. MODULE MAP al inicio de cada módulo (obligatorio)

Todo archivo Python no trivial abre con un MODULE MAP comentado que indexa sus secciones.

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

### 2. Separadores de sección

```python
# ─────────────────────────────────────────────
# SECTION 3: GLOBAL CONFIGURATION
# ─────────────────────────────────────────────

# --- 3.1 Pipeline flags ---
```

### 3. Constantes globales al inicio, antes de cualquier lógica

Orden: imports stdlib → third-party → local, luego constantes (modelos, rutas, flags, parámetros numéricos).

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

### 6. Pipelines explícitamente por fases

Nombrar y separar visualmente cada fase: carga → preparación → inferencia → evaluación → exportación.

### 7. Salida orientada a artefactos

Los scripts experimentales siempre exportan JSON de métricas, CSV por muestra y plots. Nunca solo stdout.

### 8. Naming mixto ES/EN (convención establecida)

- Funciones de dominio RAG en español: `realizar_busqueda_hibrida`, `indexar_documentos`
- Variables de configuración en inglés: `CHUNK_SIZE`, `TOP_K_FINAL`, `UMBRAL_RELEVANCIA`
- Docstrings y comentarios en inglés
- Seguir el patrón del módulo donde se trabaje; no mezclar dentro del mismo bloque.

### 9. Scripts de entrenamiento con VERSION HISTORY

Incluir un bloque `VERSION HISTORY` al inicio con cambios de versión y justificación técnica.

### 10. Script-first, no arquitectura enterprise

La lógica principal vive en módulos + `main()`. La única excepción es `MonkeyGrabCLI` (cli/app.py).

### 11. Tablas de resultados — sin desviación típica, mejora relativa

Las tablas del TFG no incluyen desviación típica. Para cuantificar el efecto del fine-tuning:
- **Δ absoluto** (pp): `adaptado − base`. Ej: `+3.2 pp`.
- **Δ relativo** (%): `(adaptado − base) / base × 100`. Ej: `+7.4 %`.

Los artefactos JSON (`evaluation_comparison.json`) incluyen ambos campos (`delta_pp`, `delta_rel_pct`).

### Docstring de módulo

```python
"""
NombreModulo -- descripción corta.

Explicación del propósito, fases del pipeline implementadas y decisiones de diseño.

Usage:
    python nombre_modulo.py [--arg valor]

Dependencies:
    - paquete1, paquete2
"""
```

### Docstring de función (estilo Google)

```python
def realizar_busqueda_hibrida(pregunta: str, collection) -> tuple:
    """Execute hybrid search combining semantic and keyword strategies.

    Args:
        pregunta: User question string.
        collection: ChromaDB collection to search against.

    Returns:
        Tuple of (ranked_fragments, best_score, metrics_dict).

    Raises:
        ValueError: If the collection is empty or None.
    """
```

### Flags de pipeline documentados en el mismo bloque

```python
# --- 3.3 Pipeline flags ---
USAR_CONTEXTUAL_RETRIEVAL = True   # enrich chunks with LLM context before indexing
USAR_LLM_QUERY_DECOMPOSITION = True  # decompose query into 3 sub-queries
USAR_RERANKER = RERANKER_AVAILABLE   # disabled automatically if sentence-transformers missing
USAR_RECOMP_SYNTHESIS = False        # experimental; toggle in this file / env
```

---

## 9. Dependencias y entorno

### Prerrequisitos del sistema

- Python 3.10+
- Ollama instalado y en ejecución local
- GPU con CUDA recomendada para reranking y fine-tuning (~24 GB VRAM para Qwen3-14B)

### Instalación

```bash
pip install -r rag/requirements.txt          # núcleo RAG (obligatorio)
pip install -r web/requirements.txt          # interfaz web (opcional)
pip install -r evaluation/requirements.txt   # evaluación RAGAS (opcional)
pip install -r scripts/requirements.txt      # fine-tuning (opcional, requiere GPU)
```

### Stack de training (versiones fijadas — no modificar sin confirmación)

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

## 10. Deuda técnica y limitaciones

### Deuda activa

- **Lógica concentrada en `chat_pdfs.py`**: supera las 1000 líneas. Refactorizar requeriría actualizar imports en `web/app.py` y `evaluation/run_eval.py`.

### Limitaciones conocidas
- **RECOMP synthesis**: implementado pero aumenta latencia; medir antes de activar en producción.
- **Evaluación RAGAS**: requiere `GOOGLE_API_KEY` (Gemini como juez LLM). No es totalmente local.
- **`enable_thinking=False` en Qwen3**: en producción se suprime el bloque `<think>` para reducir latencia. Ver `scripts/tests/test_nothink.py`.

### Decisiones de diseño fijadas (no cambiar sin evaluación)

- **RRF fusion 55/45**: ponderación semántica/léxica calibrada empíricamente.
- **`UMBRAL_RELEVANCIA = 0.50`**: umbral mínimo de relevancia RAG.
- **`TOP_K_FINAL = 6`**: fragmentos que llegan al LLM. Contexto máximo: `MAX_CONTEXTO_CHARS = 8192`.
- **ChromaDB por `(carpeta_docs, embedding_model)`**: cambiar el modelo de embedding invalida la DB y requiere reindexar.

---

## 11. Artefactos de evaluación

| Artefacto | Ubicación | Descripción |
|-----------|-----------|-------------|
| Baseline completo (320 muestras) | `training-output/baseline/baseline_evaluation.json` | 7 modelos × 5 datasets × con/sin contexto; Token F1, ROUGE-L, BERTScore, CF-léxica |
| Predicciones baseline | `training-output/baseline/predictions_{modelo}.json` | 7 archivos; permite recomputar métricas |
| Tablas de resultados | `training-output/baseline/reports/` | Markdown + CSVs + figuras; generado por `generate_reports.py` |
| Baseline 200-sample (histórico) | `training-output/baseline/200/` | Versión anterior cap=200; solo referencia |
| Artefactos LoRA Qwen3-14B | `training-output/qwen-3/` | Pendiente de training: `training_stats.json`, `evaluation_comparison.json`, `predictions_base.json`, `predictions_adapted.json` (gitignored) |
| Artefactos LoRA Phi-4 | `training-output/phi-4/` | Pendiente de training: mismos artefactos que Qwen3 (gitignored) |
| Artefactos LoRA Gemma-3-12B | `training-output/gemma-3/` | Pendiente de training: mismos artefactos que Qwen3 (gitignored) |
| Diagrama arquitectura | `docs/monkeygrab_architecture.png` / `.svg` | Generado por `generate_diagram.py` |
