# CLAUDE.md — MonkeyGrab (localOllamaRAG)

Contexto permanente para Claude Code en este repositorio.
Responde siempre en **español** en este proyecto.

---

## Descripcion del proyecto

**MonkeyGrab** es un sistema RAG (Retrieval-Augmented Generation) completamente local desarrollado como Trabajo de Fin de Grado (TFG) en la Universitat Politecnica de Valencia (UPV). El sistema indexa PDFs academicos, recupera fragmentos relevantes mediante busqueda hibrida y genera respuestas fundamentadas usando un LLM fine-tuneado servido por Ollama. Ningun dato sale de la maquina del usuario en uso normal.

El proyecto combina dos dimensiones:
- **Nucleo RAG/servicio local**: pipeline de produccion con CLI y web (chat_pdfs.py, app.py)
- **Capa de investigacion experimental**: entrenamiento LoRA, evaluacion con metricas multiples, trazabilidad de experimentos

Modelos en uso:

| Rol | Modelo | Detalle |
|-----|--------|---------|
| Generador | Qwen3-14B-FineTuned | Q4_K_M via llama.cpp/Ollama. Modelo elegido para produccion |
| Embedding | embeddinggemma | Gemma 3, 307M params, 768-d, BF16 |
| Reranker | BAAI/bge-reranker-v2-m3 | Cross-encoder ~200M. Alt rapida: ms-marco-MiniLM-L-6-v2 |
| Auxiliar | gemma3:4b | Descomposicion de query, contextual retrieval, RECOMP |
| Vision | llama3.2-vision:11b | Descripcion de figuras en PDFs |

---

## Estructura de archivos clave

```
localOllamaRAG/
├── rag/
│   ├── chat_pdfs.py              # Motor RAG principal: indexacion, recuperacion, generacion
│   ├── requirements.txt          # Dependencias del nucleo RAG
│   ├── pdfs/                     # PDFs a indexar con el sistema RAG (no versionados)
│   ├── ragbench_pdfs/            # PDFs del benchmark RAGBench — uso exclusivo de run_eval_ragbench.py (gitignored)
│   ├── mi_vector_db/             # ChromaDB persistente (gitignored)
│   ├── historial_chat.json       # Historial CHAT mode (gitignored)
│   └── cli/
│       ├── app.py                # Clase MonkeyGrabCLI: bucle interactivo y dispatch de comandos
│       ├── display.py            # Singleton `ui`: toda la salida visual Rich (panels, tables, spinners)
│       ├── renderer.py           # Renderizado ANSI de bajo nivel (legacy)
│       └── theme.py              # Paleta de colores MonkeyGrab
├── web/
│   ├── app.py                    # Backend Flask: REST + SSE, sirve React
│   ├── requirements.txt          # flask, flask-cors
│   └── zip/dist/                 # Build React (assets estaticos)
├── scripts/
│   ├── training/
│   │   ├── train-qwen3.py        # Fine-tuning LoRA Qwen3-14B (v9) - MODELO DE PRODUCCION
│   │   ├── train-llama3.1.py     # Fine-tuning LoRA Llama-3.1-8B
│   │   ├── train-gemma3.py       # Fine-tuning LoRA Gemma-3-12B-IT
│   │   ├── plot_training.py      # Visualizacion de curvas de entrenamiento
│   │   ├── run-qwen3.sh          # Script de lanzamiento en cluster (SLURM)
│   │   ├── run-llama3.sh         # Script de lanzamiento en cluster (SLURM)
│   │   ├── run-gemma3.sh         # Script de lanzamiento en cluster (SLURM)
│   │   └── requirements.txt      # torch, transformers, peft, datasets, bert-score, etc.
│   ├── evaluation/
│   │   ├── evaluate_baselines.py # Benchmark de 7 modelos base (Token F1, BERTScore, CF-lexica)
│   │   ├── compute_std.py        # Analisis mean +/- sigma sobre CSVs (historico; std no va en tablas TFG)
│   │   └── run.sh                # Script de lanzamiento en cluster (SLURM)
│   ├── conversion/
│   │   ├── merge_lora.py         # Fusiona adaptador LoRA con base para exportar a GGUF
│   │   ├── build_ollama.bat      # Automatiza creacion del modelo en Ollama (Windows)
│   │   └── quantize_to_q4km.ps1  # Cuantiza modelo merged a Q4_K_M con llama-bin
│   ├── tests/
│   │   ├── test_nothink.py       # Test supresion de <think> en Qwen3 via Ollama
│   │   └── test_ollama_stream_nothink.py
│   └── generate_diagram.py       # Renderiza el diagrama de arquitectura via Kroki.io
├── evaluation/
│   ├── run_eval.py               # Evaluacion RAGAS del pipeline RAG en vivo
│   ├── run_eval_ragbench.py      # Evaluacion RAGAS sobre PDFs de RAGBench (Vectara)
│   └── requirements.txt          # ragas, langchain-google-genai, pandas, etc.
├── training-output/
│   ├── qwen-3/                   # Adaptador LoRA + artefactos de eval (JSON, CSV, plots)
│   │   └── explore-rank/         # Artefactos de la exploracion de rango LoRA (r=32 vs r=64)
│   ├── llama-3/                  # Adaptador LoRA Llama-3.1
│   ├── gemma-3/                  # Adaptador LoRA Gemma-3
│   ├── bertscore/                # CSVs por muestra, resumen BERTScore (DATOS OBSOLETOS — re-ejecutar)
│   │   └── plots/                # Graficas BERTScore generadas
│   └── baseline/                 # Resultados del benchmark de modelos base (JSONs, plots)
├── docs/
│   ├── monkeygrab_architecture.png       # Diagrama de arquitectura (PNG)
│   ├── monkeygrab_architecture.svg       # Diagrama de arquitectura (SVG)
│   ├── tensor.pdf                        # Guia de uso del cluster tensor (UPV)
│   ├── EVALUACION_DATASETS_PUBLICOS.md  # Documentacion del protocolo de evaluacion publica
│   └── MODIFICACIONES_EVALUACION.md     # Registro de cambios en scripts de eval/training (v7.2→v9)
├── llama-bin/                    # Binarios llama.cpp compilados para Windows (llama-quantize, etc.)
├── models/
│   ├── gguf-output/              # Modelos GGUF cuantizados (gitignored)
│   └── merged-model/             # Modelos merged antes de convertir (gitignored)
├── llama.cpp/                    # Submodulo llama.cpp para conversion/cuantizacion GGUF
└── README.md
```

---

## Arquitectura del pipeline

### Pipeline de indexacion (al arrancar o con /reindex)

```
PDFs (rag/pdfs/)
  -> Extraccion texto: pymupdf4llm (preferido) / pypdf (fallback)
  -> Extraccion imagenes: fitz (PyMuPDF) + descripcion con llama3.2-vision
  -> Chunking jerarquico: 1500 chars, 350 overlap, min 80 chars
  -> [Opt] Contextual retrieval: enriquecimiento de chunks con gemma3:4b
  -> Embedding: embeddinggemma (768-d)
  -> Almacenamiento: ChromaDB persistente
```

### Pipeline de recuperacion (por cada query en modo RAG)

```
Query usuario
  -> [Opt] Descomposicion LLM: 3 sub-queries con gemma3:4b
  -> Busqueda semantica: ChromaDB top-80
  -> Busqueda keyword: $contains top-40
  -> [Opt] Busqueda exhaustiva por terminos criticos
  -> Fusion RRF: 55% semantica + 45% lexica
  -> [Opt] Reranking: BAAI/bge-reranker-v2-m3 (CrossEncoder)
  -> Top-6 fragmentos finales
  -> [Opt] Expansion con chunks adyacentes
  -> Optimizacion de contexto (limpieza de artefactos PDF)
```

### Generacion

```
System prompt + <context> + pregunta
  -> Qwen3-14B-FineTuned (Q4_K_M, temp=0.15, Ollama)
  -> Streaming token a token
```

---

## Comandos importantes

### Arrancar el sistema

```bash
# CLI (desde la raiz del proyecto)
cd rag
python chat_pdfs.py

# Web (desde la raiz del proyecto)
python web/app.py
# Abre http://localhost:5000
```

### Comandos CLI en tiempo de ejecucion

| Comando | Descripcion |
|---------|-------------|
| `/rag` | Modo RAG (consulta de documentos) |
| `/chat` | Modo CHAT (conversacion general) |
| `/docs` | Lista documentos indexados |
| `/temas` | Resumen de topicos por documento |
| `/stats` | Estadisticas de la base de datos vectorial |
| `/reindex` | Borrar DB y reindexar todos los PDFs |
| `/limpiar` o `/clear` | Limpiar historial de chat |
| `/ayuda` o `/help` | Mostrar ayuda |
| `/salir` o `/exit` | Salir guardando historial |

### Entrenamiento LoRA

```bash
# Qwen3-14B (modelo de produccion, requiere ~24GB VRAM)
python scripts/training/train-qwen3.py

# Llama-3.1-8B
python scripts/training/train-llama3.1.py

# Gemma-3-12B-IT (requiere HF_TOKEN, no compatible con Ollama via GGUF)
python scripts/training/train-gemma3.py

# Fusionar adaptador para exportar a GGUF
python scripts/conversion/merge_lora.py --model qwen-3
# Opciones: qwen-3, llama-3, gemma-3

# Visualizar curvas de entrenamiento
python scripts/training/plot_training.py --model qwen-3
```

### Evaluacion

```bash
# Benchmark de 7 modelos base — Token F1, BERTScore, CF-lexica (dev + test, con/sin contexto)
# Aina evaluado desglosado por idioma: Aina-EN, Aina-ES, Aina-CA
python scripts/evaluation/evaluate_baselines.py
# Salida en: baseline-evaluation-output/  (generado, no versionado)

# Estadisticas mean +/- std sobre los CSVs de training-output/bertscore/ (historico)
python scripts/evaluation/compute_std.py

# RAGAS sobre el pipeline en vivo (necesita GOOGLE_API_KEY en .env)
python evaluation/run_eval.py
python evaluation/run_eval.py --dataset evaluation/mi_dataset.json --verbose

# RAGAS sobre PDFs de RAGBench (Vectara)
python evaluation/run_eval_ragbench.py

# NOTA: BERTScore ya NO requiere un script separado.
# Se computa dentro de train-qwen3.py (Section 12) y evaluate_baselines.py.
```

### Generacion de diagrama de arquitectura

```bash
python scripts/generate_diagram.py
python scripts/generate_diagram.py --output docs/monkeygrab_architecture.png
python scripts/generate_diagram.py --format svg --output docs/monkeygrab_architecture.svg
```

### Variables de entorno relevantes

| Variable | Default | Descripcion |
|----------|---------|-------------|
| `OLLAMA_RAG_MODEL` | `Qwen3-FineTuned` | Modelo generador RAG |
| `OLLAMA_CHAT_MODEL` | `gemma3:4b` | Modelo modo chat |
| `OLLAMA_EMBED_MODEL` | `embeddinggemma:latest` | Modelo de embeddings |
| `OLLAMA_CONTEXTUAL_MODEL` | `gemma3:4b` | Modelo contextual retrieval |
| `OLLAMA_VISION_MODEL` | `llama3.2-vision:11b` | Modelo de vision para imagenes |
| `DOCS_FOLDER` | `rag/pdfs/` | Carpeta de PDFs a indexar |
| `RERANKER_QUALITY` | `quality` | `quality` (BAAI/bge) o `speed` (MiniLM) |
| `HF_TOKEN` | — | Token HuggingFace (necesario para Gemma-3) |
| `GOOGLE_API_KEY` | — | API key Gemini para evaluacion RAGAS |

---

## Patrones de programacion — seguir al escribir codigo nuevo

### 1. MODULE MAP al inicio de cada modulo (OBLIGATORIO)

Todo archivo Python no trivial abre con un MODULE MAP comentado que indexa sus secciones. **Todos los archivos del proyecto ya siguen este patron** (verificado 2026-03-28). Al anadir un archivo nuevo, incluir el MODULE MAP desde el inicio.

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

### 2. Separadores de seccion con nombre en mayusculas

```python
# ─────────────────────────────────────────────
# SECTION 3: GLOBAL CONFIGURATION
# ─────────────────────────────────────────────
```

Sub-secciones con guiones simples:

```python
# --- 3.1 Pipeline flags ---
```

### 3. Constantes globales al inicio, antes de cualquier logica

Orden: imports stdlib -> third-party -> local, luego constantes globales (modelos, rutas, flags, parametros numericos).

```python
MODELO_RAG = os.getenv("OLLAMA_RAG_MODEL", "Qwen3-FineTuned")
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 350
USAR_RERANKER = True
```

### 4. Inicializacion defensiva del entorno antes de imports pesados

Variables CUDA/Triton se fijan ANTES de importar torch o transformers:

```python
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TRITON_DISABLE"] = "1"

import torch  # despues
```

### 5. Dependencias opcionales con try/except + flag booleano

```python
try:
    import pymupdf4llm
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
```

La logica posterior lee el flag, nunca relanza el ImportError.

### 6. Pipelines explicitamente por fases

Nombrar y separar visualmente cada fase: carga -> preparacion -> inferencia -> evaluacion -> exportacion de artefactos.

### 7. Salida orientada a artefactos

Los scripts experimentales siempre exportan: JSON de metricas, CSV por muestra, plots. Nunca solo stdout.

### 11. Tablas de resultados — sin desviacion tipica, preferir mejora relativa

Por indicacion del tutor: las tablas de resultados del TFG **no incluyen desviacion tipica**. El formato preferido para cuantificar el efecto del fine-tuning es:

- **Δ absoluto** (en pp): diferencia entre base y adaptado. Ej: `+3.2 pp`.
- **Δ relativo** (en %): `(adaptado - base) / base × 100`. Ej: `+7.4 %`.

Los artefactos JSON (`evaluation_comparison.json`) ya incluyen ambos campos (`delta_pp` y `delta_rel_pct`). No es necesario recomputarlos.

El script `compute_std.py` existe por razones historicas; **no usarlo para tablas definitivas** del TFG.

### 8. Naming mixto ES/EN (convension establecida del proyecto)

- Funciones de dominio RAG en espanol: `realizar_busqueda_hibrida`, `indexar_documentos`, `cargar_historial`
- Variables de configuracion en ingles: `CHUNK_SIZE`, `TOP_K_FINAL`, `UMBRAL_RELEVANCIA`
- Docstrings y comentarios en ingles
- Al anadir codigo nuevo: seguir el patron del modulo donde se trabaje (no mezclar dentro del mismo bloque)

### 9. Scripts de entrenamiento con VERSION HISTORY

Los scripts de entrenamiento incluyen un bloque `VERSION HISTORY` al inicio documentando cambios de version con justificacion tecnica.

### 10. Script-first, no arquitectura enterprise

La logica principal vive en modulos + main(), no en jerarquias de clases/dominios separados. La unica excepcion es `MonkeyGrabCLI` (rag/cli/app.py) que encapsula el bucle CLI.

---

## Patrones de documentacion — seguir al escribir codigo nuevo

### Docstring de modulo

```python
"""
NombreModulo -- descripcion corta en una linea.

Explicacion extendida del proposito, las fases del pipeline que implementa,
y cualquier decision de disenyo relevante.

Usage:
    python nombre_modulo.py [--arg valor]

Dependencies:
    - paquete1, paquete2
    - modulo_interno (proyecto)
"""
```

### Docstring de funcion (estilo Google)

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

Solo para logica no obvia. Nunca comentar lo que el codigo ya dice:

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
USAR_RECOMP_SYNTHESIS = False        # experimental: disabled by default
```

---

## Reglas de comportamiento para Claude

1. **NUNCA hacer commit ni push a GitHub sin preguntar al usuario primero.** Esta regla no tiene excepciones.
2. **Responder siempre en espanol** en este proyecto.
3. **Seguir los patrones de programacion y documentacion** descritos arriba al escribir codigo nuevo.
4. **No modificar requirements.txt** sin confirmar con el usuario: las versiones estan fijadas intencionalmente (especialmente el stack de training).
5. **No activar `USAR_RECOMP_SYNTHESIS = True`** por defecto: esta deshabilitado intencionalmente en produccion.
6. **No tocar llama.cpp/**: es un submodulo externo, no codigo del proyecto.
7. Al proponer cambios en `chat_pdfs.py`: tener en cuenta que `web/app.py` importa directamente constantes y funciones de ese modulo (`PATH_DB`, `COLLECTION_NAME`, `indexar_documentos`, `evaluar_pregunta_rag`, etc.). Un renombrado rompe el backend web.
8. **Los datos en `training-output/bertscore/` son obsoletos** (version anterior del experimento). No citarlos como resultados definitivos. Los resultados definitivos vendran de re-ejecutar `train-qwen3.py` y `evaluate_baselines.py` en cluster con el nuevo esquema (Aina por idioma, BERTScore integrado).

---

## Dependencias y requisitos del entorno

### Prerrequisitos del sistema

- Python 3.10+
- [Ollama](https://ollama.ai/) instalado y en ejecucion local
- GPU con CUDA recomendada para reranking y fine-tuning; CPU funciona para inferencia
- ~24GB VRAM para fine-tuning de Qwen3-14B

### Instalacion

```bash
# Nucleo RAG (obligatorio)
pip install -r rag/requirements.txt

# Interfaz web (opcional)
pip install -r web/requirements.txt

# Evaluacion RAGAS (opcional)
pip install -r evaluation/requirements.txt

# Fine-tuning (opcional, requiere GPU)
pip install -r scripts/training/requirements.txt
```

### Modelos Ollama necesarios

```bash
ollama pull Qwen3-FineTuned        # o el nombre del modelo fine-tuneado
ollama pull embeddinggemma
ollama pull gemma3:4b
ollama pull llama3.2-vision:11b    # solo si se usan imagenes en PDFs
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

## Deuda tecnica conocida y consideraciones de disenyo

### Deuda activa

- **Duplicacion entre scripts de entrenamiento**: `train-qwen3.py` esta en v9 (con Aina por idioma, BERTScore integrado, dev+test separados). `train-llama3.1.py` y `train-gemma3.py` estan desactualizados respecto a ese esquema. Si se repiten los experimentos de Llama y Gemma, hay que alinearlos con la logica de Qwen3 v9.
- **`training-output/bertscore/`**: los CSVs y JSON existentes corresponden a la version anterior del experimento (Aina como bloque unico, caps de muestras). Son datos obsoletos; seran reemplazados cuando finalicen los experimentos en cluster. No usar para tablas definitivas del TFG.
- **Mezcla de idioma en nombres**: funciones de dominio en espanol (`realizar_busqueda_hibrida`) junto a constantes en ingles (`CHUNK_SIZE`). Es una convencion establecida, no un error, pero puede resultar sorprendente.
- **Logica concentrada en scripts largos**: `chat_pdfs.py` supera las 1000 lineas. Toda la logica RAG vive ahi. Refactorizar requeriria actualizar los imports en `web/app.py` y `evaluation/run_eval.py`.

### Limitaciones conocidas

- **Gemma-3-12B**: el adaptador LoRA no puede desplegarse via Ollama por incompatibilidad GGUF. El adaptador existe y esta publicado, pero no se usa en produccion.
- **RECOMP synthesis** (`USAR_RECOMP_SYNTHESIS`): implementado pero deshabilitado por defecto. Aumenta latencia sin mejora consistente.
- **Evaluacion RAGAS**: requiere `GOOGLE_API_KEY` (Gemini 2.0 Flash como juez LLM). No es totalmente local.
- **Re-generacion de datos de evaluacion por muestra**: los artefactos de `training-output/bertscore/` se regeneran re-ejecutando `train-qwen3.py` completo (Section 12 genera BERTScore). No hay script separado de BERTScore.

### Decisiones de disenyo relevantes

- **RRF fusion 55/45**: la ponderacion semantica/lexica fue calibrada empiricamente. No cambiar sin evaluacion.
- **`UMBRAL_RELEVANCIA = 0.50`**: umbral minimo para que un resultado RAG se considere relevante. Bajarlo aumenta recall pero introduce ruido.
- **`TOP_K_FINAL = 6`**: numero de fragmentos que llegan al LLM. El contexto maximo es `MAX_CONTEXTO_CHARS = 8192`.
- **ChromaDB persistente por combinacion `(carpeta_docs, embedding_model)`**: el path de la DB incluye el slug del modelo de embedding. Cambiar el modelo de embedding invalida la DB existente y requiere reindexar.
- **`enable_thinking=False` en Qwen3**: el modelo fine-tuneado usa razonamiento interno. La inferencia en produccion suprime el bloque `<think>` para reducir latencia y tokens de salida. Ver `scripts/tests/test_nothink.py`.

---

## Artefactos de evaluacion

Los resultados de los experimentos estan en `training-output/`. Los datos marcados como (OBSOLETOS) corresponden a la version anterior del experimento (Aina como bloque unico, caps de muestras); seran reemplazados al terminar el cluster.

| Artefacto | Ubicacion | Descripcion |
|-----------|-----------|-------------|
| Adaptador LoRA Qwen3 | `training-output/qwen-3/` | `adapter_config.json`, `adapter_model.safetensors` |
| Comparacion base/adaptado | `training-output/qwen-3/evaluation_comparison.json` | Deltas per-dataset con Token F1, BERTScore F1, CF-lexica; separados por dev/test |
| Estadisticas de training | `training-output/qwen-3/training_stats.json` | Loss, pasos, tiempo (v9) |
| Predicciones checkpoint | `training-output/qwen-3/predictions_base.json` / `predictions_adapted.json` | Predicciones + Token F1 + CF-lexica por muestra; permite recomputar BERTScore sin regenerar |
| BERTScore checkpoint | `training-output/qwen-3/bertscore_checkpoint.json` | BERTScore P/R/F1 por dataset (BASE y ADAPTED) |
| Exploracion rango LoRA | `training-output/qwen-3/explore-rank/` | Artefactos comparacion r=32 vs r=64 |
| Baseline completo | `training-output/baseline/baseline_evaluation.json` | Token F1, BERTScore F1, CF-lexica para 6 modelos x 5 datasets x dev/test x con/sin contexto |
| Baseline checkpoint | `training-output/baseline/baseline_checkpoint.json` | Checkpoint incremental (permite reanudar si falla) |
| BERTScore por muestra (OBSOLETOS) | `training-output/bertscore/bertscore_per_sample_*.csv` | Datos de evaluacion anterior; seran reemplazados |
| Metricas por muestra (OBSOLETOS) | `training-output/bertscore/metrics_per_sample_*.csv` | Datos de evaluacion anterior; seran reemplazados |
| Sigma table (OBSOLETOS) | `training-output/bertscore/sigma_table.log` / `.tex` | Basada en datos anteriores; regenear con `compute_std.py` tras nuevo cluster |
| Diagrama arquitectura | `docs/monkeygrab_architecture.png` / `.svg` | Generado por `scripts/generate_diagram.py` |
