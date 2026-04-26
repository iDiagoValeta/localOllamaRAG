---
description: 
alwaysApply: true
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<!-- Internal title: MonkeyGrab (localOllamaRAG) -->
<!-- Última actualización: 2026-04-21 -->

## 1. Descripción del proyecto

**MonkeyGrab** es un sistema RAG (Retrieval-Augmented Generation) completamente local desarrollado como TFG del **Grado en Ingeniería Informática** de la ETSINF, Universitat Politècnica de València. Autor: Ignacio Diago Valeta. Tutor: Adrià Giménez Pastor. Curso 2025-2026. Indexa PDFs, recupera fragmentos relevantes mediante búsqueda híbrida y genera respuestas fundamentadas usando un LLM servido por Ollama. Ningún dato sale de la máquina del usuario en uso normal.

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
7. Al proponer cambios en `chat_pdfs.py`: `web/app.py` importa el módulo como `rag_engine` y accede a constantes/funciones vía `rag_engine.*`; `evaluation/run_eval.py` y `scripts/tests/` también consumen símbolos directos. Un renombrado rompe el backend web, el runner de evaluación y los tests. **API pública completa usada por consumidores externos**:
   - Constantes: `PATH_DB`, `COLLECTION_NAME`, `CARPETA_DOCS`, `SYSTEM_PROMPT_CHAT`, `MAX_HISTORIAL_MENSAJES`, `MODELO_CHAT`, `MODELO_RAG`, `MIN_LONGITUD_PREGUNTA_RAG`, `UMBRAL_RELEVANCIA`, `UMBRAL_SCORE_RERANKER`, `TOP_K_FINAL`, `EXPANDIR_CONTEXTO`, `N_TOP_PARA_EXPANSION`, `MAX_CONTEXTO_CHARS`, `USAR_RERANKER`, `USAR_RECOMP_SYNTHESIS`, `RERANKER_AVAILABLE`, `STOPWORDS`
   - Funciones (CLI/web): `indexar_documentos`, `realizar_busqueda_hibrida`, `expandir_con_chunks_adyacentes`, `sintetizar_contexto_recomp`, `construir_contexto_para_modelo`, `guardar_debug_rag`, `generar_respuesta_silenciosa`, `obtener_documentos_indexados`, `cargar_historial`, `guardar_historial`, `limpiar_historial`
   - Funciones (runner de evaluación y tests): `get_pipeline_flags`, `set_pipeline_flags`, `set_docs_folder_runtime`, `set_ragbench_reranker_low_score_fallback`, `evaluar_pregunta_rag`
8. **Tras cualquier modificación, revisar si hay que actualizar `CLAUDE.md` o `README.md`**: cambios en estructura de archivos, flags de pipeline, modelos, scripts, rutas, estado del experimento o convenciones de código deben reflejarse en la documentación del proyecto para mantenerla sincronizada. **`README.md` es la puerta de entrada pública**: si el árbol de carpetas, rutas por defecto (`rag/docs/`, `rag/vector_db/`, evaluación con `evaluation/run_eval.py`, etc.) o comandos cambian en el código o en commits recientes, actualizar la sección *Repository structure* y las rutas citadas en el README en el mismo cambio (o justo después), para que no quede desfasado respecto a este archivo y al repo.
9. **Cambios en `.gitignore` o `.gitkeep`**: seguir la **Sección 11** (patrones `training-output/`, carpetas con `.gitkeep`, GGUF fuera del repo). Tras editar reglas, validar con `git check-ignore -v <ruta>` y actualizar la Sección 11 si cambia la política.

---

## 3. Estado actual del experimento

| Tarea | Estado |
|-------|--------|
| Evaluación base (7 modelos, 320 muestras/dataset) | ✅ Completada — `training-output/baseline/` |
| Fine-tuning Qwen3-14B (v10) | ✅ Artefactos locales en `training-output/qwen-3/` (checkpoints y pesos gitignored; métricas versionables) |
| Fine-tuning Phi-4 (v1) | ✅ Run principal r=32 en `training-output/phi-4/`; exploración r=64 en `phi-4/64/`; r=16 en `phi-4/16/` (artefactos completos: `training_stats.json`, `evaluation_comparison.json`) |
| Fine-tuning Gemma-3-12B (v2) | ✅ Completado — artefactos en `training-output/gemma-3/` (`training_stats.json`, `evaluation_comparison.json` versionados). Conversión GGUF bloqueada por incompatibilidad de tokenizer SentencePiece en Ollama 0.21+ (ver `scripts/conversion/GEMMA3_CONVERSION_ISSUE.md`). |
| Evaluación base/adaptado (test) | ✅ `evaluation_comparison.json` versionado en: Qwen3 (raíz), Phi r=32 (raíz), Phi r=64 (`phi-4/64/`), Phi r=16 (`phi-4/16/`), Gemma-3 (raíz). |

**Modelos viables para producción**: Qwen3-14B, Phi-4 y Gemma-3-12B (scripts actualizados y compatibles con GGUF/Ollama).

**Scripts de training** (alineados con v10): `train-qwen3.py`, `train-phi4.py`, `train-gemma3.py`.

### Estado del pipeline RAG (actualizado 2026-04-10 ~18:00)

Pipeline en estado de producción tras una sesión de mejoras completa. Reindexado con los nuevos parámetros.

| Parámetro | Valor actual |
|-----------|-------------|
| `CHUNK_SIZE` | 2000 |
| `CHUNK_OVERLAP` | 400 |
| `MIN_CHUNK_LENGTH` | 150 |
| `TOP_K_FINAL` | 8 |
| `RRF_K` | 20 |
| `UMBRAL_SCORE_RERANKER` | 0.55 |
| `MAX_CONTEXTO_CHARS` | 12000 |
| `num_ctx` (generación) | 16384 |
| `USAR_RECOMP_SYNTHESIS` | `True` |
| `USAR_CONTEXTUAL_RETRIEVAL` | `True` |
| `USAR_RERANKER` | `True` |

**Calidad validada**: 4/5 preguntas de prueba respondidas correctamente sobre `att.pdf`. El único caso parcial (Q3, pregunta multi-parte con 3 sub-tipos de atención) queda fuera del alcance de la evaluación del TFG, que se centra en preguntas de concepto único.

---

## 4. Estructura de archivos clave

Nota de evaluacion: `evaluation/evaluate_ragas_bertscore.py` es el postproceso BERTScore para salidas RAGAS ya completadas. Lee CSVs de `evaluation/runs/ragas/single/`, `evaluation/runs/ragas/comparisons/`, `evaluation/runs/ragas/ragbench/` y `evaluation/runs/ragas/ragbench_visual/`, usa `microsoft/deberta-xlarge-mnli` con `rescale_with_baseline=True` para todos los idiomas y escribe resultados separados en `evaluation/runs/bertscore/`. No ejecuta inferencia ni RAGAS. Para RagBench visual, `evaluation/run_ragbench_visual_inference.py --ragas-only` evalua con RAGAS un JSON de inferencia ya completado sin regenerar respuestas.

```
localOllamaRAG/
├── generate_diagram.py           # Diagrama de arquitectura vía Kroki.io
├── rag/
│   ├── chat_pdfs.py              # Motor RAG principal: indexación, recuperación, generación
│   ├── show_fragments/
│   │   └── export_fragments.py   # Exporta chunks de ChromaDB a TXT/JSONL para debug
│   ├── requirements.txt
│   ├── debug_rag/                # Dumps de debug de queries (runtime, gitignored)
│   ├── docs/                     # PDFs por corpus (gitignored en contenido; .gitkeep en es/ca/en, .gitignore local en los ragbench — ver §11.7)
│   │   ├── es/                   # PDFs castellano (corpus por defecto)
│   │   ├── ca/                   # PDFs catalán
│   │   ├── en/                   # PDFs inglés genérico
│   │   ├── en_ragbench_dev/      # Split dev congelado de RagBench EN
│   │   ├── en_ragbench_eval/     # Corpus final ampliable de RagBench EN
│   │   └── en_ragbench_visual/   # Corpus RagBench EN para preguntas de tablas/imágenes
│   ├── vector_db/                # ChromaDB por corpus (gitignored; creada al indexar; PATH_DB = `{basename(docs)}_{embed_slug}`)
│   │   ├── es_embeddinggemma/    # Índice castellano
│   │   ├── ca_embeddinggemma/    # Índice catalán
│   │   ├── en_ragbench_dev_embeddinggemma/   # Índice dev RagBench EN
│   │   ├── en_ragbench_eval_embeddinggemma/  # Índice eval RagBench EN
│   │   └── en_ragbench_visual_embeddinggemma/ # Índice tablas/imágenes RagBench EN
│   ├── historial_chat.json       # Historial modo CHAT (gitignored)
│   └── cli/
│       ├── app.py                # MonkeyGrabCLI: bucle interactivo, dispatch, health check Ollama, stats de sesión
│       ├── display.py            # Singleton `ui`: Rich/ANSI/plain + QueryTimer + SessionStats + Palette unificada
│       └── commands.py           # Fuente única de slash-commands (listado + alias) para dispatch/ayuda/autocompletado
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
│   ├── hf_upload_model_cards.py  # Sube model cards al Hub; --upload-qwen-q4-gguf para el GGUF de Qwen
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
│   │   ├── quantize_to_q4km.ps1  # Cuantiza modelo merged a Q4_K_M con llama-bin
│   │   └── GEMMA3_CONVERSION_ISSUE.md  # Problema tokenizer SentencePiece en Gemma-3 → GGUF (Ollama 0.21+)
│   └── tests/
│       ├── test_nothink.py                 # Test supresión de <think> en Qwen3 vía Ollama
│       ├── test_ollama_stream_nothink.py
│       ├── test_gemma4_aux_nothink.py      # Gemma 4 (ej. e4b): think en /api/generate y /api/chat
│       ├── debug_aux_subqueries.py         # Salida cruda del auxiliar (sub-queries) en terminal
│       ├── test_image_rag.py               # Tests de pipeline con imágenes en RAG
│       ├── test_cli_display_safe_tty.py    # Tests del singleton `ui` y sus modos seguros de TTY
│       └── test_run_eval_checkpoint.py     # Tests de reanudación por checkpoint en evaluation/run_eval.py
├── evaluation/
│   ├── datasets/                 # JSON de evaluación RAG (ES, CA, mix) + `ragbench_en_dev_doc_ids.json`
│   ├── run_eval.py               # Runner RAGAS: subcomandos `single`, `compare`, `list-variants`, `ragbench`, `ragbench-prepare`, `ragbench-eval`
│   ├── run_ragbench_visual_inference.py # RagBench tablas/imágenes: dataset + indexación + inferencia sin RAGAS
│   ├── runs/                     # Artefactos de evaluación bajo `ragas/`; inferencia visual en `ragas/ragbench_visual/inference/`
│   ├── aggregate_comparison_by_conjunto.py  # Post-compare: medias por conjunto (subset) desde debug JSON + dataset
│   └── requirements.txt          # ragas, langchain-google-genai, pandas…
├── training-output/
│   ├── qwen-3/                   # Adaptador LoRA Qwen3 (artefactos pesados gitignored)
│   │   ├── generate_reports.py   # Tablas + figuras train/eval → plots/ (misma lógica en cada modelo)
│   │   ├── train.py              # Copia del script de training usada en el cluster (gitignored)
│   │   └── plots/                # Curvas de training/eval (gitignored)
│   ├── phi-4/                    # LoRA Phi-4: r=32 en raíz; r≠32 en subcarpeta `<rank>/` (p. ej. 16/, 64/)
│   │   ├── generate_reports.py   # Tablas + figuras train/eval → plots/ (misma lógica en cada modelo)
│   │   ├── <rank>/               # Solo métricas JSON versionables por rank; pesos/checkpoints gitignored
│   │   └── plots/                # Curvas y reportes de training/eval (gitignored)
│   ├── gemma-3/                  # Adaptador LoRA Gemma-3 (artefactos pesados gitignored)
│   │   ├── generate_reports.py   # Tablas + figuras train/eval → plots/ (misma lógica en cada modelo)
│   │   ├── training_stats.json   # Métricas de training (versionado)
│   │   ├── evaluation_comparison.json  # Comparativa base/adaptado (versionado)
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
│   ├── investigacionMetricas.md  # Notas de métricas para el TFG
│   ├── splits.md                 # Análisis de splits de datasets
│   ├── palabras.md               # Borrador de vocabulario/terminología del TFG
│   ├── tensor.pdf                # Material académico de apoyo (referencia)
│   └── EVALUACIONES_PIPELINE.md  # Presets de corpus, variantes de ablación y notas de agregación RAGAS
├── llama-bin/                    # Binarios llama.cpp compilados para Windows (gitignored)
├── models/
│   ├── merged-model/             # Modelo HF denso post-merge LoRA (gitignored; se puede borrar tras GGUF)
│   └── gguf-output/
│       ├── qwen-3/               # GGUF Qwen3-14B + Modelfile, README, LICENSE, CONVERSION.md
│       ├── phi-4/                # GGUF Phi-4 + Modelfile + README.md + LICENSE + CONVERSION.md
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
| Chat y sub-consultas | `OLLAMA_CHAT_MODEL` | `/chat` (CLI/web) y sub-queries RAG: **`think=False`** en `ollama.chat` / `ollama.generate` para no activar trazas razonadoras (p. ej. Gemma 4). |
| Embeddings | `OLLAMA_EMBED_MODEL` | Vectoriza chunks al indexar y la pregunta al recuperar. El path de ChromaDB incluye slug del modelo. |
| Contextual retrieval | `OLLAMA_CONTEXTUAL_MODEL` | Enriquece el texto de cada chunk antes de embebedarlo. Solo en indexación con `USAR_CONTEXTUAL_RETRIEVAL`. |
| RECOMP | `OLLAMA_RECOMP_MODEL` | Sintetiza/comprime fragmentos recuperados antes del generador. Solo con `USAR_RECOMP_SYNTHESIS`. |
| Visión / OCR | `OLLAMA_OCR_MODEL` | Describe textualmente figuras raster en PDFs. Solo con `USAR_EMBEDDINGS_IMAGEN`. |
| Reranker | `RERANKER_QUALITY` | CrossEncoder local (BGE o MiniLM). No es modelo Ollama. Solo con `USAR_RERANKER`. |

**Thinking (Ollama):** en modelos con razonamiento explícito (p. ej. Gemma 4), `rag/chat_pdfs.py` y el stream RAG de `web/app.py` envían **`think=False`** en sub-consultas, contextual retrieval, RECOMP (`/api/chat`), OCR, `/api/generate` del generador y `ollama.chat` del RAG en web — para que el presupuesto de tokens vaya a la respuesta útil, no a la traza interna.

Variables de entorno de referencia (ajustar según hardware y despliegue):

| Variable | Referencia | Descripción |
|----------|------------|-------------|
| `OLLAMA_RAG_MODEL` | `phi4-finetuned:latest` | Generador RAG |
| `OLLAMA_CHAT_MODEL` | `gemma4:e2b` | Modo chat |
| `OLLAMA_EMBED_MODEL` | `embeddinggemma:latest` | Embeddings |
| `OLLAMA_CONTEXTUAL_MODEL` | `gemma4:e4b` | Contextual retrieval |
| `OLLAMA_RECOMP_MODEL` | `gemma4:e4b` | Síntesis RECOMP |
| `OLLAMA_OCR_MODEL` | `gemma4:e4b` | Descripción de imágenes (multimodal, think=False) |
| `DOCS_FOLDER` | `rag/docs/es/` | Carpeta de PDFs a indexar (es/ca/en según corpus) |
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

> **PATH_DB naming**: `rag/vector_db/{folder}_{embed_slug}` donde `folder` es el basename de `CARPETA_DOCS` (p.ej. `es`, `ca`, `en`) y `embed_slug` es la parte antes de `:` en `OLLAMA_EMBED_MODEL`. Cambiar el modelo de embedding o la carpeta genera una ruta diferente → la colección existente no se carga y hay que reindexar.


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
  -> Selección: TOP_K_FINAL (8 fragmentos)
  -> [Opt] EXPANDIR_CONTEXTO: N_TOP_PARA_EXPANSION + chunks adyacentes
  -> [Opt] USAR_OPTIMIZACION_CONTEXTO: recorte con MAX_CONTEXTO_CHARS (12000)
  -> [Opt] USAR_RECOMP_SYNTHESIS: sintetizar/comprimir contexto via OLLAMA_RECOMP_MODEL
```

### Generación

```
Pregunta + <context>
  -> Contexto final: fragmentos crudos o sintesis RECOMP
  -> OLLAMA_RAG_MODEL: streaming via Ollama (default: phi4-finetuned:latest; temperature, top_p, repeat_penalty, num_ctx)
```

---

## 7. Comandos importantes

### Arrancar el sistema

```bash
# CLI
cd rag && python chat_pdfs.py

# Web (backend Flask — sirve el build compilado de React)
python web/app.py   # http://localhost:5000
```

### Frontend React (desarrollo / build)

```bash
cd web/zip

# Modo desarrollo con hot-reload (Vite en :3000, proxy a Flask en :5000)
npm run dev

# Build de producción → web/zip/dist/ (Flask sirve este directorio)
npm run build

# Type-check sin emitir (equivalente a lint)
npm run lint
```

> El backend Flask (`web/app.py`) sirve `web/zip/dist/` directamente. Para desarrollo, arrancar Flask en :5000 **y** Vite en :3000 simultáneamente; CORS ya está configurado.

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
# Qwen3-14B (v10) — training completado
python scripts/training/train-qwen3.py

# Phi-4 (v1) — training completado (r=32, r=16, r=64)
python scripts/training/train-phi4.py

# Gemma-3-12B (v2) — training completado
python scripts/training/train-gemma3.py

# Fusionar adaptador para exportar a GGUF
python scripts/conversion/merge_lora.py --model qwen-3   # opciones: qwen-3, gemma-3, phi-4

# Model cards + carpeta reproduction/ en Hugging Face (requiere HUGGINGFACE_HUB_TOKEN; no sube .gguf)
python scripts/hf_upload_model_cards.py

# Subir solo el GGUF Q4_K_M de Qwen al Hub (~9 GB; mismo token)
python scripts/hf_upload_model_cards.py --upload-qwen-q4-gguf
```

### Evaluación

```bash
# Benchmark 7 modelos base (Token F1, ROUGE-L, BERTScore, CF-léxica; 320 muestras/dataset)
python scripts/evaluation/evaluate_baselines.py
# Salida: training-output/baseline/

# Regenerar tablas Markdown + CSVs desde baseline_evaluation.json
python training-output/baseline/generate_reports.py
# Salida: training-output/baseline/reports/

# Por cada modelo LoRA (mismo script en cada carpeta; rutas por defecto = directorio del script)
python training-output/qwen-3/generate_reports.py
python training-output/phi-4/generate_reports.py
python training-output/gemma-3/generate_reports.py
# Salida: training-output/<modelo>/plots/{train,eval}/

# Auditar splits por dataset
python scripts/evaluation/inspect_splits.py

# Exportar chunks de ChromaDB (salida por defecto: rag/show_fragments/)
python rag/show_fragments/export_fragments.py                    # ca, en, es + en_ragbench_dev/eval (omite los que no existan)
python rag/show_fragments/export_fragments.py --language es      # solo una base de idioma
python rag/show_fragments/export_fragments.py --ragbench dev     # solo el corpus RagBench EN dev
python rag/show_fragments/export_fragments.py --ragbench eval    # solo el corpus RagBench EN eval

# RAGAS sobre el pipeline en vivo (requiere GOOGLE_API_KEY)
python evaluation/run_eval.py single --corpus es
python evaluation/run_eval.py single --corpus ca
python evaluation/run_eval.py single --corpus en
python evaluation/run_eval.py compare --corpus ca --label mi_eval  # ablación comparativa
python evaluation/run_eval.py ragbench --n-papers 3 --max-q 5     # flujo legacy / exploratorio
python evaluation/run_eval.py ragbench-prepare                    # corpus EN final (25 docs / 5 q, excluye dev split)
python evaluation/run_eval.py ragbench-eval                       # indexa + infiere + RAGAS desde el manifiesto
python evaluation/run_ragbench_visual_inference.py --n-papers 25 --max-q 5  # tablas/imágenes sin RAGAS

# Tras compare: medias RAGAS por subconjunto del dataset (JSON/CSV; --etiquetas-es para informes en castellano)
python evaluation/aggregate_comparison_by_conjunto.py --dir evaluation/runs/ragas/comparisons/<label> --etiquetas-es
```

Ver `docs/EVALUACIONES_PIPELINE.md` (sección *Agregación por conjunto*).

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
MODELO_RAG = os.getenv("OLLAMA_RAG_MODEL", "phi4-finetuned:latest")
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
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

Los artefactos JSON (`evaluation_comparison.json`) incluyen ambos campos (`deltas` para el Δ absoluto en pp, `deltas_rel_pct` para el Δ relativo en %).

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
USAR_RECOMP_SYNTHESIS = True         # enabled by default; can be overridden via env
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

## 10. Artefactos de evaluación

| Artefacto | Ubicación | Descripción |
|-----------|-----------|-------------|
| Baseline completo (320 muestras) | `training-output/baseline/baseline_evaluation.json` | 7 modelos × 5 datasets × con/sin contexto; Token F1, ROUGE-L, BERTScore, CF-léxica |
| Predicciones baseline | `training-output/baseline/predictions_{modelo}.json` | 7 archivos; permite recomputar métricas |
| Tablas de resultados | `training-output/baseline/reports/` | Markdown + CSVs + figuras; generado por `generate_reports.py` |
| Baseline 200-sample (histórico) | `training-output/baseline/200/` | Versión anterior cap=200; solo referencia |
| Artefactos LoRA Qwen3-14B | `training-output/qwen-3/` | Tras training: `training_stats.json`, `evaluation_comparison.json`, predicciones (gitignored). `generate_reports.py` → `plots/{train,eval}/` |
| Artefactos LoRA Phi-4 | `training-output/phi-4/` | Convención Qwen3 + **subcarpetas por rank** (`16/`, `64/`, …) cuando `LORA_RANK≠32` en `train-phi4.py`; en raíz, run histórico r=32 |
| Artefactos LoRA Gemma-3-12B | `training-output/gemma-3/` | `training_stats.json`, `evaluation_comparison.json` versionados. `generate_reports.py` → `plots/{train,eval}/`. Conversión GGUF pendiente (incompatibilidad tokenizer; ver `GEMMA3_CONVERSION_ISSUE.md`) |
| Diagrama arquitectura | `docs/monkeygrab_architecture.png` / `.svg` | Generado por `generate_diagram.py` |
| Datasets RAGAS locales | `evaluation/datasets/local/*.json` | p. ej. `dataset_eval_es.json`, `dataset_eval_ca.json`, `dataset_eval_mix.json` |
| Datasets RagBench preparados | `evaluation/datasets/ragbench/prepared/` | datasets/manifiestos de `ragbench-prepare`, dev congelado y RagBench visual |
| Resumen RAGAS por conjunto (post-`compare`) | `evaluation/runs/ragas/comparisons/<label>/aggregates/by_conjunto_*.json` (CSV opcional bajo `evaluation/runs/ragas/comparisons/<label>/scores/`) | Script `aggregate_comparison_by_conjunto.py`: cruza `<variant>.json` con el dataset por indice y calcula medias por `source_type`, `language`, etc.; `--etiquetas-es` para claves de metricas en castellano. Detalle en `docs/EVALUACIONES_PIPELINE.md`. |
| Resultados RAGAS RagBench visual | `evaluation/runs/ragas/ragbench_visual/<tag>/` | `scores.csv` + `debug.json` por tag (p. ej. `image_table_25p_5q`); generado por `run_ragbench_visual_inference.py --ragas-only` |
| Inferencia RagBench visual (sin RAGAS) | `evaluation/runs/ragas/ragbench_visual/inference/<tag>/` | `results.csv`, `results.json`, `checkpoint.json`; generado por `run_ragbench_visual_inference.py` |
| BERTScore post-proceso | `evaluation/runs/bertscore/<label>/` | `*_bertscore.csv` + `bertscore_summary.json`/`.csv`; generado por `evaluate_ragas_bertscore.py` sobre cualquier run RAGAS |

---

## 11. Versionado Git: `.gitignore`, `.gitkeep` y pesos (GGUF)

Esta sección fija **cómo mantener el repo** para que cualquier cambio en exclusiones o carpetas vacías sea coherente con el TFG y con GitHub.

### 11.1 Principios

1. **Un solo `.gitignore` en la raíz** del monorepo y **`web/zip/.gitignore` como complemento** (`build/`, `coverage/`, `.env*` con `!.env.example`). `node_modules/` y `dist/` del frontend **solo** se listan en la raíz (`web/zip/…`); no duplicarlos en `web/zip/.gitignore`. No añadir más `.gitignore` dispersos sin motivo.
2. **Bloques numerados y comentados** en la raíz: al tocar exclusiones, coloca la regla en el bloque que corresponda (modelos, `training-output`, RAG, web, etc.) y actualiza este apartado si cambia la política.
3. **Versionar lo mínimo imprescindible para reproducir y defender el trabajo**: código, `Modelfile`, JSON de métricas pequeños, scripts; **no** pesos, índices vectoriales ni PDFs privados.
4. **Numeración en el `.gitignore` raíz:** los últimos bloques del archivo están numerados **12** y **13** (evaluación / otros) a propósito, para **no solapar** el número con este **apartado 11** de `CLAUDE.md`. La cabecera del `.gitignore` remite aquí para la política.

### 11.2 Patrón `training-output/<modelo>/`

Cada carpeta de modelo usa el patrón:

- `training-output/<slug>/*` → ignora **casi todo** lo que cuelga directamente de esa carpeta (checkpoints, adaptadores, `plots/`, predicciones voluminosas, etc.).
- Inmediatamente después, líneas `!training-output/<slug>/…` → **excepciones** para archivos que **sí** deben ir a GitHub:
  - `generate_reports.py`
  - `training_stats.json`
  - `evaluation_comparison.json`

**Reglas de mantenimiento:**

- Si añades un **nuevo modelo** bajo `training-output/`, copia el bloque de cuatro líneas (una `/*` y tres `!…`) y sustituye el slug (mismo orden que el resto de modelos).
- Si quieres versionar **otro** fichero pequeño en esa carpeta, añade **una** línea `!training-output/<slug>/nombre.ext` **debajo** de las excepciones existentes del mismo modelo.
- **No** sustituyas el patrón `/*` por ignorar solo extensiones sin más: es fácil dejar fuera del repo scripts o JSON que sí interesan (ya ocurrió con `generate_reports.py` antes de las excepciones).

**Phi-4 y varios ranks:** `train-phi4.py` usa `training-output/phi-4/` para r=32 y `training-output/phi-4/<rank>/` para otros. En `.gitignore`, por cada rank del que quieras subir métricas, replica el bloque de cuatro líneas (`!phi-4/<rank>/`, `phi-4/<rank>/*`, `!…training_stats.json`, `!…evaluation_comparison.json`). Hoy están declarados **16** y **64**; para otro rank, copia el bloque y cambia el número.

Comprobar si un path está ignorado:

```bash
git check-ignore -v ruta/al/archivo
```

### 11.3 Patrón `.gitkeep` (carpetas que deben existir al clonar)

Se usa cuando la aplicación **espera un directorio** pero su contenido **no** debe versionarse (PDFs propios, ChromaDB local).

- En `.gitignore`: `rag/<carpeta>/**` + `!rag/<carpeta>/.gitkeep` (el `**` ignora también subcarpetas; la negación solo recupera el fichero vacío).
- En disco: un archivo **vacío** `rag/<carpeta>/.gitkeep` commiteado.

**Carpetas con este patron en el repo:** `rag/docs/es/`, `rag/docs/ca/`, `rag/docs/en/`. La carpeta `rag/vector_db/` esta completamente ignorada (sin `.gitkeep`; se crea automaticamente al indexar). `rag/docs/en_ragbench_dev/`, `rag/docs/en_ragbench_eval/` y `rag/docs/en_ragbench_visual/` no versionan PDFs. Los datasets, checkpoints y resultados de `evaluation/` si deben versionarse salvo temporales bajo `evaluation/tmp/`.

**Cuándo añadir otro `.gitkeep`:** solo si aparece una ruta nueva “obligatoria” en código (replicar el mismo par `/**` + `!.gitkeep` en la raíz `.gitignore` y documentar aquí).

### 11.4 `models/gguf-output/<modelo>/`

Se versionan el **`Modelfile`**, **`README.md`**, **`LICENSE`** y **`CONVERSION.md`** (notas merge → GGUF → Ollama) donde existan — hoy en `phi-4/` y `qwen-3/` alineados con las model cards del Hub. Los `.gguf` quedan fuera por la regla global `*.gguf` y por el patrón `models/gguf-output/<modelo>/*` con esas excepciones. Cualquier despliegue debe enlazar **dónde** está el binario (ver §11.6).

**`models/merged-model/`** (salida de `merge_lora.py`) está **gitignored** por completo: no versionar; tras generar el **Q4_K_M** puedes borrar la carpeta del modelo correspondiente para liberar ~decenas de GB (ver limpieza en §7 conversión / nota abajo).

**Liberar disco tras conversión (conservar solo Q4):** opcionalmente borrar `models/merged-model/<slug>/`, el intermedio `models/gguf-output/<slug>/*-f16.gguf`, checkpoints y pesos bajo `training-output/<slug>/` que no necesites para reanudar entrenamiento; conservar **`…-Q4_K_M.gguf`**, `Modelfile` y los JSON pequeños versionables (`training_stats.json`, `evaluation_comparison.json`).

### 11.5 Cambios beneficiosos a vigilar (no obligatorios)

| Idea | Estado / beneficio |
|------|-------------------|
| PDFs y vector DB organizados por idioma bajo `rag/docs/{es,ca,en}/` y `rag/vector_db/` | **Aplicado** — migrado 2026-04-21 |
| `web/zip/.gitignore` sin duplicar `node_modules` / `dist` | **Aplicado** — solo en la raíz |
| Enlace en `README.md` al Hub u origen del GGUF | **Aplicado** — sección *Model weights* |
| Reglas más finas si aparecen `.bin` pequeños legítimos | Pendiente si surge la necesidad; hoy `*.bin` es global |
| Salida de `rag/show_fragments/export_fragments.py` (`.txt` / `.jsonl`) | **Aplicado** — bloque `12`: se ignoran `rag/show_fragments/*.txt` y `*.jsonl` salvo `!rag/show_fragments/chunks_vector_db_*.txt` (exports con nombre fijo, versionables); se mantiene `evaluation/chunks_*.txt` |

### 11.6 ¿Subir `.gguf` a GitHub?

**No como solución habitual.** Motivos:

- **Límites de GitHub:** advertencias por ficheros grandes; bloqueos por tamaño; el repo se vuelve pesado para clonar y para CI.
- **Git LFS:** cuota limitada en planes gratuitos; varios GGUF cuantizados de 7–14B multiplican GB rápidamente.

**Recomendación para el TFG y despliegue:**

1. **Hugging Face Hub** (repositorio de modelo `public` o `private`): subir el `.gguf` (o el adaptador LoRA si compartes entrenamiento), con tarjeta del modelo que cite el commit del código y la receta de cuantización. Es el estándar de facto en la comunidad ML.
2. **Almacenamiento objeto** (Azure Blob, S3, etc.) con enlace firmado o documentación en memoria si la política de la universidad lo exige.
3. **Release de GitHub** solo si el artefacto es **pequeño** (p. ej. un adapter de unos MB), no para GGUF completos de decenas de GB.

En el repo basta con **Modelfile + instrucciones** (o script existente tipo `build_ollama.bat`) y el **enlace** al binario en Hub u otra plataforma; así el código queda limpio y el modelo sigue siendo recuperable.

### 11.7 Excepción RagBench EN: corpus locales congelados/preparados

Para la evaluación final de RagBench EN hay dos carpetas locales adicionales bajo `rag/docs/`:

- `rag/docs/en_ragbench_dev/`: split dev congelado de 10 PDFs
- `rag/docs/en_ragbench_eval/`: corpus final EN ampliable para evaluación
- `rag/docs/en_ragbench_visual/`: corpus EN para inferencia RagBench con preguntas `text-image` y `text-table` sin RAGAS

Y sus índices asociados bajo `rag/vector_db/`:

- `en_ragbench_dev_embeddinggemma/`
- `en_ragbench_eval_embeddinggemma/`
- `en_ragbench_visual_embeddinggemma/`

Estas carpetas no usan `.gitkeep`. En su lugar llevan un `.gitignore` local autocontenido:

```gitignore
*
!.gitignore
```

Esta es una excepción documentada a la preferencia general de centralizar reglas en la raíz. Se acepta porque reduce el riesgo de commitear PDFs locales cuando el corpus ya existe en disco y puede regenerarse o ampliarse.
