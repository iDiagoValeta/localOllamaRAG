# Evaluaciones del pipeline RAG

Esta guía describe el protocolo de evaluación RAGAS usado en el TFG y los
comandos concretos para reproducirlo. El flujo se divide en **tres fases**
respaldadas por tres CLIs en `research/evaluation/` (más el paquete compartido
`_lib/`):

1. **`index.py`** — indexa el corpus en ChromaDB.
2. **`infer.py`** — genera respuestas RAG y persiste checkpoints (sin RAGAS).
3. **`evaluate.py`** — ejecuta RAGAS desde checkpoint con `--provider google|aws|nvidia` y agrega resultados por subconjunto del dataset.

Los scripts antiguos (`run_eval.py`, `eval_ragas_*_from_checkpoints.py`,
`run_ragbench_visual_inference.py`, `evaluate_ragas_bertscore.py`,
`judge_benchmark.py`, `aggregate_comparison_by_conjunto.py`) **han sido
sustituidos**; toda su lógica vive ahora en `_lib/` o como subcomando de los
tres CLIs.

---

## 1. Indexación

```powershell
python research\evaluation\index.py --corpus es              # rag/docs/es → ChromaDB
python research\evaluation\index.py --corpus ca --force      # reconstruye desde cero
python research\evaluation\index.py --corpus en
python research\evaluation\index.py --corpus ragbench-eval   # usa el file-filter del manifest RagBench EN
python research\evaluation\index.py --docs-dir ruta\custom --force
```

El destino se deriva de `rag.chat_pdfs.PATH_DB` (`rag/vector_db/{folder}_{embed_slug}/`).
Solo crea la base si no existe; usa `--force` para borrarla y reindexar.

---

## 2. Inferencia (checkpoints sin RAGAS)

### 2.1. Corpus locales

Presets de corpus:

| Corpus | Dataset por defecto | PDFs por defecto |
| --- | --- | --- |
| `es` | `research/evaluation/datasets/local/dataset_eval_es.json` | `rag/docs/es` |
| `ca` | `research/evaluation/datasets/local/dataset_eval_ca.json` | `rag/docs/ca` |
| `en` | `research/evaluation/datasets/local/dataset_eval_en.json` | `rag/docs/en` |
| `mix` | `research/evaluation/datasets/local/dataset_eval_mix.json` | `rag/docs/es` |

Formato de dataset (cualquiera de los soportados — JSON/CSV/Excel):
columna `question` o `pregunta`, opcional `ground_truth` (alias aceptados:
`reference`, `respuesta_esperada`, `respuesta_referencia`).

```powershell
# 1 variante (baseline_all_on) sobre el corpus indicado
python research\evaluation\infer.py single --corpus es

# Suite ablation completa (9 variantes: baseline + 7 single-flag-off + all_off)
python research\evaluation\infer.py compare --corpus ca --label mi_eval_ca_ablation --reindex

# Listar variantes
python research\evaluation\infer.py list-variants
```

### 2.2. Suite `ablation`

`baseline_all_on` activa todas las etapas opcionales de inferencia; cada
variante `no_*` desactiva exactamente una; `all_off` desactiva todas a la vez
(suelo del experimento). Las 9 variantes comparten la misma colección
ChromaDB; `--reindex` solo afecta a la primera.

| Variante | Cambio |
| --- | --- |
| `baseline_all_on` | Todas las etapas opcionales activadas |
| `no_query_decomposition` | Desactiva `USAR_LLM_QUERY_DECOMPOSITION` |
| `no_lexical_search` | Desactiva `USAR_BUSQUEDA_HIBRIDA` |
| `no_exhaustive_search` | Desactiva `USAR_BUSQUEDA_EXHAUSTIVA` |
| `no_reranker` | Desactiva `USAR_RERANKER` |
| `no_context_expansion` | Desactiva `EXPANDIR_CONTEXTO` |
| `no_context_optimization` | Desactiva `USAR_OPTIMIZACION_CONTEXTO` |
| `no_recomp_synthesis` | Desactiva `USAR_RECOMP_SYNTHESIS` |
| `all_off` | Todas las etapas opcionales desactivadas (recuperación semántica pura + filtro por umbral) |

Etapas excluidas de la suite por defecto: `USAR_CONTEXTUAL_RETRIEVAL` y
`USAR_EMBEDDINGS_IMAGEN` afectan al contenido **indexado**, no a la
inferencia. Para compararlas hace falta una colección distinta por
configuración (reindexar). Se ejecutan como experimentos separados y se
etiquetan explícitamente.

### 2.3. RagBench EN (corpus final fijo)

```powershell
python research\evaluation\infer.py ragbench-prepare   # descarga PDFs + manifest
python research\evaluation\infer.py ragbench-eval      # inferencia sobre el manifest
```

- Manifest: `research/evaluation/datasets/ragbench/prepared/en_eval/ragbench_en_eval_manifest.json`
- PDFs: `rag/docs/en_ragbench_eval/`
- Excluye el split dev congelado declarado en `research/evaluation/datasets/ragbench/ragbench_en_dev_doc_ids.json`.
- ChromaDB: `rag/vector_db/en_ragbench_eval_<embed_slug>/`

### 2.4. RagBench visual (tablas e imágenes)

```powershell
python research\evaluation\infer.py visual --n-papers 25 --max-q 5
```

- Filtra solo preguntas `text-image` y `text-table` del split.
- PDFs: `rag/docs/en_ragbench_visual/`
- ChromaDB: `rag/vector_db/en_ragbench_visual_<embed_slug>/`
- Pipeline-flags: `USAR_LLM_QUERY_DECOMPOSITION=False` y `USAR_RERANKER=False`
  (el cross-encoder tiende a infrafiltrar evidencia tabular bajo el umbral
  calibrado para chat interactivo).
- Salidas: `research/evaluation/runs/ragas/ragbench_visual/inference/<tag>/results.{csv,json}` y `checkpoint.json`.

### 2.5. Fallback de reranker en RagBench

En cualquier flujo RagBench (`ragbench-eval`, `visual` y cualquier dataset
preparado bajo `research/evaluation/datasets/ragbench/prepared/`) el runner
activa un *fallback*: si el reranker puntúa todos los candidatos por debajo
de `UMBRAL_SCORE_RERANKER`, conserva los mejores candidatos recuperados en
vez de devolver contexto vacío. **No** equivale a apagar el reranker — sigue
reordenando los fragmentos; solo se desactiva como filtro duro cuando dejaría
la pregunta sin contexto.

Motivo: RagBench contiene preguntas factuales muy cortas donde el
cross-encoder a veces puntúa evidencia útil por debajo del umbral calibrado
para evitar ruido en chat interactivo.

### 2.6. Esquema de checkpoint

Cada inferencia escribe un checkpoint JSON reanudable que `evaluate.py`
consume. Contiene como mínimo:

- `dataset_path`, `questions_count`, `eval_corpus`, `docs_dir`
- `pipeline_flags` (snapshot de las flags efectivas)
- `modelo_rag`, `modelo_chat`, `modelo_embedding`, `modelo_recomp` (invalidan
  el checkpoint si cambian entre corridas)
- `answers`, `contexts_list`, `question_statuses` (estado por pregunta)
- `ragbench_reranker_low_score_fallback` (booleano, ver §2.5)

---

## 3. RAGAS desde checkpoint (`evaluate.py`)

`evaluate.py` **nunca** genera respuestas — solo aplica RAGAS sobre los
checkpoints producidos por `infer.py`. Soporta tres jueces:

```powershell
# Google Gemini (default; requiere GOOGLE_API_KEY)
python research\evaluation\evaluate.py --provider google `
  --source-root research\evaluation\runs\ragas\comparisons\mi_eval_ca_ablation

# NVIDIA NIM (requiere NVIDIA_API_KEY; rate-limit ajustable)
python research\evaluation\evaluate.py --provider nvidia --all-known `
  --nvidia-rate-limit-per-minute 40

# AWS Bedrock (requiere AWS_BEARER_TOKEN_BEDROCK o boto3 profile)
python research\evaluation\evaluate.py --provider aws --all-known --dry-run
```

Selección de checkpoints:

- `--checkpoint PATH` — uno o varios (repetible) o un directorio (`*.json`).
- `--all-known` — descubre automáticamente los checkpoints conocidos bajo
  `--source-root` (`comparisons/*/checkpoints/*.json`, `single/**/checkpoint*.json`,
  `ragbench/**/checkpoint.json`, `ragbench_visual/**/checkpoint.json`,
  `ragbench_visual/**/results.json`).
- `--source-root PATH` — combinado con `--all-known`, restringe el descubrimiento.
- `--retry-failed` — re-evalúa **solo** las filas con celdas NaN del
  `scores.csv` previo y fusiona el resultado.
- `--limit N` — recorta a las primeras N preguntas de cada checkpoint
  (smoke tests).
- `--dry-run` — lista lo que evaluaría sin ejecutar RAGAS.

Métricas RAGAS por defecto (con ground truth):

- `answer_correctness` — precisión factual vs. referencia (TP/FP/FN, F1).
- `faithfulness` — consistencia respuesta ↔ contexto recuperado.
- `answer_relevancy` — adecuación respuesta ↔ pregunta.
- `context_precision` — orden de fragmentos recuperados.
- `context_recall` — cobertura del contexto necesario.

Filtrar a un subconjunto con `--metrics faithfulness,answer_relevancy` o
todas con `--metrics all`.

### 3.1. Salidas

Cada provider escribe bajo su propia raíz, espejando la ruta relativa del
checkpoint:

```
runs/ragas_google_revaluation/    runs/ragas_aws_revaluation/    runs/ragas_nvidia_revaluation/
└── comparisons/<label>/<variant>/
    ├── scores.csv         # tabla RAGAS (una fila por pregunta + métricas)
    ├── debug.json         # respuestas, contextos (preview), justificaciones del juez
    └── ...
└── <provider>_ragas_summary.json   # índice de todos los checkpoints evaluados
```

`scores.csv` tiene como columnas las métricas seleccionadas más los campos
canónicos RAGAS (`user_input`, `response`, `retrieved_contexts`, `reference`).
`debug.json` añade prompts internos del juez (justifications) para
trazabilidad.

---

## 4. Agregación por subconjunto (integrada en `evaluate.py`)

Tras una run de comparación, `evaluate.py` genera automáticamente medias
**variante × subconjunto × métrica** alineando cada fila del debug con la
posición del dataset. Se controla con:

```powershell
# Default: agrupa por source_type
python research\evaluation\evaluate.py --provider google --all-known

# Múltiples agrupaciones + etiquetas en castellano para la memoria
python research\evaluation\evaluate.py --provider google `
  --source-root research\evaluation\runs\ragas\comparisons\mi_eval_ca_ablation `
  --aggregate-group-by source_type,language `
  --aggregate-etiquetas-es

# Opt-out
python research\evaluation\evaluate.py --provider nvidia --all-known --no-aggregate
```

Subconjuntos soportados:

| Valor | Conjunto por |
| --- | --- |
| `source_type` | Campo `source_type` del dataset (default) |
| `language` | Campo `language` (útil en `dataset_eval_mix.json`) |
| `source_type_language` | `source_type` + `language` |
| `id_prefix` | Prefijo del `id` antes del bloque numérico final (p. ej. `wiki_es` en `wiki_es_001`) |

Salidas (junto al run de comparación, mirroring de `output_root`):

- `aggregates/by_conjunto_<criterio>.json` (o `_metricas_es.json` con etiquetas en castellano).
- `aggregates/resumen_por_conjunto_<criterio>.csv` — tabla larga
  (variante × conjunto × métrica) lista para importar en la memoria.

Solo se agregan automáticamente las runs ablation (`comparisons/<label>/`).
`single`, `ragbench-eval` y `visual` no se agregan porque por construcción
solo tienen una variante.

**Agregación acumulativa (verificado 2026-05-15):** el paso de agregación no
se limita a las variantes evaluadas en la llamada actual. Antes de agregar,
escanea `output_root/comparisons/<label>/*/debug.json` e incorpora todas las
variantes ya evaluadas en pasadas anteriores. Esto permite evaluar **variante
a variante** (`--checkpoint <variante>.json`) y obtener igualmente un aggregate
completo: cada variante vive en su propia carpeta
`comparisons/<label>/<variante>/` (nombre tomado del stem del checkpoint), las
ya evaluadas se saltan con `[skip] exists` sin volver a llamar al juez salvo
`--overwrite`, y el aggregate refleja siempre el total acumulado. No es
necesario reevaluar todo con `--source-root --overwrite`.

---

## 5. Protocolo TFG (mesa principal por idioma)

Comandos canónicos:

```powershell
# Fase 1 — inferencia (3 corridas ablation, una por idioma)
python research\evaluation\infer.py compare --corpus es --label mi_eval_es_ablation --reindex
python research\evaluation\infer.py compare --corpus ca --label mi_eval_ca_ablation --reindex
python research\evaluation\infer.py compare --corpus en --label mi_eval_en_ablation --reindex

# Fase 2 — RAGAS + agregación (juez Gemini, etiquetas en castellano)
python research\evaluation\evaluate.py --provider google `
  --source-root research\evaluation\runs\ragas\comparisons\mi_eval_es_ablation `
  --aggregate-etiquetas-es
python research\evaluation\evaluate.py --provider google `
  --source-root research\evaluation\runs\ragas\comparisons\mi_eval_ca_ablation `
  --aggregate-etiquetas-es
python research\evaluation\evaluate.py --provider google `
  --source-root research\evaluation\runs\ragas\comparisons\mi_eval_en_ablation `
  --aggregate-etiquetas-es
```

### 5.1. Interpretación recomendada

- `answer_correctness` mide cercanía a la referencia (TP/FP/FN basados en
  hechos atomizados por el juez).
- `faithfulness` mide consistencia de la respuesta con los **contextos
  exportados a RAGAS** (es decir, `retrieved_contexts`).
- `answer_relevancy` mide si la respuesta atiende la pregunta del usuario.
- `context_precision` y `context_recall` se calculan sobre `retrieved_contexts`,
  que son los chunks **crudos** devueltos por la recuperación final. Etapas
  como RECOMP u optimización de contexto pueden cambiar la respuesta
  generada sin cambiar necesariamente esos chunks → un `faithfulness` que
  baja al desactivar RECOMP indica respuestas menos fieles al contexto crudo
  *sin* que el recall del retriever se vea afectado.

Para reducir variabilidad del juez LLM, el protocolo separa generación
(`infer.py`, una vez) de evaluación (`evaluate.py`, reproducible y
relanzable contra los mismos checkpoints). Cambios de juez o de provider
no requieren regenerar respuestas.

### 5.2. Re-evaluación entre proveedores

Como la inferencia y la evaluación están desacopladas, una misma corrida
ablation puede ser puntuada por los tres jueces para reportar correlaciones
o robustez:

```powershell
python research\evaluation\evaluate.py --provider google  --source-root … --aggregate-etiquetas-es
python research\evaluation\evaluate.py --provider nvidia  --source-root … --nvidia-rate-limit-per-minute 40
python research\evaluation\evaluate.py --provider aws     --source-root … --aws-region eu-north-1
```

Cada provider escribe en `runs/ragas_<provider>_revaluation/` sin pisarse.

---

## 6. Verificación rápida

Después de cualquier cambio en el pipeline o en el runner:

```powershell
# Smoke test (10 preguntas, 1 variante)
python research\evaluation\infer.py single --corpus es
python research\evaluation\evaluate.py --provider google `
  --checkpoint research\evaluation\runs\ragas\single\dataset_eval_es_es\checkpoint_recomp_on.json `
  --limit 10 --dry-run
```

Tests automáticos sobre la plumbing (checkpoint I/O, RagBench filters,
visual export):

```powershell
pytest research\tests\evaluation\
```
