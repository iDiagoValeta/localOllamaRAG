# Evaluaciones del pipeline RAG

Este directorio usa `run_eval.py` como entrypoint principal para lanzar evaluaciones RAGAS. El objetivo es poder reportar en el TFG una comparativa clara entre el pipeline completo y variantes donde se desactiva una etapa opcional cada vez.

## Comandos principales

Evaluacion individual por corpus:

```powershell
python evaluation\run_eval.py single --corpus es
python evaluation\run_eval.py single --corpus ca
python evaluation\run_eval.py single --corpus en
```

Por compatibilidad, este comando antiguo sigue funcionando y equivale a `single --corpus ca`:

```powershell
python evaluation\run_eval.py --catalan
```

Comparativa de ablacion completa:

```powershell
python evaluation\run_eval.py compare --corpus ca --label mi_eval_ablation --reindex
```

Comparativa parcial:

```powershell
python evaluation\run_eval.py compare --corpus ca --label mi_eval_subset --variants baseline_all_on,no_recomp_synthesis,no_reranker
```

Listar variantes disponibles:

```powershell
python evaluation\run_eval.py list-variants
```

## Corpus y datasets propios

Los presets de corpus resuelven dataset y carpeta de PDFs por convencion:

| Corpus | Dataset por defecto | PDFs por defecto |
| --- | --- | --- |
| `es` | `evaluation/datasets/dataset_eval_es.json` | `rag/docs/es` |
| `ca` | `evaluation/datasets/dataset_eval_ca.json` | `rag/docs/ca` |
| `en` | `evaluation/datasets/dataset_eval_en.json` | `rag/docs/en` |
| `mix` | `evaluation/datasets/dataset_eval_mix.json` | `rag/docs/es` (default) |

Si un dataset o carpeta no existe todavia, crealo manualmente o pasa rutas explicitas:

```powershell
python evaluation\run_eval.py compare --corpus en --dataset evaluation\datasets\mi_dataset_en.json --docs-dir rag\docs\en
```

El formato esperado del dataset es el mismo para todos los idiomas: una tabla JSON/CSV/Excel con columna `question` o `pregunta`, y opcionalmente `ground_truth`, `reference`, `respuesta_esperada` o `respuesta_referencia`.

## Nota RagBench: reranker como ordenador, no filtro duro

En datasets RagBench (`ragbench` o datasets preparados en `evaluation/runs/ragas/ragbench_prepared/`) el runner activa un fallback especifico: si el reranker puntua todos los candidatos por debajo del umbral interactivo, la evaluacion conserva los mejores candidatos recuperados y genera respuesta igualmente.

Esto no equivale a apagar el reranker: el reranker sigue reordenando los fragmentos. Solo se desactiva su uso como filtro duro cuando dejaria una pregunta RagBench sin contexto. El comportamiento normal se mantiene para el resto de datasets y para uso interactivo.

Motivo: RagBench contiene preguntas factuales muy cortas donde el cross-encoder puede puntuar evidencia util por debajo del umbral calibrado para evitar ruido en chat interactivo. Sin este fallback, algunas preguntas quedan como respuesta vacia antes de llegar al modelo generador.

## Suite `ablation`

La suite por defecto compara `baseline_all_on` contra variantes que desactivan una sola etapa opcional de inferencia:

| Variante | Cambio |
| --- | --- |
| `baseline_all_on` | Todas las etapas opcionales de inferencia activadas |
| `no_query_decomposition` | Desactiva `USAR_LLM_QUERY_DECOMPOSITION` |
| `no_lexical_search` | Desactiva `USAR_BUSQUEDA_HIBRIDA` |
| `no_exhaustive_search` | Desactiva `USAR_BUSQUEDA_EXHAUSTIVA` |
| `no_reranker` | Desactiva `USAR_RERANKER` |
| `no_context_expansion` | Desactiva `EXPANDIR_CONTEXTO` |
| `no_context_optimization` | Desactiva `USAR_OPTIMIZACION_CONTEXTO` |
| `no_recomp_synthesis` | Desactiva `USAR_RECOMP_SYNTHESIS` |

Estas variantes comparten la misma coleccion ChromaDB. El flag `--reindex` solo reconstruye una vez la coleccion antes de la primera variante.

## Etapas que no se comparan por defecto

`USAR_CONTEXTUAL_RETRIEVAL` y `USAR_EMBEDDINGS_IMAGEN` afectan al contenido indexado. Para compararlas correctamente hacen falta colecciones separadas o reindexaciones controladas por perfil. Por eso no forman parte de la suite `ablation` por defecto.

Si se quieren reportar en el TFG, conviene ejecutarlas como experimentos separados y etiquetar claramente que cambian el indice, no solo la inferencia.

## Salidas

Las salidas generadas ya no se escriben en la raiz de `evaluation/`. La raiz queda para codigo, datasets y documentacion; los artefactos van bajo `evaluation/runs/`.

La comparativa guarda artefactos bajo:

```text
evaluation/runs/ragas/scores/comparison_runs/<label>/
evaluation/runs/ragas/debug/comparison_runs/<label>/
evaluation/runs/ragas/checkpoints/comparison_runs/<label>/
```

Por cada variante se genera:

| Archivo | Contenido |
| --- | --- |
| `<variant>.csv` | Tabla RAGAS con preguntas, respuestas, contextos y metricas |
| `<variant>.json` | Debug con respuestas, referencias, previews de contexto y puntuaciones |
| checkpoint `<variant>.json` | Respuestas y contextos reutilizables para reanudar |

Ademas, `comparison_summary.json` resume:

- dataset y corpus usado (`es` o `ca`);
- variantes seleccionadas;
- flags solicitados y flags efectivos por variante;
- medias RAGAS;
- `mean_score_deltas_vs_baseline`, calculado contra `baseline_all_on` si esta presente.

## Agregacion por conjunto (`aggregate_comparison_by_conjunto.py`)

Tras una comparativa, los JSON por variante (`<variant>.json`) guardan una fila por pregunta con `index` alineado a la posicion del dataset (misma ordenacion que en el JSON del dataset). El script **no** vuelve a llamar a RAGAS: solo lee esos JSON y el dataset, agrupa las muestras por un criterio de conjunto y calcula **medias** por metrica dentro de cada grupo.

**Entrada**

- Carpeta de comparativa, p. ej. `evaluation/runs/ragas/debug/comparison_runs/todas_ablacion/` (debe contener los `<variant>.json`; si existe `comparison_summary.json`, se usan sus `runs` para localizar variantes y, si hace falta, el `dataset_path`).
- Dataset JSON alineado con la evaluacion (mismo orden de preguntas que en el run). Si el `dataset_path` del resumen apunta a otra maquina o no existe, pasar `--dataset` explicito.

**Criterios de agrupacion (`--group-by`)**

| Valor | Conjunto por |
| --- | --- |
| `source_type` | Campo `source_type` del dataset (por defecto) |
| `language` | Campo `language` (util en `dataset_eval_mix.json`) |
| `source_type_language` | `source_type` + `language` |
| `id_prefix` | Prefijo del `id` antes del bloque numerico final (p. ej. `wiki_es` en `wiki_es_001`) |

**Salida**

- JSON por defecto en la misma carpeta: `by_conjunto_<criterio>.json` o, con `--etiquetas-es`, `by_conjunto_<criterio>_metricas_es.json` (claves de metricas en castellano para informes).
- Opcional `--csv <ruta>`: tabla larga (variante, conjunto, n, columnas por metrica). Requiere `pandas`.

Si el ultimo `compare` guardo un `comparison_summary.json` con **menos variantes** de las que ya tienes en disco (corrida parcial), usa **`--ignore-comparison-summary`** para agregar **todos** los `<variante>.json` de la carpeta (se ignoran `comparison_summary.json` y `by_conjunto_*.json`).

**Ejemplos**

```powershell
python evaluation\aggregate_comparison_by_conjunto.py --dir evaluation\runs\ragas\debug\comparison_runs\todas_ablacion --etiquetas-es

python evaluation\aggregate_comparison_by_conjunto.py --dir evaluation\runs\ragas\debug\comparison_runs\todas_ablacion --dataset evaluation\datasets\dataset_eval_es.json --group-by language --etiquetas-es --csv evaluation\runs\ragas\scores\comparison_runs\todas_ablacion\resumen_por_conjunto.csv

python evaluation\aggregate_comparison_by_conjunto.py --dir evaluation\runs\ragas\debug\comparison_runs\todas_ablacion_ca_ca --dataset evaluation\datasets\dataset_eval_ca.json --ignore-comparison-summary --etiquetas-es --csv evaluation\runs\ragas\scores\comparison_runs\todas_ablacion_ca_ca\resumen_por_conjunto.csv
```

**Nota para el TFG:** Si todo el dataset comparte un solo `source_type` (p. ej. solo Wikipedia en `dataset_eval_es.json`), la tabla tendra una fila por variante en ese conjunto; para contrastar subconjuntos, usar `mix` con `--group-by language` o `id_prefix`, o enriquecer el dataset con varios `source_type`.

## TFG

Para una tabla principal defendible por idioma, usar:

```powershell
python evaluation\run_eval.py compare --corpus es --label mi_eval_es_ablation --reindex
python evaluation\run_eval.py compare --corpus ca --label mi_eval_ca_ablation --reindex
python evaluation\run_eval.py compare --corpus en --label mi_eval_en_ablation --reindex
```

Interpretacion recomendada:

- `answer_correctness` mide cercania a la referencia.
- `faithfulness` mide consistencia de la respuesta con los contextos exportados a RAGAS.
- `answer_relevancy` mide si la respuesta atiende la pregunta.
- `context_precision` y `context_recall` se calculan sobre `retrieved_contexts`, que son los chunks crudos devueltos por la recuperacion final. Etapas como RECOMP u optimizacion de contexto pueden cambiar la respuesta generada sin cambiar necesariamente esos chunks.

Para reducir variabilidad del juez LLM, el runner primero genera/checkpointea todas las respuestas y despues ejecuta RAGAS de forma consecutiva para las variantes seleccionadas.

## RagBench

La logica de RagBench esta integrada directamente en `run_eval.py` (Section 7). Hay dos flujos RAGBench con propositos distintos:

- `ragbench`: flujo legacy/exploratorio. Descarga PDFs en `rag/docs/en/` y escribe salidas RAGAS genericas.
- `ragbench-prepare` + `ragbench-eval`: flujo final EN. Excluye el split dev congelado, descarga PDFs en `rag/docs/en_ragbench_eval/`, prepara dataset/manifiesto bajo `evaluation/runs/ragas/ragbench_prepared/` y evalua desde ese manifiesto.

```powershell
# Legacy / exploratorio
python evaluation\run_eval.py ragbench --n-papers 3 --max-q 5
python evaluation\run_eval.py ragbench --only-doc 2401.07294v4 --skip-download
python evaluation\run_eval.py ragbench --source text --n-papers 10 --force-reindex

# Final EN
python evaluation\run_eval.py ragbench-prepare
python evaluation\run_eval.py ragbench-eval
```

Dataset y manifiesto del flujo final EN:

- Manifiesto por defecto: `evaluation/runs/ragas/ragbench_prepared/ragbench_en_eval_manifest.json`
- Dataset por defecto: `evaluation/runs/ragas/ragbench_prepared/dataset_ragbench_en_eval_text_25p_5q_eval.json`
- Variante local existente de 40 papers: `evaluation/runs/ragas/ragbench_prepared/dataset_ragbench_en_eval_text_40p_5q_eval.json`
- PDFs: `rag/docs/en_ragbench_eval/`
- ChromaDB: `rag/vector_db/en_ragbench_eval_<embed_slug>/`

Salidas RAGAS:

- `evaluation/runs/ragas/scores/ragas_scores_ragbench_en.csv`
- `evaluation/runs/ragas/debug/ragas_debug_ragbench_en.json`
- `evaluation/runs/ragas/checkpoints/ragbench/ragbench_<tag>.json`

Para el flujo final `ragbench-eval`, los nombres incluyen `ragbench_en_final_<dataset>`:

- `evaluation/runs/ragas/scores/ragas_scores_ragbench_en_final_<dataset>.csv`
- `evaluation/runs/ragas/debug/ragas_debug_ragbench_en_final_<dataset>.json`
- `evaluation/runs/ragas/checkpoints/ragbench/ragbench_en_final_<dataset>.json`

## RagBench tablas e imagenes sin RAGAS

Para inferir solo sobre preguntas `text-image` y `text-table` sin ejecutar RAGAS, usar:

```powershell
python evaluation\run_ragbench_visual_inference.py --n-papers 25 --max-q 5
```

Este script:

- excluye el split dev congelado por defecto;
- prepara preguntas `text-image` y `text-table`;
- desactiva `USAR_RERANKER` por defecto para evitar bloqueos del cross-encoder durante inferencia larga;
- descarga PDFs en `rag/docs/en_ragbench_visual/`;
- usa una base vectorial separada: `rag/vector_db/en_ragbench_visual_<embed_slug>/`;
- guarda dataset, manifiesto y checkpoints en `evaluation/runs/inference/ragbench_visual/debug/`;
- guarda resultados sin metricas en `evaluation/runs/inference/ragbench_visual/results/`;
- no llama a RAGAS.

Para las evaluaciones del TFG sobre los datasets locales, usar `single` o `compare`.
