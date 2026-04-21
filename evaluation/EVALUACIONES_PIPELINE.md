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

La comparativa guarda artefactos bajo:

```text
evaluation/scores/comparison_runs/<label>/
evaluation/debug/comparison_runs/<label>/
evaluation/debug/checkpoints/comparison_runs/<label>/
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

La logica de RagBench esta integrada directamente en `run_eval.py` (Section 7). Descarga metadatos de HuggingFace, selecciona papers, descarga PDFs en `rag/docs/en/` y delega la generacion, checkpoints, RAGAS, CSV y debug al runner unificado. PDFs descargados se reutilizan en ejecuciones posteriores.

```powershell
python evaluation\run_eval.py ragbench --n-papers 3 --max-q 5
python evaluation\run_eval.py ragbench --only-doc 2401.07294v4 --skip-download
python evaluation\run_eval.py ragbench --source text --n-papers 10 --force-reindex
```

Salidas de RagBench:
- `evaluation/scores/ragas_scores_ragbench_en.csv`
- `evaluation/debug/ragas_debug_ragbench_en.json`
- `evaluation/debug/checkpoints/ragbench/ragbench_<tag>.json`

Para las evaluaciones del TFG sobre los datasets locales, usar `single` o `compare`.
