# Inventario de resultados RAGAS e inferencia

Localizado desde la raiz del repositorio `C:\Users\nadiv\repos\localOllamaRAG`.

Notas:
- Los CSV RAGAS principales tienen columnas como `answer_correctness`, `faithfulness`, `answer_relevancy`, `context_precision` y `context_recall`.
- Los checkpoints de inferencia son los JSON que guardan las respuestas del modelo antes de la evaluacion RAGAS. En los de ablacion aparece un array `answers`.
- No incluyo como resultados principales los CSV de `runs/bertscore`, aunque muchos contienen columnas RAGAS, porque son salidas derivadas con metricas BERTScore anadidas.

## Recalculo seleccionado con NVIDIA

Objetivo: reevaluar con RAGAS usando la API gratuita de NVIDIA Build/API, en lugar del juez anterior de Google.

Script dedicado:
- `research/evaluation/eval_ragas_nvidia_from_checkpoints.py`

Directorio de salida diferenciado por defecto:
- `research/evaluation/runs/ragas_nvidia_revaluation`

Configuracion actual del script:
- LLM juez: `mistralai/mistral-medium-3.5-128b`
- Embeddings: `nvidia/llama-3.2-nv-embedqa-1b-v2`
- Rate limit compartido: `40` llamadas/minuto
- RAGAS workers/batch por defecto: `3` / `3`
- Max tokens por defecto: `32768`
- `reasoning_effort`: valor por defecto del proveedor mediante `--reasoning-effort auto` (solo se fuerza `none` al seleccionar Mistral Small)

Checkpoints y carpetas que se quieren recalcular:

- `research/evaluation/runs/ragas/ragbench/en_eval/dataset_ragbench_en_eval_text_40p_5q_eval/checkpoint.json`
- `research/evaluation/runs/ragas/ragbench_visual/inference/image_table_25p_5q/checkpoint.json`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion/checkpoints`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/checkpoints`
- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/checkpoints`

Comando base desde la raiz del repositorio:

```powershell
python .\research\evaluation\eval_ragas_nvidia_from_checkpoints.py --checkpoint .\research\evaluation\runs\ragas\ragbench\en_eval\dataset_ragbench_en_eval_text_40p_5q_eval\checkpoint.json --checkpoint .\research\evaluation\runs\ragas\ragbench_visual\inference\image_table_25p_5q\checkpoint.json --checkpoint .\research\evaluation\runs\ragas\comparisons\todas_ablacion\checkpoints --checkpoint .\research\evaluation\runs\ragas\comparisons\todas_ablacion_ca_ca\checkpoints --checkpoint .\research\evaluation\runs\ragas\comparisons\ragbench_ablation_en_dev10_frozen\checkpoints --metrics all
```

### Comandos de ejecucion

Ejecutar una prueba pequena de 5 preguntas con todas las metricas:

```powershell
python .\research\evaluation\eval_ragas_nvidia_from_checkpoints.py --checkpoint .\research\evaluation\runs\ragas\comparisons\todas_ablacion\checkpoints\baseline_all_on.json --limit 5 --metrics all --output-root .\research\evaluation\runs\ragas_nvidia_prueba_5
```

Ejecutar solo el conjunto propio en castellano, todas las variantes de ablacion:

```powershell
python .\research\evaluation\eval_ragas_nvidia_from_checkpoints.py --checkpoint .\research\evaluation\runs\ragas\comparisons\todas_ablacion\checkpoints --metrics all
```

Ejecutar solo el conjunto propio en valenciano/catalan, todas las variantes de ablacion:

```powershell
python .\research\evaluation\eval_ragas_nvidia_from_checkpoints.py --checkpoint .\research\evaluation\runs\ragas\comparisons\todas_ablacion_ca_ca\checkpoints --metrics all
```

Ejecutar solo RagBench ablacion, todas las variantes:

```powershell
python .\research\evaluation\eval_ragas_nvidia_from_checkpoints.py --checkpoint .\research\evaluation\runs\ragas\comparisons\ragbench_ablation_en_dev10_frozen\checkpoints --metrics all
```

Ejecutar RagBench final texto 40p x 5q:

```powershell
python .\research\evaluation\eval_ragas_nvidia_from_checkpoints.py --checkpoint .\research\evaluation\runs\ragas\ragbench\en_eval\dataset_ragbench_en_eval_text_40p_5q_eval\checkpoint.json --metrics all
```

Ejecutar RagBench visual final:

```powershell
python .\research\evaluation\eval_ragas_nvidia_from_checkpoints.py --checkpoint .\research\evaluation\runs\ragas\ragbench_visual\inference\image_table_25p_5q\checkpoint.json --metrics all
```

Ejecutar el recalculo completo seleccionado, 26 checkpoints:

```powershell
python .\research\evaluation\eval_ragas_nvidia_from_checkpoints.py --checkpoint .\research\evaluation\runs\ragas\ragbench\en_eval\dataset_ragbench_en_eval_text_40p_5q_eval\checkpoint.json --checkpoint .\research\evaluation\runs\ragas\ragbench_visual\inference\image_table_25p_5q\checkpoint.json --checkpoint .\research\evaluation\runs\ragas\comparisons\todas_ablacion\checkpoints --checkpoint .\research\evaluation\runs\ragas\comparisons\todas_ablacion_ca_ca\checkpoints --checkpoint .\research\evaluation\runs\ragas\comparisons\ragbench_ablation_en_dev10_frozen\checkpoints --metrics all
```

Probar `workers=3` y `batch_size=3` guardando en un directorio alternativo:

```powershell
python .\research\evaluation\eval_ragas_nvidia_from_checkpoints.py --checkpoint .\research\evaluation\runs\ragas\comparisons\todas_ablacion\checkpoints --metrics all --ragas-max-workers 3 --ragas-batch-size 3 --output-root .\research\evaluation\runs\ragas_nvidia_revaluation_w3_b3
```

Usar una API key temporal solo para la sesion actual de PowerShell:

```powershell
$env:NVIDIA_API_KEY="PEGA_AQUI_LA_API_KEY"; python .\research\evaluation\eval_ragas_nvidia_from_checkpoints.py --checkpoint .\research\evaluation\runs\ragas\comparisons\ragbench_ablation_en_dev10_frozen\checkpoints --metrics all --ragas-max-workers 3 --ragas-batch-size 3 --output-root .\research\evaluation\runs\ragas_nvidia_revaluation_ragbench_ablation_w3_b3
```

Comprobar valores nulos al terminar una ejecucion:

```powershell
python -c "import pandas as pd; p='research/evaluation/runs/ragas_nvidia_revaluation/comparisons/todas_ablacion/baseline_all_on/scores.csv'; df=pd.read_csv(p); print(df.isna().sum())"
```

Reintentar solo filas con metricas nulas de un `scores.csv` ya existente:

```powershell
python .\research\evaluation\eval_ragas_nvidia_from_checkpoints.py --checkpoint .\research\evaluation\runs\ragas\ragbench_visual\inference\image_table_25p_5q\checkpoint.json --output-root .\research\evaluation\runs\ragas_nvidia_revaluation --retry-failed
```

## 1. Dataset propio en castellano

Dataset indicado en el resumen: `research/evaluation/datasets/local/dataset_eval_es.json`  
Carpeta de la suite: `research/evaluation/runs/ragas/comparisons/todas_ablacion`

### CSV RAGAS

- `research/evaluation/runs/ragas/comparisons/todas_ablacion/scores/baseline_all_on.csv`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion/scores/no_query_decomposition.csv`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion/scores/no_lexical_search.csv`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion/scores/no_exhaustive_search.csv`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion/scores/no_reranker.csv`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion/scores/no_context_expansion.csv`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion/scores/no_context_optimization.csv`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion/scores/no_recomp_synthesis.csv`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion/scores/resumen_por_conjunto.csv`

### Checkpoints de inferencia

- `research/evaluation/runs/ragas/comparisons/todas_ablacion/checkpoints/baseline_all_on.json`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion/checkpoints/no_query_decomposition.json`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion/checkpoints/no_lexical_search.json`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion/checkpoints/no_exhaustive_search.json`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion/checkpoints/no_reranker.json`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion/checkpoints/no_context_expansion.json`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion/checkpoints/no_context_optimization.json`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion/checkpoints/no_recomp_synthesis.json`

Auxiliares relacionados:
- `research/evaluation/runs/ragas/comparisons/todas_ablacion/comparison_summary.json`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion/debug/*.json`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion/aggregates/*.json`

## 2. Dataset propio en valenciano/catalan

Dataset indicado en el resumen: `research/evaluation/datasets/local/dataset_eval_ca.json`  
Carpeta de la suite: `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca`

### CSV RAGAS

- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/scores/baseline_all_on.csv`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/scores/no_query_decomposition.csv`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/scores/no_lexical_search.csv`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/scores/no_exhaustive_search.csv`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/scores/no_reranker.csv`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/scores/no_context_expansion.csv`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/scores/no_context_optimization.csv`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/scores/no_recomp_synthesis.csv`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/scores/resumen_por_conjunto.csv`

### Checkpoints de inferencia

- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/checkpoints/baseline_all_on.json`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/checkpoints/no_query_decomposition.json`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/checkpoints/no_lexical_search.json`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/checkpoints/no_exhaustive_search.json`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/checkpoints/no_reranker.json`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/checkpoints/no_context_expansion.json`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/checkpoints/no_context_optimization.json`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/checkpoints/no_recomp_synthesis.json`

Auxiliares relacionados:
- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/comparison_summary.json`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/debug/*.json`
- `research/evaluation/runs/ragas/comparisons/todas_ablacion_ca_ca/aggregates/*.json`

## 3. RagBench, ablacion

Dataset indicado en el resumen: `research/evaluation/datasets/ragbench/prepared/dev_frozen/dataset_ragbench_text_10p_5q_dev10_frozen.json`  
Carpeta de la suite: `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen`

### CSV RAGAS

- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/scores/baseline_all_on.csv`
- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/scores/no_query_decomposition.csv`
- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/scores/no_lexical_search.csv`
- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/scores/no_exhaustive_search.csv`
- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/scores/no_reranker.csv`
- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/scores/no_context_expansion.csv`
- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/scores/no_context_optimization.csv`
- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/scores/no_recomp_synthesis.csv`
- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/scores/resumen_por_conjunto.csv`

### Checkpoints de inferencia

- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/checkpoints/baseline_all_on.json`
- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/checkpoints/no_query_decomposition.json`
- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/checkpoints/no_lexical_search.json`
- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/checkpoints/no_exhaustive_search.json`
- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/checkpoints/no_reranker.json`
- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/checkpoints/no_context_expansion.json`
- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/checkpoints/no_context_optimization.json`
- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/checkpoints/no_recomp_synthesis.json`

Auxiliares relacionados:
- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/comparison_summary.json`
- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/debug/*.json`
- `research/evaluation/runs/ragas/comparisons/ragbench_ablation_en_dev10_frozen/aggregates/*.json`

## 4. RagBench, tests finales

### Texto, 25 papers x 5 preguntas

- CSV RAGAS: `research/evaluation/runs/ragas/ragbench/en_eval/dataset_ragbench_en_eval_text_25p_5q_eval/scores.csv`
- Checkpoint de inferencia: `research/evaluation/runs/ragas/ragbench/en_eval/dataset_ragbench_en_eval_text_25p_5q_eval/checkpoint.json`
- Debug: `research/evaluation/runs/ragas/ragbench/en_eval/dataset_ragbench_en_eval_text_25p_5q_eval/debug.json`
- Dataset preparado: `research/evaluation/datasets/ragbench/prepared/en_eval/dataset_ragbench_en_eval_text_25p_5q_eval.json`

### Texto, 40 papers x 5 preguntas

- CSV RAGAS: `research/evaluation/runs/ragas/ragbench/en_eval/dataset_ragbench_en_eval_text_40p_5q_eval/scores.csv`
- Checkpoint de inferencia: `research/evaluation/runs/ragas/ragbench/en_eval/dataset_ragbench_en_eval_text_40p_5q_eval/checkpoint.json`
- Debug: `research/evaluation/runs/ragas/ragbench/en_eval/dataset_ragbench_en_eval_text_40p_5q_eval/debug.json`
- Dataset preparado: `research/evaluation/datasets/ragbench/prepared/en_eval/dataset_ragbench_en_eval_text_40p_5q_eval.json`

### Visual, image/table 25 papers x 5 preguntas

- CSV RAGAS: `research/evaluation/runs/ragas/ragbench_visual/image_table_25p_5q/scores.csv`
- Debug RAGAS: `research/evaluation/runs/ragas/ragbench_visual/image_table_25p_5q/debug.json`
- CSV de inferencia enviado a RAGAS: `research/evaluation/runs/ragas/ragbench_visual/inference/image_table_25p_5q/results.csv`
- JSON de inferencia enviado a RAGAS: `research/evaluation/runs/ragas/ragbench_visual/inference/image_table_25p_5q/results.json`
- Checkpoint de inferencia: `research/evaluation/runs/ragas/ragbench_visual/inference/image_table_25p_5q/checkpoint.json`
- Dataset preparado: `research/evaluation/datasets/ragbench/prepared/visual/dataset_ragbench_visual_image_table_25p_5q.json`

## 5. Archivos relacionados pero no principales

### BERTScore derivado

Carpeta: `research/evaluation/runs/bertscore`

Contiene copias o extensiones de resultados con columnas RAGAS mas columnas BERTScore:
- `bertscore_precision`
- `bertscore_recall`
- `bertscore_f1`
- `bertscore_model`
- `bertscore_rescale_with_baseline`

No los he tratado como CSV RAGAS principales porque parecen postprocesados.

### Datasets fuente/locales localizados

- `research/evaluation/datasets/local/dataset_eval_es.json`
- `research/evaluation/datasets/local/dataset_eval_ca.json`
- `research/evaluation/datasets/local/dataset_eval_mix.json`
- `research/evaluation/datasets/ragbench/prepared/dev_frozen/dataset_ragbench_text_10p_5q_dev10_frozen.json`
- `research/evaluation/datasets/ragbench/prepared/en_eval/dataset_ragbench_en_eval_text_25p_5q_eval.json`
- `research/evaluation/datasets/ragbench/prepared/en_eval/dataset_ragbench_en_eval_text_40p_5q_eval.json`
- `research/evaluation/datasets/ragbench/prepared/visual/dataset_ragbench_visual_image_table_25p_5q.json`
