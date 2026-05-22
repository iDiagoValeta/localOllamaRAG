# Re-inferencia BM25 — variante `baseline_all_on` (5 conjuntos)

Comandos para regenerar la inferencia con el pipeline nuevo (BM25) usando **solo la
variante `baseline_all_on`**. No se re-ejecuta `all_off` porque desactiva todo
(incluido BM25) y es idéntico antes y después (suelo solo-semántica).

**No tocan los checkpoints antiguos:** `compare` nombra la carpeta de salida por
`--label`, así que con etiquetas nuevas se crean directorios aparte y los
`reinferencia_v3_*` / `..._visual_..._25p/` quedan intactos.

Ejecutar desde la raíz del repo, en el entorno conda `(base)` (el que tiene
chromadb/ollama/rank-bm25). Ver [BM25_MIGRATION.md](BM25_MIGRATION.md) para el detalle
del cambio.

---

## Comandos

```bash
# Castellano
python research/evaluation/infer.py compare --corpus es \
  --variants baseline_all_on --label bm25_es_all_on

# Catalán
python research/evaluation/infer.py compare --corpus ca \
  --variants baseline_all_on --label bm25_ca_all_on

# RAGBench dev (10p)
python research/evaluation/infer.py compare --corpus en \
  --dataset research/evaluation/datasets/ragbench/dev_frozen/dataset_ragbench_text_10p_5q_dev10_frozen.json \
  --docs-dir rag/docs/en_ragbench_dev \
  --variants baseline_all_on --label bm25_ragbench_dev_all_on

# RAGBench test / eval (40p)
python research/evaluation/infer.py compare --corpus en \
  --dataset research/evaluation/datasets/ragbench/en_eval/dataset_ragbench_en_eval_text_40p_5q_eval.json \
  --docs-dir rag/docs/en_ragbench_eval \
  --variants baseline_all_on --label bm25_ragbench_eval_all_on

# RAGBench image-table / visual (25p)
python research/evaluation/infer.py compare --corpus en \
  --dataset research/evaluation/datasets/ragbench/visual/dataset_ragbench_visual_image_table_25p_5q.json \
  --docs-dir rag/docs/en_ragbench_visual \
  --variants baseline_all_on --label bm25_ragbench_visual_all_on
```

---

## Dónde se guardan (espacio nuevo)

Cada comando crea una carpeta bajo `research/evaluation/runs/ragas/comparisons/<label>/`:

```
research/evaluation/runs/ragas/comparisons/<label>/
  ├── checkpoints/baseline_all_on.json
  ├── scores/baseline_all_on.csv
  ├── debug/baseline_all_on.json
  └── inference_summary.json
```

Labels: `bm25_es_all_on`, `bm25_ca_all_on`, `bm25_ragbench_dev_all_on`,
`bm25_ragbench_eval_all_on`, `bm25_ragbench_visual_all_on`.

---

## Notas

1. **Sin `--reindex` a propósito.** BM25 opera sobre los chunks ya indexados; no cambia
   la indexación. Estos comandos reutilizan las colecciones ChromaDB existentes → solo
   re-inferencia.
   - Si alguna colección no existe localmente (error de colección vacía): para es/ca
     `python research/evaluation/index.py --corpus es` (o `--corpus ca`) una vez; para
     RAGBench, añadir `--reindex` al comando una sola vez. Si reindexas el visual,
     `USAR_EMBEDDINGS_IMAGEN` debe estar activo (es el default).

2. **Evaluación RAGAS (opcional, paso siguiente).** Tras la inferencia, por cada carpeta:
   ```bash
   python research/evaluation/evaluate.py --provider google \
     --source-root research/evaluation/runs/ragas/comparisons/bm25_es_all_on
   ```
   (Repetir cambiando el `--label`/`--source-root` para cada conjunto.)

3. **Consecuencia conocida (ver BM25_MIGRATION.md §12):** los checkpoints congelados
   guardan el flag `USAR_BUSQUEDA_EXHAUSTIVA` ya eliminado, por lo que no se reutilizan
   como caché; por eso estas corridas usan etiquetas nuevas y parten de cero.
