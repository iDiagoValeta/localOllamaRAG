# Reinferencia final del pipeline RAG (sin RAGAS)

> Documento de contexto para la **segunda pasada** de inferencia sobre los tres
> corpus de evaluación del TFG. Existe porque la primera pasada
> (`research/evaluation/runs/ragas/`) se ejecutó con configuraciones de Ollama
> incompletas (`num_ctx`, `num_predict`, `repeat_last_n`) y produjo respuestas
> truncadas en una fracción de muestras. Estos resultados antiguos **no se
> tocan**: la reinferencia escribe en un árbol paralelo y, cuando termine la
> nueva evaluación RAGAS, se sustituirán.
>
> Autor del TFG: Ignacio Diago Valeta · Tutor: Adrià Giménez Pastor · ETSINF — UPV.

---

## 1. Objetivo

Volver a generar respuestas RAG sobre los siguientes corpus, **sin evaluar con
RAGAS** todavía, usando solo la comparación final `baseline_all_on` vs
`all_off`:

| Corpus | ChromaDB (ya indexado) | Dataset por defecto |
|---|---|---|
| `es` (español) | `rag/vector_db/es_embeddinggemma` | `research/evaluation/datasets/local/dataset_eval_es.json` |
| `ca` (catalán) | `rag/vector_db/ca_embeddinggemma` | `research/evaluation/datasets/local/dataset_eval_ca.json` |
| `en_ragbench_eval_40p` | `rag/vector_db/en_ragbench_eval_embeddinggemma` | `research/evaluation/datasets/ragbench/en_eval/dataset_ragbench_en_eval_text_40p_5q_eval.json` |
| `en_ragbench_visual` | `rag/vector_db/en_ragbench_visual_embeddinggemma` | `research/evaluation/datasets/ragbench/visual/dataset_ragbench_visual_image_table_25p_5q.json` |

El generador final es **Qwen3-14B fine-tuneado** (Modelfile `qwen3-finetuned-...`
en `research/models/gguf/qwen-3/`). Como el nombre contiene `finetuned`, el
prompt RAG ya está horneado en el Modelfile y `chat_pdfs.py` **no inyecta**
`SYSTEM_PROMPT_RAG` por API.

La evaluación RAGAS se hará después en otra ejecución, sobre los checkpoints
nuevos. Por ahora sólo persistimos checkpoints + debug.

---

## 2. Variantes de comparación

`research/evaluation/_lib/pipeline_flags.py` mantiene la suite legacy
`ablation`, pero la reinferencia final usa la suite por defecto `final` con
**2 variantes**:

1. `baseline_all_on` — todas las fases opcionales activas.
2. `all_off` — todas las flags opcionales desactivadas. Equivale a
   recuperación semántica pura + filtro por `UMBRAL_RELEVANCIA` + `TOP_K_FINAL`
   chunks pasados al generador.

> La suite larga puede lanzarse explícitamente con `--suite ablation`, pero no
> forma parte de la repetición final.

---

## 3. Comando exacto por corpus

Asumiendo que los Chroma ya están construidos para `embeddinggemma`, **no se
reindexa** (`--reindex` se omite). El flag `--label` envía todo a un directorio
paralelo `runs/ragas/comparisons/<label>/`, que evita pisar la primera pasada.

```powershell
# Español
python research/evaluation/infer.py compare --corpus es `
    --label reinferencia_v2_es --verbose

# Catalán
python research/evaluation/infer.py compare --corpus ca `
    --label reinferencia_v2_ca --verbose

# Inglés / RagBench eval 40p
python research/evaluation/infer.py compare --corpus en `
    --dataset research/evaluation/datasets/ragbench/en_eval/dataset_ragbench_en_eval_text_40p_5q_eval.json `
    --docs-dir rag/docs/en_ragbench_eval `
    --label reinferencia_v2_en_ragbench_eval_40p --verbose

# Inglés / RagBench visual (text-image + text-table)
python research/evaluation/infer.py compare --corpus en `
    --dataset research/evaluation/datasets/ragbench/visual/dataset_ragbench_visual_image_table_25p_5q.json `
    --docs-dir rag/docs/en_ragbench_visual `
    --label reinferencia_v2_en_ragbench_visual_image_table_25p --verbose
```

> Rutas confirmadas: `research/evaluation/datasets/ragbench/en_eval/` y
> `research/evaluation/datasets/ragbench/visual/`. El split dev congelado vive
> en `research/evaluation/datasets/ragbench/dev_frozen/` y solo se usa para
> excluir papers del held-out.

Cada ejecución produce:

```
research/evaluation/runs/ragas/comparisons/reinferencia_v2_<corpus>/
├── checkpoints/<variant>.json   # ← entrada de evaluate.py en el siguiente paso
├── scores/<variant>.csv
├── debug/<variant>.json
└── inference_summary.json
```

---

## 4. Configuración del entorno antes de lanzar

Todas son **variables de entorno** que `chat_pdfs.py` lee al import. Hay que
fijarlas en la sesión de PowerShell antes de invocar `infer.py`:

```powershell
# OLLAMA_RAG_MODEL no hace falta exportarlo: chat_pdfs.py:158 ya tiene
# default = "Qwen3-FineTuned:latest". El resto sigue el default del módulo
# salvo que se quiera forzar.

$env:OLLAMA_CHAT_MODEL       = "qwen3:14b"                # sub-queries
$env:OLLAMA_EMBED_MODEL      = "embeddinggemma:latest"
$env:OLLAMA_CONTEXTUAL_MODEL = "qwen3:14b"
$env:OLLAMA_RECOMP_MODEL     = "qwen3:14b"
$env:OLLAMA_OCR_MODEL        = "qwen3-vl:8b"

# Umbral del reranker: NO es env-overridable, es constante de módulo en
# rag/chat_pdfs.py:369. Decisión 2026-05-14: subir de 0.55 a 0.70 editando
# directamente esa línea antes de lanzar la reinferencia. Cambio versionable
# y discutible en el capítulo 4 con motivación de "precisión > recall en
# contexto: el juez RAGAS penaliza chunks irrelevantes".

# Contextos Ollama (la causa raíz del truncado de la pasada vieja)
$env:OLLAMA_NUM_CTX             = "8192"
$env:OLLAMA_RAG_NUM_CTX         = "16384"
$env:OLLAMA_AUX_NUM_CTX         = "8192"
$env:OLLAMA_QUERY_NUM_CTX       = "8192"
$env:OLLAMA_RECOMP_NUM_CTX      = "8192"
$env:OLLAMA_CONTEXTUAL_NUM_CTX  = "32768"
$env:OLLAMA_OCR_NUM_CTX         = "8192"
$env:OLLAMA_REQUEST_TIMEOUT     = "900s"
```

> **Resiliencia ante truncados (verificado 2026-05-14):**
> Al reanudar, `_lib/inference.py` aplica dos capas de detección antes de
> calcular los pendientes:
>
> 1. **Respuestas vacías** — `indices_respuestas_vacias` (`checkpoints.py:42`)
>    devuelve todos los índices sin respuesta no-vacía.
> 2. **Respuestas truncadas** — `respuesta_truncada` (`checkpoints.py`) detecta
>    respuestas no-vacías cuyo último carácter no es puntuación terminal
>    (`. ! ? » " ' ) …`). Las marca como `status=failed/respuesta_truncada`
>    antes de pasar a `indices_pendientes_generacion`, lo que las incluye en la
>    cola de reintento. El log muestra `Found N truncated answer(s). Will
>    regenerate: …` cuando hay casos detectados.
>
> Eso garantiza que **basta con relanzar `infer.py compare` con el mismo
> `--label`**: el checkpoint vive en `runs/ragas/comparisons/<label>/checkpoints/<variant>.json`,
> se carga al arrancar la variante, y sólo se generan respuestas para las
> preguntas pendientes. No hace falta `--resume`. Cuidado: si entre una pasada
> y la siguiente cambian las flags de pipeline o los modelos,
> `checkpoint_pipeline_flags_match` / `checkpoint_models_match` invalidan el
> checkpoint y la variante vuelve a empezar de cero.

---

## 5. Discusión: ¿es 0.55 un umbral razonable para el reranker?

`UMBRAL_SCORE_RERANKER = 0.55` (cross-encoder local). Argumentos en ambos
sentidos:

- **A favor de subirlo (≥ 0.6):** un juez RAGAS (`context_precision`,
  `faithfulness`) penaliza contextos irrelevantes. Si el reranker deja pasar
  fragmentos marginales, la métrica baja aunque la respuesta sea correcta. En
  cross-encoders mainstream (BGE, MiniLM, MS-MARCO) 0.55 suele estar en zona
  “relevancia débil”.
- **En contra:** subir el umbral reduce `TOP_K_FINAL` efectivo. Si quedan
  menos de los 8 chunks pretendidos, `context_recall` cae. En corpus pequeños
  (catalán, RagBench dev) el efecto puede ser brutal.

**Propuesta:** antes de cerrar la reinferencia con un umbral nuevo, hacer una
sonda con `rag/debug_rag/` sobre 8–10 preguntas representativas por corpus,
viendo qué scores reciben los chunks "buenos" vs "ruido". Si el cluster bueno
vive en 0.65–0.85 y el ruido en 0.40–0.55, subir a 0.60 es seguro. Si están
solapados, hay que tocar otra palanca (modelo de reranker más fino, RRF
weights, o `TOP_K_RERANK_CANDIDATES`).

> Confirmación del usuario pendiente: ¿lanzo la sonda de scores antes de la
> reinferencia o se hace la pasada con `0.55` actual y se valora a posteriori?
> Cambiar el umbral durante la pasada invalidaría la comparabilidad entre
> variantes.

---

## 6. Sonda de scores del reranker

Script: `research/evaluation/probe_reranker_scores.py` (creado 2026-05-14).

Para cada corpus muestrea N preguntas del dataset de evaluación, ejecuta la
recuperación híbrida **con reranker apagado** (para capturar todos los
candidatos post-RRF, no sólo los `TOP_K_AFTER_RERANK = 15` que el flujo
normal deja pasar) y después puntúa cada candidato manualmente con el
Cross-Encoder. Salida: JSON único por corpus en `rag/debug_rag/`.

```powershell
python research/evaluation/probe_reranker_scores.py --corpus es              --n 8
python research/evaluation/probe_reranker_scores.py --corpus ca              --n 8
python research/evaluation/probe_reranker_scores.py --corpus en_ragbench_dev --n 8
```

Cada `probe_<corpus>_<timestamp>.json` contiene:

- `current_umbral_score_reranker`: valor actual en `chat_pdfs.py`.
- `aggregate.umbral_cuts` y por pregunta `umbral_cuts`: cuántos candidatos
  pasan cada umbral en `{0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75}`.
- `aggregate.stats`: min/mean/median/p10/p25/p75/p90 sobre todos los candidatos.
- Por pregunta: `n_candidatos_fusion`, `sub_queries`, `keywords`, lista de
  candidatos ordenados por `score_reranker` con `source`, `page` y `preview`.

### Cómo leer la salida para validar el salto 0.55 → 0.70

1. Mirar `aggregate.umbral_cuts`: cuántos candidatos pasan a 0.55 vs 0.70.
   Si pasar de 0.55 a 0.70 corta más del ~70 %, probablemente perdamos
   `TOP_K_FINAL = 8` y `EXPANDIR_CONTEXTO` se quede sin material.
2. Por pregunta, marcar los candidatos cuyo `source`/`page` aparezca en la
   respuesta esperada. Si su `score_reranker` está consistentemente sobre
   0.70, el umbral es seguro; si caen en 0.55-0.65, 0.70 es demasiado.
3. Si los corpus difieren mucho (p.ej. CA con scores más bajos por menor
   solapamiento de vocabulario reranker↔dataset), considerar **umbrales por
   corpus** en vez de uno global.

> Decisiones que esta sonda informa: `UMBRAL_SCORE_RERANKER` (§5),
> `TOP_K_FINAL`, `MAX_CONTEXTO_CHARS`, `N_TOP_PARA_EXPANSION`.

---

## 7. Política de almacenamiento

- **No tocar** `research/evaluation/runs/ragas/comparisons/todas_ablacion*/` ni
  `runs/ragas/ragbench/en_eval/`. Son la primera pasada.
- Reinferencia nueva → siempre con `--label reinferencia_v2_<corpus>`.
- El día que la nueva pasada RAGAS supere a la vieja en `evaluation_comparison`
  consolidaremos: mover `todas_ablacion*` a `runs/ragas/_archive_v1/` y
  renombrar `reinferencia_v2_*` a su sitio.
- Los checkpoints son los únicos artefactos que `evaluate.py` consume — no
  hace falta versionar nada más en git.

---

## 8. Decisiones cerradas (2026-05-14)

1. **2 variantes finales** — la suite por defecto `final` usa
   `baseline_all_on` y `all_off`; la suite `ablation` queda como legacy.
2. **Modelo generador**: `Qwen3-FineTuned:latest` (default en `chat_pdfs.py:158`).
3. **Datasets RagBench reubicados**:
   `research/evaluation/datasets/ragbench/{dev_frozen,en_eval,visual}/`.
4. **Sonda de scores**: pendiente (ver §6 actualizado).
5. **Reranker `0.55 → 0.65`** — aplicado en `rag/chat_pdfs.py:369`. Decisión
   informada por la sonda 2026-05-14: 0.70 colapsaba CA-Q3 a 1 candidato
   (riesgo en multi-evidencia); 0.65 mantiene la meseta post-ruido en los
   tres corpus (ES 30 cand., CA 15, EN-RagBench 34 sobre 8 preguntas).
6. **Reanudación del checkpoint**: verificada en
   `_lib/checkpoints.py::indices_pendientes_generacion`. Idempotente por
   pregunta mientras flags + modelos no cambien.

---

## 9. Coste estimado

Cada corpus ejecuta **2 variantes** (`baseline_all_on` y `all_off`). Con
generación local (Ollama Qwen3-14B FT, streaming), cada variante son del orden
de horas, no minutos. Conviene lanzar
**una variante en background** con `run_in_background` y monitorizar con
`Monitor` para no bloquear la sesión interactiva. **No** lanzar todo a la vez:
saturarían VRAM / sumarían latencia de cola.
