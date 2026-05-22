# Ranking, parámetros y defensa del pipeline RAG

> Fecha: 2026-05-23. Documento único de referencia para la **recuperación
> híbrida** (semántica + BM25), la **fusión por RRF**, el **reranking**, la
> **síntesis de contexto** y la **clasificación defensiva** de todos los valores
> numéricos del pipeline. Consolida y sustituye a los antiguos
> `BM25_MIGRATION.md`, `REINFERENCIA_BM25.md` y `PIPELINE_PARAMETERS_DEFENSE.md`.
>
> Fuente de verdad: el código (`rag/chat_pdfs.py` §3.7 y `rag/engine/`). Si este
> documento y el código discrepan, manda el código.

Regla de defensa del TFG: **no presentar como "valor teórico de un paper" lo que
en realidad es un hiperparámetro operativo del sistema**, y al revés, **citar el
valor canónico cuando se usa**.

---

## 1. Mapa de citas: qué documento respalda cada etapa

Cada referencia del TFG respalda una etapa concreta del pipeline. No todas
prescriben valores numéricos; varias justifican el **diseño**, no las constantes.

| Documento | Cita | Etapa del pipeline que respalda | ¿Prescribe valores? |
|---|---|---|---|
| RAG-Fusion (Rackauckas, 2024) | [arXiv:2402.03367](https://arxiv.org/abs/2402.03367) | Descomposición multi-query + fusión por RRF | No |
| Passage Re-ranking with BERT (Nogueira & Cho, 2019) | [arXiv:1901.04085](https://arxiv.org/abs/1901.04085) | Reranking con Cross-Encoder (retrieve-then-rerank) | No (umbral propio) |
| Late Chunking (Günther et al., 2024) | [arXiv:2409.04701](https://arxiv.org/abs/2409.04701) | Problema de pérdida de contexto al trocear (**ver §1.1**) | No |
| RECOMP (Xu, Shi & Choi, 2023) | [arXiv:2310.04408](https://arxiv.org/abs/2310.04408) | Síntesis/compresión de contexto + *selective augmentation* (contexto vacío si nada es relevante) | No |
| Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, 2009) | [10.1145/1571941.1572114](https://dl.acm.org/doi/10.1145/1571941.1572114) | Fusión de listas semántica + BM25 | **Sí: `k = 60`** |
| `rank-bm25` (Brown) | [github.com/dorianbrown/rank_bm25](https://github.com/dorianbrown/rank_bm25) | Implementación de Okapi BM25 (`BM25Okapi`) | **Sí: `k1 = 1.5`, `b = 0.75`, `ε = 0.25`** |
| Improvements to BM25... (Trotman, Puurula & Burgess, 2014) | [PDF](https://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf) | Análisis de sensibilidad de `k1`/`b` en BM25 | No (justifica usar defaults) |

### 1.1 Matiz importante sobre dos citas

- **Late Chunking (2409.04701)** propone embeber el documento largo *antes* de
  trocear (chunking tardío). Este proyecto **no implementa late chunking**;
  aborda la misma pérdida de contexto mediante **Contextual Retrieval**
  (enriquecimiento de cada chunk con un resumen situacional generado por LLM,
  flag `USAR_CONTEXTUAL_RETRIEVAL`). Por tanto se cita como **trabajo
  relacionado** sobre el problema de contexto en el troceado, no como técnica
  empleada. No escribir "se aplica late chunking".

- **RAG-Fusion (2402.03367)** genera varias queries y fusiona por RRF. En este
  pipeline las sub-queries del LLM alimentan **solo la rama semántica**; BM25 se
  ejecuta **una sola vez sobre la pregunta original** (ver
  `realizar_busqueda_hibrida` → `busqueda_lexica_bm25(pregunta, collection)`).
  Es una decisión de diseño defendible (BM25 es léxico exacto; multiplicar
  variantes léxicas aporta ruido), pero **debe describirse así** para no parecer
  una incoherencia con la cita de RAG-Fusion.

---

## 2. Clasificación defensiva de los parámetros

Valores en `rag/chat_pdfs.py` §3.7. Tres categorías según cómo se defienden.

### 2.1 Valores canónicos de la literatura / implementación

| Parámetro | Default | Env var | Origen |
|---|---:|---|---|
| `RRF_K` | **60** | `RAG_RRF_K` | Valor canónico de Cormack et al. (2009) |
| `BM25_K1` | 1.5 | `RAG_BM25_K1` | Default de `BM25Okapi` en `rank-bm25` |
| `BM25_B` | 0.75 | `RAG_BM25_B` | Default de `rank-bm25`; canónico Robertson |

**`RRF_K = 60`.** Es el valor que Cormack, Clarke y Büttcher fijan en el paper
original de RRF tras su investigación piloto. Ahora se puede afirmar con rigor:
*"la fusión emplea RRF con el factor de amortiguamiento canónico `k = 60`"*.

> Histórico: el pipeline usó `RRF_K = 20` durante el desarrollo (más peso a las
> primeras posiciones, razonable con listas cortas). Se elevó a `60` para alinearlo
> con el paper. La evaluación final se regenera con `60` (ver §5).

**`BM25_K1 = 1.5`, `BM25_B = 0.75`.** Son los **defaults de la implementación**
(`BM25Okapi(corpus, k1=1.5, b=0.75, epsilon=0.25)`), no valores ajustados sobre
el conjunto de evaluación. `b = 0.75` es además el default histórico de Okapi.

Defensa recomendada: *"Se adopta Okapi BM25 como recuperador disperso clásico.
`k1` y `b` se fijan a los defaults de la implementación empleada (`rank-bm25`)
para evitar ajuste ad hoc sobre el test."* Esto lo respalda **Trotman et al.
(2014)**: tras optimizar 9 funciones de ranking con *particle swarm*, encontraron
que (a) el óptimo de `k1`/`b` **depende del corpus** (en INEX Wikipedia obtuvieron
`b = 0.3`, `k1 = 1.1`, lejos de los defaults) y (b) las diferencias de MAP entre
funciones bien ajustadas son **marginales**. Conclusión: no existe óptimo
universal, así que tunear `k1`/`b` sobre el propio test sería sobreajuste; usar el
default es lo más defendible.

Fórmula Okapi BM25 que devuelve `BM25Okapi.get_scores` (Robertson & Zaragoza, 2009):

```text
score(D,Q) = Σ_{t∈Q}  IDF(t) · [ f(t,D)·(k1+1) ] / [ f(t,D) + k1·(1 − b + b·|D|/avgdl) ]
```

- `f(t,D)`: frecuencia del término `t` en el chunk `D`.
- `|D|`, `avgdl`: longitud del chunk y longitud media → normalización por longitud.
- `k1` (saturación de frecuencia: repetir un término aporta cada vez menos).
- `b` (penalización por longitud del chunk).
- IDF en `BM25Okapi` (Robertson–Spärck-Jones):
  `IDF(t) = ln((N − n(t) + 0.5) / (n(t) + 0.5))`, con `N` = nº de chunks y `n(t)` =
  chunks con `t`. Los IDF negativos (términos casi omnipresentes) se sustituyen por
  `ε · IDF_medio` con `ε = 0.25`, para no restar peso.

### 2.2 Hiperparámetros del sistema (no son de ningún paper)

| Parámetro | Default | Env var | Naturaleza |
|---|---:|---|---|
| `PESO_SEMANTICO_RRF` | 0.55 | `RAG_PESO_SEMANTICO_RRF` | Peso de la rama semántica en la fusión |
| `PESO_BM25_RRF` | 0.45 | `RAG_PESO_BM25_RRF` | Peso de la rama BM25 en la fusión |

El RRF de Cormack es **sin pesos** (suma simple de `1/(k+rank)` sobre todas las
listas). Este pipeline aplica una **variante ponderada**:

```text
score_final(d) = 0.55 · score_semantic(d) + 0.45 · score_BM25(d)
```

Por tanto `0.55/0.45` **no** son del paper de RRF; son una decisión de fusión del
sistema. Defensa: *"la fusión ponderada mantiene una ligera prioridad para la
recuperación semántica —el objetivo es recuperar contenido conceptualmente
relacionado—, con BM25 como ancla léxica. Los pesos se mantienen fijos en todas
las variantes comparadas y se exponen como variables de entorno para
reproducibilidad."* No describirlos como "parámetros de RRF".

### 2.3 Umbrales calibrados

| Parámetro | Default | Env var | Naturaleza |
|---|---:|---|---|
| `UMBRAL_SCORE_RERANKER` | 0.65 | `RAG_UMBRAL_SCORE_RERANKER` | Filtro mínimo del Cross-Encoder |
| `UMBRAL_RELEVANCIA` | 0.50 | `RAG_UMBRAL_RELEVANCIA` | Puerta de relevancia sobre escala reranker |

`UMBRAL_SCORE_RERANKER = 0.65` es el valor mejor calibrado del grupo: tiene
protocolo de sonda (`research/evaluation/probe_reranker_scores.py`) y decisión
documentada (subido de `0.55` → `0.65` el 2026-05-14 tras observar una banda de
ruido `0.40–0.60`; `0.70` colapsaba CA-Q3 a un único candidato). Nogueira & Cho
(2019) no prescriben umbral —producen un ranking—, así que `0.65` se presenta
como **decisión calibrada**, no como valor de paper.

`UMBRAL_RELEVANCIA = 0.50` **no** se aplica al RRF puro. El código solo lo usa
cuando `USAR_RERANKER` está activo, porque entonces `score_final` ya ha sido
sustituido por `score_reranker`. No describirlo como "umbral RRF".

El comportamiento de **devolver contexto vacío cuando ningún candidato supera el
umbral** es exactamente la *selective augmentation* de **RECOMP (2310.04408)**:
no aumentar el prompt con material irrelevante. (En los flujos RAGBench se activa
un *fallback* que conserva los mejores candidatos en lugar de quedarse sin
contexto; ver `EVALUACIONES_PIPELINE.md` §2.5.)

### 2.4 Presupuestos operativos (coste / latencia / límite de contexto)

No son óptimos teóricos: son presupuestos fijados **antes** de la evaluación y
mantenidos constantes entre variantes.

| Parámetro | Default | Papel |
|---|---:|---|
| `N_RESULTADOS_SEMANTICOS` | 80 | Recall por query antes de fusión |
| `N_RESULTADOS_KEYWORD` | 40 | Top-N de BM25 antes de fusión |
| `TOP_K_RERANK_CANDIDATES` | 200 | Límite de coste del Cross-Encoder |
| `TOP_K_AFTER_RERANK` | 15 | Retención post-reranker |
| `TOP_K_FINAL` | 8 | Fragmentos base al generador |
| `N_TOP_PARA_EXPANSION` | 3 | Vecinos para continuidad textual |
| `MAX_CONTEXTO_CHARS` | 24000 | Presupuesto de entrada al LLM |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 2000 / 400 | Granularidad de chunks (solapamiento ~20%) |

---

## 3. Verificación de buenas prácticas (papers vs pipeline)

| Buena práctica | Fuente | ¿Se cumple? | Evidencia en código |
|---|---|---|---|
| Recuperar y luego rerankear (dos etapas) | Nogueira & Cho 2019 | Sí | `realizar_busqueda_hibrida` → fusión → `rerank_resultados` |
| Fusionar listas heterogéneas por RRF | Cormack 2009 | Sí, con `k=60` canónico | `retrieval.py:131,158` |
| BM25 con defaults, sin tunear sobre test | Trotman 2014 | Sí | `BM25Okapi(k1=1.5, b=0.75)` |
| Multi-query + fusión | RAG-Fusion 2024 | Sí (rama semántica) | `generar_queries_con_llm` + acumulación RRF |
| Comprimir contexto antes de generar | RECOMP 2023 | Sí | `sintetizar_contexto_recomp` (`USAR_RECOMP_SYNTHESIS`) |
| No aumentar con contexto irrelevante | RECOMP 2023 | Sí | filtro por `UMBRAL_SCORE_RERANKER` (contexto vacío) |
| Mitigar pérdida de contexto al trocear | Late Chunking 2024 | Vía alternativa | Contextual Retrieval (`USAR_CONTEXTUAL_RETRIEVAL`) |

Puntos a declarar explícitamente en el TFG (no son fallos, son decisiones):

1. La fusión es **RRF ponderado** (`0.55/0.45`), no RRF puro de Cormack.
2. **BM25 solo ve la pregunta original**; las sub-queries van a la rama semántica.
3. **No se hace late chunking**; el problema de contexto se trata con Contextual
   Retrieval.

---

## 4. Migración de la búsqueda léxica a Okapi BM25

> Afecta a la **Etapa 2 (Recuperación híbrida)** con `USAR_BUSQUEDA_HIBRIDA = True`.

### 4.1 Resumen

La capa léxica tenía **dos fases redundantes** de coincidencia de texto:

1. **Keywords** (`busqueda_por_keywords`): filtro `$contains` de ChromaDB + RRF
   por la *posición de devolución* de Chroma.
2. **Exhaustiva** (`busqueda_exhaustiva_texto`): escaneo de toda la colección
   contando términos críticos.

Ninguna producía una puntuación de relevancia léxica formal. Se han **sustituido
por una única búsqueda Okapi BM25** (`busqueda_lexica_bm25`), que sí genera un
ranking de relevancia real y citable, fusionado con la semántica por RRF. El
pipeline pasa de **3 vías** (semántica + keywords + exhaustiva) a **2 vías**
(semántica + BM25).

```
Antes:  pregunta → [semántica] + [keywords $contains] + [exhaustiva scan] → fusión
Ahora:  pregunta → [semántica] + [BM25] → fusión RRF (k=60)
```

### 4.2 Tokenización: `_tokenizar_bm25()`

Tokenizador **único** para corpus y query (requisito de BM25): minúsculas → split
por límites no alfanuméricos (Unicode, conserva acentos es/ca) → descarta
`STOPWORDS` multiidioma → descarta tokens de menos de 3 caracteres salvo que
contengan dígitos (conserva identificadores/métricas como `q4`, `f1`).

`extraer_keywords()` se conserva, pero **ya no dirige la recuperación léxica**:
solo alimenta una variante de query semántica y el campo `keywords` de
métricas/debug. BM25 usa su propia tokenización.

### 4.3 Recuperación: `busqueda_lexica_bm25(pregunta, collection, top_n=N_RESULTADOS_KEYWORD)`

- `rank-bm25` es dependencia obligatoria; `BM25_AVAILABLE` se conserva como
  constante pública de compatibilidad (`True` si `rag.chat_pdfs` importó bien).
- `query_tokens = _tokenizar_bm25(pregunta)`; si no hay tokens, devuelve `[]`.
- Escanea toda la colección en lotes de 100, tokeniza cada chunk, construye
  `BM25Okapi(corpus_tokens, k1=BM25_K1, b=BM25_B)` y puntúa con
  `get_scores(query_tokens)`.
- Ordena por score BM25 desc, **descarta score ≤ 0**, devuelve los `top_n = 40`
  mejores con `bm25_score`. El índice se **reconstruye por consulta** (sin cache).

### 4.4 Fusión final (RRF ponderado)

```text
score_keyword(d)  += 1 / (rank_BM25(d)      + RRF_K)        # RRF_K = 60
score_semantic(d) += 1 / (rank_semantico(d) + RRF_K)        # acumulado sobre variantes de query
score_final(d)     = PESO_SEMANTICO_RRF · score_semantic(d) + PESO_BM25_RRF · score_keyword(d)
```

El `bm25_score` crudo solo **ordena y filtra** (`> 0`); en la mezcla entra su
**rango**, no su valor. Una sola escala, coherente con el lado semántico.

### 4.5 Funciones y dependencias

- **Eliminadas:** `busqueda_por_keywords`, `busqueda_exhaustiva_texto`
  (`rag/engine/lexical.py`); `_filtrar_terminos_criticos`
  (`rag/engine/reranking.py`, quedaba muerta).
- **Añadidas:** `_tokenizar_bm25`, `busqueda_lexica_bm25`
  (`rag/engine/lexical.py`).
- **Flag eliminado:** `USAR_BUSQUEDA_EXHAUSTIVA` (de `chat_pdfs.py`, de
  `PIPELINE_RUNTIME_FLAGS` y de la capa de evaluación
  `research/evaluation/_lib/pipeline_flags.py`). La suite `ablation` pasa de 9 a
  8 variantes.
- **Dependencia nueva:** `rank-bm25>=0.2.2` en `rag/requirements.txt` (Python
  puro + numpy; sin GPU).

### 4.6 Implicaciones operativas

- **No requiere reindexar.** BM25 opera sobre el texto de los chunks ya presentes
  en ChromaDB (`collection.get`); funciona con el `vector_db/` actual.
- **No descarga ningún modelo.** Recuperación dispersa en CPU pura.
- **Consecuencia en checkpoints congelados:** los manifiestos `pipeline_flags` de
  `baseline_all_on.json` / `all_off.json` aún contienen `USAR_BUSQUEDA_EXHAUSTIVA`.
  Como `get_pipeline_flags()` ya no la devuelve, `checkpoint_pipeline_flags_match`
  da `False` y una re-ejecución contra esos run dirs **no reutiliza** caché
  (re-infiere). Los JSON se conservan como registro histórico.

---

## 5. Re-inferencia con BM25 (variante `baseline_all_on`)

La evaluación publicada del TFG (Qwen3, Phi-4) se obtuvo con el pipeline **léxico
antiguo** (keyword + exhaustiva) y con `RRF_K = 20`. Para que **el sistema
descrito coincida con el evaluado**, hay que regenerar la inferencia con el
pipeline actual (BM25, `RRF_K = 60`).

Solo se re-ejecuta `baseline_all_on`: `all_off` desactiva todo (incluido BM25) y
es idéntico antes y después (suelo solo-semántica). Las etiquetas nuevas crean
carpetas aparte; **no tocan** los checkpoints antiguos.

Ejecutar desde la raíz del repo, en el entorno conda `(base)` (el que tiene
chromadb/ollama/rank-bm25). Con los defaults actuales **no hace falta exportar
`RAG_RRF_K`** (ya vale 60); exportarlo explícitamente solo si se quiere dejar
constancia en el log:

```bash
# (opcional, para trazabilidad explícita)
$env:RAG_RRF_K = "60"; $env:RAG_BM25_K1 = "1.5"; $env:RAG_BM25_B = "0.75"

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

Cada comando crea `research/evaluation/runs/ragas/comparisons/<label>/` con
`checkpoints/baseline_all_on.json`, `scores/`, `debug/` e `inference_summary.json`.

**RAGAS (paso siguiente)**, por cada carpeta:

```bash
python research/evaluation/evaluate.py --provider google \
  --source-root research/evaluation/runs/ragas/comparisons/bm25_es_all_on
```

Notas:

1. **Sin `--reindex` a propósito.** BM25 opera sobre los chunks ya indexados. Si
   alguna colección no existe localmente: para es/ca,
   `python research/evaluation/index.py --corpus es` (o `ca`) una vez; para
   RAGBench, añadir `--reindex` una sola vez (si reindexas el visual,
   `USAR_EMBEDDINGS_IMAGEN` debe estar activo, que es el default).
2. Detalle del protocolo completo de evaluación: `EVALUACIONES_PIPELINE.md`.

---

## 6. Riesgos argumentales y respuesta

| Crítica posible | Respuesta defendible |
|---|---|
| "¿Por qué `BM25_K1 = 1.5`?" | Default de `rank-bm25` para `BM25Okapi`; no se ajustó contra el test. Trotman 2014: no hay óptimo universal. |
| "¿Por qué `RRF_K = 60`?" | Es el valor canónico de Cormack et al. (2009). |
| "¿Por qué pesos 55/45?" | Hiperparámetro de fusión que prioriza levemente la rama semántica; fijo en todas las variantes. No es resultado teórico. |
| "El umbral 0.65 es arbitrario" | Tiene sonda específica (`probe_reranker_scores.py`) y decisión documentada (0.55 → 0.65). |
| "BM25 no usa las sub-consultas" | Correcto y deliberado: BM25 se aplica a la pregunta original; la descomposición LLM alimenta la rama semántica. |
| "¿Aplicáis late chunking (2409.04701)?" | No; se cita como trabajo relacionado. El problema de contexto se trata con Contextual Retrieval. |
| "El sistema evaluado no es el actual" | Cierto para las métricas antiguas; por eso se regenera `baseline_all_on` con BM25 y `RRF_K = 60` (§5). |

---

## 7. Recomendación para el texto del TFG

> La recuperación híbrida combina una rama densa (semántica) y una rama dispersa
> Okapi BM25. BM25 emplea los valores por defecto de la implementación `rank-bm25`
> (`k1 = 1.5`, `b = 0.75`), evitando ajuste sobre el conjunto de evaluación
> (Trotman et al., 2014, muestran que el óptimo de estos parámetros depende del
> corpus y que las diferencias son marginales). Las listas semántica y léxica se
> fusionan mediante Reciprocal Rank Fusion (Cormack et al., 2009) con el factor de
> amortiguamiento canónico `k = 60`, combinadas con pesos fijos `0.55/0.45`
> definidos como hiperparámetros del sistema y mantenidos constantes en todas las
> variantes. El reranking emplea un Cross-Encoder (Nogueira & Cho, 2019) con un
> umbral de corte `0.65` calibrado mediante sonda previa a la evaluación. Antes de
> la generación, el contexto se condensa con una etapa de tipo RECOMP (Xu et al.,
> 2023), que además devuelve contexto vacío cuando ningún fragmento es relevante.

---

## 8. Referencias

- **RAG-Fusion** — Rackauckas, Z. (2024). *RAG-Fusion: a New Take on
  Retrieval-Augmented Generation.* https://arxiv.org/abs/2402.03367
- **Passage Re-ranking with BERT** — Nogueira, R. & Cho, K. (2019).
  https://arxiv.org/abs/1901.04085
- **Late Chunking** — Günther, M. et al. (2024). *Late Chunking: Contextual Chunk
  Embeddings Using Long-Context Embedding Models.*
  https://arxiv.org/abs/2409.04701
- **RECOMP** — Xu, F., Shi, W. & Choi, E. (2023). *RECOMP: Improving
  Retrieval-Augmented LMs with Compression and Selective Augmentation.*
  https://arxiv.org/abs/2310.04408
- **Reciprocal Rank Fusion** — Cormack, G. V., Clarke, C. L. A. & Büttcher, S.
  (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank
  Learning Methods.* SIGIR 2009. https://doi.org/10.1145/1571941.1572114 ·
  PDF: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- **BM25 (marco probabilístico)** — Robertson, S. & Zaragoza, H. (2009). *The
  Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in
  IR, 3(4), 333–389. https://doi.org/10.1561/1500000019 ·
  PDF: https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf
- **rank-bm25** — Brown, D. *rank_bm25* (clase `BM25Okapi`, `k1=1.5`, `b=0.75`,
  `epsilon=0.25`). https://github.com/dorianbrown/rank_bm25 ·
  PyPI: https://pypi.org/project/rank-bm25/
- **Improvements to BM25** — Trotman, A., Puurula, A. & Burgess, B. (2014).
  *Improvements to BM25 and Language Models Examined.* ADCS 2014.
  https://doi.org/10.1145/2682862.2682863 ·
  PDF: https://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf
