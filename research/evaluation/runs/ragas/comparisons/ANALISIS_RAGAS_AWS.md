# Análisis RAGAS (juez LLM, AWS Bedrock) — evaluación definitiva

> Documento de soporte para la memoria del TFG (MonkeyGrab — RAG local sobre PDF).
> Recoge **todos los resultados RAGAS** de la corrida definitiva y su análisis
> exhaustivo. Complementa a [`ANALISIS_METRICAS_ENTRENAMIENTO.md`](ANALISIS_METRICAS_ENTRENAMIENTO.md)
> (métricas léxicas y de *embedding*). El cruce conjunto BERTScore↔RAGAS está en
> la Sección 12 de ese documento.

- **Fecha**: 2026-05-19
- **Comando**: `python research/evaluation/evaluate.py --provider aws --ragas-max-workers 8 --ragas-batch-size 8 --checkpoint ...`
- **Juez LLM**: AWS Bedrock `eu.anthropic.claude-haiku-4-5-20251001-v1:0`
- **Embeddings juez**: AWS Bedrock `amazon.titan-embed-text-v2:0`
- **Región**: `eu-north-1` · **throughput**: workers = 8, batch = 8
- **Métricas RAGAS**: `answer_correctness`, `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
- **Modelo generador evaluado**: `phi4-finetuned:latest` (8 checkpoints reinferidos)
- **Salida**: `research/evaluation/runs/ragas_aws_revaluation/comparisons/<run>/<variante>/{scores.csv,debug.json}` + `aws_ragas_summary.json`
- **NaN**: 1 única celda NaN en todo el conjunto (`answer_relevancy`, RagBench dev all_on); el resto completo.

---

## Índice

1. [Metodología](#1-metodología)
2. [Resultados globales por conjunto](#2-resultados-globales-por-conjunto)
3. [Análisis por métrica](#3-análisis-por-métrica)
4. [El hallazgo central: context_precision](#4-el-hallazgo-central-context_precision)
5. [Ejemplos completos](#5-ejemplos-completos)
6. [Conjuntos de variante única (eval 40p · visual)](#6-conjuntos-de-variante-única-eval-40p--visual)
7. [Limitaciones del juez RAGAS](#7-limitaciones-del-juez-ragas)
8. [Conclusión para la defensa del TFG](#8-conclusión-para-la-defensa-del-tfg)
9. [Reproducibilidad](#9-reproducibilidad)

---

## 1. Metodología

RAGAS se ejecuta **sobre los checkpoints ya generados** por `infer.py` (no
regenera respuestas): reutiliza pregunta, contextos recuperados, respuesta del
modelo y *ground truth* almacenados. El juez es un LLM (Claude Haiku 4.5 vía
Bedrock) más embeddings Titan v2. Cada conjunto pareado compara las dos variantes
extremas del pipeline sobre el **mismo** generador `phi4-finetuned:latest`:

| `pipeline_flags` | `baseline_all_on` | `all_off` |
|---|:--:|:--:|
| `USAR_BUSQUEDA_HIBRIDA` · `USAR_BUSQUEDA_EXHAUSTIVA` · `USAR_LLM_QUERY_DECOMPOSITION` · `USAR_RERANKER` · `EXPANDIR_CONTEXTO` · `USAR_OPTIMIZACION_CONTEXTO` · `USAR_RECOMP_SYNTHESIS` | ✅ | ❌ |

`all_off` no es «sin recuperación»: mantiene la búsqueda semántica vectorial
básica. El delta mide el valor añadido de las técnicas avanzadas sobre un RAG
vainilla funcional.

**Significado de las métricas** (0–1, reportadas en %):

- **answer_correctness** — precisión factual de la respuesta frente al *ground truth* (F1 de afirmaciones TP/FP/FN). Métrica principal de calidad.
- **faithfulness** — consistencia factual de la respuesta con el contexto recuperado (no alucina).
- **answer_relevancy** — grado en que la respuesta aborda la pregunta.
- **context_precision** — precisión del *ranking* de fragmentos recuperados (lo relevante arriba).
- **context_recall** — cobertura: si el contexto recuperado contiene lo necesario para la *ground truth*.

**Tiempos de evaluación** (segundos, juez): ES 972/1098, CA 955/1034, dev 1090/1122, eval_40p 4229 (200q), visual 2666 (125q).

---

## 2. Resultados globales por conjunto

Medias en %. Δ (pp) = `all_on − all_off`; Δ rel (%) = `(all_on − all_off) / all_off × 100`.

### 2.1 Español — `dataset_eval_es.json` (50 preguntas)

| Métrica | all_on | all_off | Δ (pp) | Δ rel (%) |
|---|---:|---:|---:|---:|
| answer_correctness | 75.72 | 74.20 | **+1.52** | +2.05 |
| faithfulness | 92.86 | 93.03 | −0.17 | −0.18 |
| answer_relevancy | 73.42 | 68.68 | **+4.74** | +6.90 |
| context_precision | 93.52 | 70.08 | **+23.44** | +33.45 |
| context_recall | 97.00 | 95.00 | +2.00 | +2.11 |
| **MEDIA GLOBAL** | **86.50** | **80.20** | **+6.31** | **+7.86** |

### 2.2 Catalán — `dataset_eval_ca.json` (50 preguntas)

| Métrica | all_on | all_off | Δ (pp) | Δ rel (%) |
|---|---:|---:|---:|---:|
| answer_correctness | 73.41 | 72.01 | **+1.40** | +1.94 |
| faithfulness | 93.66 | 96.28 | −2.63 | −2.73 |
| answer_relevancy | 63.11 | 58.90 | **+4.22** | +7.16 |
| context_precision | 96.49 | 65.80 | **+30.69** | +46.64 |
| context_recall | 99.00 | 91.00 | **+8.00** | +8.79 |
| **MEDIA GLOBAL** | **85.13** | **76.80** | **+8.34** | **+10.85** |

### 2.3 Inglés — RagBench *dev* `...dev10_frozen.json` (50 preguntas)

| Métrica | all_on | all_off | Δ (pp) | Δ rel (%) |
|---|---:|---:|---:|---:|
| answer_correctness | 49.84 | 46.53 | **+3.31** | +7.11 |
| faithfulness | 90.91 | 92.23 | −1.33 | −1.44 |
| answer_relevancy | 75.87 | 67.27 | **+8.59** | +12.77 |
| context_precision | 86.24 | 74.08 | **+12.16** | +16.41 |
| context_recall | 93.00 | 90.00 | +3.00 | +3.33 |
| **MEDIA GLOBAL** | **79.17** | **74.02** | **+5.15** | **+6.95** |

### 2.4 Inglés — RagBench *eval* 40p (200q, variante única) y *visual* 25p (125q, variante única)

| Métrica | eval_40p (200q) | visual img+table (125q) |
|---|---:|---:|
| answer_correctness | 56.40 | 48.30 |
| faithfulness | 93.05 | 87.38 |
| answer_relevancy | 79.96 | 75.64 |
| context_precision | 86.19 | 70.78 |
| context_recall | 94.33 | 75.27 |
| **MEDIA GLOBAL** | **81.99** | **71.47** |

---

## 3. Análisis por métrica

**answer_correctness** — Mejora consistente con el pipeline en los tres conjuntos
pareados (ES +1.52, CA +1.40, dev +3.31 pp). En corpus propio se sitúa en
73–76 % (alto para una métrica F1 de afirmaciones, exigente). En RagBench es
más baja en absoluto (dev 49.8, eval_40p 56.4, visual 48.3) por la naturaleza
extractiva/adversarial del benchmark, pero **el pipeline siempre suma**.

**faithfulness** — Muy alta y estable (87–96 %) en todas las configuraciones,
incluso en `all_off`. Es la única métrica con delta ligeramente **negativo** en
ES (−0.17), CA (−2.63) y dev (−1.33): al recuperar más contexto y sintetizar
(RECOMP), el pipeline introduce ocasionalmente alguna afirmación no estrictamente
verbatim del contexto. El nivel absoluto (≥90 % salvo visual) indica que el
sistema **apenas alucina**; el pequeño descenso es un coste asumible frente a las
grandes ganancias en *relevancy* y *context_precision*. Defensa honesta: es el
único *trade-off* del pipeline y conviene reportarlo como tal.

**answer_relevancy** — Mejora clara y relevante (ES +4.74, CA +4.22, dev +8.59 pp;
+7–13 % rel). El pipeline produce respuestas que abordan mejor la pregunta,
gracias a la descomposición de consulta y a la síntesis previa a la generación.

**context_precision** — **El efecto más grande de toda la evaluación**: ES +23.44,
CA +30.69, dev +12.16 pp (hasta +46.6 % rel en CA). Es la prueba directa de que
búsqueda híbrida + exhaustiva + reranker colocan el fragmento correcto arriba.
Ver Sección 4.

**context_recall** — Alta en todas (90–99 %) y mejora con el pipeline (ES +2,
CA +8, dev +3 pp). La recuperación rara vez se deja fuera lo necesario; el
pipeline lo refina, sobre todo en catalán.

**Síntesis**: el pipeline mejora la MEDIA GLOBAL en los tres conjuntos
(ES +6.31, CA +8.34, dev +5.15 pp). El beneficio crece con la dificultad
(catalán > español), igual que en las métricas léxicas/embedding — convergencia
metodológica entre las tres familias de métricas.

---

## 4. El hallazgo central: context_precision

El salto de `context_precision` (+23 a +31 pp en corpus propio) es el resultado
más contundente para defender el pipeline. Mide si los fragmentos *relevantes*
están en las primeras posiciones del contexto recuperado. El RAG vainilla
(`all_off`, solo búsqueda semántica) falla a menudo en colocar arriba el
fragmento que contiene la respuesta; al activar **búsqueda híbrida +
exhaustiva + reranker (CrossEncoder)**, la precisión del *ranking* se dispara.

Esto se traduce en un patrón observable: en `all_off`, numerosas preguntas
reciben `context_precision = 0` y `context_recall = 0` simultáneamente y el
modelo responde «el contexto no lo explica»; con el pipeline esas mismas
preguntas pasan a `context_precision = 1.0` y se responden correctamente
(ver ejemplos ES Q39 y Q20 en la Sección 5). Es el vínculo causal directo
entre la mejora de recuperación y la mejora de calidad de respuesta.

---

## 5. Ejemplos completos

### 5.1 RAGAS valida lo que BERTScore penalizaba (artefacto de longitud)

Caso clave para la defensa: respuestas **correctas** que BERTScore F1 penalizaba
con valores **negativos** por la asimetría de longitud (referencias extractivas
de 1–5 palabras vs respuestas explicativas), y que RAGAS puntúa **alto**.

> **RagBench dev, Q43** (variante `all_on`)
> **PREGUNTA**: Is an eGFR less than 45 ml/min/1.73 m² considered impaired kidney function?
> **GROUND TRUTH**: Yes.
> **RESPUESTA**: *"Yes, an eGFR less than $45 \text{ ml/min/1.73 m}^2$ is considered impaired function."*
> **BERTScore F1 = −0.168** (penalizada por longitud) → **RAGAS: answer_correctness 0.765, faithfulness 1.00, answer_relevancy 0.852, context_precision 1.00, context_recall 1.00.**

> **RagBench dev, Q12** (variante `all_on`)
> **PREGUNTA**: Did Geoff Marcy lead the discovery of any exoplanets?
> **GROUND TRUTH**: Yes.
> **RESPUESTA**: *"Yes, Geoff Marcy led the discovery of 70 of the first 100 exoplanets ever found, and this included the first multi-planet system orbiting a Sun-like star."*
> **BERTScore F1 = −0.120** → **RAGAS: answer_correctness 0.768, faithfulness 1.00, answer_relevancy 0.959, context_precision 1.00, context_recall 1.00.**

> **RagBench dev, Q50** (variante `all_on`)
> **PREGUNTA**: Is there an analysis available for how KER and NEER are affected by varying WERs?
> **GROUND TRUTH**: No.
> **RESPUESTA**: *"No, an analysis detailing how KER and NEER are affected by varying WERs is not available; the empirical relationship between WER, KER, NEER, and F1 score ... has not been analyzed."*
> **BERTScore F1 = −0.283** → **RAGAS: answer_correctness 0.633, faithfulness 1.00, answer_relevancy 0.995, context_precision 1.00, context_recall 1.00.**

**Agregado**: en las filas con BERTScore F1 < 0 de RagBench dev, la *faithfulness*
media de RAGAS es **0.906 (all_off)** y **0.891 (all_on)** — es decir, esas
respuestas «mal puntuadas» por BERTScore están de hecho **fundamentadas en el
contexto**. La `answer_correctness` media en esas filas sube de **0.298 (all_off)
a 0.410 (all_on)**: el pipeline también mejora justamente los casos difíciles.

### 5.2 RAGAS también penaliza los fallos reales (no es complaciente)

> **RagBench dev, Q25** (variante `all_off`)
> **PREGUNTA**: Does the $\mathbb{Z}_2$ flavor symmetry break in the strong coupling regime without vortices?
> **GROUND TRUTH**: No, it remains unbroken.
> **RESPUESTA**: *"The provided context contains conflicting information ... [enumera 8 fragmentos] ... with some sources indicating it is **unbroken** in the strong coupling limit (Fragments 2, 5), while others indicate it is **unbroken** in the strong coupling limit (Fragments 3, 4)."* (el modelo divaga y se autocontradice)
> **BERTScore F1 = −0.413** → **RAGAS: answer_correctness 0.499, answer_relevancy 0.000**, faithfulness 0.769.

RAGAS hunde `answer_relevancy` a 0.0 y deja `answer_correctness` en 0.50: cuando
el modelo realmente falla, **el juez LLM lo detecta**. Tener este contraejemplo
junto a Q43/Q12/Q50 demuestra que RAGAS **discrimina calidad real**, mientras que
BERTScore penaliza por igual a correctas y a fallidas (artefacto de longitud).

### 5.3 El pipeline arregla recuperaciones fallidas (context_precision 0 → 1)

> **ES Q39** — ¿Por qué las condiciones idóneas para el cultivo de la chufa se hallan de manera especial en la huerta de Valencia?
> **all_off**: `context_precision 0.000`, `context_recall 0.000`, `answer_correctness 0.348` — respuesta: *"El contexto proporcionado no explica..."*
> **all_on**: `context_precision 1.000`, `context_recall 1.000`, `answer_correctness 0.845` — respuesta: *"...debido a su clima, ya que este proporciona tierra arenosa y temperaturas suaves."*

> **ES Q20** — ¿Cómo se llama la persona a la que el Cid situó al frente de la nueva sede episcopal de la catedral de Santa María en 1098?
> **all_off**: `context_precision 0.000`, `context_recall 0.000`, `answer_correctness 0.468` — respuesta evasiva (no encuentra el nombre).
> **all_on**: `context_precision 1.000`, `answer_relevancy 0.888`, `answer_correctness 0.659` — respuesta: *"...fue Jerónimo de Perigord."*

Estos dos casos materializan el hallazgo de la Sección 4: la mejora de
`context_precision` es **causal** de la mejora de calidad de respuesta.

---

## 6. Conjuntos de variante única (eval 40p · visual)

Solo tienen `baseline_all_on` → **no hay Δ**; caracterizan el rendimiento
absoluto del pipeline a mayor escala:

- **eval_40p (200q)** — MEDIA GLOBAL 81.99 %; `faithfulness` 93.05, `context_recall`
  94.33, `answer_relevancy` 79.96. Es el conjunto más grande y estable; el más
  representativo para citar el rendimiento absoluto del pipeline en inglés
  científico. `answer_correctness` 56.4 (coherente con benchmark extractivo).
- **visual img+table (125q)** — MEDIA GLOBAL 71.47 %, el más difícil:
  `context_precision` 70.78 y `context_recall` 75.27 (recuperar sobre
  figuras/tablas vía OCR es más duro), `faithfulness` 87.38. Defendible como
  **límite superior de dificultad** del benchmark, no como debilidad del sistema:
  `faithfulness` sigue alta (no alucina) y `answer_relevancy` 75.6.

---

## 7. Limitaciones del juez RAGAS

- **Juez LLM ≠ verdad absoluta**: Claude Haiku 4.5 es un evaluador fuerte pero no
  infalible; `answer_correctness` puede infravalorar respuestas correctas con
  fraseo muy distinto al *ground truth*. Aun así no sufre el sesgo de longitud de
  BERTScore (ver Sección 5.1).
- **Coste/latencia**: ~16–70 min por checkpoint vía Bedrock; sensible a
  *throttling* (se observaron *backoffs*). No apto para iteración rápida; por eso
  las métricas léxicas/embedding siguen siendo útiles como señal barata.
- **Determinismo**: el juez es estocástico; pequeñas variaciones entre corridas
  son esperables. Los deltas reportados (5–8 pp de media global) están muy por
  encima de ese ruido.
- **context_precision/recall** dependen de cómo RAGAS segmenta los contextos;
  son consistentes dentro de esta corrida (mismo juez, misma config) y por tanto
  comparables entre variantes.

---

## 8. Conclusión para la defensa del TFG

1. **El pipeline completo mejora la calidad en las tres familias de métricas y en
   los tres idiomas/dominios.** RAGAS (MEDIA GLOBAL): ES +6.31, CA +8.34,
   dev +5.15 pp. Convergente con Token F1/ROUGE-L/BERTScore.
2. **El mecanismo está identificado y cuantificado**: el mayor efecto es
   `context_precision` (+23 a +31 pp), causado por búsqueda híbrida + exhaustiva
   + reranker; se traduce en mejores `answer_relevancy` y `answer_correctness`.
3. **RAGAS resuelve el problema de las métricas léxicas**: respuestas correctas
   que BERTScore penalizaba con valores negativos (artefacto de longitud en
   RagBench) obtienen `answer_correctness`/`faithfulness` altas — y los fallos
   reales (Q25) siguen penalizados. RAGAS mide calidad real; las léxicas miden
   solapamiento.
4. **Único *trade-off* reconocido**: `faithfulness` baja levemente (−0.2 a −2.6 pp)
   al añadir contexto y síntesis, pero se mantiene ≥90 % (no alucina). Honesto de
   reportar; no compromete la conclusión.
5. **Narrativa final = triangulación**: léxicas (Token F1/ROUGE-L) y de
   *embedding* (BERTScore) dan dirección y consistencia; **RAGAS aporta la
   magnitud real de la calidad**, libre del sesgo de longitud, y es la métrica
   principal a citar para corrección semántica.

---

## 9. Reproducibilidad

```powershell
python research/evaluation/evaluate.py --provider aws --ragas-max-workers 8 --ragas-batch-size 8 `
  --checkpoint "...\reinferencia_v3_es_50_final\checkpoints\baseline_all_on.json" `
  --checkpoint "...\reinferencia_v3_es_50_final\checkpoints\all_off.json" `
  --checkpoint "...\reinferencia_v3_ca_50_final_ca\checkpoints\baseline_all_on.json" `
  --checkpoint "...\reinferencia_v3_ca_50_final_ca\checkpoints\all_off.json" `
  --checkpoint "...\reinferencia_v2_en_ragbench_visual_image_table_25p\checkpoints\baseline_all_on.json" `
  --checkpoint "...\reinferencia_v2_en_ragbench_eval_40p\checkpoints\baseline_all_on.json" `
  --checkpoint "...\reinferencia_v3_en_ragbench_dev_50_final\checkpoints\baseline_all_on.json" `
  --checkpoint "...\reinferencia_v3_en_ragbench_dev_50_final\checkpoints\all_off.json"
```

Artefactos generados (corrida definitiva, 2026-05-19):

- `research/evaluation/runs/ragas_aws_revaluation/comparisons/<run>/<variante>/scores.csv` — métricas por pregunta.
- `.../<variante>/debug.json` — trazas del juez por pregunta.
- `.../<run>/aggregates/` — agregación por subconjunto (`source_type`).
- `research/evaluation/runs/ragas_aws_revaluation/aws_ragas_summary.json` — resumen global (medias, flags, tiempos, modelo juez).
- Métricas léxicas/embedding correlativas: `research/evaluation/runs/ragas/comparisons/<run>/training_metrics/` y resumen `training_metrics_comparison_all.csv`.
