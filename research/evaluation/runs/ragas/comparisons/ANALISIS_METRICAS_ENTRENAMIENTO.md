# Análisis de métricas léxicas y de *embedding* (Token F1 · ROUGE-L · BERTScore)

> Documento de soporte para la memoria del TFG (MonkeyGrab — RAG local sobre PDF).
> Recoge **todos los resultados** producidos por `research/evaluation/training_metrics.py`
> y su **análisis exhaustivo**. El cruce con RAGAS (juez LLM, AWS Bedrock) está en
> la [Sección 12](#12-cruce-bertscore--ragas-triangulación); el informe RAGAS
> completo en [`ANALISIS_RAGAS_AWS.md`](ANALISIS_RAGAS_AWS.md).

- **Fecha**: 2026-05-18
- **Script**: `research/evaluation/training_metrics.py`
- **Modelo de scoring BERTScore**: `microsoft/deberta-xlarge-mnli`, `lang="en"`, `rescale_with_baseline=True`, `batch_size=32`
- **Normalización Token F1 / ROUGE-L**: idéntica a `research/training/train_*.py` (minúsculas, eliminación de artículos EN/ES/CA, sin puntuación; ROUGE-L = LCS a nivel de token)
- **Resumen global**: `training_metrics_comparison_all.csv` (junto a este documento)
- **CSV por muestra**: `<run>/training_metrics/<variante>.csv`

---

## Índice

1. [Metodología y definición de variantes](#1-metodología-y-definición-de-variantes)
2. [Resultados globales](#2-resultados-globales)
3. [Análisis pareado: efecto del pipeline](#3-análisis-pareado-efecto-del-pipeline)
4. [Distribuciones y estadística por conjunto](#4-distribuciones-y-estadística-por-conjunto)
5. [Por qué BERTScore es bajo en RagBench (y por qué no es un mal resultado)](#5-por-qué-bertscore-es-bajo-en-ragbench-y-por-qué-no-es-un-mal-resultado)
6. [Fenómeno «respuesta correcta penalizada» (ejemplos completos)](#6-fenómeno-respuesta-correcta-penalizada-ejemplos-completos)
7. [Patologías del RAG vainilla corregidas por el pipeline](#7-patologías-del-rag-vainilla-corregidas-por-el-pipeline)
8. [Conjuntos de variante única (eval 40p · visual)](#8-conjuntos-de-variante-única-eval-40p--visual)
9. [Limitaciones de las métricas y rol de RAGAS](#9-limitaciones-de-las-métricas-y-rol-de-ragas)
10. [Guion de defensa](#10-guion-de-defensa)
11. [Reproducibilidad](#11-reproducibilidad)
12. [Cruce BERTScore ↔ RAGAS (triangulación)](#12-cruce-bertscore--ragas-triangulación)

---

## 1. Metodología y definición de variantes

Las dos variantes evaluadas son los **dos extremos del pipeline** sobre el mismo
modelo generador, embeddings y modelo RECOMP (no son modelos distintos). Se
aíslan así, de forma limpia, la contribución conjunta de las técnicas avanzadas
de recuperación y síntesis.

| `pipeline_flags` | `baseline_all_on` | `all_off` |
|---|:--:|:--:|
| `USAR_BUSQUEDA_HIBRIDA` | ✅ | ❌ |
| `USAR_BUSQUEDA_EXHAUSTIVA` | ✅ | ❌ |
| `USAR_LLM_QUERY_DECOMPOSITION` | ✅ | ❌ |
| `USAR_RERANKER` | ✅ | ❌ |
| `EXPANDIR_CONTEXTO` | ✅ | ❌ |
| `USAR_OPTIMIZACION_CONTEXTO` | ✅ | ❌ |
| `USAR_RECOMP_SYNTHESIS` | ✅ | ❌ |

**Matiz importante para la defensa**: `all_off` **no es «sin recuperación»** —
mantiene la búsqueda semántica vectorial básica + generación. Es un *RAG vainilla*.
Por tanto el delta mide el valor añadido de las técnicas avanzadas sobre una
línea base ya funcional, no «RAG vs no-RAG».

**Conjuntos evaluados** (8 checkpoints, 6 *runs*):

| Conjunto | Dataset | Idioma | n | Variantes |
|---|---|:--:|--:|---|
| `reinferencia_v3_es_50_final` | `datasets/local/dataset_eval_es.json` | ES | 50 | all_on, all_off |
| `reinferencia_v3_ca_50_final_ca` | `datasets/local/dataset_eval_ca.json` | CA | 50 | all_on, all_off |
| `reinferencia_v3_en_ragbench_dev_50_final` | `datasets/ragbench/dev_frozen/...dev10_frozen.json` | EN | 50 | all_on, all_off |
| `reinferencia_v2_en_ragbench_eval_40p` | `datasets/ragbench/en_eval/...40p_5q_eval.json` | EN | 200 | all_on |
| `reinferencia_v2_en_ragbench_visual_image_table_25p` | `datasets/ragbench/visual/...image_table_25p_5q.json` | EN | 125 | all_on |

---

## 2. Resultados globales

Medias en %. `P/R/F1` = precisión / *recall* / F1 de BERTScore.
Δ (pp) = `all_on − all_off`; Δ rel (%) = `(all_on − all_off) / all_off × 100`
(convención del proyecto: regla 11 de `CLAUDE.md`).

### 2.1 Español — `dataset_eval_es.json` (50 preguntas)

| Variante | Token F1 | ROUGE-L | BERTScore P | BERTScore R | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| baseline_all_on | 62.11 | 55.14 | 62.48 | 51.40 | 56.21 |
| all_off | 59.84 | 52.66 | 56.70 | 51.90 | 53.60 |
| **Δ (pp)** | **+2.27** | **+2.48** | +5.78 | −0.50 | **+2.61** |
| **Δ rel (%)** | **+3.79** | **+4.71** | +10.19 | −0.96 | **+4.87** |

### 2.2 Catalán — `dataset_eval_ca.json` (50 preguntas)

| Variante | Token F1 | ROUGE-L | BERTScore P | BERTScore R | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| baseline_all_on | 59.65 | 53.92 | 64.36 | 48.18 | 55.44 |
| all_off | 56.49 | 47.87 | 55.01 | 44.45 | 49.12 |
| **Δ (pp)** | **+3.16** | **+6.05** | +9.35 | +3.73 | **+6.32** |
| **Δ rel (%)** | **+5.59** | **+12.64** | +17.00 | +8.39 | **+12.87** |

### 2.3 Inglés — RagBench *dev* (50 preguntas)

| Variante | Token F1 | ROUGE-L | BERTScore P | BERTScore R | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| baseline_all_on | 25.96 | 23.34 | −4.53 | 51.02 | 13.71 |
| all_off | 22.50 | 19.62 | −10.61 | 46.73 | 8.19 |
| **Δ (pp)** | **+3.46** | **+3.72** | +6.08 | +4.29 | **+5.52** |
| **Δ rel (%)** | **+15.38** | **+18.96** | — | +9.18 | **+67.40** |

### 2.4 Inglés — RagBench *eval* 40p (200 preguntas, variante única)

| Variante | Token F1 | ROUGE-L | BERTScore P | BERTScore R | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| baseline_all_on | 36.91 | 33.37 | 11.34 | 55.47 | 27.00 |

### 2.5 Inglés — RagBench *visual* imagen+tabla 25p (125 preguntas, variante única)

| Variante | Token F1 | ROUGE-L | BERTScore P | BERTScore R | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| baseline_all_on | 24.32 | 19.86 | 1.17 | 32.44 | 14.14 |

---

## 3. Análisis pareado: efecto del pipeline

No basta con comparar medias: se analiza **pregunta a pregunta** (misma pregunta
en ambas variantes) para descartar que la mejora sea un artefacto de outliers.

| Conjunto | Δ BERTScore F1 (media / mediana) | Mejora / Empate / Empeora | Δ Token F1 (media / mediana) | Δ ROUGE-L (media / mediana) |
|---|---|---|---|---|
| ES local (50q) | **+2.61 / +1.64 pp** | 26 / 9 / 15 | +2.27 / +1.66 pp | +2.48 / +1.34 pp |
| CA local (50q) | **+6.32 / +3.83 pp** | 30 / 7 / 13 | +3.16 / +3.26 pp | +6.05 / +4.36 pp |
| RagBench dev (50q) | **+5.52 / +3.99 pp** | 31 / 5 / 14 | +3.46 / +3.65 pp | +3.72 / +3.26 pp |

**Conclusiones defendibles:**

1. **Consistencia direccional**: el pipeline mejora las tres métricas en los tres
   idiomas/dominios. La **mediana también es positiva** → no es un sesgo de cola.
2. **Balance ≈ 2:1 a favor**: en 26–31 de 50 preguntas el pipeline mejora; en
   13–15 empeora ligeramente. Se reporta con honestidad: la descomposición de
   consulta y RECOMP introducen ruido en preguntas que el RAG vainilla ya
   resolvía bien. El argumento sólido es el **balance neto positivo y
   consistente**, no «mejora siempre».
3. **El beneficio crece con la dificultad**: catalán (recursos lingüísticos
   peores) y RagBench (dominio científico externo, fuera de distribución) se
   benefician más que español. Coherente con la hipótesis de que las técnicas
   avanzadas aportan más donde la recuperación básica falla más.

---

## 4. Distribuciones y estadística por conjunto

Estadísticos sobre el CSV por muestra (media · mediana · mín · máx en %). Se
incluye además: nº de muestras con BERTScore F1 < 0, longitudes medias de
respuesta y *ground truth* (caracteres), su ratio, y la correlación de Pearson
entre longitud de respuesta y BERTScore F1.

### 4.1 ES local

| Variante | Métrica | media | mediana | mín | máx |
|---|---|--:|--:|--:|--:|
| all_on | Token F1 | 62.11 | 62.69 | 27.61 | 90.91 |
| all_on | ROUGE-L | 55.14 | 54.36 | 20.15 | 90.91 |
| all_on | BERTScore F1 | 56.21 | 55.34 | 18.86 | 91.46 |
| all_off | Token F1 | 59.84 | 58.04 | 30.99 | 97.56 |
| all_off | ROUGE-L | 52.66 | 51.94 | 21.60 | 97.56 |
| all_off | BERTScore F1 | 53.60 | 53.41 | 22.42 | 94.53 |

- Muestras con BERTScore F1 < 0: **0/50 (all_on)**, **0/50 (all_off)** → escala «sana».
- `status=ok`: 50/50 (all_on), 50/50 (all_off).
- Longitud media respuesta/GT: all_on 226/232 (**ratio 1.0×**), all_off 264/232 (1.1×).
- Pearson r(long. resp., BERTScore F1): all_on **−0.357**, all_off **−0.464**.

### 4.2 CA local

| Variante | Métrica | media | mediana | mín | máx |
|---|---|--:|--:|--:|--:|
| all_on | Token F1 | 59.65 | 60.00 | 23.26 | 95.65 |
| all_on | ROUGE-L | 53.92 | 54.41 | 13.95 | 95.65 |
| all_on | BERTScore F1 | 55.44 | 53.49 | 29.22 | 93.97 |
| all_off | Token F1 | 56.49 | 59.33 | 23.81 | 86.36 |
| all_off | ROUGE-L | 47.87 | 48.59 | 14.29 | 85.71 |
| all_off | BERTScore F1 | 49.12 | 51.03 | 20.41 | 92.02 |

- Muestras con BERTScore F1 < 0: **0/50** en ambas → escala «sana».
- `status=ok`: 50/50 (all_on), **48/50 (all_off)** (2 fallos del vainilla).
- Longitud media respuesta/GT: all_on 186/228 (**ratio 0.8×**), all_off 220/228 (1.0×).
- Pearson r(long. resp., BERTScore F1): all_on **−0.237**, all_off **−0.333**.

### 4.3 RagBench dev

| Variante | Métrica | media | mediana | mín | máx |
|---|---|--:|--:|--:|--:|
| all_on | Token F1 | 25.96 | 22.92 | 0.00 | 100.00 |
| all_on | ROUGE-L | 23.34 | 17.89 | 0.00 | 100.00 |
| all_on | BERTScore P | −4.53 | −2.61 | −58.77 | 100.00 |
| all_on | BERTScore R | 51.02 | 59.64 | 4.62 | 100.00 |
| all_on | BERTScore F1 | 13.71 | 12.28 | −32.57 | 100.00 |
| all_off | Token F1 | 22.50 | 20.21 | 0.00 | 74.19 |
| all_off | ROUGE-L | 19.62 | 13.07 | 0.00 | 70.97 |
| all_off | BERTScore P | −10.61 | −10.41 | −60.87 | 54.45 |
| all_off | BERTScore R | 46.73 | 43.52 | 2.06 | 92.05 |
| all_off | BERTScore F1 | 8.19 | 8.06 | −41.27 | 69.61 |

- Muestras con BERTScore F1 < 0: **19/50 (all_on)**, **21/50 (all_off)**.
- `status=ok`: 49/50 (all_on), **46/50 (all_off)** (4 truncamientos del vainilla).
- Longitud media respuesta/GT: all_on 385/110 (**ratio 3.5×**), all_off 533/110 (**ratio 4.9×**).
- Pearson r(long. resp., BERTScore F1): all_on −0.043, all_off −0.206.

### 4.4 RagBench eval 40p (variante única)

- Token F1 36.91 (med 33.33) · ROUGE-L 33.37 (med 29.01) · BERTScore F1 27.00 (med 26.57).
- BERTScore F1 < 0: **41/200**. Longitud resp./GT 447/130 (**ratio 3.4×**).
- Pearson r(long. resp., BERTScore F1): **−0.403**.

### 4.5 RagBench visual imagen+tabla 25p (variante única)

- Token F1 24.32 (med 22.54) · ROUGE-L 19.86 (med 17.19) · BERTScore F1 14.14 (med 13.22).
- BERTScore F1 < 0: **25/125**. Longitud resp./GT 629/183 (**ratio 3.4×**).
- Pearson r(long. resp., BERTScore F1): **−0.445**.

---

## 5. Por qué BERTScore es bajo en RagBench (y por qué no es un mal resultado)

Tres causas, cada una con evidencia cuantitativa.

### 5.1 Causa principal — asimetría de longitud referencia ↔ respuesta

RagBench es un benchmark **extractivo/binario**: las *ground truth* tienen 1–5
palabras (`"Yes."`, `"No, it remains unbroken."`), mientras el RAG **explica** la
respuesta.

| Conjunto | Long. media resp. | Long. media GT | Ratio |
|---|--:|--:|--:|
| ES local | 226 | 232 | **1.0×** |
| CA local | 186 | 228 | **0.8×** |
| RagBench dev (all_off) | 533 | 110 | **4.9×** |
| RagBench dev (all_on) | 385 | 110 | **3.5×** |
| RagBench eval 40p | 447 | 130 | **3.4×** |
| RagBench visual | 629 | 183 | **3.4×** |

BERTScore-**precisión** = fracción de tokens *generados* que casan con la
referencia. Con referencias de pocas palabras y respuestas de cientos de
caracteres, casi ningún token generado encuentra par → la precisión se hunde.
Se ve al descomponer P/R en RagBench dev:

- `all_off`: **P = −10.6**, R = 46.7, F1 = 8.2
- `all_on` : **P = −4.5**, R = 51.0, F1 = 13.7

La **recall es positiva y razonable** (la información correcta está presente); es
la **precisión** la que hunde el F1, y lo hace **por verbosidad, no por error**.
De hecho, **el principal mecanismo por el que el pipeline sube BERTScore en
RagBench es que RECOMP/síntesis acorta la respuesta** (ratio 4.9× → 3.5×),
elevando la precisión de −10.6 a −4.5.

Evidencia transversal: la correlación de Pearson entre longitud de respuesta y
BERTScore F1 es **negativa** en todos los conjuntos con variación de longitud
(ES −0.36/−0.46, eval 40p −0.40, visual −0.45). A mayor longitud, peor
BERTScore → artefacto de longitud, no de calidad.

**Ejemplo completo — el pipeline mejora *porque concentra la respuesta* (RagBench dev, Q2):**

> **PREGUNTA**: Is detailed justification required when refining problems during the curation process?
>
> **GROUND TRUTH**: Yes, detailed justifications are recorded as comments when refining problems.
>
> **RESPUESTA `all_off`** — BERTScore F1 = 0.092:
> *"When a problem needed to be refined during the dataset curation process, the reason for the refinement was recorded in detail as a comment. For instance, in the case of IVQD, one annotator might leave a comment detailing "The reason the image relates to the question is..." or "If we change this image into ..., the irrelevance is guaranteed." If another annotator agreed with the comment, the problem was refined; if there was disagreement, all four annotators engaged in discussions to reach a consensus."*
>
> **RESPUESTA `all_on`** — BERTScore F1 = 0.488:
> *"Yes, if a problem needed to be refined, the reason was recorded in detail as a comment, and detailed justifications with collective decisions ensure curation consistency."*

Ambas respuestas son **factualmente correctas**; la única diferencia es la
concisión. El pipeline gana **+0.40 de BERTScore F1** solo por sintetizar. El
ejemplo ilustra simultáneamente (a) el artefacto de longitud y (b) el valor real
del pipeline.

### 5.2 Causa secundaria — `rescale_with_baseline=True`: los negativos son normales

Se usa `microsoft/deberta-xlarge-mnli` con reescalado por baseline (idéntico a
los scripts de entrenamiento, por **consistencia metodológica**). BERTScore crudo
rara vez baja de ~0.80, por lo que se resta un baseline empírico: ~0.85 crudo →
≈0 reescalado, y los desajustes fuertes se vuelven **negativos**. Proporción de
muestras con BERTScore F1 < 0:

| Conjunto | BERTScore F1 < 0 |
|---|---|
| ES / CA | **0 / 50** (ambas variantes) |
| RagBench dev | 19/50 (all_on), 21/50 (all_off) |
| RagBench eval 40p | 41/200 |
| RagBench visual | 25/125 |

A documentar explícitamente: **un BERTScore negativo no es un error de cálculo ni
una respuesta «peor que aleatoria» en sentido absoluto; es la consecuencia
matemática del reescalado cuando la similitud cae por debajo del baseline del
modelo**, amplificada aquí por la asimetría de longitud.

### 5.3 Lo que descarta que el problema sea el idioma

El scorer es inglés (`lang="en"`) **también para ES/CA**. Aun así, ES y CA
puntúan alto (BERTScore F1 56 y 55, **0 negativos**). Por tanto el bajo
rendimiento en RagBench **no se explica por el idioma ni por el modelo de
scoring**, sino por la naturaleza extractiva del benchmark: mismo scorer, mismo
modelo RAG → 56 en corpus propio frente a 14 en RagBench. La diferencia es el
**formato de la referencia**, no la calidad del sistema. Argumento fuerte para la
defensa.

---

## 6. Fenómeno «respuesta correcta penalizada» (ejemplos completos)

Más allá de la longitud, hay casos donde la respuesta es **correcta** y aun así
BERTScore es negativo. Sostiene que las métricas léxicas/embedding *subestiman*
la calidad real y motiva metodológicamente el uso de RAGAS.

**Caso A — respuesta correcta penalizada (RagBench dev, Q15):**
Token F1 = 0.014 · BERTScore P/R/F1 = −0.61 / 0.60 / **−0.34**

> **PREGUNTA**: Did some papers focus on cold dark matter models struggling with galaxy formation timing?
>
> **GROUND TRUTH**: Yes.
>
> **RESPUESTA**: *"Yes, some papers focused on cold dark matter (CDM) models struggling with galaxy formation timing. Specifically, by 1990, three major conflicts emerged for a Universe composed only of cold dark matter (CDM) and a small amount of ordinary matter. One of these conflicts was that the measured ages of the oldest stars, which ranged from 13 to 16 billion years, were older than the calculated expansion time since the Big Bang, which was about 10.5 billion years, resulting in an "impossibility." This age discrepancy was noted as being evident by 1993 and worsening by 1995. Furthermore, simulations of galaxy formation and the large-scale structure of the Universe struggled to reproduce the observed large-scale structure and were often able to achieve a good fit by introducing a cosmological constant. One specific challenge mentioned was simulating the gravitational collapse of the primordial mass clumps into the filamentary structure observed within only 10.5 billion years."*

La respuesta **empieza literalmente con "Yes", es correcta y está bien
fundamentada**, pero recibe BERTScore −0.34 y Token F1 ≈ 0 porque la referencia
es solo `"Yes."`. Ninguna métrica léxica/embedding puede dar crédito aquí; un
juez LLM (RAGAS *answer correctness* / *faithfulness*) sí.

**Caso B — fallo real correctamente penalizado (RagBench dev, Q25):**
BERTScore F1 = **−0.41**

> **PREGUNTA**: Does the $\mathbb{Z}_2$ flavor symmetry break in the strong coupling regime without vortices?
>
> **GROUND TRUTH**: No, it remains unbroken.
>
> **RESPUESTA**: *"The provided context contains conflicting information regarding whether the $\mathbb{Z}_2$ flavor symmetry breaks in the strong coupling regime without vortices. Specifically: [enumera 8 fragmentos]... Therefore, based on the context, the symmetry breaking pattern differs between the strong and weak coupling regimes, with some sources indicating it is **unbroken** in the strong coupling limit (Fragments 2, 5), while others indicate it is **unbroken** in the strong coupling limit (Fragments 3, 4)."*

Aquí el modelo **no resuelve**: divaga e incluso se autocontradice
(«unbroken… while others indicate it is unbroken»). Penalización **justificada**.

**Por qué incluir ambos casos**: tener un correcto-penalizado (A) y un
fallo-penalizado (B) permite argumentar con honestidad que la métrica **no
distingue calidad real en formato extractivo** —ni para bien ni para mal— y que
ese es exactamente el hueco que cubre RAGAS. No se oculta que la métrica a veces
acierta (B): se delimita *cuándo* es fiable y *cuándo* no.

---

## 7. Patologías del RAG vainilla corregidas por el pipeline

Inspeccionando estados (`status` / `reason`) aparecen dos patrones del RAG
vainilla que el pipeline corrige —beneficio operativo, no solo de métrica.

### 7.1 Truncamiento por contexto

- RagBench dev: `all_off` tiene **4** preguntas `status=failed / reason=respuesta_truncada`
  (Q27, Q31, Q32, Q43) frente a **1** en `all_on`.
- CA local: `all_off` tiene **2** preguntas no-`ok` frente a **0** en `all_on`.

Sin `USAR_OPTIMIZACION_CONTEXTO`/RECOMP el prompt es más largo y la generación se
corta. El pipeline reduce el truncamiento.

### 7.2 Respuestas elusivas por mala recuperación

Sin búsqueda híbrida/exhaustiva/reranker, el vainilla responde «el contexto no lo
explica» en preguntas que el pipeline sí resuelve. Aquí el delta de BERTScore
**sí refleja calidad real** (mejora de fondo, no solo de longitud) — contraejemplo
perfecto a la sección 5.

**Ejemplo completo (ES, Q39) — mejora pareada +0.376 BERTScore F1 (0.319 → 0.694):**

> **PREGUNTA**: ¿Por qué las condiciones idóneas para el cultivo de la chufa se hallan de manera especial en la huerta de Valencia?
>
> **GROUND TRUTH**: Las condiciones para el cultivo de la chufa se hallan de manera especial en la huerta de Valencia debido a su clima. Este entorno geográfico proporciona específicamente la tierra arenosa y las temperaturas suaves que requiere dicho tubérculo.
>
> **RESPUESTA `all_off`**: *"El contexto proporcionado no explica por qué las condiciones idóneas para el cultivo de la chufa se hallan de manera especial en la huerta de Valencia. Sin embargo, sí menciona la existencia de un enlace externo del 'Consejo Regulador de la D.O. Chufa de Valencia' que podría contener dicha información."*
>
> **RESPUESTA `all_on`**: *"Las condiciones idóneas para el cultivo de la chufa se encuentran de manera especial en la huerta de Valencia debido a su clima, ya que este proporciona tierra arenosa y temperaturas suaves."*

**Ejemplo completo (ES, Q20) — mejora pareada +0.305 BERTScore F1 (0.304 → 0.609):**

> **PREGUNTA**: ¿Cómo se llama la persona a la que el Cid situó al frente de la nueva sede episcopal de la catedral de Santa María en 1098?
>
> **GROUND TRUTH**: La persona a la que el Cid situó al frente de la nueva sede episcopal de la catedral de Santa María fue Jerónimo de Perigord. Ocupó este cargo eclesiástico en detrimento del antiguo metropolitano mozárabe tras el asedio de la ciudad de Valencia.
>
> **RESPUESTA `all_off`**: *"El contexto proporcionado menciona la datación de un diploma de Valencia como 'LXXXXº VIIIº post millesimum', es decir, 1098, y hace referencia a un autógrafo de Rodrigo Díaz en relación con este evento, pero no especifica el nombre de la persona a la que el Cid situó al frente de la nueva sede episcopal de la catedral de Santa María en 1098."*
>
> **RESPUESTA `all_on`**: *"La persona a la que el Cid situó al frente de la nueva sede episcopal de la catedral de Santa María en 1098 fue Jerónimo de Perigord."*

Usar estos junto al Q2 (sección 5.1) blinda el argumento: el pipeline mejora
**tanto por concisión (RECOMP) como por recuperación (híbrida + exhaustiva +
reranker)**.

**Ejemplo de techo de calidad (ES, Q28) — BERTScore F1 = 0.915:**

> **PREGUNTA**: ¿Cuáles son las cuatro comarcas en las que se divide el conjunto de municipios que comprenden el territorio del parque natural de la Albufera?
>
> **GROUND TRUTH**: El conjunto de municipios que conforman el territorio del parque natural de la Albufera se divide en cuatro comarcas concretas. Estas son la Ribera Alta, la Ribera Baja, la Huerta Sur y la Ciudad de Valencia.
>
> **RESPUESTA `all_on`**: *"El conjunto de municipios que comprenden el territorio del parque natural de la Albufera se divide en cuatro comarcas: la Ribera Alta, la Ribera Baja, la Huerta Sur y la Ciudad de Valencia."*

Cuando longitud y fraseo se alinean (ratio ≈ 1×), BERTScore alcanza 0.9 — prueba
de que la métrica funciona bien cuando el formato de referencia es comparable.

---

## 8. Conjuntos de variante única (eval 40p · visual)

`reinferencia_v2_en_ragbench_eval_40p` (200q) y
`reinferencia_v2_en_ragbench_visual_image_table_25p` (125q) **solo tienen
`baseline_all_on`** → **no hay Δ que reportar**. Se presentan como
caracterización absoluta del pipeline en RagBench a mayor escala, no como
comparación.

- **eval 40p** (200q, el mayor): BERTScore F1 27.0, **P 11.3 positiva**, solo
  41/200 negativos. Al haber más volumen y referencias algo más largas (130
  car.), el comportamiento es más estable; es el conjunto más representativo
  para citar el rendimiento absoluto del pipeline en inglés científico.
- **visual img+table** (125q): el más duro (BERTScore F1 14.1, Token F1 24.3).
  Coherente: requiere razonar sobre figuras/tablas vía OCR; referencias binarias
  + respuestas largas (ratio 3.4×, r = −0.45). Defendible como **límite superior
  de dificultad del benchmark**, no como debilidad del sistema.

---

## 9. Limitaciones de las métricas y rol de RAGAS

- Token F1 y ROUGE-L comparten el **mismo sesgo de longitud** que BERTScore: en
  formato extractivo penalizan respuestas correctas pero explicativas.
- BERTScore reescalado produce negativos por diseño; no son errores.
- El scorer inglés no es el factor limitante (ES/CA puntúan alto con él).
- **RAGAS (juez LLM) será la métrica principal de corrección semántica**: evalúa
  *answer correctness* / *faithfulness* sin penalizar verbosidad ni depender de
  solapamiento léxico con una referencia de 3 palabras.
- **Narrativa final = triangulación de tres familias de métricas**: las léxicas
  (Token F1/ROUGE-L) y de *embedding* (BERTScore) demuestran la **dirección y
  consistencia** de la mejora del pipeline; RAGAS confirmará la **magnitud real**
  de la calidad, libre del sesgo de longitud.
- **Cruce realizado** (2026-05-19, RAGAS AWS Bedrock): confirmado pregunta a
  pregunta que los casos con BERTScore F1 negativo y respuesta correcta obtienen
  *answer_correctness/faithfulness* alta en RAGAS. Detalle en la
  [Sección 12](#12-cruce-bertscore--ragas-triangulación).

---

## 10. Guion de defensa

1. **No se ocultan los valores bajos; se explican.** BERTScore bajo en RagBench =
   (a) referencias extractivas de 1–5 palabras vs respuestas explicativas (ratio
   3–5×), (b) `rescale_with_baseline` que vuelve negativos los desajustes, (c)
   benchmark externo fuera de distribución. **No** es el idioma (ES/CA con el
   mismo scorer dan 55–56 y 0 negativos).
2. **La hipótesis se sostiene ya con estas métricas**: el pipeline completo
   mejora Token F1, ROUGE-L y BERTScore en los tres conjuntos pareados, con
   mediana positiva y balance pregunta-a-pregunta ≈ 2:1.
3. **Mecanismo identificado**: descomponiendo P/R, el pipeline sube BERTScore
   sobre todo elevando la **precisión** al acortar la respuesta (RECOMP) y al
   recuperar mejor (híbrida + reranker). Demostrado con Q2 (concisión) y ES-Q39 /
   ES-Q20 (recuperación).
4. **Límites de la métrica reconocidos**: Q15 (correcto y penalizado) justifica
   que las métricas léxicas/embedding subestiman la calidad en formato
   extractivo y motiva el uso de RAGAS.
5. **Triangulación**: «tres familias de métricas convergentes» — dirección
   (léxicas/embedding) + magnitud real (RAGAS).

---

## 11. Reproducibilidad

```powershell
python research/evaluation/training_metrics.py `
  --checkpoint-dir "...\reinferencia_v3_es_50_final\checkpoints\baseline_all_on.json" `
  --checkpoint-dir "...\reinferencia_v3_es_50_final\checkpoints\all_off.json" `
  --checkpoint-dir "...\reinferencia_v3_ca_50_final_ca\checkpoints\baseline_all_on.json" `
  --checkpoint-dir "...\reinferencia_v3_ca_50_final_ca\checkpoints\all_off.json" `
  --checkpoint-dir "...\reinferencia_v2_en_ragbench_visual_image_table_25p\checkpoints\baseline_all_on.json" `
  --checkpoint-dir "...\reinferencia_v2_en_ragbench_eval_40p\checkpoints\baseline_all_on.json" `
  --checkpoint-dir "...\reinferencia_v3_en_ragbench_dev_50_final\checkpoints\baseline_all_on.json" `
  --checkpoint-dir "...\reinferencia_v3_en_ragbench_dev_50_final\checkpoints\all_off.json" `
  --overwrite
```

Artefactos generados:

- `<run>/training_metrics/<variante>.csv` — métricas por pregunta (incluye
  `question`, `ground_truth`, `answer`, `status`, `reason`).
- `<run>/training_metrics/comparison_training_metrics.csv` — agregado por *run*.
- `ragas/comparisons/training_metrics_comparison_all.csv` — resumen global.

---

## 12. Cruce BERTScore ↔ RAGAS (triangulación)

> Completado el 2026-05-19 con la evaluación RAGAS definitiva (juez AWS Bedrock
> `eu.anthropic.claude-haiku-4-5-20251001-v1:0` + `amazon.titan-embed-text-v2:0`,
> `eu-north-1`, workers=8, batch=8). Informe completo:
> [`ANALISIS_RAGAS_AWS.md`](ANALISIS_RAGAS_AWS.md). Modelo generador de los 8
> checkpoints: `phi4-finetuned:latest`.

### 12.1 Las tres familias de métricas convergen

| Conjunto | Δ Token F1 (pp) | Δ ROUGE-L (pp) | Δ BERTScore F1 (pp) | Δ RAGAS media global (pp) |
|---|---:|---:|---:|---:|
| ES local | +2.27 | +2.48 | +2.61 | **+6.31** |
| CA local | +3.16 | +6.05 | +6.32 | **+8.34** |
| RagBench dev | +3.46 | +3.72 | +5.52 | **+5.15** |

Las cuatro métricas apuntan en la **misma dirección** (el pipeline mejora) y con
la **misma jerarquía** (catalán se beneficia más que español). Las léxicas y de
*embedding* dan la **dirección y consistencia**; RAGAS aporta la **magnitud
real** de la calidad, sin el sesgo de longitud, y es la métrica principal a citar.

### 12.2 RAGAS resuelve el artefacto de longitud de BERTScore

Verificación pregunta a pregunta en RagBench dev: en las filas con **BERTScore
F1 < 0** (penalizadas por la asimetría longitud-respuesta vs referencia
extractiva), RAGAS asigna **faithfulness media 0.906 (all_off) / 0.891 (all_on)**
— esas respuestas están de hecho fundamentadas — y la **answer_correctness media
sube de 0.298 (all_off) a 0.410 (all_on)** (el pipeline también mejora los casos
difíciles). Ejemplos donde BERTScore penaliza y RAGAS valida lo correcto:

| Caso | Ground truth | BERTScore F1 | RAGAS answer_correctness | RAGAS faithfulness |
|---|---|---:|---:|---:|
| dev Q43 (all_on) eGFR<45 → impaired | `Yes.` | **−0.168** | **0.765** | **1.00** |
| dev Q12 (all_on) Geoff Marcy exoplanets | `Yes.` | **−0.120** | **0.768** | **1.00** |
| dev Q50 (all_on) análisis KER/NEER | `No.` | **−0.283** | **0.633** | **1.00** |

Contraejemplo de control — fallo real, penalizado por **ambas**: dev Q25
(el modelo divaga y se autocontradice) → BERTScore F1 −0.413 y RAGAS
`answer_relevancy` 0.000, `answer_correctness` 0.499. RAGAS **discrimina calidad
real**; BERTScore penaliza por igual a correctas y a fallidas en formato
extractivo. Esta es la prueba más contundente de que el BERTScore bajo en
RagBench es un **artefacto de medida**, no una deficiencia del sistema.

### 12.3 Vínculo causal recuperación → calidad

El mayor efecto RAGAS es `context_precision` (+23 a +31 pp en corpus propio,
+12 pp en dev): búsqueda híbrida + exhaustiva + reranker colocan arriba el
fragmento correcto. Ejemplos ES Q39 y ES Q20 (Sección 7): pasan de
`context_precision = 0` (respuesta evasiva «el contexto no lo explica») a
`context_precision = 1.0` y respuesta correcta. La mejora de recuperación es
**causal** de la mejora de calidad de respuesta — el mismo par de preguntas
sostenía el argumento de recuperación en las métricas léxicas (Sección 7).

### 12.4 Trade-off honesto

`faithfulness` es la única métrica con delta levemente negativo (ES −0.17,
CA −2.63, dev −1.33 pp) al añadir más contexto + síntesis RECOMP, pero se
mantiene ≥90 % (el sistema no alucina). Se reporta como el único *trade-off* del
pipeline; no compromete la conclusión global (media global RAGAS +5 a +8 pp).

---

Artefactos RAGAS (corrida definitiva 2026-05-19):
`research/evaluation/runs/ragas_aws_revaluation/comparisons/<run>/<variante>/{scores.csv,debug.json}`,
`.../<run>/aggregates/`, y `ragas_aws_revaluation/aws_ragas_summary.json`.
