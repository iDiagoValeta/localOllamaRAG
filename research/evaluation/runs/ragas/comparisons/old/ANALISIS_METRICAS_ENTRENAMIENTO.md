# Lexical and embedding metric analysis (Token F1 · ROUGE-L · BERTScore)

> Supporting document for the TFG report (MonkeyGrab — local RAG over PDFs).
> Collects **all results** produced by `research/evaluation/training_metrics.py`
> and an **exhaustive analysis**. The cross-check with RAGAS (LLM judge,
> AWS Bedrock) lives in [Section 12](#12-bertscore--ragas-cross-check-triangulation);
> the full RAGAS report lives in [`ANALISIS_RAGAS_AWS.md`](ANALISIS_RAGAS_AWS.md).

> **Pending re-evaluation under BM25 / `RRF_K = 60`.** The figures below
> correspond to the **pre-BM25** lexical pipeline (`$contains` keyword +
> exhaustive search) and `RRF_K = 20`. Production now uses Okapi BM25
> (`rank-bm25`, `k1 = 1.5`, `b = 0.75`) fused with RRF at the canonical
> `RRF_K = 60`. Reproduction commands and pipeline-divergence notes live in
> [`../../../../docs/EVALUACIONES_PIPELINE.md`](../../../../docs/EVALUACIONES_PIPELINE.md).
> This report will be regenerated with the new checkpoints when that pass
> completes.

- **Date**: 2026-05-18
- **Script**: `research/evaluation/training_metrics.py`
- **BERTScore scoring model**: `microsoft/deberta-xlarge-mnli`, `lang="en"`, `rescale_with_baseline=True`, `batch_size=32`
- **Token F1 / ROUGE-L normalization**: identical to `research/training/train_*.py` (lowercased, EN/ES/CA articles removed, punctuation stripped; ROUGE-L = LCS at token level)
- **Global summary**: `training_metrics_comparison_all.csv` (next to this document)
- **Per-sample CSV**: `<run>/training_metrics/<variant>.csv`

> **Project convention**: Δ (pp) = `all_on − all_off` (absolute difference in percentage points). Δ rel (%) = `(all_on − all_off) / all_off × 100` (relative change with respect to the `all_off` baseline).

---

## Contents

1. [Methodology and variant definition](#1-methodology-and-variant-definition)
2. [Global results](#2-global-results)
3. [Paired analysis: pipeline effect](#3-paired-analysis-pipeline-effect)
4. [Distributions and statistics per set](#4-distributions-and-statistics-per-set)
5. [Why BERTScore is low on RagBench (and why that is not a bad result)](#5-why-bertscore-is-low-on-ragbench-and-why-that-is-not-a-bad-result)
6. ["Correct answer penalized" phenomenon (full examples)](#6-correct-answer-penalized-phenomenon-full-examples)
7. [Vanilla RAG pathologies fixed by the pipeline](#7-vanilla-rag-pathologies-fixed-by-the-pipeline)
8. [Single-variant sets (eval 40p · visual)](#8-single-variant-sets-eval-40p--visual)
9. [Metric limitations and the role of RAGAS](#9-metric-limitations-and-the-role-of-ragas)
10. [Defense script](#10-defense-script)
11. [Reproducibility](#11-reproducibility)
12. [BERTScore ↔ RAGAS cross-check (triangulation)](#12-bertscore--ragas-cross-check-triangulation)

---

## 1. Methodology and variant definition

The two evaluated variants are the **two extremes of the pipeline** over the
same generator model, embeddings and RECOMP model (they are not different
models). This cleanly isolates the joint contribution of the advanced
retrieval and synthesis techniques.

| `pipeline_flags` | `baseline_all_on` | `all_off` |
|---|:--:|:--:|
| `USAR_BUSQUEDA_HIBRIDA` | enabled | disabled |
| `USAR_BUSQUEDA_EXHAUSTIVA` | enabled | disabled |
| `USAR_LLM_QUERY_DECOMPOSITION` | enabled | disabled |
| `USAR_RERANKER` | enabled | disabled |
| `EXPANDIR_CONTEXTO` | enabled | disabled |
| `USAR_OPTIMIZACION_CONTEXTO` | enabled | disabled |
| `USAR_RECOMP_SYNTHESIS` | enabled | disabled |

**Important defense nuance**: `all_off` is **not "no retrieval"** —
it keeps basic vector semantic search + generation. It is a *vanilla RAG*.
The delta therefore measures the added value of the advanced techniques on
top of an already functional baseline, not "RAG vs no-RAG".

**Evaluated sets** (8 checkpoints, 6 runs):

| Set | Dataset | Language | n | Variants |
|---|---|:--:|--:|---|
| `reinferencia_v3_es_50_final` | `datasets/local/dataset_eval_es.json` | ES | 50 | all_on, all_off |
| `reinferencia_v3_ca_50_final_ca` | `datasets/local/dataset_eval_ca.json` | CA | 50 | all_on, all_off |
| `reinferencia_v3_en_ragbench_dev_50_final` | `datasets/ragbench/dev_frozen/...dev10_frozen.json` | EN | 50 | all_on, all_off |
| `reinferencia_v2_en_ragbench_eval_40p` | `datasets/ragbench/en_eval/...40p_5q_eval.json` | EN | 200 | all_on |
| `reinferencia_v2_en_ragbench_visual_image_table_25p` | `datasets/ragbench/visual/...image_table_25p_5q.json` | EN | 125 | all_on |

---

## 2. Global results

Means in %. `P/R/F1` = BERTScore precision / recall / F1.
Δ (pp) = `all_on − all_off`; Δ rel (%) = `(all_on − all_off) / all_off × 100`.

### 2.1 Spanish — `dataset_eval_es.json` (50 questions)

| Variant | Token F1 | ROUGE-L | BERTScore P | BERTScore R | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| baseline_all_on | 62.11 | 55.14 | 62.48 | 51.40 | 56.21 |
| all_off | 59.84 | 52.66 | 56.70 | 51.90 | 53.60 |
| **Δ (pp)** | **+2.27** | **+2.48** | +5.78 | −0.50 | **+2.61** |
| **Δ rel (%)** | **+3.79** | **+4.71** | +10.19 | −0.96 | **+4.87** |

### 2.2 Catalan — `dataset_eval_ca.json` (50 questions)

| Variant | Token F1 | ROUGE-L | BERTScore P | BERTScore R | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| baseline_all_on | 59.65 | 53.92 | 64.36 | 48.18 | 55.44 |
| all_off | 56.49 | 47.87 | 55.01 | 44.45 | 49.12 |
| **Δ (pp)** | **+3.16** | **+6.05** | +9.35 | +3.73 | **+6.32** |
| **Δ rel (%)** | **+5.59** | **+12.64** | +17.00 | +8.39 | **+12.87** |

### 2.3 English — RagBench *dev* (50 questions)

| Variant | Token F1 | ROUGE-L | BERTScore P | BERTScore R | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| baseline_all_on | 25.96 | 23.34 | −4.53 | 51.02 | 13.71 |
| all_off | 22.50 | 19.62 | −10.61 | 46.73 | 8.19 |
| **Δ (pp)** | **+3.46** | **+3.72** | +6.08 | +4.29 | **+5.52** |
| **Δ rel (%)** | **+15.38** | **+18.96** | — | +9.18 | **+67.40** |

### 2.4 English — RagBench *eval* 40p (200 questions, single variant)

| Variant | Token F1 | ROUGE-L | BERTScore P | BERTScore R | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| baseline_all_on | 36.91 | 33.37 | 11.34 | 55.47 | 27.00 |

### 2.5 English — RagBench *visual* image+table 25p (125 questions, single variant)

| Variant | Token F1 | ROUGE-L | BERTScore P | BERTScore R | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| baseline_all_on | 24.32 | 19.86 | 1.17 | 32.44 | 14.14 |

---

## 3. Paired analysis: pipeline effect

Comparing means is not enough: the analysis is performed **question by
question** (same question across both variants) to rule out an outlier
artifact.

| Set | Δ BERTScore F1 (mean / median) | Improves / Tie / Degrades | Δ Token F1 (mean / median) | Δ ROUGE-L (mean / median) |
|---|---|---|---|---|
| ES local (50q) | **+2.61 / +1.64 pp** | 26 / 9 / 15 | +2.27 / +1.66 pp | +2.48 / +1.34 pp |
| CA local (50q) | **+6.32 / +3.83 pp** | 30 / 7 / 13 | +3.16 / +3.26 pp | +6.05 / +4.36 pp |
| RagBench dev (50q) | **+5.52 / +3.99 pp** | 31 / 5 / 14 | +3.46 / +3.65 pp | +3.72 / +3.26 pp |

**Defensible conclusions:**

1. **Directional consistency**: the pipeline improves all three metrics on
   all three languages/domains. The **median is also positive** → no
   long-tail bias.
2. **Roughly 2:1 in favor**: in 26–31 out of 50 questions the pipeline
   improves; in 13–15 it slightly degrades. Reported honestly: query
   decomposition and RECOMP add noise on questions that the vanilla RAG
   already solved well. The strong argument is the **positive and
   consistent net balance**, not "always improves".
3. **The benefit grows with difficulty**: Catalan (worse language
   resources) and RagBench (out-of-distribution scientific domain) benefit
   more than Spanish. Consistent with the hypothesis that the advanced
   techniques contribute more where basic retrieval fails more.

---

## 4. Distributions and statistics per set

Statistics over the per-sample CSV (mean · median · min · max in %).
Additionally: number of samples with BERTScore F1 < 0, mean answer and
ground-truth lengths (characters), their ratio, and the Pearson correlation
between answer length and BERTScore F1.

### 4.1 ES local

| Variant | Metric | mean | median | min | max |
|---|---|--:|--:|--:|--:|
| all_on | Token F1 | 62.11 | 62.69 | 27.61 | 90.91 |
| all_on | ROUGE-L | 55.14 | 54.36 | 20.15 | 90.91 |
| all_on | BERTScore F1 | 56.21 | 55.34 | 18.86 | 91.46 |
| all_off | Token F1 | 59.84 | 58.04 | 30.99 | 97.56 |
| all_off | ROUGE-L | 52.66 | 51.94 | 21.60 | 97.56 |
| all_off | BERTScore F1 | 53.60 | 53.41 | 22.42 | 94.53 |

- Samples with BERTScore F1 < 0: **0/50 (all_on)**, **0/50 (all_off)** → "healthy" scale.
- `status=ok`: 50/50 (all_on), 50/50 (all_off).
- Mean answer/GT length: all_on 226/232 (**ratio 1.0×**), all_off 264/232 (1.1×).
- Pearson r(answer length, BERTScore F1): all_on **−0.357**, all_off **−0.464**.

### 4.2 CA local

| Variant | Metric | mean | median | min | max |
|---|---|--:|--:|--:|--:|
| all_on | Token F1 | 59.65 | 60.00 | 23.26 | 95.65 |
| all_on | ROUGE-L | 53.92 | 54.41 | 13.95 | 95.65 |
| all_on | BERTScore F1 | 55.44 | 53.49 | 29.22 | 93.97 |
| all_off | Token F1 | 56.49 | 59.33 | 23.81 | 86.36 |
| all_off | ROUGE-L | 47.87 | 48.59 | 14.29 | 85.71 |
| all_off | BERTScore F1 | 49.12 | 51.03 | 20.41 | 92.02 |

- Samples with BERTScore F1 < 0: **0/50** in both → "healthy" scale.
- `status=ok`: 50/50 (all_on), **48/50 (all_off)** (2 vanilla failures).
- Mean answer/GT length: all_on 186/228 (**ratio 0.8×**), all_off 220/228 (1.0×).
- Pearson r(answer length, BERTScore F1): all_on **−0.237**, all_off **−0.333**.

### 4.3 RagBench dev

| Variant | Metric | mean | median | min | max |
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

- Samples with BERTScore F1 < 0: **19/50 (all_on)**, **21/50 (all_off)**.
- `status=ok`: 49/50 (all_on), **46/50 (all_off)** (4 vanilla truncations).
- Mean answer/GT length: all_on 385/110 (**ratio 3.5×**), all_off 533/110 (**ratio 4.9×**).
- Pearson r(answer length, BERTScore F1): all_on −0.043, all_off −0.206.

### 4.4 RagBench eval 40p (single variant)

- Token F1 36.91 (med 33.33) · ROUGE-L 33.37 (med 29.01) · BERTScore F1 27.00 (med 26.57).
- BERTScore F1 < 0: **41/200**. Answer/GT length 447/130 (**ratio 3.4×**).
- Pearson r(answer length, BERTScore F1): **−0.403**.

### 4.5 RagBench visual image+table 25p (single variant)

- Token F1 24.32 (med 22.54) · ROUGE-L 19.86 (med 17.19) · BERTScore F1 14.14 (med 13.22).
- BERTScore F1 < 0: **25/125**. Answer/GT length 629/183 (**ratio 3.4×**).
- Pearson r(answer length, BERTScore F1): **−0.445**.

---

## 5. Why BERTScore is low on RagBench (and why that is not a bad result)

Three causes, each with quantitative evidence.

### 5.1 Main cause — length asymmetry reference ↔ answer

RagBench is an **extractive/binary** benchmark: ground truths have 1–5
words (`"Yes."`, `"No, it remains unbroken."`), whereas the RAG **explains**
the answer.

| Set | Mean answer length | Mean GT length | Ratio |
|---|--:|--:|--:|
| ES local | 226 | 232 | **1.0×** |
| CA local | 186 | 228 | **0.8×** |
| RagBench dev (all_off) | 533 | 110 | **4.9×** |
| RagBench dev (all_on) | 385 | 110 | **3.5×** |
| RagBench eval 40p | 447 | 130 | **3.4×** |
| RagBench visual | 629 | 183 | **3.4×** |

BERTScore **precision** = fraction of *generated* tokens that match the
reference. With references of few words and answers of hundreds of
characters, almost no generated token finds a match → precision collapses.
Visible when decomposing P/R on RagBench dev:

- `all_off`: **P = −10.6**, R = 46.7, F1 = 8.2
- `all_on` : **P = −4.5**, R = 51.0, F1 = 13.7

**Recall is positive and reasonable** (the correct information is
present); it is the **precision** that sinks F1, and it does so **by
verbosity, not by error**. In fact, **the main mechanism by which the
pipeline raises BERTScore on RagBench is that RECOMP/synthesis shortens
the answer** (ratio 4.9× → 3.5×), lifting precision from −10.6 to −4.5.

Cross-cutting evidence: the Pearson correlation between answer length and
BERTScore F1 is **negative** in every set with length variation (ES
−0.36/−0.46, eval 40p −0.40, visual −0.45). Longer answer → worse
BERTScore → length artifact, not quality artifact.

**Full example — the pipeline improves *because it concentrates the answer* (RagBench dev, Q2):**

> **QUESTION**: Is detailed justification required when refining problems during the curation process?
>
> **GROUND TRUTH**: Yes, detailed justifications are recorded as comments when refining problems.
>
> **`all_off` ANSWER** — BERTScore F1 = 0.092:
> *"When a problem needed to be refined during the dataset curation process, the reason for the refinement was recorded in detail as a comment. For instance, in the case of IVQD, one annotator might leave a comment detailing "The reason the image relates to the question is..." or "If we change this image into ..., the irrelevance is guaranteed." If another annotator agreed with the comment, the problem was refined; if there was disagreement, all four annotators engaged in discussions to reach a consensus."*
>
> **`all_on` ANSWER** — BERTScore F1 = 0.488:
> *"Yes, if a problem needed to be refined, the reason was recorded in detail as a comment, and detailed justifications with collective decisions ensure curation consistency."*

Both answers are **factually correct**; the only difference is brevity.
The pipeline gains **+0.40 of BERTScore F1** simply by synthesizing. The
example illustrates simultaneously (a) the length artifact and (b) the
real value of the pipeline.

### 5.2 Secondary cause — `rescale_with_baseline=True`: negatives are normal

`microsoft/deberta-xlarge-mnli` is used with baseline rescaling (identical
to the training scripts, for **methodological consistency**). Raw
BERTScore rarely drops below ~0.80, so an empirical baseline is
subtracted: ~0.85 raw → ≈0 rescaled, and strong mismatches become
**negative**. Proportion of samples with BERTScore F1 < 0:

| Set | BERTScore F1 < 0 |
|---|---|
| ES / CA | **0 / 50** (both variants) |
| RagBench dev | 19/50 (all_on), 21/50 (all_off) |
| RagBench eval 40p | 41/200 |
| RagBench visual | 25/125 |

To document explicitly: **a negative BERTScore is not a computation error
or a "worse than random" answer in absolute terms; it is the mathematical
consequence of rescaling when the similarity falls below the model's
baseline**, amplified here by the length asymmetry.

### 5.3 What rules out the language as the cause

The scorer is English (`lang="en"`) **even for ES/CA**. Even so, ES and CA
score high (BERTScore F1 56 and 55, **0 negatives**). Therefore the low
performance on RagBench **is not explained by the language or the scoring
model**, but by the extractive nature of the benchmark: same scorer, same
RAG model → 56 on the own corpus vs 14 on RagBench. The difference is the
**reference format**, not the system quality. Strong argument for the
defense.

---

## 6. "Correct answer penalized" phenomenon (full examples)

Beyond length, there are cases where the answer is **correct** and
BERTScore is still negative. This supports the claim that lexical/embedding
metrics *underestimate* real quality and methodologically motivates the use
of RAGAS.

**Case A — correct answer penalized (RagBench dev, Q15):**
Token F1 = 0.014 · BERTScore P/R/F1 = −0.61 / 0.60 / **−0.34**

> **QUESTION**: Did some papers focus on cold dark matter models struggling with galaxy formation timing?
>
> **GROUND TRUTH**: Yes.
>
> **ANSWER**: *"Yes, some papers focused on cold dark matter (CDM) models struggling with galaxy formation timing. Specifically, by 1990, three major conflicts emerged for a Universe composed only of cold dark matter (CDM) and a small amount of ordinary matter. One of these conflicts was that the measured ages of the oldest stars, which ranged from 13 to 16 billion years, were older than the calculated expansion time since the Big Bang, which was about 10.5 billion years, resulting in an "impossibility." This age discrepancy was noted as being evident by 1993 and worsening by 1995. Furthermore, simulations of galaxy formation and the large-scale structure of the Universe struggled to reproduce the observed large-scale structure and were often able to achieve a good fit by introducing a cosmological constant. One specific challenge mentioned was simulating the gravitational collapse of the primordial mass clumps into the filamentary structure observed within only 10.5 billion years."*

The answer **begins literally with "Yes", is correct and well grounded**,
but gets BERTScore −0.34 and Token F1 ≈ 0 because the reference is only
`"Yes."`. No lexical/embedding metric can give credit here; an LLM judge
(RAGAS *answer correctness* / *faithfulness*) can.

**Case B — real failure correctly penalized (RagBench dev, Q25):**
BERTScore F1 = **−0.41**

> **QUESTION**: Does the $\mathbb{Z}_2$ flavor symmetry break in the strong coupling regime without vortices?
>
> **GROUND TRUTH**: No, it remains unbroken.
>
> **ANSWER**: *"The provided context contains conflicting information regarding whether the $\mathbb{Z}_2$ flavor symmetry breaks in the strong coupling regime without vortices. Specifically: [enumerates 8 fragments]... Therefore, based on the context, the symmetry breaking pattern differs between the strong and weak coupling regimes, with some sources indicating it is **unbroken** in the strong coupling limit (Fragments 2, 5), while others indicate it is **unbroken** in the strong coupling limit (Fragments 3, 4)."*

Here the model **does not resolve**: it rambles and even contradicts
itself ("unbroken… while others indicate it is unbroken"). Penalty
**justified**.

**Why include both cases**: having a correct-and-penalized (A) and a
failed-and-penalized (B) case lets us argue honestly that the metric
**does not distinguish real quality in extractive format** — neither for
good nor for bad — and that this is exactly the gap RAGAS fills. We do
not hide that the metric sometimes hits the mark (B); we delimit *when*
it is reliable and *when* it is not.

---

## 7. Vanilla RAG pathologies fixed by the pipeline

Inspecting states (`status` / `reason`) reveals two patterns of the
vanilla RAG that the pipeline fixes — an operational benefit, not just a
metric one.

### 7.1 Context-driven truncation

- RagBench dev: `all_off` has **4** questions with `status=failed / reason=respuesta_truncada` (Q27, Q31, Q32, Q43) vs **1** in `all_on`.
- CA local: `all_off` has **2** non-`ok` questions vs **0** in `all_on`.

Without `USAR_OPTIMIZACION_CONTEXTO`/RECOMP the prompt is longer and the
generation is cut off. The pipeline reduces truncation.

### 7.2 Evasive answers caused by poor retrieval

Without hybrid/exhaustive search/reranker, the vanilla model answers "the
context does not explain it" on questions the pipeline does solve. Here
the BERTScore delta **does reflect real quality** (deep improvement, not
just length) — a perfect counter-example to Section 5.

**Full example (ES, Q39) — paired improvement +0.376 BERTScore F1 (0.319 → 0.694):**

> **QUESTION**: ¿Por qué las condiciones idóneas para el cultivo de la chufa se hallan de manera especial en la huerta de Valencia?
>
> **GROUND TRUTH**: Las condiciones para el cultivo de la chufa se hallan de manera especial en la huerta de Valencia debido a su clima. Este entorno geográfico proporciona específicamente la tierra arenosa y las temperaturas suaves que requiere dicho tubérculo.
>
> **`all_off` ANSWER**: *"El contexto proporcionado no explica por qué las condiciones idóneas para el cultivo de la chufa se hallan de manera especial en la huerta de Valencia. Sin embargo, sí menciona la existencia de un enlace externo del 'Consejo Regulador de la D.O. Chufa de Valencia' que podría contener dicha información."*
>
> **`all_on` ANSWER**: *"Las condiciones idóneas para el cultivo de la chufa se encuentran de manera especial en la huerta de Valencia debido a su clima, ya que este proporciona tierra arenosa y temperaturas suaves."*

**Full example (ES, Q20) — paired improvement +0.305 BERTScore F1 (0.304 → 0.609):**

> **QUESTION**: ¿Cómo se llama la persona a la que el Cid situó al frente de la nueva sede episcopal de la catedral de Santa María en 1098?
>
> **GROUND TRUTH**: La persona a la que el Cid situó al frente de la nueva sede episcopal de la catedral de Santa María fue Jerónimo de Perigord. Ocupó este cargo eclesiástico en detrimento del antiguo metropolitano mozárabe tras el asedio de la ciudad de Valencia.
>
> **`all_off` ANSWER**: *"El contexto proporcionado menciona la datación de un diploma de Valencia como 'LXXXXº VIIIº post millesimum', es decir, 1098, y hace referencia a un autógrafo de Rodrigo Díaz en relación con este evento, pero no especifica el nombre de la persona a la que el Cid situó al frente de la nueva sede episcopal de la catedral de Santa María en 1098."*
>
> **`all_on` ANSWER**: *"La persona a la que el Cid situó al frente de la nueva sede episcopal de la catedral de Santa María en 1098 fue Jerónimo de Perigord."*

Using these together with Q2 (Section 5.1) reinforces the argument: the
pipeline improves **both via brevity (RECOMP) and via retrieval (hybrid +
exhaustive + reranker)**.

**Quality-ceiling example (ES, Q28) — BERTScore F1 = 0.915:**

> **QUESTION**: ¿Cuáles son las cuatro comarcas en las que se divide el conjunto de municipios que comprenden el territorio del parque natural de la Albufera?
>
> **GROUND TRUTH**: El conjunto de municipios que conforman el territorio del parque natural de la Albufera se divide en cuatro comarcas concretas. Estas son la Ribera Alta, la Ribera Baja, la Huerta Sur y la Ciudad de Valencia.
>
> **`all_on` ANSWER**: *"El conjunto de municipios que comprenden el territorio del parque natural de la Albufera se divide en cuatro comarcas: la Ribera Alta, la Ribera Baja, la Huerta Sur y la Ciudad de Valencia."*

When length and phrasing align (ratio ≈ 1×), BERTScore reaches 0.9 —
proof that the metric works well when the reference format is comparable.

---

## 8. Single-variant sets (eval 40p · visual)

`reinferencia_v2_en_ragbench_eval_40p` (200q) and
`reinferencia_v2_en_ragbench_visual_image_table_25p` (125q) **only have
`baseline_all_on`** → **no Δ to report**. They are presented as an
absolute characterization of the pipeline on larger-scale RagBench, not as
a comparison.

- **eval 40p** (200q, the largest): BERTScore F1 27.0, **P 11.3 positive**,
  only 41/200 negatives. With more volume and slightly longer references
  (130 chars), behavior is more stable; the most representative set to
  cite absolute pipeline performance on scientific English.
- **visual img+table** (125q): the hardest (BERTScore F1 14.1, Token F1
  24.3). Consistent: it requires reasoning over figures/tables via OCR;
  binary references + long answers (ratio 3.4×, r = −0.45). Defensible as
  **upper difficulty bound of the benchmark**, not as a system weakness.

---

## 9. Metric limitations and the role of RAGAS

- Token F1 and ROUGE-L share the **same length bias** as BERTScore: on
  extractive format they penalize correct but explanatory answers.
- Rescaled BERTScore produces negatives by design; they are not errors.
- The English scorer is not the limiting factor (ES/CA score high with
  it).
- **RAGAS (LLM judge) will be the main metric of semantic correctness**:
  it evaluates *answer correctness* / *faithfulness* without penalizing
  verbosity or depending on lexical overlap with a 3-word reference.
- **Final narrative = triangulation across three metric families**: the
  lexical ones (Token F1/ROUGE-L) and the embedding one (BERTScore) show
  the **direction and consistency** of the pipeline improvement; RAGAS
  confirms the **real magnitude** of quality, free of the length bias.
- **Cross-check performed** (2026-05-19, RAGAS AWS Bedrock): confirmed
  question-by-question that cases with negative BERTScore F1 and a
  correct answer obtain high *answer_correctness/faithfulness* in RAGAS.
  Detail in [Section 12](#12-bertscore--ragas-cross-check-triangulation).

---

## 10. Defense script

1. **Low values are not hidden; they are explained.** Low BERTScore on
   RagBench = (a) extractive references of 1–5 words vs explanatory
   answers (ratio 3–5×), (b) `rescale_with_baseline` that turns mismatches
   negative, (c) external benchmark out of distribution. **It is not** the
   language (ES/CA with the same scorer give 55–56 and 0 negatives).
2. **The hypothesis holds already with these metrics**: the complete
   pipeline improves Token F1, ROUGE-L and BERTScore on the three paired
   sets, with positive median and a question-by-question balance of about
   2:1.
3. **Mechanism identified**: decomposing P/R, the pipeline lifts
   BERTScore mostly by raising **precision** through shorter answers
   (RECOMP) and better retrieval (hybrid + reranker). Demonstrated with
   Q2 (brevity) and ES-Q39 / ES-Q20 (retrieval).
4. **Metric limits acknowledged**: Q15 (correct and penalized) justifies
   that lexical/embedding metrics underestimate quality on extractive
   format and motivates the use of RAGAS.
5. **Triangulation**: "three convergent metric families" — direction
   (lexical/embedding) + real magnitude (RAGAS).

---

## 11. Reproducibility

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

Generated artifacts:

- `<run>/training_metrics/<variant>.csv` — per-question metrics (includes `question`, `ground_truth`, `answer`, `status`, `reason`).
- `<run>/training_metrics/comparison_training_metrics.csv` — per-run aggregate.
- `ragas/comparisons/training_metrics_comparison_all.csv` — global summary.

---

## 12. BERTScore ↔ RAGAS cross-check (triangulation)

> Completed on 2026-05-19 with the definitive RAGAS evaluation (AWS Bedrock
> judge `eu.anthropic.claude-haiku-4-5-20251001-v1:0` +
> `amazon.titan-embed-text-v2:0`, `eu-north-1`, workers=8, batch=8). Full
> report: [`ANALISIS_RAGAS_AWS.md`](ANALISIS_RAGAS_AWS.md). Generator of
> the 8 checkpoints: `phi4-finetuned:latest`.

### 12.1 The three metric families converge

| Set | Δ Token F1 (pp) | Δ ROUGE-L (pp) | Δ BERTScore F1 (pp) | Δ RAGAS global mean (pp) |
|---|---:|---:|---:|---:|
| ES local | +2.27 | +2.48 | +2.61 | **+6.31** |
| CA local | +3.16 | +6.05 | +6.32 | **+8.34** |
| RagBench dev | +3.46 | +3.72 | +5.52 | **+5.15** |

All four metrics point in the **same direction** (the pipeline improves)
and follow the **same hierarchy** (Catalan benefits more than Spanish).
The lexical and embedding metrics give **direction and consistency**;
RAGAS contributes the **real magnitude** of quality, free of the length
bias, and is the main metric to cite.

### 12.2 RAGAS solves the BERTScore length artifact

Question-by-question verification on RagBench dev: in rows with **BERTScore
F1 < 0** (penalized by the answer-length vs extractive-reference
asymmetry), RAGAS assigns **mean faithfulness 0.906 (all_off) / 0.891
(all_on)** — those answers are in fact grounded — and the **mean
answer_correctness rises from 0.298 (all_off) to 0.410 (all_on)** (the
pipeline also improves the difficult cases). Examples where BERTScore
penalizes and RAGAS validates the correct answer:

| Case | Ground truth | BERTScore F1 | RAGAS answer_correctness | RAGAS faithfulness |
|---|---|---:|---:|---:|
| dev Q43 (all_on) eGFR<45 → impaired | `Yes.` | **−0.168** | **0.765** | **1.00** |
| dev Q12 (all_on) Geoff Marcy exoplanets | `Yes.` | **−0.120** | **0.768** | **1.00** |
| dev Q50 (all_on) KER/NEER analysis | `No.` | **−0.283** | **0.633** | **1.00** |

Control counter-example — real failure, penalized by **both**: dev Q25
(the model rambles and contradicts itself) → BERTScore F1 −0.413 and
RAGAS `answer_relevancy` 0.000, `answer_correctness` 0.499. RAGAS
**discriminates real quality**; BERTScore penalizes correct and failed
answers alike on extractive format. This is the strongest proof that the
low BERTScore on RagBench is a **measurement artifact**, not a system
deficiency.

### 12.3 Causal link retrieval → quality

The largest RAGAS effect is `context_precision` (+23 to +31 pp on the own
corpus, +12 pp on dev): hybrid + exhaustive search + reranker put the
correct fragment on top. Examples ES Q39 and ES Q20 (Section 7) go from
`context_precision = 0` (evasive answer "the context does not explain
it") to `context_precision = 1.0` and a correct answer. The retrieval
improvement is **causal** for the answer-quality improvement — the same
question pair sustained the retrieval argument in the lexical metrics
(Section 7).

### 12.4 Honest trade-off

`faithfulness` is the only metric with a slightly negative delta (ES
−0.17, CA −2.63, dev −1.33 pp) when adding more context + RECOMP
synthesis, but it stays ≥90 % (the system does not hallucinate). Reported
as the only trade-off of the pipeline; it does not compromise the global
conclusion (RAGAS global mean +5 to +8 pp).

---

RAGAS artifacts (definitive run 2026-05-19):
`research/evaluation/runs/ragas_aws_revaluation/comparisons/<run>/<variant>/{scores.csv,debug.json}`,
`.../<run>/aggregates/`, and `ragas_aws_revaluation/aws_ragas_summary.json`.
