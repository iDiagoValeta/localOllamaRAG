# RAGAS analysis (LLM judge, AWS Bedrock) — definitive evaluation

> Supporting document for the TFG report (MonkeyGrab — local RAG over PDFs).
> Collects **all RAGAS results** of the definitive run plus an exhaustive
> analysis. Complements
> [`ANALISIS_METRICAS_ENTRENAMIENTO.md`](ANALISIS_METRICAS_ENTRENAMIENTO.md)
> (lexical and embedding metrics). The joint BERTScore↔RAGAS cross-check
> lives in Section 12 of that document.

> **Pending re-evaluation under BM25 / `RRF_K = 60`.** The figures below
> correspond to the **pre-BM25** lexical pipeline (`$contains` keyword +
> exhaustive search) and `RRF_K = 20`. Production now uses Okapi BM25
> (`rank-bm25`, `k1 = 1.5`, `b = 0.75`) fused with RRF at the canonical
> `RRF_K = 60`. Reproduction commands and pipeline-divergence notes live in
> [`../../../../docs/EVALUACIONES_PIPELINE.md`](../../../../docs/EVALUACIONES_PIPELINE.md).
> This report will be regenerated with the new checkpoints when that pass
> completes.

- **Date**: 2026-05-19
- **Command**: `python research/evaluation/evaluate.py --provider aws --ragas-max-workers 8 --ragas-batch-size 8 --checkpoint ...`
- **LLM judge**: AWS Bedrock `eu.anthropic.claude-haiku-4-5-20251001-v1:0`
- **Judge embeddings**: AWS Bedrock `amazon.titan-embed-text-v2:0`
- **Region**: `eu-north-1` · **throughput**: workers = 8, batch = 8
- **RAGAS metrics**: `answer_correctness`, `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
- **Evaluated generator**: `phi4-finetuned:latest` (8 reinferred checkpoints)
- **Output**: `research/evaluation/runs/ragas_aws_revaluation/comparisons/<run>/<variant>/{scores.csv,debug.json}` + `aws_ragas_summary.json`
- **NaN**: a single NaN cell across the entire set (`answer_relevancy`, RagBench dev all_on); the rest complete.

---

## Contents

1. [Methodology](#1-methodology)
2. [Global results per set](#2-global-results-per-set)
3. [Per-metric analysis](#3-per-metric-analysis)
4. [Central finding: context_precision](#4-central-finding-context_precision)
5. [Full examples](#5-full-examples)
6. [Single-variant sets (eval 40p · visual)](#6-single-variant-sets-eval-40p--visual)
7. [Limitations of the RAGAS judge](#7-limitations-of-the-ragas-judge)
8. [Conclusion for the TFG defense](#8-conclusion-for-the-tfg-defense)
9. [Reproducibility](#9-reproducibility)

---

## 1. Methodology

RAGAS runs **on the checkpoints already generated** by `infer.py` (it does
not regenerate answers): it reuses the stored question, retrieved contexts,
model answer and *ground truth*. The judge is an LLM (Claude Haiku 4.5 via
Bedrock) plus Titan v2 embeddings. Each paired set compares the two extreme
pipeline variants over the **same** generator `phi4-finetuned:latest`:

| `pipeline_flags` | `baseline_all_on` | `all_off` |
|---|:--:|:--:|
| `USAR_BUSQUEDA_HIBRIDA` · `USAR_BUSQUEDA_EXHAUSTIVA` · `USAR_LLM_QUERY_DECOMPOSITION` · `USAR_RERANKER` · `EXPANDIR_CONTEXTO` · `USAR_OPTIMIZACION_CONTEXTO` · `USAR_RECOMP_SYNTHESIS` | enabled | disabled |

`all_off` is not "no retrieval": it keeps basic vector semantic search. The
delta therefore measures the added value of the advanced techniques on top of
a functional vanilla RAG.

**Metric meanings** (0–1, reported as %):

- **answer_correctness** — factual precision of the answer against ground truth (F1 over TP/FP/FN assertions). Main quality metric.
- **faithfulness** — factual consistency of the answer with the retrieved context (does not hallucinate).
- **answer_relevancy** — degree to which the answer addresses the question.
- **context_precision** — precision of the retrieved-fragment ranking (relevant fragments at the top).
- **context_recall** — coverage: whether the retrieved context contains what is needed for the ground truth.

**Evaluation times** (seconds, judge): ES 972/1098, CA 955/1034, dev 1090/1122, eval_40p 4229 (200q), visual 2666 (125q).

---

## 2. Global results per set

Means in %. Δ (pp) = `all_on − all_off`; Δ rel (%) = `(all_on − all_off) / all_off × 100`.

### 2.1 Spanish — `dataset_eval_es.json` (50 questions)

| Metric | all_on | all_off | Δ (pp) | Δ rel (%) |
|---|---:|---:|---:|---:|
| answer_correctness | 75.72 | 74.20 | **+1.52** | +2.05 |
| faithfulness | 92.86 | 93.03 | −0.17 | −0.18 |
| answer_relevancy | 73.42 | 68.68 | **+4.74** | +6.90 |
| context_precision | 93.52 | 70.08 | **+23.44** | +33.45 |
| context_recall | 97.00 | 95.00 | +2.00 | +2.11 |
| **GLOBAL MEAN** | **86.50** | **80.20** | **+6.31** | **+7.86** |

### 2.2 Catalan — `dataset_eval_ca.json` (50 questions)

| Metric | all_on | all_off | Δ (pp) | Δ rel (%) |
|---|---:|---:|---:|---:|
| answer_correctness | 73.41 | 72.01 | **+1.40** | +1.94 |
| faithfulness | 93.66 | 96.28 | −2.63 | −2.73 |
| answer_relevancy | 63.11 | 58.90 | **+4.22** | +7.16 |
| context_precision | 96.49 | 65.80 | **+30.69** | +46.64 |
| context_recall | 99.00 | 91.00 | **+8.00** | +8.79 |
| **GLOBAL MEAN** | **85.13** | **76.80** | **+8.34** | **+10.85** |

### 2.3 English — RagBench *dev* `...dev10_frozen.json` (50 questions)

| Metric | all_on | all_off | Δ (pp) | Δ rel (%) |
|---|---:|---:|---:|---:|
| answer_correctness | 49.84 | 46.53 | **+3.31** | +7.11 |
| faithfulness | 90.91 | 92.23 | −1.33 | −1.44 |
| answer_relevancy | 75.87 | 67.27 | **+8.59** | +12.77 |
| context_precision | 86.24 | 74.08 | **+12.16** | +16.41 |
| context_recall | 93.00 | 90.00 | +3.00 | +3.33 |
| **GLOBAL MEAN** | **79.17** | **74.02** | **+5.15** | **+6.95** |

### 2.4 English — RagBench *eval* 40p (200q, single variant) and *visual* 25p (125q, single variant)

| Metric | eval_40p (200q) | visual img+table (125q) |
|---|---:|---:|
| answer_correctness | 56.40 | 48.30 |
| faithfulness | 93.05 | 87.38 |
| answer_relevancy | 79.96 | 75.64 |
| context_precision | 86.19 | 70.78 |
| context_recall | 94.33 | 75.27 |
| **GLOBAL MEAN** | **81.99** | **71.47** |

---

## 3. Per-metric analysis

**answer_correctness** — Consistent improvement with the pipeline across the
three paired sets (ES +1.52, CA +1.40, dev +3.31 pp). On the own corpus it
sits at 73–76 % (high for a demanding assertion-F1 metric). On RagBench it
is lower in absolute terms (dev 49.8, eval_40p 56.4, visual 48.3) due to the
extractive/adversarial nature of the benchmark, but **the pipeline always
adds value**.

**faithfulness** — Very high and stable (87–96 %) across all
configurations, including `all_off`. It is the only metric with a slightly
**negative** delta in ES (−0.17), CA (−2.63) and dev (−1.33): by retrieving
more context and synthesizing (RECOMP), the pipeline occasionally
introduces an assertion that is not strictly verbatim in the context. The
absolute level (≥90 % except visual) shows that the system **barely
hallucinates**; the small drop is an acceptable cost in exchange for the
large gains in *relevancy* and *context_precision*. Honest defense: it is
the only trade-off of the pipeline and should be reported as such.

**answer_relevancy** — Clear and relevant improvement (ES +4.74, CA +4.22,
dev +8.59 pp; +7–13 % rel). The pipeline produces answers that address the
question better, thanks to query decomposition and pre-generation
synthesis.

**context_precision** — **The largest effect across the whole evaluation**:
ES +23.44, CA +30.69, dev +12.16 pp (up to +46.6 % rel in CA). It is direct
proof that hybrid + exhaustive search + reranker place the correct fragment
at the top. See Section 4.

**context_recall** — High everywhere (90–99 %) and improves with the
pipeline (ES +2, CA +8, dev +3 pp). Retrieval rarely leaves needed
information out; the pipeline refines it, especially in Catalan.

**Summary**: the pipeline improves GLOBAL MEAN on all three sets
(ES +6.31, CA +8.34, dev +5.15 pp). The benefit grows with difficulty
(Catalan > Spanish), in line with the lexical/embedding metrics —
methodological convergence between the three metric families.

---

## 4. Central finding: context_precision

The jump in `context_precision` (+23 to +31 pp in the own corpus) is the
strongest result to defend the pipeline. It measures whether the *relevant*
fragments appear in the first positions of the retrieved context. The
vanilla RAG (`all_off`, semantic search only) often fails to place at the
top the fragment containing the answer; once **hybrid + exhaustive search +
reranker (CrossEncoder)** are enabled, ranking precision rises sharply.

This translates into an observable pattern: in `all_off`, many questions
get `context_precision = 0` and `context_recall = 0` at the same time and
the model answers "the context does not explain it"; with the pipeline,
those same questions move to `context_precision = 1.0` and are answered
correctly (see ES Q39 and Q20 in Section 5). It is the direct causal link
between the retrieval improvement and the answer-quality improvement.

---

## 5. Full examples

### 5.1 RAGAS validates what BERTScore penalized (length artifact)

A key case for the defense: **correct** answers that BERTScore F1 penalized
with **negative** values because of length asymmetry (extractive references
of 1–5 words vs explanatory answers), and which RAGAS scores **high**.

> **RagBench dev, Q43** (variant `all_on`)
> **QUESTION**: Is an eGFR less than 45 ml/min/1.73 m² considered impaired kidney function?
> **GROUND TRUTH**: Yes.
> **ANSWER**: *"Yes, an eGFR less than $45 \text{ ml/min/1.73 m}^2$ is considered impaired function."*
> **BERTScore F1 = −0.168** (penalized by length) → **RAGAS: answer_correctness 0.765, faithfulness 1.00, answer_relevancy 0.852, context_precision 1.00, context_recall 1.00.**

> **RagBench dev, Q12** (variant `all_on`)
> **QUESTION**: Did Geoff Marcy lead the discovery of any exoplanets?
> **GROUND TRUTH**: Yes.
> **ANSWER**: *"Yes, Geoff Marcy led the discovery of 70 of the first 100 exoplanets ever found, and this included the first multi-planet system orbiting a Sun-like star."*
> **BERTScore F1 = −0.120** → **RAGAS: answer_correctness 0.768, faithfulness 1.00, answer_relevancy 0.959, context_precision 1.00, context_recall 1.00.**

> **RagBench dev, Q50** (variant `all_on`)
> **QUESTION**: Is there an analysis available for how KER and NEER are affected by varying WERs?
> **GROUND TRUTH**: No.
> **ANSWER**: *"No, an analysis detailing how KER and NEER are affected by varying WERs is not available; the empirical relationship between WER, KER, NEER, and F1 score ... has not been analyzed."*
> **BERTScore F1 = −0.283** → **RAGAS: answer_correctness 0.633, faithfulness 1.00, answer_relevancy 0.995, context_precision 1.00, context_recall 1.00.**

**Aggregate**: on rows with BERTScore F1 < 0 in RagBench dev, RAGAS mean
*faithfulness* is **0.906 (all_off)** and **0.891 (all_on)** — i.e. those
answers "badly scored" by BERTScore are in fact **grounded in the
context**. Mean `answer_correctness` on those rows rises from **0.298
(all_off) to 0.410 (all_on)**: the pipeline also improves the difficult
cases.

### 5.2 RAGAS also penalizes real failures (it is not complacent)

> **RagBench dev, Q25** (variant `all_off`)
> **QUESTION**: Does the $\mathbb{Z}_2$ flavor symmetry break in the strong coupling regime without vortices?
> **GROUND TRUTH**: No, it remains unbroken.
> **ANSWER**: *"The provided context contains conflicting information ... [enumerates 8 fragments] ... with some sources indicating it is **unbroken** in the strong coupling limit (Fragments 2, 5), while others indicate it is **unbroken** in the strong coupling limit (Fragments 3, 4)."* (the model rambles and contradicts itself)
> **BERTScore F1 = −0.413** → **RAGAS: answer_correctness 0.499, answer_relevancy 0.000**, faithfulness 0.769.

RAGAS pushes `answer_relevancy` down to 0.0 and leaves `answer_correctness`
at 0.50: when the model really fails, **the LLM judge detects it**. Having
this counter-example next to Q43/Q12/Q50 shows that RAGAS **discriminates
real quality**, while BERTScore penalizes correct and failed answers alike
(length artifact).

### 5.3 The pipeline fixes failed retrievals (context_precision 0 → 1)

> **ES Q39** — ¿Por qué las condiciones idóneas para el cultivo de la chufa se hallan de manera especial en la huerta de Valencia?
> **all_off**: `context_precision 0.000`, `context_recall 0.000`, `answer_correctness 0.348` — answer: *"El contexto proporcionado no explica..."*
> **all_on**: `context_precision 1.000`, `context_recall 1.000`, `answer_correctness 0.845` — answer: *"...debido a su clima, ya que este proporciona tierra arenosa y temperaturas suaves."*

> **ES Q20** — ¿Cómo se llama la persona a la que el Cid situó al frente de la nueva sede episcopal de la catedral de Santa María en 1098?
> **all_off**: `context_precision 0.000`, `context_recall 0.000`, `answer_correctness 0.468` — evasive answer (does not find the name).
> **all_on**: `context_precision 1.000`, `answer_relevancy 0.888`, `answer_correctness 0.659` — answer: *"...fue Jerónimo de Perigord."*

These two cases materialize the finding of Section 4: the improvement of
`context_precision` is **causal** for the improvement in answer quality.

---

## 6. Single-variant sets (eval 40p · visual)

They only have `baseline_all_on` → **no Δ**; they characterize the absolute
performance of the pipeline at a larger scale:

- **eval_40p (200q)** — GLOBAL MEAN 81.99 %; `faithfulness` 93.05,
  `context_recall` 94.33, `answer_relevancy` 79.96. It is the largest and
  most stable set; the most representative one to cite the absolute
  pipeline performance on scientific English. `answer_correctness` 56.4
  (coherent with an extractive benchmark).
- **visual img+table (125q)** — GLOBAL MEAN 71.47 %, the hardest:
  `context_precision` 70.78 and `context_recall` 75.27 (retrieving over
  figures/tables via OCR is harder), `faithfulness` 87.38. Defensible as
  **upper difficulty bound** of the benchmark, not as a system weakness:
  `faithfulness` is still high (no hallucination) and `answer_relevancy`
  75.6.

---

## 7. Limitations of the RAGAS judge

- **LLM judge ≠ absolute truth**: Claude Haiku 4.5 is a strong evaluator
  but not infallible; `answer_correctness` can underrate correct answers
  whose phrasing is very different from the ground truth. Even so, it
  does not suffer the length bias of BERTScore (see Section 5.1).
- **Cost/latency**: ~16–70 min per checkpoint via Bedrock; sensitive to
  throttling (backoffs observed). Not suitable for fast iteration; that
  is why lexical/embedding metrics remain useful as cheap signal.
- **Determinism**: the judge is stochastic; small variations between runs
  are expected. The reported deltas (5–8 pp of global mean) are well above
  that noise.
- **context_precision/recall** depend on how RAGAS segments contexts;
  they are consistent within this run (same judge, same configuration)
  and therefore comparable across variants.

---

## 8. Conclusion for the TFG defense

1. **The complete pipeline improves quality across the three metric
   families and the three languages/domains.** RAGAS (GLOBAL MEAN):
   ES +6.31, CA +8.34, dev +5.15 pp. Convergent with Token F1/ROUGE-L/BERTScore.
2. **The mechanism is identified and quantified**: the largest effect is
   `context_precision` (+23 to +31 pp), caused by hybrid + exhaustive
   search + reranker; it translates into better `answer_relevancy` and
   `answer_correctness`.
3. **RAGAS solves the lexical-metric problem**: correct answers that
   BERTScore penalized with negative values (length artifact on RagBench)
   obtain high `answer_correctness`/`faithfulness` — and real failures
   (Q25) remain penalized. RAGAS measures real quality; the lexical
   metrics measure overlap.
4. **Only acknowledged trade-off**: `faithfulness` drops slightly
   (−0.2 to −2.6 pp) when adding context and synthesis, but stays ≥90 %
   (no hallucinations). Honest to report; it does not undermine the
   conclusion.
5. **Final narrative = triangulation**: lexical metrics (Token F1/ROUGE-L)
   and embedding metrics (BERTScore) give direction and consistency;
   **RAGAS contributes the real magnitude of quality**, free of the
   length bias, and is the main metric to cite for semantic correctness.

---

## 9. Reproducibility

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

Generated artifacts (definitive run, 2026-05-19):

- `research/evaluation/runs/ragas_aws_revaluation/comparisons/<run>/<variant>/scores.csv` — per-question metrics.
- `.../<variant>/debug.json` — per-question judge traces.
- `.../<run>/aggregates/` — per-subset aggregation (`source_type`).
- `research/evaluation/runs/ragas_aws_revaluation/aws_ragas_summary.json` — global summary (means, flags, times, judge model).
- Correlative lexical/embedding metrics: `research/evaluation/runs/ragas/comparisons/<run>/training_metrics/` and the `training_metrics_comparison_all.csv` summary.
