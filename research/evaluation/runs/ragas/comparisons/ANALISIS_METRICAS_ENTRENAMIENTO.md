# Lexical and embedding metric analysis (BM25 rerun)

> Collects the Token F1 (TF1), ROUGE-L and BERTScore results produced by `research/evaluation/training_metrics.py` for the final BM25 inference checkpoints.

- **Date**: 2026-05-25
- **Script**: `research/evaluation/training_metrics.py`
- **Generator model recorded in corrected checkpoints**: `phi4-finetuned:latest`
- **BERTScore scoring model**: `microsoft/deberta-xlarge-mnli`, `lang="en"`, `rescale_with_baseline=True`, `batch_size=32`
- **Token F1 / ROUGE-L normalization**: identical to `research/training/train_*.py` (lowercase, EN/ES/CA articles removed, punctuation stripped; ROUGE-L = token-level LCS)
- **Global summary**: `training_metrics_comparison_all.csv`
- **Per-sample CSV**: `<run>/training_metrics/<variant>.csv`
- **Previous pre-BM25 artifacts**: archived under `old/` together with their checkpoints, summaries and analysis reports.

> **RAGAS status**: completed for this BM25 rerun. The semantic judge report lives in [`ANALISIS_RAGAS_AWS.md`](ANALISIS_RAGAS_AWS.md); previous pre-BM25 RAGAS artifacts remain archived under `old/`.

> **Project convention**: Delta (pp) = `baseline_all_on - all_off` in percentage points. Delta rel (%) = `(baseline_all_on - all_off) / all_off * 100` when the `all_off` value is positive.

---

## Contents

1. [Methodology and variant definition](#1-methodology-and-variant-definition)
2. [Global results](#2-global-results)
3. [Paired analysis: pipeline effect](#3-paired-analysis-pipeline-effect)
4. [Distributions and statistics per set](#4-distributions-and-statistics-per-set)
5. [Interpretation for the defense](#5-interpretation-for-the-defense)
6. [Reproducibility](#6-reproducibility)

---

## 1. Methodology and variant definition

The two variants compare the complete inference-time pipeline against the semantic-only floor over the same datasets and stored answers.

| pipeline_flags | baseline_all_on | all_off |
| --- | ---: | ---: |
| USAR_BUSQUEDA_HIBRIDA | enabled | disabled |
| USAR_LLM_QUERY_DECOMPOSITION | enabled | disabled |
| USAR_RERANKER | enabled | disabled |
| EXPANDIR_CONTEXTO | enabled | disabled |
| USAR_OPTIMIZACION_CONTEXTO | enabled | disabled |
| USAR_RECOMP_SYNTHESIS | enabled | disabled |

**Important defense nuance**: `all_off` is not "no retrieval". It keeps the basic vector-semantic retrieval path plus generation, so the deltas measure the incremental value of the optional advanced stages.

**Evaluated sets** (10 checkpoints, 5 paired runs):

| Set | Dataset | Language | n | Variants |
| --- | ---: | ---: | ---: | ---: |
| `bm25rerun_es` | dataset_eval_es.json | ES | 50 | baseline_all_on, all_off |
| `bm25rerun_ca_ca` | dataset_eval_ca.json | CA | 50 | baseline_all_on, all_off |
| `bm25rerun_ragbench_dev` | dataset_ragbench_text_10p_5q_dev10_frozen.json | EN | 50 | baseline_all_on, all_off |
| `bm25rerun_ragbench_eval` | dataset_ragbench_en_eval_text_40p_5q_eval.json | EN | 200 | baseline_all_on, all_off |
| `bm25rerun_ragbench_visual` | dataset_ragbench_visual_image_table_25p_5q.json | EN | 125 | baseline_all_on, all_off |

---

## 2. Global results

Means are percentages. `P/R/F1` means BERTScore precision, recall and F1.

### 2.1 Spanish local - `dataset_eval_es.json` (50 questions)

| Variant | Token F1 | ROUGE-L | BERTScore P | BERTScore R | BERTScore F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_all_on | 62.09 | 55.18 | 61.81 | 51.47 | 55.89 |
| all_off | 59.46 | 52.63 | 56.71 | 50.72 | 52.97 |
| **Delta (pp)** | **+2.63** | **+2.55** | **+5.10** | **+0.75** | **+2.92** |
| **Delta rel (%)** | **+4.42** | **+4.85** | **+8.99** | **+1.48** | **+5.51** |

### 2.2 Catalan local - `dataset_eval_ca.json` (50 questions)

| Variant | Token F1 | ROUGE-L | BERTScore P | BERTScore R | BERTScore F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_all_on | 58.73 | 53.20 | 64.14 | 47.20 | 54.77 |
| all_off | 56.22 | 46.99 | 54.14 | 44.77 | 48.82 |
| **Delta (pp)** | **+2.51** | **+6.21** | **+10.00** | **+2.43** | **+5.95** |
| **Delta rel (%)** | **+4.46** | **+13.22** | **+18.47** | **+5.43** | **+12.19** |

### 2.3 RagBench dev frozen - `dataset_ragbench_text_10p_5q_dev10_frozen.json` (50 questions)

| Variant | Token F1 | ROUGE-L | BERTScore P | BERTScore R | BERTScore F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_all_on | 27.24 | 24.54 | -2.56 | 53.59 | 15.83 |
| all_off | 22.62 | 20.07 | -11.99 | 47.01 | 6.88 |
| **Delta (pp)** | **+4.62** | **+4.47** | **+9.43** | **+6.58** | **+8.95** |
| **Delta rel (%)** | **+20.42** | **+22.27** | **N/A** | **+14.00** | **+130.09** |

### 2.4 RagBench eval 40p - `dataset_ragbench_en_eval_text_40p_5q_eval.json` (200 questions)

| Variant | Token F1 | ROUGE-L | BERTScore P | BERTScore R | BERTScore F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_all_on | 37.25 | 33.85 | 11.74 | 56.37 | 27.69 |
| all_off | 27.34 | 24.29 | -3.06 | 45.82 | 14.54 |
| **Delta (pp)** | **+9.91** | **+9.56** | **+14.80** | **+10.55** | **+13.15** |
| **Delta rel (%)** | **+36.25** | **+39.36** | **N/A** | **+23.02** | **+90.44** |

### 2.5 RagBench visual image+table 25p - `dataset_ragbench_visual_image_table_25p_5q.json` (125 questions)

| Variant | Token F1 | ROUGE-L | BERTScore P | BERTScore R | BERTScore F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_all_on | 23.35 | 18.95 | 1.05 | 31.51 | 13.67 |
| all_off | 20.58 | 15.96 | -4.53 | 29.29 | 9.64 |
| **Delta (pp)** | **+2.77** | **+2.99** | **+5.58** | **+2.22** | **+4.03** |
| **Delta rel (%)** | **+13.46** | **+18.73** | **N/A** | **+7.58** | **+41.80** |

---

## 3. Paired analysis: pipeline effect

The paired analysis compares the same question across `baseline_all_on` and `all_off`, which avoids reading the aggregate delta as an outlier artifact.

| Set | Delta BERTScore F1 mean / median | Improves / Tie / Degrades | Delta Token F1 mean / median | Delta ROUGE-L mean / median |
| --- | ---: | ---: | ---: | ---: |
| Spanish local | +2.92 / +0.93 pp | 28 / 4 / 18 | +2.63 / +1.58 pp | +2.55 / +1.91 pp |
| Catalan local | +5.95 / +2.55 pp | 31 / 1 / 18 | +2.51 / +1.72 pp | +6.21 / +2.90 pp |
| RagBench dev frozen | +8.94 / +7.38 pp | 38 / 1 / 11 | +4.62 / +4.12 pp | +4.47 / +3.50 pp |
| RagBench eval 40p | +13.15 / +10.88 pp | 155 / 0 / 45 | +9.91 / +5.98 pp | +9.56 / +5.47 pp |
| RagBench visual image+table 25p | +4.03 / +3.07 pp | 79 / 1 / 45 | +2.77 / +1.43 pp | +2.98 / +1.31 pp |

**Defensible conclusions:**

1. The complete pipeline improves Token F1, ROUGE-L and BERTScore F1 on all five paired sets.
2. The largest absolute gains are on RagBench eval (+13.15 pp BERTScore F1) and RagBench dev (+8.95 pp), where the advanced retrieval/synthesis path has more room to help.
3. The visual set remains the hardest absolute benchmark, but the full pipeline still improves BERTScore F1 by +4.03 pp and improves most paired questions.
4. The RAGAS pass confirms that the improvement is semantically meaningful and mostly driven by better retrieval quality, especially `context_precision`.

---

## 4. Distributions and statistics per set

Statistics over the per-sample CSVs. Means, medians, min and max are percentages.

### 4.1 Spanish local

| Variant | Metric | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_all_on | Token F1 | 62.09 | 62.69 | 23.11 | 90.91 |
| baseline_all_on | ROUGE-L | 55.18 | 54.85 | 18.67 | 90.91 |
| baseline_all_on | BERTScore F1 | 55.89 | 54.03 | 15.21 | 91.46 |
| all_off | Token F1 | 59.46 | 56.68 | 31.33 | 97.56 |
| all_off | ROUGE-L | 52.63 | 52.23 | 26.67 | 97.56 |
| all_off | BERTScore F1 | 52.97 | 51.64 | 15.88 | 94.53 |

- `baseline_all_on`: BERTScore F1 < 0 = **0/50**; `status=ok` = **50/50**; mean answer/GT length = 234/232 (1.0x); Pearson r(answer length, BERTScore F1) = -0.414.
- `all_off`: BERTScore F1 < 0 = **0/50**; `status=ok` = **50/50**; mean answer/GT length = 256/232 (1.1x); Pearson r(answer length, BERTScore F1) = -0.444.

### 4.2 Catalan local

| Variant | Metric | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_all_on | Token F1 | 58.73 | 56.80 | 21.82 | 95.65 |
| baseline_all_on | ROUGE-L | 53.20 | 51.97 | 13.95 | 95.65 |
| baseline_all_on | BERTScore F1 | 54.77 | 53.08 | 31.11 | 93.97 |
| all_off | Token F1 | 56.22 | 56.21 | 26.80 | 85.71 |
| all_off | ROUGE-L | 46.99 | 45.98 | 14.29 | 85.71 |
| all_off | BERTScore F1 | 48.82 | 50.96 | 17.63 | 92.02 |

- `baseline_all_on`: BERTScore F1 < 0 = **0/50**; `status=ok` = **50/50**; mean answer/GT length = 177/228 (0.8x); Pearson r(answer length, BERTScore F1) = -0.087.
- `all_off`: BERTScore F1 < 0 = **0/50**; `status=ok` = **48/50**; mean answer/GT length = 218/228 (1.0x); Pearson r(answer length, BERTScore F1) = -0.255.

### 4.3 RagBench dev frozen

| Variant | Metric | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_all_on | Token F1 | 27.24 | 23.91 | 0.00 | 77.97 |
| baseline_all_on | ROUGE-L | 24.54 | 18.88 | 0.00 | 77.97 |
| baseline_all_on | BERTScore F1 | 15.83 | 15.65 | -31.75 | 75.46 |
| all_off | Token F1 | 22.62 | 18.32 | 0.00 | 100.00 |
| all_off | ROUGE-L | 20.07 | 13.02 | 0.00 | 100.00 |
| all_off | BERTScore F1 | 6.88 | 6.80 | -34.53 | 100.00 |

- `baseline_all_on`: BERTScore F1 < 0 = **18/50**; `status=ok` = **49/50**; mean answer/GT length = 372/110 (3.4x); Pearson r(answer length, BERTScore F1) = -0.084.
- `all_off`: BERTScore F1 < 0 = **23/50**; `status=ok` = **48/50**; mean answer/GT length = 558/110 (5.1x); Pearson r(answer length, BERTScore F1) = -0.227.

### 4.4 RagBench eval 40p

| Variant | Metric | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_all_on | Token F1 | 37.25 | 33.03 | 0.00 | 100.00 |
| baseline_all_on | ROUGE-L | 33.85 | 28.11 | 0.00 | 100.00 |
| baseline_all_on | BERTScore F1 | 27.69 | 28.45 | -35.92 | 92.91 |
| all_off | Token F1 | 27.34 | 25.95 | 0.00 | 83.64 |
| all_off | ROUGE-L | 24.29 | 20.94 | 0.00 | 83.64 |
| all_off | BERTScore F1 | 14.54 | 16.63 | -40.44 | 87.64 |

- `baseline_all_on`: BERTScore F1 < 0 = **39/200**; `status=ok` = **198/200**; mean answer/GT length = 429/130 (3.3x); Pearson r(answer length, BERTScore F1) = -0.353.
- `all_off`: BERTScore F1 < 0 = **51/200**; `status=ok` = **187/200**; mean answer/GT length = 623/130 (4.8x); Pearson r(answer length, BERTScore F1) = -0.268.

### 4.5 RagBench visual image+table 25p

| Variant | Metric | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_all_on | Token F1 | 23.35 | 21.28 | 0.00 | 87.50 |
| baseline_all_on | ROUGE-L | 18.94 | 16.19 | 0.00 | 87.50 |
| baseline_all_on | BERTScore F1 | 13.67 | 13.77 | -50.20 | 80.03 |
| all_off | Token F1 | 20.58 | 19.44 | 0.00 | 60.00 |
| all_off | ROUGE-L | 15.96 | 14.29 | 0.00 | 51.85 |
| all_off | BERTScore F1 | 9.64 | 9.72 | -46.89 | 57.51 |

- `baseline_all_on`: BERTScore F1 < 0 = **22/125**; `status=ok` = **125/125**; mean answer/GT length = 648/183 (3.5x); Pearson r(answer length, BERTScore F1) = -0.394.
- `all_off`: BERTScore F1 < 0 = **34/125**; `status=ok` = **120/125**; mean answer/GT length = 848/183 (4.6x); Pearson r(answer length, BERTScore F1) = -0.374.

---

## 5. Interpretation for the defense

1. **The BM25 rerun preserves the main conclusion**: the complete pipeline beats the semantic-only floor consistently across local ES, local CA, RagBench dev, RagBench eval and RagBench visual.
2. **The gains are not limited to one metric family**: Token F1, ROUGE-L and BERTScore F1 move in the same direction for every set.
3. **RagBench still needs careful explanation**: short extractive references versus explanatory RAG answers depress precision-based metrics, especially BERTScore precision. Negative rescaled BERTScore values are expected and should not be described as computation errors.
4. **RAGAS closes the loop**: the AWS judge confirms the same direction semantically, with global RAGAS gains on ES, CA and RagBench dev and strong `baseline_all_on` validation on RagBench eval/visual.

---

## 6. Reproducibility

```powershell
python research/evaluation/training_metrics.py `
  --checkpoint-dir "research/evaluation/runs/ragas/comparisons/bm25rerun_es/checkpoints" `
  --checkpoint-dir "research/evaluation/runs/ragas/comparisons/bm25rerun_ca_ca/checkpoints" `
  --checkpoint-dir "research/evaluation/runs/ragas/comparisons/bm25rerun_ragbench_dev/checkpoints" `
  --checkpoint-dir "research/evaluation/runs/ragas/comparisons/bm25rerun_ragbench_eval/checkpoints" `
  --checkpoint-dir "research/evaluation/runs/ragas/comparisons/bm25rerun_ragbench_visual/checkpoints" `
  --global-summary "research/evaluation/runs/ragas/comparisons/training_metrics_comparison_all.csv" `
  --overwrite
```

Generated artifacts:

- `<run>/training_metrics/<variant>.csv` - per-question metrics with `question`, `ground_truth`, `answer`, `status` and `reason`.
- `<run>/training_metrics/comparison_training_metrics.csv` - per-run aggregate.
- `training_metrics_comparison_all.csv` - global BM25 rerun summary.
- [`ANALISIS_RAGAS_AWS.md`](ANALISIS_RAGAS_AWS.md) - semantic RAGAS analysis for the BM25 rerun.
