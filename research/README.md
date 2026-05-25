# MonkeyGrab — Research workspace

Everything needed to reproduce the TFG: [RAGAS](https://docs.ragas.io/) evaluation, RAGBench benchmark, [LoRA](https://arxiv.org/abs/2106.09685) fine-tuning, GGUF conversion and Hugging Face upload.

**Not required** to use the product (`python rag/chat_pdfs.py` / `python rag/web/app.py`).

> Detailed references live in sub-docs:
> - Evaluation + reinference protocol → [`docs/EVALUACIONES_PIPELINE.md`](docs/EVALUACIONES_PIPELINE.md)
> - Gemma-3 post-mortem → [`docs/GEMMA3_CONVERSION_ISSUE.md`](docs/GEMMA3_CONVERSION_ISSUE.md)
> - Documentation audit → [`docs/DOCS_AUDIT.md`](docs/DOCS_AUDIT.md)
> - Directory map → the "Contents" table below

---

## Contents

| Path | Purpose |
|------|---------|
| `evaluation/` | 3 CLIs (`index.py`, `infer.py`, `evaluate.py`) + `probe_reranker_scores.py` + shared `_lib/` |
| `training/` | LoRA fine-tuning (`train_qwen3.py`, `train_phi4.py`, `train_gemma3.py`) |
| `baselines/` | 7-model benchmark (`evaluate_baselines.py`) and split utilities |
| `conversion/` | LoRA merge (`merge_lora.py`) + Gemma-3 post-mortem |
| `training-output/` | LoRA metrics + per-model `generate_reports.py` (weights gitignored) |
| `models/gguf/` | Per-model `Modelfile`, `README.md` and `CONVERSION.md` (`.gguf` gitignored) |
| `utils/` | Diagrams, packagers, HF push |
| `tests/` | Pytest: `core/` (pipeline) and `evaluation/` (offline runners) |

---

## Installation

```bash
pip install -r research/evaluation/requirements.txt   # RAGAS + Gemini judge
pip install -r research/training/requirements.txt     # LoRA training stack (GPU)
pip install langchain-openai openai                   # extra: NVIDIA provider
```

---

## Datasets

**Training** (LoRA, rank 32 on [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B), [Phi-4](https://huggingface.co/microsoft/phi-4), [Gemma-3-12B](https://huggingface.co/google/gemma-3-12b-it)):

| Source | HF | Notes |
|---|---|---|
| Neural-Bridge RAG | [neural-bridge/rag-dataset-12000](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000) | 12k EN triplets |
| Filtered Dolly QA | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | 4,467 grounded-EN triplets |
| Aina RAG Multilingual | [projecte-aina/RAG-multilingual](https://huggingface.co/datasets/projecte-aina/RAG-multilingual) | Aina-EN, Aina-ES, Aina-CA splits |

**Evaluation** (do not update weights, only score the RAG pipeline):

| Dataset | HF | Role |
|---|---|---|
| Wikipedia ES + Valencian/CA (50 QA × 5 PDFs per language) | [nadiva1243/wikipediaEs-Ca4RAG](https://huggingface.co/datasets/nadiva1243/wikipediaEs-Ca4RAG) | E2E in Spanish and Catalan |
| Vectara RAGBench (`pdf/arxiv`) | [vectara/open_ragbench](https://huggingface.co/datasets/vectara/open_ragbench) | Academic EN benchmark |

---

## RAGAS evaluation flow

Three steps, three CLIs (`research/evaluation/`):

```bash
# 1. Index
python research/evaluation/index.py --corpus es        # es | ca | en | ragbench-eval
python research/evaluation/index.py --corpus ca --force

# 2. Generate answers (no RAGAS)
python research/evaluation/infer.py single  --corpus es
python research/evaluation/infer.py compare --corpus ca --label my_eval     # baseline_all_on + all_off
python research/evaluation/infer.py compare --corpus ca --suite ablation    # long legacy suite
python research/evaluation/infer.py list-variants
python research/evaluation/infer.py ragbench-prepare && python research/evaluation/infer.py ragbench-eval

# 3. Evaluate checkpoints with RAGAS
python research/evaluation/evaluate.py --provider google --source-root research/evaluation/runs/ragas/comparisons/my_eval
python research/evaluation/evaluate.py --provider nvidia --all-known --nvidia-rate-limit-per-minute 40
python research/evaluation/evaluate.py --provider aws    --all-known
```

Checkpoints → `research/evaluation/runs/ragas/{single,comparisons,ragbench,ragbench_visual}/`.
RAGAS outputs → `research/evaluation/runs/ragas_<provider>_revaluation/`.

Training-style metrics: `python research/evaluation/training_metrics.py --checkpoint-dir <comparisons/label/checkpoints>`.

> **Definitive run (2026-05-25)** — generator `phi4-finetuned:latest`, Okapi BM25 hybrid retrieval, RAGAS judge AWS Bedrock. Labels: `bm25rerun_{es,ca_ca,ragbench_dev,ragbench_eval,ragbench_visual}`. Results and analysis: [`runs/ragas/comparisons/ANALISIS_RAGAS_AWS.md`](evaluation/runs/ragas/comparisons/ANALISIS_RAGAS_AWS.md) and [`ANALISIS_METRICAS_ENTRENAMIENTO.md`](evaluation/runs/ragas/comparisons/ANALISIS_METRICAS_ENTRENAMIENTO.md) (§6 BERTScore↔RAGAS cross-check).

Providers, environment variables (`GOOGLE_API_KEY`, `NVIDIA_API_KEY`, `AWS_BEARER_TOKEN_BEDROCK`), default judge models, aggregation options (`--aggregate-group-by`, `--aggregate-etiquetas-es`) and the RagBench flow details: see [`docs/EVALUACIONES_PIPELINE.md`](docs/EVALUACIONES_PIPELINE.md).

---

## Reranker score probe and reinference

```bash
# Measures the cross-encoder score distribution per corpus to calibrate UMBRAL_SCORE_RERANKER
python research/evaluation/probe_reranker_scores.py --corpus es              --n 8
python research/evaluation/probe_reranker_scores.py --corpus ca              --n 8
python research/evaluation/probe_reranker_scores.py --corpus en_ragbench_dev --n 8
```

Output in `rag/debug_rag/probe_<corpus>_<timestamp>.json`. Protocol and the 0.55 → 0.65 decision in [`docs/EVALUACIONES_PIPELINE.md`](docs/EVALUACIONES_PIPELINE.md) §2.7.

---

## LoRA fine-tuning

```bash
python research/training/train_qwen3.py   # Qwen3-14B   (LoRA r=32, GPU)
python research/training/train_phi4.py    # Phi-4
python research/training/train_gemma3.py  # Gemma-3-12B (not importable into Ollama)

python research/conversion/merge_lora.py --model qwen-3   # options: qwen-3 | phi-4 | gemma-3
# then: research/conversion/quantize_to_q4km.ps1   (Windows)
```

Curves and tables:

```bash
python research/training-output/qwen-3/generate_reports.py
python research/training-output/phi-4/generate_reports.py
python research/training-output/gemma-3/generate_reports.py
python research/training-output/baseline/generate_reports.py
```

---

## Baselines, diagrams, HF, tests

```bash
python research/baselines/evaluate_baselines.py                # 7 models × 320 samples
python research/utils/generate_diagram.py --output research/docs/monkeygrab_architecture.png
python research/utils/hf_upload_model_cards.py                 # add --upload-qwen-q4-gguf
pytest                                                          # research/tests/
```

---

## Trained models

All three use LoRA rank 32 on the *Datasets → Training* mixture.

| Model | HF | Ollama |
|---|---|---|
| [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) RAG | [nadiva1243/qwen3RAG](https://huggingface.co/nadiva1243/qwen3RAG) | Modelfile in `research/models/gguf/qwen-3/`, importable |
| [Phi-4](https://huggingface.co/microsoft/phi-4) RAG | [nadiva1243/phi4RAG](https://huggingface.co/nadiva1243/phi4RAG) | Modelfile in `research/models/gguf/phi-4/`, importable |
| [Gemma-3-12B](https://huggingface.co/google/gemma-3-12b-it) | — | **Not importable into Ollama** (GGUF/tokenizer vs Gemma3 stack). See [`docs/GEMMA3_CONVERSION_ISSUE.md`](docs/GEMMA3_CONVERSION_ISSUE.md). |

---

## RagBench corpus

PDFs live under `rag/docs/en_ragbench_*` so `DOCS_FOLDER` and the manifests need no rewriting. Lighter clone without these trees: "Lighter clone (sparse checkout)" section of the root [`README.md`](../README.md) or `research/utils/package_user_bundle.{ps1,sh}`.

---

