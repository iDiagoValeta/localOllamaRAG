# MonkeyGrab — Research workspace

This folder contains everything needed to reproduce and extend the thesis work: [RAGAS](https://docs.ragas.io/) evaluation, RAGBench benchmarks, [LoRA](https://arxiv.org/abs/2106.09685) fine-tuning, model conversion and upload helpers.

It is **not required** to run the user-facing RAG stack (`python rag/chat_pdfs.py` or `python rag/web/app.py`).

---

## Contents

| Path | Purpose |
|------|---------|
| `evaluation/` | 3 operative CLIs (`index.py`, `infer.py`, `evaluate.py`) consuming `_lib/` (datasets, checkpoints, inference, RAGAS runner, providers, aggregation). `evaluate.py` auto-aggregates by subset after comparison runs. Artifacts under `runs/` |
| `training/` | LoRA fine-tuning scripts (Qwen3-14B, Phi-4, Gemma-3-12B) and `requirements.txt` |
| `baselines/` | 7-model baseline benchmark, split inspection, SLURM helpers |
| `conversion/` | LoRA merge, GGUF build, quantization (`quantize_to_q4km.ps1`), Modelfile notes |
| `utils/` | Architecture diagram, HF upload helpers, user bundle packaging, dataset push scripts |
| `training-output/` | LoRA metrics, `generate_reports.py` and `evaluation_comparison.json` per model (weights gitignored) |
| `models/gguf/` | `Modelfile`, `README.md` and `CONVERSION.md` per model (`.gguf` binaries gitignored) |
| `docs/` | `EVALUACIONES_PIPELINE.md`, layout guide, sparse-checkout notes, architecture diagrams |
| `tests/` | [pytest](https://docs.pytest.org/) suites: `core/` (pipeline + CLI) and `evaluation/` (eval runners) |

---

## Install

```bash
pip install -r research/evaluation/requirements.txt   # RAGAS + Gemini judge
pip install langchain-openai openai                   # extra deps for NVIDIA evaluation provider (see below)
pip install -r research/training/requirements.txt     # LoRA training stack (GPU, pinned versions)
```

---

## Training data (LoRA)

[LoRA](https://arxiv.org/abs/2106.09685) adapters (**rank 32** for [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B), [Phi-4](https://huggingface.co/microsoft/phi-4) and [Gemma-3-12B](https://huggingface.co/google/gemma-3-12b-it)) are trained on a **mixture of public Hugging Face corpora** — not on the proprietary Wikipedia evaluation set below.

| Source | Hugging Face | Notes |
|--------|--------------|-------|
| Neural-Bridge RAG | [neural-bridge/rag-dataset-12000](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000) | 12k EN triplets (question, context, answer); technical / encyclopaedic synthesis |
| Dolly QA (filtered) | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | EN; only document-grounded task types (`closed_qa`, `information_extraction`, `summarization`) → **4 467** training triplets |
| Aina RAG Multilingual | [projecte-aina/RAG-multilingual](https://huggingface.co/datasets/projecte-aina/RAG-multilingual) | Multilingual news / Wikipedia-style RAG; used as **Aina-EN**, **Aina-ES** and **Aina-CA** splits |

How these pools are filtered, shuffled and split for training/validation is described in the thesis (Chapter 4). The scripts under `research/training/` implement the concrete recipe.

---

## Evaluation datasets

These corpora **do not** update LoRA weights; they score the **full PDF-indexed RAG pipeline** (retrieval + generation).

| Dataset | Hugging Face / location | Role |
|---------|-------------------------|------|
| Proprietary Wikipedia **ES** & **Valencian/Catalan** (50 QA each, 5 PDFs per language) | **[nadiva1243/wikipediaEs-Ca4RAG](https://huggingface.co/datasets/nadiva1243/wikipediaEs-Ca4RAG)** | End-to-end evaluation on real PDFs in Spanish and Valencian/Catalan; published for reuse |
| Vectara RAGBench (**arXiv / pdf** subset) | [vectara/open_ragbench](https://huggingface.co/datasets/vectara/open_ragbench) | External English academic-PDF benchmark (1000 PDFs, 3045 QA pairs in this work) |

The **RagBench** PDF trees under `rag/docs/en_ragbench_*` are used by `research/evaluation/infer.py` (subcommands `ragbench-prepare`, `ragbench-eval`, `visual`) for the English benchmark workflow; see `research/docs/EVALUACIONES_PIPELINE.md`.

---

## RAGAS evaluation

El flujo se divide en tres pasos operativos respaldados por tres CLIs en
`research/evaluation/` que comparten el paquete `_lib/`:

1. **`index.py`** — indexa un corpus en ChromaDB.
2. **`infer.py`** — genera respuestas RAG y persiste checkpoints (sin RAGAS).
3. **`evaluate.py`** — ejecuta RAGAS sobre checkpoints con `--provider google|aws|nvidia`.

### Paso 1 — Indexación

```bash
python research/evaluation/index.py --corpus es                 # rag/docs/es → ChromaDB
python research/evaluation/index.py --corpus ca --force         # re-indexa desde cero
python research/evaluation/index.py --corpus ragbench-eval      # usa el file-filter del manifest preparado
python research/evaluation/index.py --docs-dir custom/path --force
```

### Paso 2 — Inferencia (genera checkpoints)

```bash
# Baseline sobre un corpus local
python research/evaluation/infer.py single --corpus es
python research/evaluation/infer.py single --corpus ca

# Ablation (8 variantes, una por flag desactivado)
python research/evaluation/infer.py compare --corpus ca --label mi_eval --reindex
python research/evaluation/infer.py list-variants

# RagBench EN (corpus fijo — 25 papers, 5 q each)
python research/evaluation/infer.py ragbench-prepare
python research/evaluation/infer.py ragbench-eval

# RagBench visual (tablas/imágenes)
python research/evaluation/infer.py visual --n-papers 25 --max-q 5
```

Los checkpoints siguen el mismo schema que antes y se guardan bajo
`research/evaluation/runs/ragas/{single,comparisons,ragbench,ragbench_visual}/`.

### Paso 3 — RAGAS desde checkpoint (Google / NVIDIA / AWS)

`evaluate.py` no genera respuestas: solo lee checkpoints y aplica RAGAS con el
backend pedido. Soporta un único checkpoint (`--checkpoint`), descubrimiento
automático (`--all-known`) o un árbol concreto (`--source-root`).

#### Google Gemini (`GOOGLE_API_KEY`)

```bash
python research/evaluation/evaluate.py --provider google \
  --source-root research/evaluation/runs/ragas/comparisons/mi_eval
python research/evaluation/evaluate.py --provider google --all-known
```

Salida: `research/evaluation/runs/ragas_google_revaluation/`.

#### NVIDIA Build API (`NVIDIA_API_KEY`)

Modelos por defecto: `mistralai/mistral-medium-3.5-128b` (juez),
`nvidia/llama-3.2-nv-embedqa-1b-v2` (embeddings). Rate limit ajustable.

```bash
python research/evaluation/evaluate.py --provider nvidia --all-known --dry-run
python research/evaluation/evaluate.py --provider nvidia --all-known \
  --nvidia-rate-limit-per-minute 40
python research/evaluation/evaluate.py --provider nvidia \
  --checkpoint research/evaluation/runs/ragas/comparisons/<label>/checkpoints/<file>.json \
  --limit 5
python research/evaluation/evaluate.py --provider nvidia \
  --checkpoint <file>.json --retry-failed
```

Salida: `research/evaluation/runs/ragas_nvidia_revaluation/`.

#### Amazon Bedrock (boto3 / `AWS_BEARER_TOKEN_BEDROCK`)

Modelos por defecto: `eu.anthropic.claude-sonnet-4-20250514-v1:0` (juez),
`amazon.titan-embed-text-v2:0` (embeddings). Requiere acceso habilitado en la
consola de Bedrock.

```bash
python research/evaluation/evaluate.py --provider aws --all-known --dry-run
python research/evaluation/evaluate.py --provider aws --all-known
python research/evaluation/evaluate.py --provider aws \
  --checkpoint research/evaluation/runs/ragas/comparisons/<label>/checkpoints/<file>.json \
  --limit 5
```

Salida: `research/evaluation/runs/ragas_aws_revaluation/`.

### Agregación por subconjunto (integrada en `evaluate.py`)

Tras una run de comparación ablation, `evaluate.py` genera automáticamente
`by_conjunto_<group_by>.json` (+ CSV) por etiqueta, con medias variante ×
subconjunto × métrica. Se controla con tres flags:

```bash
# Agregar por source_type (default) y por idioma, con etiquetas en castellano
python research/evaluation/evaluate.py --provider google \
  --source-root research/evaluation/runs/ragas/comparisons/<label> \
  --aggregate-group-by source_type,language \
  --aggregate-etiquetas-es

# Saltar la agregación
python research/evaluation/evaluate.py --provider nvidia --all-known --no-aggregate
```

Subconjuntos soportados: `source_type`, `language`, `source_type_language`,
`id_prefix`.

### 2026 repetition runs

```bash
python repeticion_run_eval.py repeticion --corpus all
python repeticion_run_eval.py repeticion --corpus ragbench_eval --final-variant baseline_all_on
```

Artifacts are written under `research/evaluation/runs/`. See `research/docs/EVALUACIONES_PIPELINE.md` for corpus presets, variant definitions and artifact layout.

---

## Baseline benchmark

```bash
python research/baselines/evaluate_baselines.py   # 7-model benchmark (320 samples/dataset)
python research/training-output/baseline/generate_reports.py
```

---

## LoRA fine-tuning

```bash
python research/training/train_qwen3.py    # Qwen3-14B  (LoRA r=32, GPU)
python research/training/train_phi4.py     # Phi-4      (LoRA r=32)
python research/training/train_gemma3.py   # Gemma-3-12B (LoRA r=32)
```

After training, merge and quantize:

```bash
python research/conversion/merge_lora.py --model qwen-3   # options: qwen-3, phi-4, gemma-3
# then run quantize_to_q4km.ps1 (Windows) or llama-quantize manually
```

Generate training curves and evaluation tables:

```bash
python research/training-output/qwen-3/generate_reports.py
python research/training-output/phi-4/generate_reports.py
python research/training-output/gemma-3/generate_reports.py
```

---

## Upload to Hugging Face

```bash
python research/utils/hf_upload_model_cards.py
python research/utils/hf_upload_model_cards.py --upload-qwen-q4-gguf
```

---

## Architecture diagram

```bash
python research/utils/generate_diagram.py --output research/docs/monkeygrab_architecture.png
```

Renders the Mermaid flowchart via [Kroki.io](https://kroki.io/) and saves the result locally.

---

## Tests

```bash
pytest                          # runs research/tests/ (configured in pytest.ini)
pytest research/tests/core/     # pipeline + CLI smoke tests (require Ollama running)
pytest research/tests/evaluation/ # eval runners (offline, no Ollama needed)
```

---

## Trained models

All three fine-tunes use **LoRA rank 32** on the **training mixture** in the section *Training data (LoRA)* above (Neural-Bridge RAG + filtered Dolly QA + Aina splits).

| Model | HuggingFace model card | Ollama |
|-------|------------------------|--------|
| [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) RAG | [nadiva1243/qwen3RAG](https://huggingface.co/nadiva1243/qwen3RAG) | GGUF + Modelfile in repo; importable |
| [Phi-4](https://huggingface.co/microsoft/phi-4) RAG | [nadiva1243/phi4RAG](https://huggingface.co/nadiva1243/phi4RAG) | GGUF + Modelfile in repo; importable |
| [Gemma-3-12B](https://huggingface.co/google/gemma-3-12b-it) | — | **Not importable into Ollama** — technical limitations (GGUF / tokenizer vs Ollama's Gemma3 stack). See `research/conversion/GEMMA3_CONVERSION_ISSUE.md`. |

Modelfiles and conversion notes (where applicable): `research/models/gguf/<model>/`.

---

## 2026 Repetition Commands

PowerShell commands for the inference-only repetition run:

```powershell
# Terminal 1: Ollama server
cd C:\Users\nadiv\repos\localOllamaRAG
$env:OLLAMA_CONTEXT_LENGTH="32768"
$env:OLLAMA_KEEP_ALIVE="-1"
$env:OLLAMA_MAX_LOADED_MODELS="3"
ollama serve

# Terminal 2: repetition inference, no RAGAS
# Ollama context, RAG model, context char caps and eval timeouts use repo defaults
# from rag/chat_pdfs.py (and EVAL_* in run_eval / repeticion_run_eval). Optional:
# copy .env at repo root — repeticion_run_eval loads it before importing chat_pdfs.
cd C:\Users\nadiv\repos\localOllamaRAG
python .\repeticion_run_eval.py repeticion --run-id repeticion_20260512_132549 --corpus all
```

## RagBench PDF corpora

PDFs remain under `rag/docs/en_ragbench_*` so that `DOCS_FOLDER` and evaluation manifests keep working without path changes. For a lighter clone without these trees, see `research/docs/USER_SPARSE_CHECKOUT.md` or run `research/utils/package_user_bundle.ps1` / `package_user_bundle.sh`.

---

*Bachelor's thesis (TFG) — Grado en Ingeniería Informática, ETSINF, Universitat Politècnica de València. Author: Ignacio Diago Valeta. Tutor: Adrià Giménez Pastor. 2025–2026.*
