# MonkeyGrab — Research workspace

Todo lo necesario para reproducir el TFG: evaluación [RAGAS](https://docs.ragas.io/), benchmark RAGBench, fine-tuning [LoRA](https://arxiv.org/abs/2106.09685), conversión a GGUF y subida a Hugging Face.

**No es necesario** para usar el producto (`python rag/chat_pdfs.py` / `python rag/web/app.py`).

> Las referencias detalladas viven en sub-docs:
> - Protocolo de evaluación → [`docs/EVALUACIONES_PIPELINE.md`](docs/EVALUACIONES_PIPELINE.md)
> - Protocolo de reinferencia → [`docs/REINFERENCIA_FINAL.md`](docs/REINFERENCIA_FINAL.md)
> - Mapa de directorios → [`docs/PROJECT_LAYOUT.md`](docs/PROJECT_LAYOUT.md)
> - Auditoría de toda la documentación → [`docs/DOCS_AUDIT.md`](docs/DOCS_AUDIT.md)

---

## Contenido

| Path | Propósito |
|------|-----------|
| `evaluation/` | 3 CLIs (`index.py`, `infer.py`, `evaluate.py`) + `probe_reranker_scores.py` + `_lib/` compartido |
| `training/` | Fine-tuning LoRA (`train_qwen3.py`, `train_phi4.py`, `train_gemma3.py`) |
| `baselines/` | Benchmark de 7 modelos (`evaluate_baselines.py`) y utilidades de splits |
| `conversion/` | Merge LoRA (`merge_lora.py`) + post-mortem Gemma-3 |
| `training-output/` | Métricas LoRA + `generate_reports.py` por modelo (pesos gitignored) |
| `models/gguf/` | `Modelfile`, `README.md` y `CONVERSION.md` por modelo (`.gguf` gitignored) |
| `utils/` | Diagramas, packagers, push a HF |
| `tests/` | Pytest: `core/` (pipeline) y `evaluation/` (runners offline) |

---

## Instalación

```bash
pip install -r research/evaluation/requirements.txt   # RAGAS + Gemini judge
pip install -r research/training/requirements.txt     # LoRA training stack (GPU)
pip install langchain-openai openai                   # extra: NVIDIA provider
```

---

## Datasets

**Training** (LoRA, rank 32 sobre [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B), [Phi-4](https://huggingface.co/microsoft/phi-4), [Gemma-3-12B](https://huggingface.co/google/gemma-3-12b-it)):

| Fuente | HF | Notas |
|---|---|---|
| Neural-Bridge RAG | [neural-bridge/rag-dataset-12000](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000) | 12k tripletas EN |
| Dolly QA filtrado | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | 4 467 tripletas grounded-EN |
| Aina RAG Multilingual | [projecte-aina/RAG-multilingual](https://huggingface.co/datasets/projecte-aina/RAG-multilingual) | Splits Aina-EN, Aina-ES, Aina-CA |

**Evaluación** (no actualizan pesos, sólo puntúan el pipeline RAG):

| Dataset | HF | Rol |
|---|---|---|
| Wikipedia ES + Valencian/CA (50 QA × 5 PDFs por idioma) | [nadiva1243/wikipediaEs-Ca4RAG](https://huggingface.co/datasets/nadiva1243/wikipediaEs-Ca4RAG) | E2E en español y catalán |
| Vectara RAGBench (`pdf/arxiv`) | [vectara/open_ragbench](https://huggingface.co/datasets/vectara/open_ragbench) | Benchmark EN académico |

---

## Flujo de evaluación RAGAS

Tres pasos, tres CLIs (`research/evaluation/`):

```bash
# 1. Indexar
python research/evaluation/index.py --corpus es        # es | ca | en | ragbench-eval
python research/evaluation/index.py --corpus ca --force

# 2. Generar respuestas (sin RAGAS)
python research/evaluation/infer.py single  --corpus es
python research/evaluation/infer.py compare --corpus ca --label mi_eval     # baseline_all_on + all_off
python research/evaluation/infer.py compare --corpus ca --suite ablation    # suite larga legacy
python research/evaluation/infer.py list-variants
python research/evaluation/infer.py ragbench-prepare && python research/evaluation/infer.py ragbench-eval

# 3. Evaluar checkpoints con RAGAS
python research/evaluation/evaluate.py --provider google --source-root research/evaluation/runs/ragas/comparisons/mi_eval
python research/evaluation/evaluate.py --provider nvidia --all-known --nvidia-rate-limit-per-minute 40
python research/evaluation/evaluate.py --provider aws    --all-known
```

Checkpoints → `research/evaluation/runs/ragas/{single,comparisons,ragbench,ragbench_visual}/`.
Salidas RAGAS → `research/evaluation/runs/ragas_<provider>_revaluation/`.

Métricas adicionales tipo training: `python research/evaluation/training_metrics.py --checkpoint-dir <comparisons/label/checkpoints>`.

> **Corrida definitiva (2026-05-19)** — generador `phi4-finetuned:latest`, juez RAGAS AWS Bedrock. Resultados y análisis: [`runs/ragas/comparisons/ANALISIS_RAGAS_AWS.md`](evaluation/runs/ragas/comparisons/ANALISIS_RAGAS_AWS.md) y [`ANALISIS_METRICAS_ENTRENAMIENTO.md`](evaluation/runs/ragas/comparisons/ANALISIS_METRICAS_ENTRENAMIENTO.md) (§12 cruce BERTScore↔RAGAS).

Providers, variables de entorno (`GOOGLE_API_KEY`, `NVIDIA_API_KEY`, `AWS_BEARER_TOKEN_BEDROCK`), modelos juez por defecto, opciones de agregación (`--aggregate-group-by`, `--aggregate-etiquetas-es`) y los detalles del flujo RagBench: ver [`docs/EVALUACIONES_PIPELINE.md`](docs/EVALUACIONES_PIPELINE.md).

---

## Sonda y reinferencia (umbrales del reranker)

```bash
# Mide la distribución de scores del cross-encoder por corpus para calibrar UMBRAL_SCORE_RERANKER
python research/evaluation/probe_reranker_scores.py --corpus es              --n 8
python research/evaluation/probe_reranker_scores.py --corpus ca              --n 8
python research/evaluation/probe_reranker_scores.py --corpus en_ragbench_dev --n 8
```

Salida en `rag/debug_rag/probe_<corpus>_<timestamp>.json`. Protocolo y decisión 0.55 → 0.65 en [`docs/REINFERENCIA_FINAL.md`](docs/REINFERENCIA_FINAL.md).

---

## Fine-tuning LoRA

```bash
python research/training/train_qwen3.py   # Qwen3-14B   (LoRA r=32, GPU)
python research/training/train_phi4.py    # Phi-4
python research/training/train_gemma3.py  # Gemma-3-12B (no importable en Ollama)

python research/conversion/merge_lora.py --model qwen-3   # opciones: qwen-3 | phi-4 | gemma-3
# luego: research/conversion/quantize_to_q4km.ps1   (Windows)
```

Curvas y tablas:

```bash
python research/training-output/qwen-3/generate_reports.py
python research/training-output/phi-4/generate_reports.py
python research/training-output/gemma-3/generate_reports.py
python research/training-output/baseline/generate_reports.py
```

---

## Baselines, diagramas, HF, tests

```bash
python research/baselines/evaluate_baselines.py                # 7 modelos × 320 muestras
python research/utils/generate_diagram.py --output research/docs/monkeygrab_architecture.png
python research/utils/hf_upload_model_cards.py                 # añade --upload-qwen-q4-gguf
pytest                                                          # research/tests/
```

---

## Modelos entrenados

Los tres usan LoRA rank 32 sobre la mezcla de la sección *Datasets → Training*.

| Modelo | HF | Ollama |
|---|---|---|
| [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) RAG | [nadiva1243/qwen3RAG](https://huggingface.co/nadiva1243/qwen3RAG) | Modelfile en `research/models/gguf/qwen-3/`, importable |
| [Phi-4](https://huggingface.co/microsoft/phi-4) RAG | [nadiva1243/phi4RAG](https://huggingface.co/nadiva1243/phi4RAG) | Modelfile en `research/models/gguf/phi-4/`, importable |
| [Gemma-3-12B](https://huggingface.co/google/gemma-3-12b-it) | — | **No importable en Ollama** (GGUF/tokenizer vs stack de Gemma3). Ver [`conversion/GEMMA3_CONVERSION_ISSUE.md`](conversion/GEMMA3_CONVERSION_ISSUE.md). |

---

## Corpus RagBench

Los PDFs viven en `rag/docs/en_ragbench_*` para que `DOCS_FOLDER` y los manifests no requieran reescritura. Clon ligero sin estos árboles: [`docs/USER_SPARSE_CHECKOUT.md`](docs/USER_SPARSE_CHECKOUT.md) o `research/utils/package_user_bundle.{ps1,sh}`.

---

*TFG — Grado en Ingeniería Informática, ETSINF, Universitat Politècnica de València. Autor: Ignacio Diago Valeta. Tutor: Adrià Giménez Pastor. 2025–2026.*
