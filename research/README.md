# MonkeyGrab — Research workspace

This folder contains everything needed to reproduce and extend the thesis work: RAGAS evaluation, RAGBench benchmarks, BERTScore post-processing, LoRA fine-tuning, model conversion and upload helpers.

It is **not required** to run the user-facing RAG stack (`python rag/chat_pdfs.py` or `python rag/web/app.py`).

---

## Contents

| Path | Purpose |
|------|---------|
| `evaluation/` | RAGAS runner (`run_eval.py`), RagBench preparation, BERTScore post-processing, datasets and `runs/` artifacts |
| `scripts/training/` | LoRA fine-tuning scripts (Qwen3-14B, Phi-4, Gemma-3-12B) |
| `scripts/evaluation/` | 7-model baseline benchmark, split inspection, SLURM helpers |
| `scripts/conversion/` | LoRA merge, GGUF build, quantization (`quantize_to_q4km.ps1`), Modelfile notes |
| `scripts/generate_diagram.py` | Architecture diagram via Kroki.io |
| `scripts/hf_upload_model_cards.py` | Upload weights + model cards to Hugging Face Hub |
| `scripts/package_user_bundle.*` | Build a minimal user zip (excludes `research/` and RagBench PDFs) |
| `training-output/` | LoRA metrics, `generate_reports.py` and `evaluation_comparison.json` per model (weights gitignored) |
| `models/gguf-output/` | `Modelfile`, `README.md` and `CONVERSION.md` per model (`.gguf` binaries gitignored) |
| `docs/` | `EVALUACIONES_PIPELINE.md`, layout guide, sparse-checkout notes, architecture diagrams |
| `tests/` | Pytest suites: `core/` (pipeline + CLI) and `research/` (eval runners) |

---

## Install

```bash
pip install -r research/evaluation/requirements.txt   # RAGAS + Gemini judge
pip install -r research/scripts/requirements.txt      # LoRA training stack (GPU, pinned versions)
```

---

## Training dataset (Hugging Face)

LoRA fine-tuning is built from **[nadiva1243/wikipediaEs-Ca4RAG](https://huggingface.co/datasets/nadiva1243/wikipediaEs-Ca4RAG)** — Wikipedia-style ES/CA material for RAG-oriented QA.

---

## RAGAS evaluation

Requires `GOOGLE_API_KEY` in `.env` (Gemini as judge LLM).

```bash
# Single run on a local corpus
python research/evaluation/run_eval.py single --corpus es
python research/evaluation/run_eval.py single --corpus ca

# Ablation comparison (multiple pipeline variants)
python research/evaluation/run_eval.py compare --corpus ca --label mi_eval --reindex
python research/evaluation/run_eval.py list-variants

# RagBench EN (fixed eval corpus — 25 papers, 5 q each)
python research/evaluation/run_eval.py ragbench-prepare
python research/evaluation/run_eval.py ragbench-eval

# RagBench visual (tables and images)
python research/evaluation/run_ragbench_visual_inference.py --n-papers 25 --max-q 5
python research/evaluation/run_ragbench_visual_inference.py --ragas-only

# BERTScore post-process over all completed RAGAS runs
python research/evaluation/evaluate_ragas_bertscore.py --all-completed

# Aggregate ablation by dataset subset
python research/evaluation/aggregate_comparison_by_conjunto.py \
  --dir research/evaluation/runs/ragas/comparisons/<label> \
  --etiquetas-es
```

Artifacts are written under `research/evaluation/runs/`. See `research/docs/EVALUACIONES_PIPELINE.md` for corpus presets, variant definitions and artifact layout.

---

## Baseline benchmark

```bash
python research/scripts/evaluation/evaluate_baselines.py   # 7-model benchmark (320 samples/dataset)
python research/training-output/baseline/generate_reports.py
```

---

## LoRA fine-tuning

```bash
python research/scripts/training/train-qwen3.py    # Qwen3-14B  (LoRA r=32, GPU)
python research/scripts/training/train-phi4.py     # Phi-4      (LoRA r=32)
python research/scripts/training/train-gemma3.py   # Gemma-3-12B (LoRA r=32)
```

After training, merge and quantize:

```bash
python research/scripts/conversion/merge_lora.py --model qwen-3   # options: qwen-3, phi-4, gemma-3
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
python research/scripts/hf_upload_model_cards.py
python research/scripts/hf_upload_model_cards.py --upload-qwen-q4-gguf
```

---

## Architecture diagram

```bash
python research/scripts/generate_diagram.py --output research/docs/monkeygrab_architecture.png
```

---

## Tests

```bash
pytest                          # runs research/tests/ (configured in pytest.ini)
pytest research/tests/core/     # pipeline + CLI smoke tests (require Ollama running)
pytest research/tests/research/ # eval runners (offline, no Ollama needed)
```

---

## Trained models

All three fine-tunes use **LoRA rank 32** on [nadiva1243/wikipediaEs-Ca4RAG](https://huggingface.co/datasets/nadiva1243/wikipediaEs-Ca4RAG).

| Model | HuggingFace model card | Ollama |
|-------|------------------------|--------|
| Qwen3-14B RAG | [nadiva1243/qwen3RAG](https://huggingface.co/nadiva1243/qwen3RAG) | GGUF + Modelfile in repo; importable |
| Phi-4 RAG | [nadiva1243/phi4RAG](https://huggingface.co/nadiva1243/phi4RAG) | GGUF + Modelfile in repo; importable |
| Gemma-3-12B | — | **Not importable into Ollama** — technical limitations (GGUF / tokenizer vs Ollama’s Gemma3 stack). See `research/scripts/conversion/GEMMA3_CONVERSION_ISSUE.md`. |

Modelfiles and conversion notes (where applicable): `research/models/gguf-output/<model>/`.

---

## RagBench PDF corpora

PDFs remain under `rag/docs/en_ragbench_*` so that `DOCS_FOLDER` and evaluation manifests keep working without path changes. For a lighter clone without these trees, see `research/docs/USER_SPARSE_CHECKOUT.md` or run `research/scripts/package_user_bundle.ps1` / `package_user_bundle.sh`.

---

*Bachelor's thesis (TFG) — Grado en Ingeniería Informática, ETSINF, Universitat Politècnica de València. Author: Ignacio Diago Valeta. Tutor: Adrià Giménez Pastor. 2025–2026.*
