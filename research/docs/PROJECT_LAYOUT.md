# Repository layout — user vs research

This document maps **top-level** folders. Thesis material, metrics, Modelfiles, methodology docs, and automated tests now live **inside** [`research/`](../../README.md) so the repository root stays focused on the runnable product.

## End users (minimal root)

| Path | Purpose |
|------|---------|
| `rag/` | RAG engine, CLI, optional web UI under `rag/web/`, PDF corpora under `rag/docs/`, `rag/requirements.txt` |
| `README.md` / `CLAUDE.md` / `pytest.ini` | Entry docs and pytest config |

You do **not** need anything under `research/` to run `python rag/chat_pdfs.py` or `python rag/web/app.py`.

## Inside `research/` (thesis / tooling)

| Path | Purpose |
|------|---------|
| `research/evaluation/` | RAGAS (`run_eval.py`), datasets, `runs/`, BERTScore, aggregates |
| `research/training/` | LoRA fine-tuning scripts + `requirements.txt` |
| `research/conversion/` | LoRA merge, GGUF export, quantization helpers |
| `research/baselines/` | 7-model baseline benchmark and split inspection |
| `research/utils/` | Architecture diagram, HF upload, user bundle packaging |
| `research/training-output/` | LoRA runs: `generate_reports.py` + small JSON versioned; weights gitignored |
| `research/models/` | `merged-model/` (gitignored) and `gguf/` Modelfiles + cards |
| `research/docs/` | This file, `EVALUACIONES_PIPELINE.md`, sparse-checkout notes, diagrams |
| `research/tests/` | Pytest (`core/` + `evaluation/`) |

## Corpora

PDFs stay under `rag/docs/` (including RagBench trees) so `DOCS_FOLDER` and manifests keep working.

## Tests

Run from repository root: `pytest` (see `pytest.ini` → `research/tests`).
