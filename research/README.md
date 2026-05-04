# Research / TFG workspace

Everything under `research/` supports **reproducibility and evaluation** of the MonkeyGrab thesis work. It is optional for running the production RAG stack (`rag/`, including `rag/web/` for the UI).

## Contents

- **`evaluation/`** — RAGAS on the live pipeline, RagBench preparation, BERTScore post-processing, dataset JSONs, and `runs/` artifacts (partially gitignored where large).
- **`scripts/`** — LoRA training, baseline benchmarks, GGUF conversion orchestration, Hugging Face upload helper.
- **`docs/`** — Methodology (`EVALUACIONES_PIPELINE.md`), layout guide, sparse-checkout notes, architecture assets.
- **`training-output/`** — LoRA metrics and `generate_reports.py` per model.
- **`models/`** — GGUF `Modelfile` trees and merge output paths (large files gitignored).
- **`tests/`** — Pytest suites for pipeline and evaluation.

## Quick commands (from repo root)

```bash
pip install -r research/evaluation/requirements.txt
python research/evaluation/run_eval.py single --corpus es
```

Training stack (GPU, pinned versions):

```bash
pip install -r research/scripts/requirements.txt
python research/scripts/training/train-qwen3.py
```

## RagBench PDF corpora

RagBench PDFs remain under `rag/docs/en_ragbench_*` by design: moving them would require updating every prepared manifest and dataset path. If you need a **lighter clone**, use [sparse checkout](docs/USER_SPARSE_CHECKOUT.md) or `research/scripts/package_user_bundle.ps1` / `research/scripts/package_user_bundle.sh`.
