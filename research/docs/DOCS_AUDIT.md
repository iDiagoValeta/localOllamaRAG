# Documentation inventory and audit

> Single index of all repository documentation. Each row states purpose,
> audience and the date the doc was last verified against the code. Update it
> when the corresponding doc changes.
>
> **What is documented where** (to avoid drift between docs):
>
> - `README.md` (root) → product install and usage (CLI + Web), including the
>   lighter sparse-checkout clone.
> - `rag/README.md` → RAG pipeline technical reference (modules, signatures, parameters).
> - `rag/engine/ENGINE_MAP.md` → engine module map: purpose, functions and dependency graph of every file under `rag/engine/`.
> - `research/README.md` → experimental workspace map; details live in sub-docs.
> - `research/docs/EVALUACIONES_PIPELINE.md` → RAG pipeline evaluation + reinference protocol.
> - `research/docs/GEMMA3_CONVERSION_ISSUE.md` → Gemma-3 conversion failure post-mortem.
> - Analysis reports under `research/evaluation/runs/ragas/comparisons/ANALISIS_*.md`.
> - Model cards in `research/models/gguf/<model>/README.md` → self-contained, published on the HF Hub.

| Doc | Purpose | Audience | Language | Last verified | Drift risk |
|---|---|---|---|---|---|
| `README.md` | Product install and usage (CLI + Web + sparse checkout) | End user | English | 2026-05-24 | Medium |
| `rag/README.md` | RAG pipeline technical reference | Developer | English | 2026-05-24 | High |
| `rag/engine/ENGINE_MAP.md` | Engine module map: purposes, functions, dependency graph | Developer | English | 2026-05-24 | High |
| `rag/web/frontend/README.md` | React frontend `npm` commands | Developer | English | 2026-05-24 | Low |
| `research/README.md` | Experimental workspace map | Maintainer | English | 2026-05-25 | Medium |
| `research/docs/EVALUACIONES_PIPELINE.md` | RAG pipeline evaluation + reinference protocol | Maintainer | English | 2026-05-25 | High |
| `research/docs/GEMMA3_CONVERSION_ISSUE.md` | Gemma-3 conversion failure post-mortem (problems only) | Maintainer | English | 2026-05-24 | Low (frozen) |
| `research/docs/DOCS_AUDIT.md` | This index | Maintainer | English | 2026-05-25 | — |
| `research/evaluation/runs/ragas/comparisons/ANALISIS_RAGAS_AWS.md` | Definitive RAGAS report — `bm25rerun_*` labels (AWS Bedrock judge) | Maintainer | English | 2026-05-25 | Low (frozen) |
| `research/evaluation/runs/ragas/comparisons/ANALISIS_METRICAS_ENTRENAMIENTO.md` | Token F1 / ROUGE-L / BERTScore + §6 RAGAS cross-check — `bm25rerun_*` labels | Maintainer | English | 2026-05-25 | Low (frozen) |
| `research/evaluation/runs/ragas/comparisons/old/ANALISIS_RAGAS_AWS.md` | Pre-BM25 RAGAS report (archived, not updated) | — | English | — | Archived |
| `research/evaluation/runs/ragas/comparisons/old/ANALISIS_METRICAS_ENTRENAMIENTO.md` | Pre-BM25 Token F1 / ROUGE-L / BERTScore (archived, not updated) | — | English | — | Archived |
| `research/models/gguf/gemma-3/README.md` | Gemma-3 final state (abandoned) | HF Hub | English | 2026-05-24 | Low |
| `research/models/gguf/phi-4/CONVERSION.md` | LoRA → GGUF → Ollama merge checklist | Maintainer | English | 2026-05-24 | Low |
| `research/models/gguf/phi-4/README.md` | Phi-4 RAG fine-tuned model card | HF Hub | English | 2026-05-24 | Medium |
| `research/models/gguf/qwen-3/CONVERSION.md` | LoRA → GGUF → Ollama merge checklist | Maintainer | English | 2026-05-24 | Low |
| `research/models/gguf/qwen-3/README.md` | Qwen3-14B RAG fine-tuned model card | HF Hub | English | 2026-05-24 | Medium |

## How to use this index

When code changes (signatures in `rag/chat_pdfs.py`, CLIs in
`research/evaluation/*.py`, metrics in `training-output/`), check the **Drift
risk** column:

- **High** → the doc almost certainly needs updating.
- **Medium** → review the commands/paths touched by the change.
- **Low** → unlikely, but a quick glance is cheap.

Update the **Last verified** column when done.
