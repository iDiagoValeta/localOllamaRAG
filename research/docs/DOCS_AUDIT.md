# Documentation inventory and audit

> Single index of all repository documentation. Each row states purpose,
> audience and the last date it was verified against the code. Update it after
> any change to the corresponding doc.
>
> **What is documented where** (to avoid drift between docs):
>
> - `CLAUDE.md` / `AGENTS.md` → operating rules for agents (public symbols,
>   forbidden flags, git policy) and the repository layout tree.
> - `README.md` (root) → product install and usage (CLI + Web), including the
>   lighter sparse-checkout clone.
> - `rag/README.md` → RAG pipeline technical reference (modules, signatures, parameters).
> - `research/README.md` → experimental workspace map; details live in sub-docs.
> - `research/docs/EVALUACIONES_PIPELINE.md` → TFG evaluation + reinference protocol (English).
> - `research/docs/RANKING_Y_PARAMETROS.md` → ranking/BM25/RRF reference + parameter defense + citation map (Spanish).
> - `research/docs/GEMMA3_CONVERSION_ISSUE.md` → Gemma-3 conversion failure post-mortem (English).
> - Analysis reports under `research/evaluation/runs/ragas/comparisons/ANALISIS_*.md`.
> - Model cards in `research/models/gguf/<model>/README.md` → self-contained, HF Hub.

| Doc | Purpose | Audience | Last verified | Drift risk |
|---|---|---|---|---|
| `CLAUDE.md` / `AGENTS.md` | Operating rules + layout (public symbols, flags, git) | Agents / author | 2026-05-19 | High |
| `README.md` | Product install and usage (CLI + Web + sparse checkout) | End user | 2026-05-19 | Medium |
| `rag/README.md` | RAG pipeline technical reference | Developer | 2026-05-14 | High |
| `rag/web/frontend/README.md` | React frontend `npm` commands | Frontend dev | 2026-05-14 | Low |
| `research/README.md` | TFG experimental workspace map | Tutor / author | 2026-05-19 | Medium |
| `research/docs/EVALUACIONES_PIPELINE.md` | TFG evaluation + reinference protocol (English) | Tutor / author | 2026-05-23 | High |
| `research/docs/RANKING_Y_PARAMETROS.md` | Ranking/BM25/RRF reference + parameter defense + citation map | Tutor / author | 2026-05-23 | High |
| `research/docs/GEMMA3_CONVERSION_ISSUE.md` | Gemma-3 conversion failure post-mortem (English, problems only) | Author / tutor | 2026-05-19 | Low (frozen) |
| `research/docs/DOCS_AUDIT.md` | This index | Author | 2026-05-23 | — |
| `research/evaluation/runs/ragas/comparisons/ANALISIS_RAGAS_AWS.md` | Definitive RAGAS report (AWS Bedrock judge) | Tutor / author | 2026-05-19 | High |
| `research/evaluation/runs/ragas/comparisons/ANALISIS_METRICAS_ENTRENAMIENTO.md` | Token F1 / ROUGE-L / BERTScore + §12 RAGAS cross-check | Tutor / author | 2026-05-19 | High |
| `research/models/gguf/gemma-3/README.md` | Gemma-3 final state (abandoned) | HF Hub / tutor | 2026-05-14 | Low |
| `research/models/gguf/phi-4/CONVERSION.md` | LoRA → GGUF → Ollama merge checklist | Author | 2026-05-14 | Low |
| `research/models/gguf/phi-4/README.md` | Phi-4 RAG fine-tuned model card | HF Hub / tutor | 2026-05-14 | Medium |
| `research/models/gguf/qwen-3/CONVERSION.md` | LoRA → GGUF → Ollama merge checklist | Author | 2026-05-14 | Low |
| `research/models/gguf/qwen-3/README.md` | Qwen3-14B RAG fine-tuned model card | HF Hub / tutor | 2026-05-14 | Medium |

## Deletion history

**2026-05-23 (ranking/parameters consolidation)**

- `research/docs/BM25_MIGRATION.md`, `research/docs/REINFERENCIA_BM25.md` and
  `research/docs/PIPELINE_PARAMETERS_DEFENSE.md` — merged into the single
  `research/docs/RANKING_Y_PARAMETROS.md` (BM25 migration detail + reinference
  commands + parameter defense + paper→stage citation map), updated for the new
  canonical `RRF_K = 60` (Cormack et al., 2009). Originals removed.

**2026-05-14**

- `research/docs/palabras.md` — TFG administrative form, out of repo scope.
- `research/docs/investigacionMetricas.md` — already captured in the thesis.
- `research/docs/splits.md` — already captured in the thesis.

**2026-05-19 (documentation consolidation)**

- `research/docs/PROJECT_LAYOUT.md` — redundant with the "Layout" section of `CLAUDE.md`/`AGENTS.md`; removed.
- `research/docs/USER_SPARSE_CHECKOUT.md` — content moved into the root `README.md` ("Lighter clone"); removed.
- `research/docs/REINFERENCIA_FINAL.md` — essentials (definitive labels, generator `phi4-finetuned:latest`, env config, reranker-threshold 0.65 decision, truncation resilience) consolidated into `EVALUACIONES_PIPELINE.md`; removed.
- `research/conversion/GEMMA3_CONVERSION_ISSUE.md` — rewritten in English (problems only, no unblock options) and moved to `research/docs/GEMMA3_CONVERSION_ISSUE.md`.
- `research/docs/EVALUACIONES_PIPELINE.md` — translated to English and merged with the reinference essentials above.

## How to use this index

When you change code (signatures in `rag/chat_pdfs.py`, CLIs in
`research/evaluation/*.py`, metrics in `training-output/`), check the **Drift
risk** column:

- **High** → the doc almost certainly needs updating.
- **Medium** → review commands/paths touched by your change.
- **Low** → unlikely, but a quick glance is cheap.

Update the **Last verified** column when done.
