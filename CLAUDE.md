---
description: 
alwaysApply: true
---

# CLAUDE.md — MonkeyGrab (localOllamaRAG)

**MonkeyGrab** is a fully local RAG system built as a Computer Engineering final thesis (TFG) at ETSINF, UPV. Author: Ignacio Diago Valeta. Tutor: Adrià Giménez Pastor. It indexes PDFs, retrieves relevant chunks via hybrid search, and generates grounded answers using a local Ollama LLM. Two dimensions: (1) experimental layer — LoRA training, multi-metric evaluation; (2) production system — full RAG pipeline with CLI and Web UI.

---

## 1. Behavior rules

1. **Never commit or push without asking the user first.** No exceptions.
2. **Always respond in Spanish** to the user.
3. **Follow code patterns** in Section 7 when writing new code.
4. **Do not modify `requirements.txt`** without confirmation — versions are intentionally pinned (especially the training stack).
5. **Do not change pipeline flags** (`USAR_RECOMP_SYNTHESIS`, etc.) without agreement — they directly affect latency and cost.
6. **Do not touch `llama.cpp/`** — external submodule.
7. **`chat_pdfs.py` public API** — `rag/web/app.py` imports it as `rag_engine`; `research/evaluation/run_eval.py` and `research/tests/` also consume its symbols directly. Renaming anything breaks the web backend, eval runner, and tests. Full public API:
   - Constants: `PATH_DB`, `COLLECTION_NAME`, `CARPETA_DOCS`, `SYSTEM_PROMPT_CHAT`, `SYSTEM_PROMPT_RAG`, `MAX_HISTORIAL_MENSAJES`, `MODELO_CHAT`, `MODELO_RAG`, `MIN_LONGITUD_PREGUNTA_RAG`, `UMBRAL_RELEVANCIA`, `UMBRAL_SCORE_RERANKER`, `TOP_K_FINAL`, `EXPANDIR_CONTEXTO`, `N_TOP_PARA_EXPANSION`, `MAX_CONTEXTO_CHARS`, `USAR_RERANKER`, `USAR_RECOMP_SYNTHESIS`, `RERANKER_AVAILABLE`, `STOPWORDS`
   - Functions (CLI/web): `indexar_documentos`, `realizar_busqueda_hibrida`, `expandir_con_chunks_adyacentes`, `sintetizar_contexto_recomp`, `construir_contexto_para_modelo`, `guardar_debug_rag`, `generar_respuesta_silenciosa`, `obtener_documentos_indexados`, `cargar_historial`, `guardar_historial`, `limpiar_historial`
   - Functions (eval/tests): `get_pipeline_flags`, `set_pipeline_flags`, `set_docs_folder_runtime`, `set_ragbench_reranker_low_score_fallback`, `evaluar_pregunta_rag`
8. **After any change, check whether `CLAUDE.md` or the relevant README needs updating** — `README.md` is the user-facing entry point (installation, run, configuration, CLI commands only); `research/README.md` covers evaluation, training, benchmarks and thesis details. Update the correct file; do not add thesis/research content to the root README.
9. **Changes to `.gitignore` or `.gitkeep`**: follow Section 9 (git policy). After editing rules, validate with `git check-ignore -v <path>` and update Section 9 if policy changes.

---

## 2. Experiment status

| Task | Status |
|------|--------|
| Baseline evaluation (7 models, 320 samples/dataset) | ✅ Done — `research/training-output/baseline/` |
| Fine-tuning Qwen3-14B (v10) | ✅ `research/training-output/qwen-3/` — LoRA **r=32** (weights gitignored; metrics versioned) |
| Fine-tuning Phi-4 (v1) | ✅ `research/training-output/phi-4/` — LoRA **r=32** (optional exploration outputs may exist under `phi-4/16/`, `phi-4/64/`) |
| Fine-tuning Gemma-3-12B (v2) | ✅ `research/training-output/gemma-3/` — LoRA **r=32**. **Not importable into Ollama** (technical limitations; see `research/scripts/conversion/GEMMA3_CONVERSION_ISSUE.md`) |
| Base/adapted test evaluation | ✅ `evaluation_comparison.json` versioned in all model dirs |

**LoRA training corpora (public HF):** [neural-bridge/rag-dataset-12000](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000), filtered [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (document-grounded categories only, 4 467 samples), and [projecte-aina/RAG-multilingual](https://huggingface.co/datasets/projecte-aina/RAG-multilingual) (Aina-EN / Aina-ES / Aina-CA).

**Evaluation-only HF dataset (not used in LoRA loss):** [nadiva1243/wikipediaEs-Ca4RAG](https://huggingface.co/datasets/nadiva1243/wikipediaEs-Ca4RAG) — proprietary Spanish + Valencian/Catalan Wikipedia-PDF QA for end-to-end pipeline evaluation.

**External EN PDF benchmark:** [vectara/open_ragbench](https://huggingface.co/datasets/vectara/open_ragbench) (`pdf`/arXiv subset used in this work).

**Ollama:** Qwen3-14B and Phi-4 RAG builds are importable as Ollama models. The Gemma-3-12B fine-tune is **not importable into Ollama** today — technical limitations in `research/scripts/conversion/GEMMA3_CONVERSION_ISSUE.md`.

### RAG pipeline state (2026-04-10)

| Parameter | Value |
|-----------|-------|
| `CHUNK_SIZE` | 2000 |
| `CHUNK_OVERLAP` | 400 |
| `MIN_CHUNK_LENGTH` | 150 |
| `TOP_K_FINAL` | 8 |
| `RRF_K` | 20 |
| `UMBRAL_SCORE_RERANKER` | 0.55 |
| `MAX_CONTEXTO_CHARS` | 12000 |
| `num_ctx` (generation) | 16384 |
| `USAR_RECOMP_SYNTHESIS` | `True` |
| `USAR_CONTEXTUAL_RETRIEVAL` | `True` |
| `USAR_RERANKER` | `True` |
| `USAR_EMBEDDINGS_IMAGEN` | `True` |

**Response truncation (Phi-4 / Qwen3):** both models occasionally cut responses mid-sentence (e.g. "el año 2…"). This is not a render bug — it reflects the Modelfile `num_ctx` limit (16k by default) interacting with long contexts. Set `OLLAMA_RAG_NUM_CTX=32768` in evaluation runs to reduce occurrence. Ollama's default if no context env var is set is 4096, which causes systematic truncation.

---

## 3. File structure

`research/evaluation/evaluate_ragas_bertscore.py` is BERTScore post-processing for completed RAGAS runs. Reads CSVs from `research/evaluation/runs/ragas/`, uses `microsoft/deberta-xlarge-mnli` with `rescale_with_baseline=True`, writes to `research/evaluation/runs/bertscore/`. Does **not** run inference or RAGAS. For RagBench visual, `run_ragbench_visual_inference.py --ragas-only` runs RAGAS on a completed inference JSON without regenerating responses.

```
localOllamaRAG/
├── pytest.ini                    # Pytest: pythonpath + research/tests
├── rag/
│   ├── chat_pdfs.py              # Public facade + global config; implementation in rag/engine/
│   ├── engine/
│   │   ├── runtime.py            # Sync layer: exposes chat_pdfs globals to child modules
│   │   ├── chunking.py           # Markdown chunking, neighbor chunk IDs
│   │   ├── lexical.py            # Stopwords, keyword extraction, lexical/exhaustive search
│   │   ├── reranking.py          # LLM query decomposition, CrossEncoder reranking
│   │   ├── retrieval.py          # Hybrid search orchestration (semantic + lexical + RRF)
│   │   ├── context.py            # Context cleaning, model formatting, RECOMP synthesis
│   │   ├── debug.py              # Per-query RAG debug dumps
│   │   ├── generation.py         # Ollama generation (streaming + silent eval path)
│   │   ├── contextual.py         # Contextual retrieval (chunk enrichment at indexing)
│   │   ├── images.py             # PDF image extraction + LLM OCR description
│   │   ├── history.py            # Chat history persistence
│   │   └── indexing.py           # PDF indexing into ChromaDB
│   ├── show_fragments/export_fragments.py  # salida por defecto: show_fragments/exports/ (versionada)
│   ├── requirements.txt
│   ├── debug_rag/                # Query debug dumps (runtime, gitignored)
│   ├── docs/                     # PDFs por corpus — todos versionados como evidencia del TFG
│   │   ├── libre/                # Corpus de uso libre (default DOCS_FOLDER)
│   │   ├── es/  ca/  en/         # Corpora Wikipedia ES/CA + carpeta EN vacía
│   │   ├── en_ragbench_dev/      # Frozen RagBench EN dev split
│   │   ├── en_ragbench_eval/     # Final RagBench EN eval corpus
│   │   └── en_ragbench_visual/   # RagBench EN visual — tables/images
│   ├── vector_db/                # ChromaDB per corpus (gitignored; PATH_DB = {folder}_{embed_slug})
│   ├── cli/
│       ├── app.py                # MonkeyGrabCLI: interactive loop, dispatch, Ollama health check
│       ├── display.py            # `ui` singleton: Rich/ANSI/plain + QueryTimer + SessionStats
│       ├── commands.py           # Slash-command registry (list + aliases)
│       └── strings.py            # ES/EN/CA string tables; s(key, lang) for i18n
│   └── web/                      # Flask backend + React (Vite in zip/)
│       ├── app.py                # REST + SSE; serves zip/dist/
│       └── zip/                  # React source + build (dist/, gitignored)
├── research/
│   ├── README.md
│   ├── evaluation/               # RAG evaluation + run_eval.py + NVIDIA re-evaluator + runs/
│   ├── docs/                     # EVALUACIONES_PIPELINE.md, PROJECT_LAYOUT, sparse-checkout, diagrams
│   ├── training-output/          # LoRA runs + baseline benchmark artifacts
│   ├── models/                   # merged-model/ (gitignored) + gguf-output/ Modelfiles
│   ├── tests/                    # pytest: core/ + research/
│   └── scripts/
│       ├── training/             # train-qwen3.py, train-phi4.py, train-gemma3.py, run-general.sh
│       ├── evaluation/           # evaluate_baselines.py, inspect_splits.py, run-baselines.sh
│       ├── conversion/           # merge_lora.py, build_ollama.bat, quantize_to_q4km.ps1, GEMMA3_CONVERSION_ISSUE.md
│       ├── generate_diagram.py   # Kroki architecture diagram
│       ├── package_user_bundle.ps1
│       ├── package_user_bundle.sh
│       ├── hf_upload_model_cards.py
│       └── requirements.txt      # Training stack (pinned)
├── llama-bin/                    # Compiled llama.cpp binaries for Windows (gitignored)
├── llama.cpp/                    # External submodule — do not modify
├── README.md
└── CLAUDE.md
```

---

## 4. Model roles

Configured via env vars; defaults are the second arg of `os.getenv` in `rag/chat_pdfs.py`. **The running process environment always takes precedence.**

| Role | Env var | Notes |
|------|---------|-------|
| RAG generator | `OLLAMA_RAG_MODEL` | Streaming response in `/rag` mode |
| Chat + sub-queries | `OLLAMA_CHAT_MODEL` | `/chat` and RAG sub-queries; `think=False` to suppress reasoning traces |
| Embeddings | `OLLAMA_EMBED_MODEL` | Indexes chunks and queries; slug included in ChromaDB path |
| Contextual retrieval | `OLLAMA_CONTEXTUAL_MODEL` | Enriches chunk text at indexing time (`USAR_CONTEXTUAL_RETRIEVAL`) |
| RECOMP | `OLLAMA_RECOMP_MODEL` | Synthesizes fragments before generation (`USAR_RECOMP_SYNTHESIS`) |
| Vision / OCR | `OLLAMA_OCR_MODEL` | Describes raster images in PDFs (`USAR_EMBEDDINGS_IMAGEN`); multimodal, `think=False` |
| Reranker | `RERANKER_QUALITY` | Local CrossEncoder (`quality`=BGE, `speed`=MiniLM); not an Ollama model |

Reference environment (adjust per hardware):

| Variable | Default | |
|----------|---------|--|
| `OLLAMA_RAG_MODEL` | `phi4-finetuned:latest` | RAG generator |
| `OLLAMA_CHAT_MODEL` | `gemma4:e2b` | Chat mode |
| `OLLAMA_EMBED_MODEL` | `embeddinggemma:latest` | Embeddings |
| `OLLAMA_CONTEXTUAL_MODEL` | `gemma4:e4b` | Contextual retrieval |
| `OLLAMA_RECOMP_MODEL` | `gemma4:e4b` | RECOMP synthesis |
| `OLLAMA_OCR_MODEL` | `gemma4:e4b` | Image description |
| `DOCS_FOLDER` | `rag/docs/libre/` | PDF folder to index |
| `RERANKER_QUALITY` | `quality` | Reranker tier |
| `MONKEYGRAB_LANG` | `es` | CLI language: `es`, `en` or `ca` |
| `HF_TOKEN` | — | HuggingFace token (required for Gemma-3) |
| `GOOGLE_API_KEY` | — | Gemini API key for RAGAS evaluation (provider A) |
| `NVIDIA_API_KEY` | — | NVIDIA Build API key for RAGAS re-evaluation (provider B, free) |
| `OLLAMA_NUM_CTX` | 16384 | Global context window for all Ollama models; Ollama defaults to 4096 if not set |
| `OLLAMA_RAG_NUM_CTX` | 32768 | Context override for the RAG generator (takes precedence over `OLLAMA_NUM_CTX`) |
| `OLLAMA_AUX_NUM_CTX` | 32768 | Context override for auxiliary models (RECOMP, contextual retrieval, OCR) |
| `OLLAMA_REQUEST_TIMEOUT` | 900 | HTTP timeout (seconds) for Ollama requests in evaluation runs |

**`embeddinggemma` context limit:** Ollama caps this model at ~2048 tokens. The pipeline's `CHUNK_SIZE` (2000 chars) is usually within the limit, but monitor `ollama serve` output for `truncated` warnings on unusually long chunks (e.g. contextual-retrieval-enriched chunks).

---

## 5. Pipeline architecture

### Indexing (on startup or `/reindex`)

```
PDFs (CARPETA_DOCS)
  -> Text extraction: pymupdf4llm (preferred) / pypdf (fallback)
  -> [Opt] USAR_EMBEDDINGS_IMAGEN: fitz + OLLAMA_OCR_MODEL captions
  -> Chunking: CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH
  -> [Opt] USAR_CONTEXTUAL_RETRIEVAL: LLM enrichment via OLLAMA_CONTEXTUAL_MODEL
  -> Embedding: OLLAMA_EMBED_MODEL (EMBED_PREFIX_* if required by model)
  -> Persistence: ChromaDB at PATH_DB = rag/vector_db/{folder}_{embed_slug}
```

### Retrieval (per RAG query)

```
User query (>= MIN_LONGITUD_PREGUNTA_RAG chars)
  -> [Opt] USAR_LLM_QUERY_DECOMPOSITION: up to 3 sub-queries via OLLAMA_CHAT_MODEL
  -> Semantic search: OLLAMA_EMBED_MODEL, N_RESULTADOS_SEMANTICOS results
  -> [Opt] USAR_BUSQUEDA_HIBRIDA: keyword/lexical search, N_RESULTADOS_KEYWORD results
  -> [Opt] USAR_BUSQUEDA_EXHAUSTIVA: critical-term deep scan
  -> RRF fusion: semantic + keyword -> score_final (55/45 weights — do not change without evaluation)
  -> [Opt] USAR_RERANKER: CrossEncoder, TOP_K_RERANK_CANDIDATES -> TOP_K_AFTER_RERANK
  -> Filtering: UMBRAL_RELEVANCIA (0.50); with reranker: UMBRAL_SCORE_RERANKER
  -> Selection: TOP_K_FINAL (8 fragments)
  -> [Opt] EXPANDIR_CONTEXTO: N_TOP_PARA_EXPANSION + adjacent chunks
  -> [Opt] USAR_OPTIMIZACION_CONTEXTO: trim to MAX_CONTEXTO_CHARS (12000)
  -> [Opt] USAR_RECOMP_SYNTHESIS: synthesize context via OLLAMA_RECOMP_MODEL
```

### Generation

```
Question + <context>
  -> Final context: raw chunks or RECOMP synthesis
  -> [Opt] SYSTEM_PROMPT_RAG: sent via API if model name does not contain "finetuned"
      (fine-tuned models have it baked in their Modelfile; all others receive it explicitly)
  -> OLLAMA_RAG_MODEL: streaming via Ollama (temperature, top_p, repeat_penalty, num_ctx)
```

---

## 6. Key commands

```bash
# CLI (from repo root)
python rag/chat_pdfs.py
MONKEYGRAB_LANG=en python rag/chat_pdfs.py           # English UI (bash)
$env:MONKEYGRAB_LANG = "en"; python rag/chat_pdfs.py # English UI (PowerShell)
MONKEYGRAB_LANG=ca python rag/chat_pdfs.py           # Valencian UI (bash)
$env:MONKEYGRAB_LANG = "ca"; python rag/chat_pdfs.py # Valencian UI (PowerShell)

# Web — http://localhost:5000
python rag/web/app.py
# Web UI has an ES / EN / VAL selector persisted in localStorage.

# PowerShell note: use $env:MONKEYGRAB_LANG = "en" / "ca".
# set MONKEYGRAB_LANG=... is cmd.exe syntax and does not update PowerShell env.

# React dev (Vite on :3000, proxied to Flask on :5000)
cd rag/web/zip && npm run dev
cd rag/web/zip && npm run build      # production build -> rag/web/zip/dist/

# LoRA training (all runs complete)
python research/scripts/training/train-{qwen3,phi4,gemma3}.py
python research/scripts/conversion/merge_lora.py --model qwen-3    # options: qwen-3, phi-4, gemma-3
python research/scripts/hf_upload_model_cards.py [--upload-qwen-q4-gguf]

# Evaluation
python research/scripts/evaluation/evaluate_baselines.py             # 7-model baseline benchmark
python research/training-output/baseline/generate_reports.py         # regenerate tables/CSVs
python research/training-output/{qwen-3,phi-4,gemma-3}/generate_reports.py
python research/evaluation/run_eval.py single --corpus es            # RAGAS on live pipeline (GOOGLE_API_KEY required)
python research/evaluation/run_eval.py compare --corpus ca --label my_eval
python research/evaluation/run_eval.py ragbench-prepare
python research/evaluation/run_eval.py ragbench-eval
python research/evaluation/run_ragbench_visual_inference.py --n-papers 25 --max-q 5
python research/evaluation/run_ragbench_visual_inference.py --ragas-only  # RAGAS on existing inference JSON
python research/evaluation/aggregate_comparison_by_conjunto.py --dir research/evaluation/runs/ragas/comparisons/<label> --etiquetas-es

# RAGAS re-evaluation using NVIDIA Build API (NVIDIA_API_KEY required; free, rate limit 40 req/min)
python research/evaluation/eval_ragas_nvidia_from_checkpoints.py --all-known --dry-run
python research/evaluation/eval_ragas_nvidia_from_checkpoints.py --all-known
python research/evaluation/eval_ragas_nvidia_from_checkpoints.py --checkpoint <path> --limit 5

# Misc
python research/scripts/generate_diagram.py --output research/docs/monkeygrab_architecture.png
python rag/show_fragments/export_fragments.py [--language es] [--ragbench dev|eval]
git check-ignore -v <path>      # validate gitignore rules
```

### CLI runtime commands

| Command | Description |
|---------|-------------|
| `/rag` | RAG mode (document query) |
| `/chat` | Chat mode (general conversation) |
| `/docs` | List indexed documents |
| `/temas` `/topics` `/temes` | Topic summary per document |
| `/stats` | Vector DB statistics |
| `/reindex` | Drop DB and re-index all PDFs |
| `/limpiar` `/clear` `/netejar` | Clear chat history |
| `/ayuda` `/help` `/ajuda` | Show help |
| `/salir` `/exit` `/eixir` | Exit saving history |

---

## 7. Code patterns

1. **MODULE MAP at the top of every non-trivial Python file** — ASCII tree indexing all sections.
2. **Section separators** — `# ─────────────────────────────────────────────` + `# SECTION N: NAME`, then `# --- N.1 subsection ---`.
3. **Global constants before any logic** — order: stdlib → third-party → local imports, then constants (models, paths, flags, numeric params).
4. **Defensive env setup before heavy imports** — set `TORCH_COMPILE_DISABLE`, `TRITON_DISABLE`, etc. *before* importing torch/transformers.
5. **Optional dependencies** — `try/except ImportError` with a boolean availability flag (`PYMUPDF_AVAILABLE`, etc.).
6. **Explicit pipeline phases** — name and visually separate: load → prepare → infer → evaluate → export.
7. **Artifact-oriented output** — experimental scripts always write metrics JSON + per-sample CSV + plots; never stdout only.
8. **Mixed ES/EN naming (established convention)** — RAG-domain functions in Spanish (`realizar_busqueda_hibrida`), config constants in English (`CHUNK_SIZE`, `TOP_K_FINAL`), docstrings and comments in English. Follow the module's existing pattern; do not mix within a block.
9. **Training scripts include VERSION HISTORY** — block at the top with version changes and technical rationale.
10. **Script-first, not enterprise architecture** — main logic in modules + `main()`. Only exception: `MonkeyGrabCLI` in `cli/app.py`.
11. **Results tables — no std dev, use relative improvement** — Δ absolute (pp): `adapted − base`; Δ relative (%): `(adapted − base) / base × 100`. JSON fields: `deltas` (pp), `deltas_rel_pct` (%).
12. **Pipeline flags documented in their own block** — inline comment per flag explaining what it does.
13. **Google-style docstrings** — Args / Returns / Raises sections; module-level docstring with Usage and Dependencies.

---

## 8. Dependencies

```bash
pip install -r rag/requirements.txt          # RAG core (required)
pip install -r rag/web/requirements.txt          # Web UI (optional)
pip install -r research/evaluation/requirements.txt   # RAGAS evaluation (optional)
pip install -r research/scripts/requirements.txt      # LoRA fine-tuning (optional, GPU required)
```

**Training stack (pinned — do not change without confirmation):**

```
torch==2.6.0  transformers==4.57.6  peft==0.18.1  datasets==4.3.0
accelerate==1.12.0  bitsandbytes==0.49.1  safetensors==0.7.0  bert-score>=0.3.13
```

System: Python 3.10+, Ollama running locally, CUDA GPU recommended (~24 GB VRAM for Qwen3-14B fine-tuning).

---

## 9. Git versioning policy

**Single root `.gitignore`** plus `rag/web/zip/.gitignore` as a complement. No additional scattered `.gitignore` files. Version the minimum needed to reproduce and defend the work: code, `Modelfile`, small metric JSONs, scripts, corpus PDFs — not weights or vector indices.

### rag/docs/

All PDF corpora are versioned directly — no gitignore rules for `rag/docs/`. This applies to `es/`, `ca/`, `en/`, `libre/`, `en_ragbench_dev/`, `en_ragbench_eval/`, `en_ragbench_visual/`. The `rag/vector_db/` directory is fully ignored (auto-created at indexing time). Default chunk dumps from `export_fragments.py` go to `rag/show_fragments/exports/` (versioned). Any `*.txt` / `*.jsonl` placed directly under `rag/show_fragments/` (not under `exports/`) remains ignored. The `.gitkeep` in `rag/docs/en/` keeps the empty folder tracked.

### research/training-output/\<model\>/

Pattern: `research/training-output/<slug>/*` ignores everything; explicit exceptions keep the three small versioned files: `generate_reports.py`, `training_stats.json`, `evaluation_comparison.json`.

- **New model**: copy the 4-line block (one `/*` + three `!…`) and replace the slug.
- **Phi-4 alternate ranks**: optional `research/training-output/phi-4/16/` and `phi-4/64/` trees (exploration) each have their own 4-line `.gitignore` block alongside the main `phi-4/` block.
- Do not replace `/*` with extension globs — easy to accidentally exclude scripts.

### research/models/gguf-output/\<model\>/

Versioned: `Modelfile`, `README.md`, `LICENSE`, `CONVERSION.md` (where they exist). `.gguf` binaries excluded by `*.gguf` global rule. Host binaries on Hugging Face Hub and link from README — do not commit to GitHub.

`.gitignore` block numbering uses **12** and **13** at the end of the root file to avoid colliding with Section 9 of this file.
