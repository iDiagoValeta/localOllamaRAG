<p align="center">
  <img src="logo-circular.png" alt="MonkeyGrab Logo" width="180" />
</p>

<h1 align="center">MonkeyGrab</h1>

<p align="center">
  <strong>A local, multilingual RAG system for querying PDF documents with open language models</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-000000?style=for-the-badge" alt="Ollama">
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B35?style=for-the-badge" alt="ChromaDB">
  <img src="https://img.shields.io/badge/RAG-Hybrid-28A745?style=for-the-badge" alt="RAG">
</p>

---

> **Última actualización:** 2026-04-21

## What is MonkeyGrab?

MonkeyGrab is a Retrieval-Augmented Generation (RAG) system that runs entirely on your own hardware. You point it at a folder of PDF documents, and it lets you ask questions about them in natural language — receiving answers that are grounded in the actual content of those files.

No data leaves your machine. All inference, indexing and retrieval happens locally through [Ollama](https://ollama.ai/). MonkeyGrab is designed for researchers and students who need to query academic documents in English, Spanish or Catalan without sending their data to external services.

The system works with any instruction-tuned language model available in Ollama. You configure which models to use via environment variables, so it adapts to whatever hardware you have available.

This project was developed as a Bachelor's thesis (TFG) for the Grado en Ingeniería Informática at ETSINF, Universitat Politècnica de València (UPV), by Ignacio Diago Valeta, tutored by Adrià Giménez Pastor (2025–2026). It combines a functional RAG production system with a research layer for LoRA fine-tuning and evaluation of open language models.

## Video

This video shows a sample query I ran on my own laptop against a database containing five Wikipedia articles, indexed using my preferred settings. Every user can fully customise their experience; my models and settings are unique, so I encourage you to discover your own favourites!

https://github.com/user-attachments/assets/cc36fc27-e4b9-49ac-a1f1-131f6e6afe4f

---

## Features

- Fully local — no external API calls during normal operation
- Multilingual support: English, Spanish, Catalan
- Hybrid retrieval: semantic search + keyword search + optional cross-encoder reranking
- Two interaction modes: document Q&A (RAG) and general conversation (CHAT)
- Two interfaces: terminal CLI (Rich) and web interface (Flask + React)
- Optional indexing of images and figures found in PDFs, described by a vision model
- All model roles configurable via environment variables
- Debug output included with every query: retrieved fragments, relevance scores, sub-queries generated

---

## How it works

```
PDF corpus  (rag/docs/es/ by default; set DOCS_FOLDER or use ca/en)
      |
      v
  INDEXING
      Text extraction    pymupdf4llm  /  pypdf (fallback)
      Chunking           configurable size and overlap
      Enrichment         [optional]  OLLAMA_CONTEXTUAL_MODEL
      Image description  [optional]  OLLAMA_OCR_MODEL
      Embedding                       OLLAMA_EMBED_MODEL
      Storage            ChromaDB  (rag/vector_db/<folder>_<embed_slug>/)
      |
      v
  RETRIEVAL
      Query decomposition  [optional]  sub-queries via OLLAMA_CHAT_MODEL
      Semantic search                  OLLAMA_EMBED_MODEL  (top-80)
      Keyword search       [optional]  text filter         (top-40)
      Exhaustive scan      [optional]  critical terms
      Score fusion                     RRF  55% semantic + 45% lexical
      Reranking            [optional]  cross-encoder  RERANKER_QUALITY
      Context selection               top-8 fragments
      Chunk expansion      [optional]  adjacent chunks
      |
      v
  GENERATION
      Prompt assembly   system prompt + context + question
      Response          OLLAMA_RAG_MODEL  —  streaming via Ollama
```

---

## Requirements

- Python 3.10 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- A GPU with CUDA is recommended for reranking; CPU works for inference but is slower

---

## Installation

```bash
git clone https://github.com/your-username/localOllamaRAG
cd localOllamaRAG

# Core RAG system (required)
pip install -r rag/requirements.txt

# Web interface (optional)
pip install -r web/requirements.txt

# RAGAS evaluation (optional; requires GOOGLE_API_KEY)
pip install -r evaluation/requirements.txt
```

### Pull your models

MonkeyGrab needs at minimum a generator model and an embedding model running in Ollama. Pull whichever models fit your hardware and assign them via environment variables:

```bash
# Required
ollama pull <your OLLAMA_RAG_MODEL>
ollama pull <your OLLAMA_EMBED_MODEL>

# Optional pipeline stages
ollama pull <your OLLAMA_CHAT_MODEL>         # chat mode and query decomposition (sub-queries)
ollama pull <your OLLAMA_CONTEXTUAL_MODEL>   # contextual chunk enrichment during indexing
ollama pull <your OLLAMA_RECOMP_MODEL>       # context synthesis before generation
ollama pull <your OLLAMA_OCR_MODEL>          # must be a vision-language model
```

### Model weights (GGUF)

Large **`.gguf`** files are **not** committed to this repository (size and clone cost). The repo keeps **`Modelfile`** files under `models/gguf-output/<model>/` plus conversion scripts in `scripts/conversion/`. Build or quantize locally, or download weights from **Hugging Face Hub** (or another object store) and point Ollama at the file path you use. Document any public model URL in your thesis or deployment notes for reproducibility.

**Qwen3-14B RAG (LoRA, Q4_K_M GGUF):** [nadiva1243/qwen3RAG](https://huggingface.co/nadiva1243/qwen3RAG) — model card and weights on Hugging Face.

**Phi-4 RAG (LoRA, Q4_K_M GGUF):** [nadiva1243/phi4RAG](https://huggingface.co/nadiva1243/phi4RAG) — model card and weights on Hugging Face.

After building locally, you can remove `models/merged-model/<slug>/` and the intermediate `*-f16.gguf` to save disk; keep the `*-Q4_K_M.gguf` you use with Ollama.

---

## Configuration

All pipeline behaviour is controlled via environment variables. Set them in your shell or in a `.env` file at the project root.

| Variable | Description |
|----------|-------------|
| `OLLAMA_RAG_MODEL` | Generator model for RAG mode (document Q&A) |
| `OLLAMA_CHAT_MODEL` | Generator model for CHAT mode and query decomposition (sub-queries) |
| `OLLAMA_EMBED_MODEL` | Embedding model used for indexing and retrieval |
| `OLLAMA_CONTEXTUAL_MODEL` | Auxiliary model for contextual chunk enrichment during indexing (`USAR_CONTEXTUAL_RETRIEVAL`) |
| `OLLAMA_RECOMP_MODEL` | Model used to synthesise/compress retrieved fragments before generation (`USAR_RECOMP_SYNTHESIS`) |
| `OLLAMA_OCR_MODEL` | Vision model for describing images found in PDFs (`USAR_EMBEDDINGS_IMAGEN`) |
| `DOCS_FOLDER` | Path to the folder containing PDFs to index (default: `rag/docs/es/`) |
| `RERANKER_QUALITY` | Cross-encoder reranker tier: `quality` (BAAI/bge) or `speed` (MiniLM) |
| `USAR_RECOMP_SYNTHESIS` | Enables/disables RECOMP context synthesis (`true`/`false`, default: `true`) |

> **Note on ChromaDB paths**: indexes live under `rag/vector_db/<folder>_<embed_slug>/`, where `<folder>` is the basename of `DOCS_FOLDER` (e.g. `es`, `ca`, `en`) and `<embed_slug>` comes from `OLLAMA_EMBED_MODEL`. Changing the embedding model or the docs folder selects a different path — re-index when you intentionally switch either.

---

## Usage

### Terminal CLI

Place your PDF files under **`rag/docs/es/`** by default (Spanish corpus). Catalan and English corpora use **`rag/docs/ca/`** and **`rag/docs/en/`** respectively; set `DOCS_FOLDER` or use the evaluation presets in `evaluation/run_eval.py` to point at the right folder. Each language folder is kept in Git with `.gitkeep` only; PDF content stays local. The ChromaDB directory **`rag/vector_db/`** is gitignored and created when you index.

```bash
cd rag
python chat_pdfs.py
```

| Command | Description |
|---------|-------------|
| `/rag` | Switch to RAG mode — answers are grounded in your documents |
| `/chat` | Switch to CHAT mode — general conversation without document context |
| `/docs` | List indexed documents |
| `/temas` | Show a topic summary per document |
| `/stats` | Show vector database statistics |
| `/reindex` | Delete the current index and re-index all documents |
| `/limpiar` or `/clear` | Clear the conversation history |
| `/ayuda` or `/help` | Show all available commands |
| `/salir` or `/exit` | Exit and save history |

### Web Interface

```bash
python web/app.py
```

Opens at `http://localhost:5000`. Supports document upload, streaming responses and access to all pipeline settings through the UI.

### LoRA train/eval reports (per model)

Each fine-tuned model folder under `training-output/` includes the same `generate_reports.py` (invoked from that folder so defaults point at the right artifacts). After training (e.g. `scripts/training/train-qwen3.py`, `train-phi4.py`, `train-gemma3.py`), regenerate tables and figures under `training-output/<model>/plots/`:

```bash
python training-output/qwen-3/generate_reports.py
python training-output/phi-4/generate_reports.py
python training-output/gemma-3/generate_reports.py
```

Optional flags: `--model-dir`, `--eval-input`, `--train-input`, `--plots-dir`, `--no-figures` (see the script docstring).

Output layout is the same for every model:
- `plots/train/` — training curves (`loss`, `learning_rate`, `grad_norm`) and CSV summaries from `log_history`
- `plots/eval/` — per-metric CSV tables, markdown report tables, and comparison figures (`base` vs `adapted`)

### RAGAS evaluation (`evaluation/run_eval.py`)

Requires `pip install -r evaluation/requirements.txt` and a **`GOOGLE_API_KEY`** in `.env` (Gemini as judge LLM).

Single-corpus runs (loads bundled datasets under `evaluation/datasets/` and the matching `rag/docs/` folder):

```bash
python evaluation/run_eval.py single --corpus es
python evaluation/run_eval.py single --corpus ca
python evaluation/run_eval.py single --corpus mix
```

For English, supply your own dataset path if you do not have `evaluation/datasets/dataset_eval_en.json`, or use the **RAGBench** flow:

```bash
python evaluation/run_eval.py ragbench --n-papers 10 --max-q 5   # PDFs under rag/docs/en
```

RAGBench prepared datasets enable a documented evaluation-only reranker fallback:
the reranker still orders candidates, but if every candidate falls below the
interactive relevance threshold, the best retrieved candidates are kept so the
answer generator can run. Other datasets keep the normal hard reranker filter.

Ablation-style comparison (multiple pipeline variants, shared index; optional `--reindex` before the first variant):

```bash
python evaluation/run_eval.py compare --corpus ca --label mi_eval --reindex
python evaluation/run_eval.py list-variants
```

Legacy alias for Catalan-only presets: `python evaluation/run_eval.py --catalan` (same as `single --corpus ca`).

Artifacts go under **`evaluation/scores/`** (CSVs), **`evaluation/debug/`** (JSON traces), and **`evaluation/debug/checkpoints/`** (resume state). Comparison runs additionally use `evaluation/scores/comparison_runs/<label>/` and `evaluation/debug/comparison_runs/<label>/`. See `evaluation/EVALUACIONES_PIPELINE.md` for corpus presets and variant names.

After a comparison run, **`evaluation/aggregate_comparison_by_conjunto.py`** aggregates per-variant debug JSONs with the question dataset and writes subset means (e.g. by `source_type` or `language`) to JSON/CSV; use `--etiquetas-es` for Spanish metric labels in the report. Details: `evaluation/EVALUACIONES_PIPELINE.md` (section *Agregación por conjunto*).

---

## Repository structure

```
localOllamaRAG/
├── generate_diagram.py           # Architecture diagram (Kroki.io)
├── rag/
│   ├── chat_pdfs.py              # Main RAG engine (indexing, retrieval, generation)
│   ├── show_fragments/
│   │   └── export_fragments.py   # Export ChromaDB chunks to TXT/JSONL for debug
│   ├── docs/
│   │   ├── es/                   # Spanish PDF corpus (default DOCS_FOLDER; .gitkeep only in Git)
│   │   ├── ca/                   # Catalan PDF corpus
│   │   └── en/                   # English corpus / RAGBench PDFs
│   ├── vector_db/                # ChromaDB per language + embedding slug (gitignored at runtime)
│   ├── debug_rag/                # Optional per-query debug dumps (gitignored)
│   ├── historial_chat.json       # CHAT mode history (gitignored)
│   ├── cli/                      # Rich terminal UI (MonkeyGrabCLI)
│   └── requirements.txt
├── web/
│   ├── app.py                    # Flask backend (REST + SSE); serves React build
│   └── zip/                      # React source (src/) + Vite config; production build → dist/
├── scripts/
│   ├── hf_upload_model_cards.py  # Hugging Face model cards / optional GGUF upload helper
│   ├── training/                 # LoRA fine-tuning (Qwen3, Phi-4, Gemma-3)
│   ├── evaluation/               # Baseline benchmark + split inspection + SLURM helpers
│   ├── conversion/               # LoRA merge, GGUF build, quantization notes
│   └── tests/                    # Ollama / pipeline smoke tests
├── evaluation/
│   ├── datasets/                 # Question datasets (ES, CA, mix; add EN as needed)
│   ├── scores/                   # RAGAS CSV outputs (keep large runs out of Git manually if needed)
│   ├── debug/                    # Debug JSON + resumable checkpoints
│   ├── run_eval.py               # RAGAS entrypoint: single | compare | ragbench
│   ├── aggregate_comparison_by_conjunto.py  # Post-compare: subset means from debug JSON + dataset
│   ├── EVALUACIONES_PIPELINE.md  # Detailed eval presets, ablation variants, aggregation notes
│   └── requirements.txt
├── models/
│   ├── merged-model/             # Dense HF weights after LoRA merge (gitignored)
│   └── gguf-output/              # Modelfile + docs per model (GGUF binaries gitignored)
├── training-output/
│   ├── qwen-3/
│   ├── phi-4/                    # Other LoRA ranks may live under phi-4/<rank>/
│   ├── gemma-3/
│   └── baseline/                 # Seven-model baseline benchmark artifacts
├── docs/                         # Architecture assets, methodology notes (thesis)
├── README.md
└── CLAUDE.md                     # Contributor / internal conventions
```

---

## Known limitations

- Vector graphics embedded in PDFs (SVG-based figures) are not extracted during indexing and will not be retrievable.
- The RAGAS evaluation pipeline requires a `GOOGLE_API_KEY` and is therefore not fully local.

---

*Bachelor's thesis (TFG) — Grado en Ingeniería Informática, ETSINF, Universitat Politècnica de València. Author: Ignacio Diago Valeta. Tutor: Adrià Giménez Pastor. 2025–2026.*
