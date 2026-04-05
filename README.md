<p align="center">
  <img src="logo-circular.png" alt="MonkeyGrab Logo" width="180" />
</p>

<h1 align="center">MonkeyGrab</h1>

<p align="center">
  <strong>A local, multilingual RAG system for academic PDF consultation with LoRA-fine-tuned LLMs</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-000000?style=for-the-badge" alt="Ollama">
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B35?style=for-the-badge" alt="ChromaDB">
  <img src="https://img.shields.io/badge/RAG-Hybrid-28A745?style=for-the-badge" alt="RAG">
</p>

---

## Overview

MonkeyGrab is a Retrieval-Augmented Generation (RAG) system that runs entirely on local hardware. It indexes PDF documents, retrieves relevant passages through a hybrid search pipeline, and generates grounded answers using a large language model served via Ollama. All data stays on the user's machine — no external API calls are made during normal operation.

The system was built as a Bachelor's Thesis (TFG) project at the Universitat Politecnica de Valencia (UPV). It targets researchers and students who need to query collections of academic documents in English, Spanish, and Catalan without sending their data to cloud services. Every model role (generator, embedder, reranker, auxiliary, vision) is configurable via environment variables, so MonkeyGrab adapts to whatever models are available on your hardware.

MonkeyGrab offers two interaction modes: **RAG mode** (document-grounded Q&A) and **CHAT mode** (general conversation), accessible through both a terminal CLI (Rich) and a web interface (Flask + React).

---

## Architecture

```
PDF Corpus (rag/pdfs/)
       |
       v
┌─────────────────────────────────────────────────────────┐
│                    INDEXING PIPELINE                    │
│                                                         │
│  Text extraction (pymupdf4llm / pypdf)                  │
│       → Hierarchical chunking (1500 chars, 350 overlap) │
│       → [Optional] Contextual retrieval (OLLAMA_CONTEXTUAL_MODEL)  │
│       → [Optional] Image description (OLLAMA_OCR_MODEL) │
│       → Embedding (OLLAMA_EMBED_MODEL)                  │
│       → ChromaDB persistent storage                     │
└─────────────────────────────────────────────────────────┘
       |
       v
┌─────────────────────────────────────────────────────────┐
│                   RETRIEVAL PIPELINE                    │
│                                                         │
│  User query                                             │
│       → [Optional] LLM query decomposition (3 variants) │
│       → Semantic search (ChromaDB, top-80)              │
│       → Keyword search ($contains, top-40)              │
│       → [Optional] Exhaustive term scan                 │
│       → RRF score fusion (55% semantic + 45% lexical)   │
│       → [Optional] Cross-encoder reranking (RERANKER_QUALITY)      │
│       → Top-6 fragments                                 │
│       → [Optional] Neighbor chunk expansion             │
│       → Context optimization (PDF artifact removal)     │
└─────────────────────────────────────────────────────────┘
       |
       v
┌─────────────────────────────────────────────────────────┐
│                      GENERATION                         │
│                                                         │
│  System prompt + context + question                     │
│       → OLLAMA_RAG_MODEL (via Ollama, streaming)        │
│       → Streaming response                              │
└─────────────────────────────────────────────────────────┘
```

---

## Models

All model roles are configured via environment variables, so you can use any Ollama-compatible model that fits your hardware. See the [Configuration](#configuration) section for the full list of variables.

| Role | Env variable | Description |
|------|-------------|-------------|
| **Generator** | `OLLAMA_RAG_MODEL` | Answers questions in RAG mode. Any instruction-tuned LLM served by Ollama. |
| **Chat** | `OLLAMA_CHAT_MODEL` | Answers questions in CHAT mode (no document context). |
| **Embedding** | `OLLAMA_EMBED_MODEL` | Converts text to vectors for indexing and retrieval. |
| **Auxiliary** | `OLLAMA_CONTEXTUAL_MODEL` | Query decomposition and contextual retrieval enrichment. |
| **Vision** | `OLLAMA_OCR_MODEL` | Describes images found in PDFs during indexing (optional). |
| **Reranker** | `RERANKER_QUALITY` | Cross-encoder loaded from HuggingFace. `quality` or `speed`. |

### Fine-tuning

Models fine-tuned with LoRA as part of the TFG research:

| Model | Parameters | LoRA r | LoRA alpha | Status |
|-------|-----------|--------|------------|--------|
| Qwen3-14B | 14B | 32 | 64 | Selected for production |
| Phi-4 | 14B | 32 | 64 | Pending cluster run |
| Llama-3.1-8B-Instruct | 8B | 32 | 64 | Published as open adapter |
| Gemma-3-12B-IT | 12B | 32 | 64 | Published (GGUF incompatible with Ollama) |

All adapters: dropout=0.05, 7 target modules (q/k/v/o_proj + gate/up/down_proj), trained on 5 datasets (Neural-Bridge RAG, Dolly QA, Aina-EN, Aina-ES, Aina-CA), ~16,000 training samples.

---

## Repository Structure

```
localOllamaRAG/
├── rag/
│   ├── chat_pdfs.py              # Main RAG engine (indexing, retrieval, generation)
│   ├── export_fragments.py       # Export ChromaDB chunks to TXT/JSONL for inspection
│   ├── requirements.txt          # Core dependencies
│   ├── debug_context_issues.md   # Analysis of minor context presentation issues
│   ├── debug_rag/                # Query debug dumps (runtime, not versioned)
│   └── cli/                      # Rich terminal interface
│       ├── app.py                # CLI main loop and command dispatch
│       ├── display.py            # Rich UI singleton (panels, tables, spinners)
│       ├── renderer.py           # Low-level ANSI rendering (legacy)
│       └── theme.py              # Color palette and visual identity
├── web/
│   ├── app.py                    # Flask backend (REST + SSE API)
│   ├── requirements.txt          # Web dependencies
│   └── zip/                      # React frontend build
├── scripts/
│   ├── training/
│   │   ├── train-qwen3.py        # Qwen3-14B LoRA fine-tuning (v10, production model)
│   │   ├── train-phi4.py         # Phi-4 LoRA fine-tuning (v1, pending cluster run)
│   │   ├── train-llama3.1.py     # Llama-3.1-8B LoRA fine-tuning
│   │   ├── train-gemma3.py       # Gemma-3-12B LoRA fine-tuning
│   │   └── plot_training.py      # Training curve visualization
│   ├── evaluation/
│   │   ├── evaluate_baselines.py # 7-model baseline benchmark (Token F1, ROUGE-L, BERTScore, CF)
│   │   ├── inspect_splits.py     # Audit dataset split sizes before/after filters
│   │   └── compute_std.py        # Mean +/- sigma analysis (historical)
│   ├── conversion/
│   │   └── merge_lora.py         # Merge LoRA adapter for GGUF export
│   └── tests/                    # Ollama streaming/thinking tests
├── evaluation/
│   ├── run_eval.py               # RAGAS evaluation of live RAG pipeline
│   ├── run_eval_ragbench.py      # RAGAS evaluation over RAGBench PDFs
│   └── requirements.txt          # Evaluation dependencies
├── training-output/
│   ├── qwen-3/                   # Qwen3 LoRA adapter + evaluation artifacts
│   ├── llama-3/                  # Llama LoRA adapter + evaluation artifacts
│   ├── gemma-3/                  # Gemma LoRA adapter + evaluation artifacts
│   └── baseline/                 # 7-model baseline results
│       ├── baseline_evaluation.json       # Aggregate results (all models/datasets/metrics)
│       ├── predictions_{model}.json       # Per-model predictions (7 files)
│       ├── generate_reports.py            # Generate Markdown tables + CSVs
│       └── reports/                       # Generated tables, CSVs, figures
├── docs/                         # Architecture diagrams and documentation
├── scripts/generate_diagram.py   # Architecture diagram generator (Kroki.io)
├── models/gguf-output/           # Quantized GGUF models (gitignored)
├── llama-bin/                    # llama.cpp binaries for quantization
├── CLAUDE.md                     # AI assistant context and contribution guide
└── README.md                     # This file
```

---

## Installation & Requirements

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally
- GPU recommended (CUDA) for reranking and fine-tuning; CPU works for inference

### Setup

```bash
git clone https://github.com/your-username/localOllamaRAG
cd localOllamaRAG

# Core RAG dependencies
pip install -r rag/requirements.txt

# Web interface (optional)
pip install -r web/requirements.txt

# Evaluation tools (optional)
pip install -r evaluation/requirements.txt

# Fine-tuning (optional — requires GPU)
pip install -r scripts/training/requirements.txt
```

### Ollama Models

MonkeyGrab uses Ollama for all LLM inference. You choose which models to run — pull whatever fits your hardware and set the corresponding environment variables (see [Configuration](#configuration)).

```bash
# Minimum required: a generator and an embedding model
ollama pull <your OLLAMA_RAG_MODEL>
ollama pull <your OLLAMA_EMBED_MODEL>

# Needed for query decomposition and contextual retrieval (optional pipeline stages)
ollama pull <your OLLAMA_CONTEXTUAL_MODEL>

# Needed for describing images found in PDFs (optional)
ollama pull <your OLLAMA_OCR_MODEL>    # must be a vision-language model
```

---

## Configuration

All pipeline behaviour is controlled via environment variables. Set them in your shell or a `.env` file in the project root.

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_RAG_MODEL` | `Qwen3-FineTuned` | Generator model for RAG mode |
| `OLLAMA_CHAT_MODEL` | `gemma3:4b` | Generator model for CHAT mode |
| `OLLAMA_EMBED_MODEL` | `embeddinggemma:latest` | Embedding model |
| `OLLAMA_CONTEXTUAL_MODEL` | `gemma3:4b` | Auxiliary model (query decomposition, contextual retrieval) |
| `OLLAMA_OCR_MODEL` | `qwen3-vl:8b` | Vision model for PDF image descriptions |
| `DOCS_FOLDER` | `rag/pdfs/` | Folder containing PDFs to index |
| `RERANKER_QUALITY` | `quality` | Reranker tier: `quality` (BAAI/bge-reranker-v2-m3) or `speed` (MiniLM) |
| `HF_TOKEN` | — | HuggingFace token (required for Gemma-3 fine-tuning) |
| `GOOGLE_API_KEY` | — | Gemini API key (required for RAGAS evaluation) |

---

## Usage

### CLI

```bash
cd rag
python chat_pdfs.py
```

Place PDF files in `rag/pdfs/` before starting. The system indexes them automatically on first launch.

**Commands:**
| Command | Description |
|---------|-------------|
| `/rag` | Switch to RAG mode (document Q&A) |
| `/chat` | Switch to CHAT mode (general conversation) |
| `/docs` | List indexed documents |
| `/temas` | Show document topics summary |
| `/stats` | Display database statistics |
| `/reindex` | Re-index all documents |
| `/help` | Show help |
| `/salir` | Exit |

### Web Interface

```bash
python web/app.py
```

Opens at `http://localhost:5000`. Supports file upload, streaming responses, and all pipeline settings via the UI.

---

## Fine-tuning

```bash
# Train Qwen3-14B (requires GPU with ~24GB VRAM) — production model, v10
python scripts/training/train-qwen3.py

# Train Phi-4 (14B, same protocol as Qwen3 v10, pending cluster run)
python scripts/training/train-phi4.py

# Train Llama-3.1-8B
python scripts/training/train-llama3.1.py

# Train Gemma-3-12B
python scripts/training/train-gemma3.py

# Merge adapter for GGUF export
python scripts/conversion/merge_lora.py --model qwen-3

# Visualize training curves
python scripts/training/plot_training.py --model qwen-3
```

Training uses: lr=5e-5, cosine scheduler, warmup=0.05, batch=1x16 gradient accumulation, BF16, AdamW 8-bit, max_seq=4096, 3 epochs with early stopping.

---

## Evaluation

### Baseline Benchmark

```bash
# Run full benchmark (7 models × 5 datasets × dev/test × with/without context)
python scripts/evaluation/evaluate_baselines.py

# Generate Markdown tables + CSVs from results
python training-output/baseline/generate_reports.py

# Audit dataset split sizes before/after filters
python scripts/evaluation/inspect_splits.py
```

Evaluates 7 base models (Llama-3.1-8B, Qwen3-14B, Qwen3.5-14B, Qwen2.5-14B, Gemma-3-12B, Phi-4, and fine-tuned Qwen3-14B) across 5 datasets (Aina-EN, Aina-ES, Aina-CA, Neural-Bridge RAG, Dolly QA), with and without context. Metrics: Token F1, ROUGE-L F1, BERTScore F1, Context Faithfulness.

### RAGAS (live pipeline evaluation)

```bash
python evaluation/run_eval.py
```

Metrics: Context Recall, Factual Correctness, Context Precision, Faithfulness, Response Relevancy. Uses Gemini 2.0 Flash as judge LLM.

### ChromaDB chunk inspection

```bash
# Export indexed chunks to text for manual inspection
python rag/export_fragments.py              # both stores (mi_vector_db + ragbench_vector_db)
python rag/export_fragments.py --mi-only    # own PDFs only
```

---

## Contributing

For contributors and AI assistants working on this codebase, see [`CLAUDE.md`](CLAUDE.md) for:

- Code and documentation style conventions
- Pipeline architecture details
- Known technical debt and design decisions
- Environment setup and build commands

**Code style at a glance:**
- **Module docstrings**: English, triple-quoted, with short description + longer explanation + Usage + Dependencies
- **Section separators**: `# ─────` lines with `# SECTION NAME` in caps for logical code groupings
- **Function docstrings**: Google-style with Args, Returns, and Raises sections
- **Inline comments**: English, only for non-obvious logic

---

## Status

This project is under active development as part of a Bachelor's Thesis (TFG) at UPV.

**Implemented:**
- Full RAG pipeline (indexing, hybrid retrieval, reranking, generation)
- CLI interface (Rich) and web interface (Flask + React)
- LoRA fine-tuning pipeline for 4 models (Qwen3-14B, Phi-4, Llama-3.1-8B, Gemma-3-12B)
- Token F1 + ROUGE-L F1 + BERTScore + Context Faithfulness evaluation
- 7-model baseline benchmark across 5 datasets (dev + test splits, with/without context)
- RAGAS evaluation integration
- Per-sample metric logging and statistical analysis
- PDF image indexing with vision-language model

**Known limitations:**
- Gemma-3-12B adapter cannot be deployed via Ollama (GGUF incompatibility)
- Phi-4 adapter pending cluster run
- Vector graphics in PDFs (e.g. SVG-embedded figures) are not extracted by `page.get_images()` and are not indexed
- RECOMP synthesis stage is disabled by default (opt-in via `USAR_RECOMP_SYNTHESIS = True`)

---

*Bachelor's Thesis — Universitat Politècnica de València (UPV)*
