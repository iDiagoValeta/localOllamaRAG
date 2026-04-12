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

## What is MonkeyGrab?

MonkeyGrab is a Retrieval-Augmented Generation (RAG) system that runs entirely on your own hardware. You point it at a folder of PDF documents, and it lets you ask questions about them in natural language — receiving answers that are grounded in the actual content of those files.

No data leaves your machine. All inference, indexing and retrieval happens locally through [Ollama](https://ollama.ai/). MonkeyGrab is designed for researchers and students who need to query academic documents in English, Spanish or Catalan without sending their data to external services.

The system works with any instruction-tuned language model available in Ollama. You configure which models to use via environment variables, so it adapts to whatever hardware you have available.

This project was developed as a Bachelor's thesis (TFG) at the Universitat Politècnica de València (UPV), combining a functional RAG production system with a research layer for LoRA fine-tuning and evaluation of open language models.

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
PDF corpus  (rag/pdfs/)
      |
      v
  INDEXING
      Text extraction    pymupdf4llm  /  pypdf (fallback)
      Chunking           configurable size and overlap
      Enrichment         [optional]  OLLAMA_CONTEXTUAL_MODEL
      Image description  [optional]  OLLAMA_OCR_MODEL
      Embedding                       OLLAMA_EMBED_MODEL
      Storage            ChromaDB  (rag/mi_vector_db/)
      |
      v
  RETRIEVAL
      Query decomposition  [optional]  sub-queries via OLLAMA_CHAT_MODEL
      Semantic search                  OLLAMA_EMBED_MODEL  (top-80)
      Keyword search       [optional]  text filter         (top-40)
      Exhaustive scan      [optional]  critical terms
      Score fusion                     RRF  55% semantic + 45% lexical
      Reranking            [optional]  cross-encoder  RERANKER_QUALITY
      Context selection               top-6 fragments
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
```

### Pull your models

MonkeyGrab needs at minimum a generator model and an embedding model running in Ollama. Pull whichever models fit your hardware and assign them via environment variables:

```bash
# Required
ollama pull <your OLLAMA_RAG_MODEL>
ollama pull <your OLLAMA_EMBED_MODEL>

# Optional pipeline stages
ollama pull <your OLLAMA_CONTEXTUAL_MODEL>   # query decomposition and contextual retrieval
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
| `OLLAMA_CHAT_MODEL` | Generator model for CHAT mode (general conversation) |
| `OLLAMA_EMBED_MODEL` | Embedding model used for indexing and retrieval |
| `OLLAMA_CONTEXTUAL_MODEL` | Auxiliary model for query decomposition and contextual retrieval |
| `OLLAMA_OCR_MODEL` | Vision model for describing images found in PDFs |
| `DOCS_FOLDER` | Path to the folder containing PDFs to index (default: `rag/pdfs/`) |
| `RERANKER_QUALITY` | Cross-encoder reranker tier: `quality` (higher accuracy) or `speed` (lower latency) |

> **Note on ChromaDB paths**: the vector database path includes a slug derived from `OLLAMA_EMBED_MODEL`. If you change the embedding model, the existing index is no longer valid and you will need to re-index your documents.

---

## Usage

### Terminal CLI

Place your PDF files in `rag/pdfs/` (the folder exists after clone via `.gitkeep`). The system indexes them automatically on first launch. RAGBench evaluation uses `rag/ragbench_pdfs/` and `rag/ragbench_vector_db/` in the same way.

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

---

## Repository structure

```
localOllamaRAG/
├── rag/
│   ├── chat_pdfs.py              # Main RAG engine (indexing, retrieval, generation)
│   ├── pdfs/                     # Your PDFs (only .gitkeep in Git; content ignored)
│   ├── mi_vector_db/             # ChromaDB production index (same .gitkeep pattern)
│   ├── ragbench_pdfs/            # RAGBench PDFs (same pattern; see run_eval_ragbench.py)
│   ├── ragbench_vector_db/       # ChromaDB for RAGBench eval (same pattern)
│   └── cli/                      # Rich terminal interface
├── web/
│   ├── app.py                    # Flask backend (REST + SSE)
│   └── zip/dist/                 # React frontend build
├── scripts/
│   ├── training/                 # LoRA fine-tuning scripts
│   ├── evaluation/               # Baseline benchmark and dataset inspection tools
│   └── conversion/               # LoRA adapter merge and GGUF quantization
├── evaluation/
│   └── run_eval.py               # RAGAS evaluation of the live pipeline
├── models/
│   ├── merged-model/             # Dense HF weights after LoRA merge (gitignored; safe to delete after Q4 GGUF)
│   └── gguf-output/              # GGUF + Modelfile per model (only small files tracked in Git)
├── training-output/
│   ├── qwen-3/                   # Qwen3 LoRA artifacts + generate_reports.py → plots/
│   ├── phi-4/                    # Phi-4 LoRA artifacts + generate_reports.py → plots/
│   ├── gemma-3/                  # Gemma-3 LoRA artifacts + generate_reports.py → plots/
│   └── baseline/                 # Baseline benchmark results (JSON, reports)
├── docs/                         # Architecture diagrams
└── CLAUDE.md                     # Internal development guide
```

---

## Known limitations

- Vector graphics embedded in PDFs (SVG-based figures) are not extracted during indexing and will not be retrievable.
- The RAGAS evaluation pipeline requires a `GOOGLE_API_KEY` and is therefore not fully local.

---

*Bachelor's thesis — Universitat Politècnica de València (UPV)*
