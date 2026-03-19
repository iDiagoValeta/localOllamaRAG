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

MonkeyGrab is a Retrieval-Augmented Generation (RAG) system that runs entirely on local hardware. It indexes PDF documents, retrieves relevant passages through a hybrid search pipeline, and generates grounded answers using a fine-tuned large language model served via Ollama. All data stays on the user's machine — no external API calls are made during normal operation.

The system was built as a Bachelor's Thesis (TFG) project at the Universitat Politecnica de Valencia (UPV). It targets researchers and students who need to query collections of academic documents in English, Spanish, and Catalan without sending their data to cloud services. The generator model (Qwen3-14B) was fine-tuned with LoRA on three RAG-specific datasets to improve context adherence and response quality compared to the base model.

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
│       → [Optional] Contextual retrieval (gemma3:4b)     │
│       → [Optional] Image description (llama3.2-vl:11b)  │
│       → Embedding (embeddinggemma, 768d)                │
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
│       → [Optional] Cross-encoder reranking (BAAI/bge)   │
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
│       → Qwen3-14B (Q4_K_M via Ollama, temp=0.15)        │
│       → Streaming response                              │
└─────────────────────────────────────────────────────────┘
```

---

## Models

| Role | Model | Details |
|------|-------|---------|
| **Generator** | Qwen3-14B-FineTuned | **Generation.** Q4_K_M (llama.cpp); served by Ollama. |
| **Embedding** | embeddinggemma | **Embeddings.** Gemma 3, 307M params, 768-d, BF16. |
| **Reranker** | BAAI/bge-reranker-v2-m3 | **Reranking.** ~200M cross-encoder. Faster alt: ms-marco-MiniLM-L-6-v2. |
| **Auxiliary** | gemma3:4b | **Orchestration.** Query decomposition, contextual retrieval, RECOMP synthesis. |
| **Vision (image chunks)** | llama3.2-vision:11b | **Figure captioning.** Llama 3.2 Vision, 11B params, multimodal, Ollama. |

### Fine-tuning

Three base models were fine-tuned with LoRA and evaluated:

| Model | Parameters | LoRA r | LoRA alpha | Status |
|-------|-----------|--------|------------|--------|
| Qwen3-14B | 14B | 64 | 128 | Selected for production |
| Llama-3.1-8B-Instruct | 8B | 32 | 64 | Published as open adapter |
| Gemma-3-12B-IT | 12B | 32 | 64 | Published (GGUF incompatible with Ollama) |

All adapters: dropout=0.05, 7 target modules (q/k/v/o_proj + gate/up/down_proj), trained on 3 datasets (Neural-Bridge RAG, Dolly QA, Aina RAG Multilingual) totaling ~20,500 samples.

---

## Repository Structure

```
localOllamaRAG/
├── rag/
│   ├── chat_pdfs.py              # Main RAG engine (indexing, retrieval, generation)
│   ├── requirements.txt          # Core dependencies
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
│   │   ├── train-qwen3.py        # Qwen3-14B LoRA fine-tuning
│   │   ├── train-llama3.1.py     # Llama-3.1-8B LoRA fine-tuning
│   │   ├── train-gemma3.py       # Gemma-3-12B LoRA fine-tuning
│   │   └── plot_training.py      # Training curve visualization
│   ├── evaluation/
│   │   ├── eval_bertscore.py     # BERTScore evaluation (per-sample + summary)
│   │   ├── plot_bertscore.py     # BERTScore visualization
│   │   ├── plot_baseline_results.py  # Baseline model comparison plots
│   │   └── plot_comparison.py    # Base vs fine-tuned comparison plots
│   ├── conversion/
│   │   └── merge_lora.py         # Merge LoRA adapter for GGUF export
│   └── tests/                    # Ollama streaming/thinking tests
├── evaluation/
│   ├── run_eval.py               # RAGAS evaluation of RAG pipeline
│   └── requirements.txt          # Evaluation dependencies
├── training-output/
│   ├── qwen-3/                   # Qwen3 adapter + evaluation artifacts
│   ├── llama-3/                  # Llama adapter + evaluation artifacts
│   ├── gemma-3/                  # Gemma adapter + evaluation artifacts
│   ├── bertscore/                # BERTScore summary and per-sample CSVs
│   └── baseline/
│       └── evaluate_baselines.py # 6-model baseline benchmark
├── compute_std.py                # Mean +/- sigma statistical analysis
├── models/gguf-output/           # Quantized GGUF models (gitignored)
├── AUDIT.md                      # Repository audit
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
git clone <repository-url>
cd localOllamaRAG

# Core RAG dependencies
pip install -r rag/requirements.txt

# Web interface (optional)
pip install -r web/requirements.txt

# Evaluation tools (optional)
pip install -r evaluation/requirements.txt

# Fine-tuning (optional — requires GPU)
pip install torch transformers peft datasets tqdm
```

### Ollama Models

```bash
# Required for RAG mode
ollama pull Qwen3-FineTuned        # Or your fine-tuned model name

# Required for embedding
ollama pull embeddinggemma

# Required for auxiliary tasks (query decomposition, contextual retrieval)
ollama pull gemma3:4b

# Required for optional image indexing (PDF figures → text descriptions)
ollama pull llama3.2-vision:11b
```

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
# Train Qwen3-14B (requires GPU with ~24GB VRAM)
python scripts/training/train-qwen3.py

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

### BERTScore + Token F1 + Faithfulness

```bash
python scripts/evaluation/eval_bertscore.py                 # All 3 models
python scripts/evaluation/eval_bertscore.py --model qwen-3  # Single model
python scripts/evaluation/plot_bertscore.py                  # Generate plots
python compute_std.py                                        # Compute mean +/- std
```

### RAGAS (live pipeline evaluation)

```bash
python evaluation/run_eval.py
```

Metrics: Context Recall, Factual Correctness, Context Precision, Faithfulness, Response Relevancy. Uses Gemini 2.0 Flash as judge LLM.

### Baseline Benchmark

```bash
python training-output/baseline/evaluate_baselines.py
python scripts/evaluation/plot_baseline_results.py
```

Evaluates 6 base models (Llama-3.1-8B, Qwen3/3.5/2.5-14B, Gemma-3-12B, Phi-4) across 3 datasets with/without context.

---

## Code Documentation Style

All Python files follow a consistent documentation convention:

- **Module docstrings**: English, triple-quoted, with short description + longer explanation + Usage + Dependencies
- **Section separators**: `# ─────` lines with `# SECTION NAME` in caps for logical code groupings
- **Function docstrings**: Google-style with Args, Returns, and Raises sections
- **Inline comments**: English, only for non-obvious logic; trivially obvious comments are omitted

---

## Status

This project is under active development as part of a Bachelor's Thesis (TFG) at UPV.

**Implemented:**
- Full RAG pipeline (indexing, hybrid retrieval, reranking, generation)
- CLI interface (Rich) and web interface (Flask + React)
- LoRA fine-tuning pipeline for 3 models
- BERTScore + Token F1 + Faithfulness evaluation
- RAGAS evaluation integration
- Per-sample metric logging and statistical analysis

**Known limitations:**
- Gemma-3-12B adapter cannot be deployed via Ollama (GGUF incompatibility)
- Per-sample evaluation data requires re-running `eval_bertscore.py` to generate
- RECOMP synthesis stage is disabled by default
