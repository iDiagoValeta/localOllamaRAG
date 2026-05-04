<p align="center">
  <img src="logo-circular.png" alt="MonkeyGrab Logo" width="180" />
</p>

<h1 align="center">MonkeyGrab</h1>

<p align="center">
  <strong>A local, multilingual RAG system for querying PDF documents with open language models.</strong><br/>
  All indexing, retrieval and generation runs on your own hardware — no data leaves your machine.
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://ollama.com/"><img src="https://img.shields.io/badge/Ollama-Local%20LLM-000000?style=for-the-badge" alt="Ollama"></a>
  <a href="https://www.trychroma.com/"><img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B35?style=for-the-badge" alt="ChromaDB"></a>
  <img src="https://img.shields.io/badge/RAG-Hybrid-28A745?style=for-the-badge" alt="RAG">
  <img src="https://img.shields.io/badge/Languages-ES%20%7C%20CA%20%7C%20EN-5C6BC0?style=for-the-badge" alt="Multilingual">
  <img src="https://img.shields.io/badge/Privacy-100%25%20Local-orange?style=for-the-badge" alt="Local">
</p>

<p align="center">
  <a href="#1-overview">Overview</a> ·
  <a href="#2-demo">Demo</a> ·
  <a href="#3-getting-started">Getting started</a> ·
  <a href="#4-configuration">Configuration</a> ·
  <a href="#5-usage">Usage</a>
</p>

---

## 1. Overview

MonkeyGrab lets you ask questions about your PDF documents in natural language. Point it at a folder of PDFs, start the CLI or the web interface, and get answers grounded in the actual content of those files — no data sent to any cloud.

| | |
|---|---|
| **Local-first** | All indexing, retrieval and generation runs on your hardware. No API keys required for the core pipeline. |
| **Any model** | Works with any instruction-tuned model in [Ollama](https://ollama.com/) — `llama3.2`, `mistral`, `gemma4`, `qwen3`, etc. |
| **Hybrid retrieval** | Semantic search + keyword search fused with RRF, followed by optional cross-encoder reranking. |
| **Multilingual** | Spanish, Catalan and English out of the box. Active corpus selected via environment variable. |
| **Two interfaces** | Rich-based terminal CLI and a Flask + React web UI with streaming responses. |
| **Image-aware** | Optionally describes raster images and figures in PDFs with a vision model, making visual content retrievable. |

---

## 2. Demo

**Web interface — querying a local document corpus**

https://github.com/user-attachments/assets/22582283-1b28-4054-a341-3aa1cbdc5057

**CLI — querying a local document corpus**

https://github.com/user-attachments/assets/4b8a84ca-422f-44a6-a0d5-a8078fa5e17a

**LaTeX rendering — math formulas rendered natively in the web UI**

<img width="1277" height="674" alt="mathRender" src="https://github.com/user-attachments/assets/0a9dbcdf-fe91-4992-af82-e4e945fbf766" />

---

## 3. Getting started

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com/) running locally.

### Install

```bash
git clone https://github.com/iDiagoValeta/localOllamaRAG
cd localOllamaRAG

pip install -r rag/requirements.txt        # core RAG pipeline (required)
pip install -r rag/web/requirements.txt    # web interface (optional)
```

### Pull models

You need at minimum a generator and an embedding model:

```bash
ollama pull <OLLAMA_RAG_MODEL>      # document Q&A generator (required)
ollama pull <OLLAMA_EMBED_MODEL>    # indexing and retrieval (required)

ollama pull <OLLAMA_CHAT_MODEL>     # chat mode + query decomposition (optional)
ollama pull <OLLAMA_RECOMP_MODEL>   # context synthesis before generation (optional)
ollama pull <OLLAMA_OCR_MODEL>      # vision model for PDF images (optional)
```

**Fine-tuned weights trained specifically for RAG (recommended):**

- **Qwen3-14B RAG** — [nadiva1243/qwen3RAG](https://huggingface.co/nadiva1243/qwen3RAG)
- **Phi-4 RAG** — [nadiva1243/phi4RAG](https://huggingface.co/nadiva1243/phi4RAG)

### Run

Drop your PDFs into `rag/docs/libre/` and start:

```bash
# CLI (Spanish UI by default)
python rag/chat_pdfs.py

# CLI in English
MONKEYGRAB_LANG=en python rag/chat_pdfs.py          # bash/zsh
$env:MONKEYGRAB_LANG = "en"; python rag/chat_pdfs.py # PowerShell

# Web interface → http://localhost:5000
python rag/web/app.py
```

The vector index is created automatically in `rag/vector_db/` on first run.

---

## 4. Configuration

Set these in your shell or in a `.env` file at the project root.

| Variable | Description |
|----------|-------------|
| `OLLAMA_RAG_MODEL` | Generator model for RAG mode |
| `OLLAMA_CHAT_MODEL` | Generator for chat mode and query decomposition |
| `OLLAMA_EMBED_MODEL` | Embedding model for indexing and retrieval |
| `OLLAMA_RECOMP_MODEL` | Model for context synthesis before generation |
| `OLLAMA_OCR_MODEL` | Vision model for PDF image descriptions |
| `OLLAMA_CONTEXTUAL_MODEL` | Auxiliary model for contextual chunk enrichment at indexing |
| `DOCS_FOLDER` | PDF folder to index (default: `rag/docs/libre/`) |
| `RERANKER_QUALITY` | Cross-encoder tier: `quality` (BAAI/bge) or `speed` (MiniLM) |
| `MONKEYGRAB_LANG` | CLI language: `es` (default) or `en` |

> Changing `DOCS_FOLDER` or `OLLAMA_EMBED_MODEL` selects a different ChromaDB path — run `/reindex` when you intentionally switch either.

<details>
<summary><strong>Advanced pipeline flags</strong></summary>

These constants live in `rag/chat_pdfs.py`. Edit them directly to toggle pipeline stages.

| Flag | Default | Effect |
|------|---------|--------|
| `USAR_CONTEXTUAL_RETRIEVAL` | `True` | Enrich chunks with LLM context at indexing time |
| `USAR_LLM_QUERY_DECOMPOSITION` | `True` | Decompose query into sub-queries |
| `USAR_BUSQUEDA_HIBRIDA` | `True` | Add keyword search alongside semantic search |
| `USAR_RERANKER` | `True` | Cross-encoder reranking |
| `USAR_RECOMP_SYNTHESIS` | `True` | RECOMP context compression before generation |
| `EXPANDIR_CONTEXTO` | `True` | Include adjacent chunks around top results |
| `USAR_EMBEDDINGS_IMAGEN` | `False` | Describe raster images in PDFs with a vision model |

</details>

---

## 5. Usage

### CLI commands

| Command | Description |
|---------|-------------|
| `/rag` | RAG mode — answers grounded in your documents |
| `/chat` | Chat mode — free conversation without document context |
| `/docs` | List indexed documents |
| `/temas` | Topic summary per document |
| `/stats` | Vector database statistics |
| `/reindex` | Drop the current index and re-index all documents |
| `/limpiar` `/clear` | Clear conversation history |
| `/ayuda` `/help` | Show all available commands |
| `/salir` `/exit` | Exit and save history |

### Web interface

Open `http://localhost:5000`. Supports document upload, streaming responses and pipeline settings through the UI.

For development with hot-reload: run `npm run dev` inside `rag/web/zip/` (Vite on :3000 proxies to Flask on :5000).

---

## Known limitations

- Vector graphics (SVG-based figures) embedded in PDFs are not extracted.
- Increasing `CHUNK_SIZE` or enabling all optional stages increases indexing time and memory usage.

---

*For thesis reproduction, RAGAS evaluation, LoRA fine-tuning and benchmark results see [`research/`](research/README.md).*
