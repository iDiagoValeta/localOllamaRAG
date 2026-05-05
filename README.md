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
  <a href="#2-architecture">Architecture</a> ·
  <a href="#3-demo">Demo</a> ·
  <a href="#4-getting-started">Getting started</a> ·
  <a href="#5-configuration">Configuration</a> ·
  <a href="#6-usage">Usage</a>
</p>

---

## 1. Overview

MonkeyGrab lets you ask questions about your PDF documents in natural language. Point it at a folder of PDFs, start the CLI or the web interface, and get answers grounded in the actual content of those files — no data sent to any cloud.

| | |
|---|---|
| **Local-first** | All indexing, retrieval and generation runs on your hardware. No API keys required for the core pipeline. |
| **Any model** | Works with any instruction-tuned model in [Ollama](https://ollama.com/) — `llama3.2`, `mistral`, `gemma4`, `qwen3`, etc. |
| **Hybrid retrieval** | Semantic search + keyword search fused with RRF, followed by optional [cross-encoder reranking](https://www.sbert.net/). |
| **Multilingual UI** | Spanish, English and Valencian out of the box. The CLI uses `MONKEYGRAB_LANG`; the web UI has an `ES / EN / VAL` selector. |
| **Two interfaces** | [Rich](https://rich.readthedocs.io/en/stable/)-based terminal CLI and a [Flask](https://flask.palletsprojects.com/) + [React](https://react.dev/) web UI with streaming responses. |
| **Image-aware** | Optionally describes raster images in PDFs with a vision model via [pymupdf4llm](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/), making visual content retrievable. |

---

## 2. Architecture

PDFs are indexed once into a [ChromaDB](https://www.trychroma.com/) vector store. Each query then passes through a configurable multi-stage retrieval pipeline before reaching the generator — all running locally via [Ollama](https://ollama.com/).

```mermaid
flowchart TD
    WEB["React Web App\nserved by Flask or Vite dev server"]
    CLI["Rich CLI"]

    API["Flask API  ·  rag/web/app.py\nREST + SSE streaming"]

    subgraph IDX["  Indexing Pipeline  "]
        direction TB
        EXT["PDF Extraction\npymupdf4llm · pypdf fallback"]
        IMG["Image Description\nPyMuPDF + OLLAMA_OCR_MODEL"]
        CHUNK["Chunking\nconfigurable size + overlap"]
        CTX["Contextual Enrichment\noptional · OLLAMA_CONTEXTUAL_MODEL"]
        EMB["Embedding\nOLLAMA_EMBED_MODEL"]
        EXT --> CHUNK
        IMG --> CHUNK
        CHUNK --> CTX --> EMB
    end

    subgraph RET["  Hybrid Retrieval Pipeline  "]
        direction TB
        D1["① Query Decomposition\noptional · OLLAMA_CHAT_MODEL"]
        D2["② Semantic + Keyword + Exhaustive Search\nChromaDB · top-80 + top-40 + critical terms"]
        D3["③ RRF Fusion + Cross-Encoder\n55% semantic · 45% lexical"]
        D4["④ Context Expansion + Cleanup\nadjacent chunks · artifact removal"]
        D5["⑤ RECOMP Synthesis\noptional · OLLAMA_RECOMP_MODEL"]
        D1 --> D2 --> D3 --> D4 --> D5
    end

    DB[("ChromaDB\nPersistent Vector Store\nrag/vector_db/<folder>_<embed_slug>")]
    GEN["Generation\nOLLAMA_RAG_MODEL\ndefault: phi4-finetuned:latest"]

    subgraph OLL["  Ollama / Local Models  "]
        direction LR
        M1["embeddinggemma\nEmbeddings"]
        M2["gemma4:e2b\nChat / decomposition"]
        M3["BAAI/bge-reranker\nCross-Encoder"]
        M4["gemma4:e4b\nOCR / contextual / RECOMP"]
        M5["phi4-finetuned\nRAG generator"]
    end

    WEB & CLI -->|"query / PDF upload"| API

    API -->|"PDF files"| IDX
    EMB -->|"store vectors"| DB

    API -->|"user question"| RET
    D2 <-->|"vector + lexical lookup"| DB
    D5 -->|"compressed context"| GEN
    D4 -. "fallback: raw chunks" .-> GEN
    GEN -->|"answer + sources"| API
    API -->|"SSE tokens"| WEB

    M1 -. embeddings .-> EMB
    M2 -. orchestration .-> D1
    M3 -. reranking .-> D3
    M4 -. "OCR / contextual / RECOMP" .-> IMG
    M4 -. "OCR / contextual / RECOMP" .-> CTX
    M4 -. "OCR / contextual / RECOMP" .-> D5
    M5 -. generation .-> GEN

    classDef client  fill:#4A90D9,stroke:#2C5F8A,color:#fff,font-weight:bold
    classDef api     fill:#5BAD6F,stroke:#3A7A4A,color:#fff,font-weight:bold
    classDef idx     fill:#E8A838,stroke:#B07820,color:#fff
    classDef ret     fill:#8B6BB1,stroke:#5E4080,color:#fff
    classDef gen     fill:#D45F5F,stroke:#9A3535,color:#fff,font-weight:bold
    classDef db      fill:#2D7D9A,stroke:#1A5570,color:#fff
    classDef model   fill:#3A3A3A,stroke:#111,color:#eee

    class WEB,CLI client
    class API api
    class EXT,IMG,CHUNK,CTX,EMB idx
    class D1,D2,D3,D4,D5 ret
    class GEN gen
    class DB db
    class M1,M2,M3,M4,M5 model
```

---

## 3. Demo

**Web interface — querying a local document corpus**

https://github.com/user-attachments/assets/22582283-1b28-4054-a341-3aa1cbdc5057

**CLI — querying a local document corpus**

https://github.com/user-attachments/assets/4b8a84ca-422f-44a6-a0d5-a8078fa5e17a

**LaTeX rendering — math formulas rendered natively in the web UI**

<img width="1277" height="674" alt="mathRender" src="https://github.com/user-attachments/assets/0a9dbcdf-fe91-4992-af82-e4e945fbf766" />

The web interface uses [KaTeX](https://katex.org/) to render inline (`$...$`) and display (`$$...$$`) LaTeX expressions generated by the model.

---

## 4. Getting started

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

- **Qwen3-14B RAG** — [nadiva1243/qwen3RAG](https://huggingface.co/nadiva1243/qwen3RAG) on [Hugging Face](https://huggingface.co/)
- **Phi-4 RAG** — [nadiva1243/phi4RAG](https://huggingface.co/nadiva1243/phi4RAG) on [Hugging Face](https://huggingface.co/)

### Run

Drop your PDFs into `rag/docs/libre/` and start:

```bash
# CLI (Spanish UI by default)
python rag/chat_pdfs.py

# CLI in English
MONKEYGRAB_LANG=en python rag/chat_pdfs.py          # bash/zsh
$env:MONKEYGRAB_LANG = "en"; python rag/chat_pdfs.py # PowerShell

# CLI in Valencian
MONKEYGRAB_LANG=ca python rag/chat_pdfs.py          # bash/zsh
$env:MONKEYGRAB_LANG = "ca"; python rag/chat_pdfs.py # PowerShell

# Web interface → http://localhost:5000
python rag/web/app.py
```

PowerShell note: use `$env:MONKEYGRAB_LANG = "en"` / `"ca"`. The `set MONKEYGRAB_LANG=...` syntax is for `cmd.exe`, not PowerShell.

The vector index is created automatically in `rag/vector_db/` on first run.

---

## 5. Configuration

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
| `RERANKER_QUALITY` | Cross-encoder tier: `quality` ([BAAI/bge](https://huggingface.co/BAAI/bge-reranker-v2-m3)) or `speed` (MiniLM) |
| `MONKEYGRAB_LANG` | CLI language: `es` (default), `en` or `ca` |

> [ChromaDB](https://www.trychroma.com/) paths follow the pattern `rag/vector_db/<folder>_<embed_slug>/`. Changing `DOCS_FOLDER` or `OLLAMA_EMBED_MODEL` selects a different index — run `/reindex` when you intentionally switch either.

<details>
<summary><strong>Advanced pipeline flags</strong></summary>

These constants live in `rag/chat_pdfs.py`. Edit them directly to toggle pipeline stages.

| Flag | Default | Effect |
|------|---------|--------|
| `USAR_CONTEXTUAL_RETRIEVAL` | `True` | Enrich chunks with LLM context at indexing time |
| `USAR_LLM_QUERY_DECOMPOSITION` | `True` | Decompose query into sub-queries |
| `USAR_BUSQUEDA_HIBRIDA` | `True` | Add keyword search alongside semantic search |
| `USAR_RERANKER` | `True` | [Cross-encoder reranking](https://www.sbert.net/) |
| `USAR_RECOMP_SYNTHESIS` | `True` | RECOMP context compression before generation |
| `EXPANDIR_CONTEXTO` | `True` | Include adjacent chunks around top results |
| `USAR_EMBEDDINGS_IMAGEN` | `True` | Describe raster images in PDFs with a vision model |

</details>

---

## 6. Usage

### CLI commands

| Command | Description |
|---------|-------------|
| `/rag` | RAG mode — answers grounded in your documents |
| `/chat` | Chat mode — free conversation without document context |
| `/docs` | List indexed documents |
| `/temas` `/topics` `/temes` | Topic summary per document |
| `/stats` | Vector database statistics |
| `/reindex` | Drop the current index and re-index all documents |
| `/limpiar` `/clear` `/netejar` | Clear conversation history |
| `/ayuda` `/help` `/ajuda` | Show all available commands |
| `/salir` `/exit` `/eixir` | Exit and save history |

### Web interface

Open `http://localhost:5000`. Supports document upload, streaming responses, pipeline settings and an `ES / EN / VAL` language selector through the UI. The selected web language is stored in the browser.

For development with hot-reload: run `npm run dev` inside `rag/web/zip/` ([Vite](https://vitejs.dev/) on :3000 proxies to [Flask](https://flask.palletsprojects.com/) on :5000).

---

## Known limitations

- **Vector graphics** (SVG figures) embedded in PDFs are not extracted.
- **Math, tables and images** are not plain text. Even with OCR and image captions, formulas and complex layouts can be misread, chunked awkwardly or poorly retrieved — expect occasional errors or incomplete answers on those pages.
- **Indexing cost** grows with `CHUNK_SIZE`, contextual enrichment, image captions and similar options (time and memory).

---

*For thesis reproduction, RAGAS evaluation, LoRA fine-tuning and benchmark results see [`research/`](research/README.md).*
