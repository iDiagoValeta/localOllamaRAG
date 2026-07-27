<p align="center">
  <img src="assets/logo-light-circular.png" alt="MonkeyGrab Logo" width="180" />
</p>

<h1 align="center">MonkeyGrab</h1>

<p align="center">
  <strong>A local, multilingual RAG system for querying PDF documents with open language models.</strong><br/>
  All indexing, retrieval and generation runs on your own hardware — no data leaves your machine.
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://ollama.com/"><img src="https://img.shields.io/badge/Ollama-Local%20LLM-000000?style=flat-square&logo=ollama&logoColor=white" alt="Ollama"></a>
  <a href="https://www.trychroma.com/"><img src="https://img.shields.io/badge/ChromaDB-Vector%20store-4B32C3?style=flat-square" alt="ChromaDB"></a>
  <a href="https://react.dev/"><img src="https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB" alt="React"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Retrieval-Hybrid%20(BM25%20%2B%20vector)-2EA44F?style=flat-square" alt="Hybrid retrieval">
  <img src="https://img.shields.io/badge/Languages-ES%20%7C%20CA%20%7C%20EN-5C6BC0?style=flat-square" alt="Multilingual">
  <img src="https://img.shields.io/badge/Privacy-100%25%20local-FF8C00?style=flat-square" alt="Local">
</p>

<p align="center">
  <a href="#1-overview">Overview</a> ·
  <a href="#2-getting-started">Getting started</a> ·
  <a href="#3-configuration">Configuration</a> ·
  <a href="#4-usage">Usage</a> ·
  <a href="#5-license">License</a>
</p>

---

## 1. Overview

MonkeyGrab lets you ask questions about your PDF documents in natural language. Point it at a folder of PDFs, start the CLI or the web interface, and get answers grounded in the actual content of those files — no data sent to any cloud.

- **Local-first** — all indexing, retrieval and generation runs on your hardware; no API keys required.
- **Any model** — works with any instruction-tuned model in [Ollama](https://ollama.com/).
- **Hybrid retrieval** — semantic search + BM25 lexical search fused with RRF, plus optional [cross-encoder reranking](https://www.sbert.net/).
- **Multilingual** — Spanish, English and Valencian UI and retrieval out of the box.
- **Image-aware** — optionally describes raster images in PDFs with a vision model, making visual content retrievable.
- **Three interfaces** — terminal CLI, [Flask](https://flask.palletsprojects.com/) + [React](https://react.dev/) web UI, and a packaged Windows desktop app.
- **Hexagonal core** — retrieval and generation logic lives behind swappable ports (`src/monkeygrab/`), so the underlying storage or model tech can change without touching the interfaces above it.

PDFs are indexed once into a [ChromaDB](https://www.trychroma.com/) vector store. Each query passes through a configurable multi-stage retrieval pipeline before reaching the generator, all running locally via [Ollama](https://ollama.com/). Both front-ends share the same engine and stream the answer back token by token; in the web UI, cited sources open in an inline PDF viewer.

<p align="center">
  <img src="assets/userInteraction.svg" alt="User interaction flow across the web and CLI interfaces" width="900" />
</p>

<details>
<summary><strong>See it in action</strong></summary>
<br/>

**Web interface:** https://github.com/user-attachments/assets/f5f8fa1d-b193-4f94-85c2-8f903afa2348

**CLI:** https://github.com/user-attachments/assets/a27b6fef-52c1-4d4a-846e-7c4cd36863fa

**LaTeX rendering** (formulas via [KaTeX](https://katex.org/) in the web UI):

<img width="918" height="563" alt="LaTeX rendering in the web UI" src="assets/latexRender.png" />

</details>

---

## 2. Getting started

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

### Run

Drop your PDFs into `rag/docs/en/` and start:

```bash
# CLI (Spanish UI by default)
python rag/chat_pdfs.py

# CLI in English / Valencian
MONKEYGRAB_LANG=en python rag/chat_pdfs.py          # bash/zsh
$env:MONKEYGRAB_LANG = "en"; python rag/chat_pdfs.py # PowerShell

# Web interface → http://localhost:5000
python rag/web/app.py
```

The vector index is created automatically in `rag/vector_db/` on first run.

---

## 3. Configuration

MonkeyGrab is configured entirely through environment variables. Copy the bundled template and edit only what you need:

```bash
cp .env.example .env          # macOS / Linux
Copy-Item .env.example .env   # Windows PowerShell
```

The `.env` file at the project root is loaded automatically on startup. Anything exported in your shell still takes precedence over it.

> [!TIP]
> [`.env.example`](.env.example) documents **every** supported variable with its default value and a one-line description. Start there for anything beyond the essentials below.

| Variable | Description |
|----------|-------------|
| `OLLAMA_RAG_MODEL` | Generator model for RAG mode |
| `OLLAMA_CHAT_MODEL` | Generator for chat mode and query decomposition |
| `OLLAMA_EMBED_MODEL` | Embedding model for indexing and retrieval |
| `DOCS_FOLDER` | PDF folder to index (default: `rag/docs/en/`) |
| `RERANKER_QUALITY` | Cross-encoder tier: `quality` or `fast` |
| `MONKEYGRAB_LANG` | CLI language: `es` (default), `en` or `ca` |

> [!IMPORTANT]
> [ChromaDB](https://www.trychroma.com/) paths follow the pattern `rag/vector_db/<folder>_<embed_slug>/`. Changing `DOCS_FOLDER` or `OLLAMA_EMBED_MODEL` selects a different index — run `/reindex` when you intentionally switch either.

---

## 4. Usage

### CLI

| Command | Description |
|---------|-------------|
| `/rag` | RAG mode — answers grounded in your documents |
| `/chat` | Chat mode — free conversation without document context |
| `/docs` | List indexed documents |
| `/reindex` | Drop the current index and re-index all documents |
| `/ayuda` `/help` `/ajuda` | Show all available commands |
| `/salir` `/exit` `/eixir` | Exit and save history |

### Web interface

Open `http://localhost:5000`. The sidebar covers **Documents**, **Models** and **RAG Pipeline** control, plus PDF upload, streaming responses and an `ES / EN / VAL` language selector. There are three fixed language stores — English (default), Castellano, Valencià — each bound to `rag/docs/{en,es,ca}/`; pick one to switch the active corpus at runtime.

Ollama starts automatically at launch if installed but not running. Assign any installed model to a pipeline role (generator, chat, embeddings, contextual, RECOMP, OCR) from the **Models** tab — changes apply on the next query without a restart. The reranker is a local CrossEncoder selected via `RERANKER_QUALITY`, not an Ollama model role.

### Desktop app

MonkeyGrab can also be packaged as a standalone Windows `.exe` (PyInstaller + pywebview) — no browser, no terminal, no Python install needed on the target machine. See [`packaging/README.md`](packaging/README.md).

---

## Known limitations

> [!WARNING]
> - **Vector graphics** (SVG figures) embedded in PDFs are not extracted.
> - **Math, tables and images** are not plain text — expect occasional errors or incomplete answers on those pages even with OCR and image captions.
> - **Indexing cost** grows with chunk size, contextual enrichment, image captions and similar options.

---

## Development

Every change is checked by two CI gates: a fast one (lint, architecture
rules, frontend build) on every pull request, and a full one that runs the
real pipeline against a set of gold question/answer cases before a merge to
`main`. See [`.claude/CLAUDE.md`](.claude/CLAUDE.md) for the architecture and
contributor rules.

---

## 5. License

[MIT](LICENSE) © Ignacio Diago Valeta.
