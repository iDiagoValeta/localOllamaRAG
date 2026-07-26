"""
MonkeyGrab -- RAG engine for PDF document queries.

Interactive application with two operating modes: CHAT (free conversation with
a base model, persistent history, and project identity) and RAG (document
queries over indexed PDFs with hybrid retrieval and source-backed answers).

Pipeline stages (each togglable via flags):
    1. Indexing        chunking + embeddings + contextual retrieval (opt.)
    2. Retrieval       semantic + query decomposition (opt.) + BM25 lexical
    3. Ranking         RRF fusion + Cross-Encoder reranking (opt.)
    4. Context         neighbor expansion + optimization
    5. Generation      RECOMP synthesis (opt.) + streaming
    6. Observability   metrics and debug dumps

How to run (interactive CLI):
    From the repository root (recommended, matches docs and imports):

        python rag/chat_pdfs.py

    From inside ``rag/``:

        cd rag
        python chat_pdfs.py

    On Windows (PowerShell), same commands from the project root or ``rag/``.

    This starts ``MonkeyGrabCLI`` (see ``rag/cli/app.py``): slash commands
    such as ``/rag``, ``/chat``, ``/reindex``, ``/docs``, ``/salir``.

    The Flask web app does **not** execute this file as ``__main__``; it imports
    functions and constants from here. Start the UI with ``python rag/web/app.py``
    from the repository root.

    Prerequisites: Ollama running; PDFs under ``rag/docs/en/`` unless ``DOCS_FOLDER``
    points elsewhere. Model names via ``OLLAMA_RAG_MODEL``, ``OLLAMA_EMBED_MODEL``,
    ``OLLAMA_OCR_MODEL`` (image indexing), etc., as documented in the project README
    and in ``rag/README.md``.
"""

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  CONFIGURATION (startup)
#  +-- 1. Imports             stdlib -> third-party (ollama, chromadb, pypdf) -> local
#  +-- 2. Required deps       PDF extraction, reranking and BM25
#  +-- 3. Global config
#  |      +-- 3.1 Terminal runtime (UTF-8)
#  |      +-- 3.2 Environment readers
#  |      +-- 3.3 Model roles and Ollama runtime
#  |      +-- 3.4 Pipeline flags
#  |      +-- 3.5 Paths and persistence
#  |      +-- 3.6 Embedding prefixes
#  |      +-- 3.7 Indexing, retrieval and ranking parameters
#  |      +-- 3.8 Logging and process environment
#  |      +-- 3.9 Model role runtime switching
#  |
#  PUBLIC FACADE
#  +-- 4. System prompts      CHAT (identity + language); RAG prompt baked into Modelfile
#  +-- 5. Engine reexports    public API preserved from rag.engine.* modules
#  |
#  SPLIT IMPLEMENTATION (rag/engine/)
#  +-- history.py             CHAT history persistence
#  +-- chunking.py            Markdown chunking and neighbor IDs
#  +-- lexical.py             keywords, stopwords and BM25 lexical search
#  +-- reranking.py           query decomposition and CrossEncoder reranking
#  +-- retrieval.py           hybrid retrieval orchestration
#  +-- context.py             context cleanup, formatting and RECOMP synthesis
#  +-- debug.py               debug_rag interaction dumps
#  +-- generation.py          Ollama generation and silent evaluation path
#  +-- contextual.py          contextual retrieval helpers
#  +-- images.py              PDF image extraction and OCR descriptions
#  +-- indexing.py            PDF indexing and collection document listing
#  +-- runtime.py             sync layer for mutable runtime flags/config
#  |
#  ENTRY
#  +-- main()                 MonkeyGrabCLI.run()
#
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# SECTION 1: IMPORTS
# ─────────────────────────────────────────────


import base64
import io
import json
import logging
import os
import re
import requests
import sys
import warnings
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, List, Optional, Tuple


import chromadb
import ollama
from pypdf import PdfReader


_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# Load configuration from a ``.env`` file at the project root, if present.
# ``override=False`` keeps the process environment authoritative over the file,
# matching the precedence documented in the README. ``python-dotenv`` is an
# optional convenience: when absent, variables must be exported in the shell.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_project_root, ".env"), override=False)
except ImportError:
    pass


from rag.cli.display import ui


# ─────────────────────────────────────────────
# SECTION 2: REQUIRED DEPENDENCIES
# ─────────────────────────────────────────────


with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
    import pymupdf4llm
import fitz
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# Public compatibility constants. The imports above are mandatory and fail at
# startup when the Python environment is incomplete.

PYMUPDF_AVAILABLE = True
FITZ_DISPONIBLE = True
RERANKER_AVAILABLE = True
BM25_AVAILABLE = True


# ─────────────────────────────────────────────
# SECTION 3: GLOBAL CONFIGURATION
# ─────────────────────────────────────────────


# 3.1 Terminal runtime

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 3.2 Environment readers


def _leer_env_int(nombre_variable: str, default: int) -> int:
    """Parse an integer environment variable with a safe fallback.

    Args:
        nombre_variable: Environment variable name to inspect.
        default: Fallback value when the variable is undefined or invalid.

    Returns:
        Parsed integer value, or ``default``.
    """
    try:
        return int(os.getenv(nombre_variable, str(default)))
    except (TypeError, ValueError):
        return default


def _leer_env_float(nombre_variable: str, default: float) -> float:
    """Parse a float environment variable with a safe fallback.

    Args:
        nombre_variable: Environment variable name to inspect.
        default: Fallback value when the variable is undefined or invalid.

    Returns:
        Parsed float value, or ``default``.
    """
    try:
        return float(os.getenv(nombre_variable, str(default)))
    except (TypeError, ValueError):
        return default


def _inferir_descripcion_modelo(nombre_modelo: str) -> str:
    """Extract the base model name by stripping the tag suffix.

    Args:
        nombre_modelo: Full model identifier (e.g. ``"gemma4:e4b"``).

    Returns:
        Model name without the colon-separated tag.
    """
    return nombre_modelo.split(":")[0]


# 3.3 Model roles and Ollama runtime

MODELO_RAG = os.getenv("OLLAMA_RAG_MODEL", "gemma4:e4b")
MODELO_CHAT = os.getenv("OLLAMA_CHAT_MODEL", "gemma4:e4b")
MODELO_EMBEDDING = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")
MODELO_CONTEXTUAL = os.getenv("OLLAMA_CONTEXTUAL_MODEL", "gemma4:e4b")
MODELO_RECOMP = os.getenv("OLLAMA_RECOMP_MODEL", "gemma4:e4b")
MODELO_OCR = os.getenv("OLLAMA_OCR_MODEL", "gemma4:e4b")
MODELO_DESC = os.getenv("MODELO_DESC", _inferir_descripcion_modelo(MODELO_RAG))

OLLAMA_NUM_CTX = _leer_env_int("OLLAMA_NUM_CTX", 8192)
OLLAMA_RAG_NUM_CTX = _leer_env_int("OLLAMA_RAG_NUM_CTX", 16384)
OLLAMA_AUX_NUM_CTX = _leer_env_int("OLLAMA_AUX_NUM_CTX", 8192)
OLLAMA_QUERY_NUM_CTX = _leer_env_int("OLLAMA_QUERY_NUM_CTX", 2048)
OLLAMA_RECOMP_NUM_CTX = _leer_env_int("OLLAMA_RECOMP_NUM_CTX", 8192)
OLLAMA_CONTEXTUAL_NUM_CTX = _leer_env_int("OLLAMA_CONTEXTUAL_NUM_CTX", 32768)
OLLAMA_OCR_NUM_CTX = _leer_env_int("OLLAMA_OCR_NUM_CTX", 8192)
OLLAMA_REQUEST_TIMEOUT = _leer_env_int("OLLAMA_REQUEST_TIMEOUT", 900)
# Seconds to keep weights in VRAM after each Ollama call; 0 unloads immediately.
OLLAMA_KEEP_ALIVE = _leer_env_int("OLLAMA_KEEP_ALIVE", 0)
OLLAMA_GENERATE_RETRIES = _leer_env_int("OLLAMA_GENERATE_RETRIES", 2)
OLLAMA_GENERATE_RETRY_DELAY = _leer_env_int("OLLAMA_GENERATE_RETRY_DELAY", 3)


# 3.4 Pipeline flags

USAR_CONTEXTUAL_RETRIEVAL = True
USAR_LLM_QUERY_DECOMPOSITION = True
USAR_BUSQUEDA_HIBRIDA = True
USAR_RERANKER = True
EXPANDIR_CONTEXTO = True
USAR_OPTIMIZACION_CONTEXTO = True
USAR_RECOMP_SYNTHESIS = True
USAR_EMBEDDINGS_IMAGEN = True
LOGGING_METRICAS = True
GUARDAR_DEBUG_RAG = True

# Runtime toggles are inference-time only; indexing flags require a fresh
# Chroma collection to compare fairly.
PIPELINE_RUNTIME_FLAGS = (
    "USAR_LLM_QUERY_DECOMPOSITION",
    "USAR_BUSQUEDA_HIBRIDA",
    "USAR_RERANKER",
    "EXPANDIR_CONTEXTO",
    "USAR_OPTIMIZACION_CONTEXTO",
    "USAR_RECOMP_SYNTHESIS",
)


def get_pipeline_flags() -> Dict[str, bool]:
    """Return the runtime-toggleable pipeline flags used during inference."""
    return {name: bool(globals()[name]) for name in PIPELINE_RUNTIME_FLAGS}


def set_pipeline_flags(overrides: Dict[str, bool]) -> Dict[str, bool]:
    """Override inference-time pipeline flags for the current Python process.

    Index-time flags are intentionally excluded because they require rebuilding
    a Chroma collection to be compared fairly.
    """
    invalid = sorted(set(overrides) - set(PIPELINE_RUNTIME_FLAGS))
    if invalid:
        valid = ", ".join(PIPELINE_RUNTIME_FLAGS)
        raise ValueError(f"Unsupported pipeline flag(s): {', '.join(invalid)}. Valid: {valid}")

    previous = get_pipeline_flags()
    for name, value in overrides.items():
        globals()[name] = bool(value)
    return previous


# 3.5 Paths and persistence

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Writable data root. Defaults to the package dir for source/dev runs (paths stay
# identical to before), but the packaged desktop app points it at a per-user
# location via ``MONKEYGRAB_DATA_DIR`` (e.g. ``%LOCALAPPDATA%/MonkeyGrab``) so the
# vector DBs, history and debug dumps live outside the read-only application bundle.
DATA_DIR = os.path.abspath(os.getenv("MONKEYGRAB_DATA_DIR", BASE_DIR))

CARPETA_DOCS = os.getenv("DOCS_FOLDER", os.path.join(BASE_DIR, "docs", "en"))


def _derivar_paths_db(carpeta: str, modelo_embedding: str) -> tuple[str, str]:
    """Derive ``(PATH_DB, COLLECTION_NAME)`` from a docs folder and embedding model.

    The embedding slug is part of the DB path so vector stores built with
    different embedding models never collide on disk. The DB lives under
    ``DATA_DIR`` so packaged builds write to a user-writable location.

    Args:
        carpeta: PDF directory (absolute or relative).
        modelo_embedding: Ollama embedding model name; the tag is stripped for the slug.

    Returns:
        Tuple ``(path_db, collection_name)``.
    """
    nombre = os.path.basename(os.path.abspath(carpeta))
    slug = modelo_embedding.split(":")[0].replace("/", "_")
    return (
        os.path.join(DATA_DIR, "vector_db", f"{nombre}_{slug}"),
        f"docs_{nombre}",
    )


PATH_DB, COLLECTION_NAME = _derivar_paths_db(CARPETA_DOCS, MODELO_EMBEDDING)

_DEFAULT_CARPETA_DOCS = CARPETA_DOCS
_DEFAULT_PATH_DB = PATH_DB
_DEFAULT_COLLECTION_NAME = COLLECTION_NAME


def set_docs_folder_runtime(carpeta: str | None) -> tuple[str, str, str]:
    """Switch ``CARPETA_DOCS`` and derived Chroma paths (for tests).

    Restores module-level defaults when ``carpeta`` is ``None`` (values captured
    at import from ``DOCS_FOLDER`` / ``rag/docs/es``).

    Args:
        carpeta: Absolute or relative path to a PDF directory, or ``None`` to restore defaults.

    Returns:
        Previous ``(CARPETA_DOCS, PATH_DB, COLLECTION_NAME)`` tuple before this call.
    """
    global CARPETA_DOCS, PATH_DB, COLLECTION_NAME
    previous = (CARPETA_DOCS, PATH_DB, COLLECTION_NAME)
    if carpeta is None:
        CARPETA_DOCS = _DEFAULT_CARPETA_DOCS
        PATH_DB = _DEFAULT_PATH_DB
        COLLECTION_NAME = _DEFAULT_COLLECTION_NAME
    else:
        CARPETA_DOCS = os.path.abspath(carpeta)
        PATH_DB, COLLECTION_NAME = _derivar_paths_db(CARPETA_DOCS, MODELO_EMBEDDING)
    return previous


HISTORIAL_PATH = os.path.join(DATA_DIR, "historial_chat.json")
MAX_HISTORIAL_MENSAJES = 40

CARPETA_DEBUG_RAG = os.path.join(DATA_DIR, "debug_rag")


# 3.6 Embedding prefixes

def _derivar_prefijos_embedding(modelo_embedding: str) -> tuple[str, str]:
    """Return ``(query_prefix, doc_prefix)`` task prefixes for an embedding model.

    Nomic embedding models expect ``search_query:`` / ``search_document:``
    prefixes; other models embed the text unmodified.

    Args:
        modelo_embedding: Ollama embedding model name.

    Returns:
        Tuple ``(query_prefix, doc_prefix)``.
    """
    nombre = modelo_embedding.lower().split(":")[0]
    if "nomic" in nombre:
        return "search_query: ", "search_document: "
    return "", ""


EMBED_PREFIX_QUERY, EMBED_PREFIX_DOC = _derivar_prefijos_embedding(MODELO_EMBEDDING)


# 3.7 Indexing, retrieval and ranking parameters

CONTEXTUAL_DOC_CHARS = _leer_env_int("CONTEXTUAL_DOC_CHARS", 24000)
CHUNK_SIZE = _leer_env_int("RAG_CHUNK_SIZE", 2000)
CHUNK_OVERLAP = _leer_env_int("RAG_CHUNK_OVERLAP", 400)
MIN_CHUNK_LENGTH = _leer_env_int("RAG_MIN_CHUNK_LENGTH", 150)
MAX_IMAGENES_POR_PAGINA = _leer_env_int("RAG_MAX_IMAGES_PER_PAGE", 5)
MIN_IMAGEN_SIZE_PX = _leer_env_int("RAG_MIN_IMAGE_SIZE_PX", 100)
CAPTION_MARGIN_PX = _leer_env_int("RAG_CAPTION_MARGIN_PX", 80)
_IMAGEN_CHUNK_OFFSET = 10_000

N_RESULTADOS_SEMANTICOS = _leer_env_int("RAG_N_RESULTADOS_SEMANTICOS", 80)
N_RESULTADOS_KEYWORD = _leer_env_int("RAG_N_RESULTADOS_KEYWORD", 40)
TOP_K_RERANK_CANDIDATES = _leer_env_int("RAG_TOP_K_RERANK_CANDIDATES", 200)
TOP_K_FINAL = _leer_env_int("RAG_TOP_K_FINAL", 8)
N_TOP_PARA_EXPANSION = _leer_env_int("RAG_N_TOP_PARA_EXPANSION", 3)

RERANKER_MODEL_QUALITY = os.getenv("RERANKER_QUALITY", "quality")
UMBRAL_SCORE_RERANKER = _leer_env_float("RAG_UMBRAL_SCORE_RERANKER", 0.65)

RRF_K = _leer_env_int("RAG_RRF_K", 60)
BM25_K1 = _leer_env_float("RAG_BM25_K1", 1.5)
BM25_B = _leer_env_float("RAG_BM25_B", 0.75)
PESO_SEMANTICO_RRF = _leer_env_float("RAG_PESO_SEMANTICO_RRF", 0.55)
PESO_BM25_RRF = _leer_env_float("RAG_PESO_BM25_RRF", 0.45)

MIN_LONGITUD_PREGUNTA_RAG = _leer_env_int("RAG_MIN_LONGITUD_PREGUNTA", 10)
MAX_CONTEXTO_CHARS = _leer_env_int("MAX_CONTEXTO_CHARS", 24000)


# 3.8 Logging and process environment

LOG_LEVEL = logging.ERROR

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s: %(message)s")

for _logger_name in (
    "httpx", "chromadb", "chromadb.telemetry", "urllib3", "requests",
    "sentence_transformers", "transformers", "huggingface_hub",
    "tqdm", "filelock",
):
    logging.getLogger(_logger_name).setLevel(logging.CRITICAL)

warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*huggingface.*")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


# 3.9 Model role runtime switching

# Maps the public role keys used by the web UI / API to the module-level model
# variables. Engine modules read these lazily through ``cfg``, so reassigning
# them takes effect on the next pipeline call without a restart.
MODEL_ROLE_VARS = {
    "rag": "MODELO_RAG",
    "chat": "MODELO_CHAT",
    "embedding": "MODELO_EMBEDDING",
    "contextual": "MODELO_CONTEXTUAL",
    "recomp": "MODELO_RECOMP",
    "ocr": "MODELO_OCR",
}


def get_model_roles() -> Dict[str, str]:
    """Return the Ollama model currently assigned to each pipeline role."""
    return {role: globals()[var] for role, var in MODEL_ROLE_VARS.items()}


def set_model_roles_runtime(overrides: Dict[str, str]) -> Dict[str, str]:
    """Reassign pipeline model roles for the current process.

    Changing the ``embedding`` role also recomputes ``PATH_DB``,
    ``COLLECTION_NAME`` and the embedding task prefixes, because the vector store
    path is namespaced by the embedding model. Callers must rebind any cached
    Chroma collection afterwards and re-index, since embeddings from different
    models are not comparable.

    Args:
        overrides: Mapping of role keys (see ``MODEL_ROLE_VARS``) to Ollama model
            names. Empty values are ignored; unknown keys raise ``ValueError``.

    Returns:
        The full role -> model mapping after applying the overrides.

    Raises:
        ValueError: If ``overrides`` contains an unsupported role key.
    """
    invalid = sorted(set(overrides) - set(MODEL_ROLE_VARS))
    if invalid:
        valid = ", ".join(MODEL_ROLE_VARS)
        raise ValueError(f"Unsupported model role(s): {', '.join(invalid)}. Valid: {valid}")

    global MODELO_RAG, MODELO_CHAT, MODELO_EMBEDDING, MODELO_CONTEXTUAL
    global MODELO_RECOMP, MODELO_OCR, MODELO_DESC
    global PATH_DB, COLLECTION_NAME, EMBED_PREFIX_QUERY, EMBED_PREFIX_DOC

    embedding_changed = False
    for role, modelo in overrides.items():
        modelo = (modelo or "").strip()
        if not modelo:
            continue
        globals()[MODEL_ROLE_VARS[role]] = modelo
        if role == "embedding":
            embedding_changed = True

    if (overrides.get("rag") or "").strip():
        MODELO_DESC = _inferir_descripcion_modelo(MODELO_RAG)
    if embedding_changed:
        PATH_DB, COLLECTION_NAME = _derivar_paths_db(CARPETA_DOCS, MODELO_EMBEDDING)
        EMBED_PREFIX_QUERY, EMBED_PREFIX_DOC = _derivar_prefijos_embedding(MODELO_EMBEDDING)

    return get_model_roles()


# ─────────────────────────────────────────────
# SECTION 4: SYSTEM PROMPTS
# ─────────────────────────────────────────────


SYSTEM_PROMPT_CHAT = f"""
You are MonkeyGrab, the conversational assistant for a local academic RAG system (TFG project).
Your purpose is to help users query indexed PDF documents and understand the system itself.

---

### SYSTEM OVERVIEW
- **Architecture:** Runs fully locally using Ollama (LLM inference) and ChromaDB (vector store).
- **Model configuration:** All model roles are configurable through environment variables. Explain roles and variables, not fixed model names.
- **Modes:**
  1. **CHAT**: General conversation, project guidance, and command help. Maintains local history.
  2. **RAG**: Document-grounded answers from indexed PDFs using hybrid retrieval.

---

### RAG PIPELINE ARCHITECTURE (Technical Knowledge Base)

Use this reference to explain how the system works or which parts are mandatory vs. configurable.

#### 1. INDEXING PHASE
* **CORE (Mandatory):**
    * **Extraction & Chunking:** Reading PDFs and splitting text (`dividir_en_chunks`).
    * **Embeddings:** Converting text to vectors with `MODELO_EMBEDDING` configured through `OLLAMA_EMBED_MODEL`, then saving them to ChromaDB.
* **OPTIONAL (Flag: `USAR_CONTEXTUAL_RETRIEVAL`):**
    * **Contextual Retrieval:** Uses `MODELO_CONTEXTUAL` configured through `OLLAMA_CONTEXTUAL_MODEL` to generate summary/context for each chunk before indexing to improve retrieval accuracy.
* **OPTIONAL (Flag: `USAR_EMBEDDINGS_IMAGEN`):**
    * **Image Indexing:** Extracts raster images from each PDF page with PyMuPDF (fitz), describes them with `MODELO_OCR` configured through `OLLAMA_OCR_MODEL` using a structured OCR prompt that transcribes tables cell by cell, chart axes and legends, diagram components, and equations, then stores the result as a regular text chunk in ChromaDB.

#### 2. RETRIEVAL PHASE
Orchestrated by `realizar_busqueda_hibrida`. Core is semantic (vector) search; optional components extend it.
* **CORE (Mandatory):**
    * **Semantic Search:** Vector distance lookup using `MODELO_EMBEDDING`; always performed.
* **OPTIONAL (execution order):**
    * **Query Decomposition** (`USAR_LLM_QUERY_DECOMPOSITION`): Uses `MODELO_CHAT` configured through `OLLAMA_CHAT_MODEL` to generate sub-queries before semantic search; activates for long questions (>60 chars).
    * **Hybrid Search** (`USAR_BUSQUEDA_HIBRIDA`): Adds Okapi BM25 lexical search (Robertson & Zaragoza 2009) over the indexed chunks, then fuses the BM25 ranking with semantic retrieval using Reciprocal Rank Fusion (RRF).

#### 3. RANKING & REFINEMENT
* **OPTIONAL:**
    * **Reranking** (`USAR_RERANKER`): Uses a Cross-Encoder (requires `sentence-transformers`) to re-score the top results for higher precision.

#### 4. CONTEXT & GENERATION
* **CORE (Mandatory):**
    * **Generation:** `MODELO_RAG`, configured through `OLLAMA_RAG_MODEL`, generates the final answer based on the retrieved text.
* **OPTIONAL:**
    * **Context Optimization** (`USAR_OPTIMIZACION_CONTEXTO`): Cleans PDF artifacts (headers, footers, noise) before sending to the LLM.
    * **Neighbor Expansion** (`EXPANDIR_CONTEXTO`): Retrieves adjacent chunks to provide continuous context.
    * **RECOMP Synthesis** (`USAR_RECOMP_SYNTHESIS`): Uses `MODELO_RECOMP`, configured through `OLLAMA_RECOMP_MODEL` and separate from `MODELO_RAG`, to summarize/synthesize the context instead of feeding raw chunks (default: True).

---

### BEHAVIOR RULES
1. **Conciseness:** Be concise by default. Expand only when asked.
2. **Honesty:** Never fabricate system state or document contents. If you don't know, say so.
3. **Guidance:** If a user asks "what should I do?", provide concrete next steps (e.g., suggest switching to RAG mode to search their PDFs).
4. **Mode Enforcement:** If the user asks for information contained in the documents while in CHAT mode, redirect them to use RAG mode for document-grounded answers.
5. **Configuration framing:** When explaining the pipeline, describe configurable roles (`OLLAMA_*`, `RERANKER_QUALITY`) instead of presenting any specific model as required.
6. **Language:** Always respond in the exact same language the user uses. If they write in Spanish, respond in Spanish. If they write in Catalan, respond in Catalan. If they write in English, respond in English. Never switch languages mid-conversation.
7. **Tone:** Professional, academic, yet approachable.
8. **Math formatting:** Use LaTeX notation for all formulas: $...$ inline, $$...$$ for display equations.
"""

SYSTEM_PROMPT_RAG = """You are a professional document analysis assistant. Your role is to answer questions accurately based on the provided document context.

Guidelines:
- Base your answers strictly on the information within the <context> tags.
- Do not add information beyond what the context provides.
- Preserve technical terms, notation, formulas, and numbers exactly as they appear.
- Formulate clear, well-structured responses in complete sentences.
- For factual questions, be direct and precise.
- For analytical or complex questions, provide detailed explanations referencing specific information from the context.
- Always respond in the same language as the context (English, Spanish/Castellano, or Catalan/Català).
- For mathematical expressions, always use LaTeX notation: $...$ for inline math and $$...$$ for display/block equations."""


# ─────────────────────────────────────────────

# -----------------------------------------------------------------------------
# BUSINESS LOGIC FACADE
# -----------------------------------------------------------------------------
# The implementation lives in rag.engine.*. Keep these imports explicit so all
# existing callers can continue to use rag.chat_pdfs as the public runtime API.

from rag.engine.history import cargar_historial, guardar_historial, limpiar_historial
from rag.engine.chunking import dividir_en_chunks, expandir_con_chunks_adyacentes
from rag.engine.lexical import (
    STOPWORDS,
    GENERIC_TERMS_BLACKLIST,
    extraer_keywords,
    _tokenizar_bm25,
    busqueda_lexica_bm25,
)
from rag.engine.reranking import (
    _detectar_dispositivo_reranker,
    obtener_modelo_reranker,
    rerank_resultados,
    generar_queries_con_llm,
    _validar_coherencia_query,
)
from rag.engine.retrieval import realizar_busqueda_hibrida
from rag.engine.context import (
    _es_continuacion_parrafo,
    _reunir_parrafos,
    optimizar_texto_contexto,
    _marcar_fragmento_incompleto,
    _texto_fuente_fragmento,
    _strip_ollama_think_blocks,
    _normalizar_salida_recomp,
    construir_contexto_para_modelo,
    sintetizar_contexto_recomp,
)
from rag.engine.debug import guardar_debug_rag
from rag.engine.generation import (
    OLLAMA_BASE_URL,
    _ollama_generate_stream,
    _preparar_mensaje_usuario_rag,
    generar_tokens_respuesta,
    _generar_respuesta_stream,
    preparar_fragmentos_para_generacion,
    generar_respuesta,
    generar_respuesta_silenciosa,
    evaluar_pregunta_rag,
)
from rag.engine.contextual import _detectar_idioma, generar_contexto_situacional
from rag.engine.images import (
    _es_descripcion_spam,
    _es_prompt_echo,
    _es_solo_caption,
    extraer_imagenes_pdf,
    describir_imagen_con_llm,
)
from rag.engine.indexing import indexar_documentos, obtener_documentos_indexados


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

def main():
    """Launch the MonkeyGrab CLI application."""
    import rag.chat_pdfs as rag_engine
    from rag.cli import MonkeyGrabCLI
    cli = MonkeyGrabCLI(rag_engine)
    cli.run()


if __name__ == "__main__":
    main()
