"""
MonkeyGrab -- RAG engine for PDF document queries.

Interactive application with two operating modes: CHAT (free conversation with
a base model, persistent history, and project identity) and RAG (document
queries over indexed PDFs with hybrid retrieval and source-backed answers).

Pipeline stages (each togglable via flags):
    1. Indexing        chunking + embeddings + contextual retrieval (opt.)
    2. Retrieval       semantic + query decomposition (opt.) + keywords
    3. Deep scan       exhaustive search by critical terms
    4. Ranking         RRF fusion + Cross-Encoder reranking (opt.)
    5. Context         neighbor expansion + optimization
    6. Generation      RECOMP synthesis (opt.) + streaming
    7. Observability   metrics and debug dumps

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
    functions and constants from here. Start the UI with ``python web/app.py``
    from the repository root.

    Prerequisites: Ollama running; PDFs under ``rag/pdfs/`` unless ``DOCS_FOLDER``
    points elsewhere. Model names via ``OLLAMA_RAG_MODEL``, ``OLLAMA_EMBED_MODEL``,
    ``OLLAMA_OCR_MODEL`` (image indexing), etc., as documented in the project README
    / ``CLAUDE.md``.
"""

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  CONFIGURATION (startup)
#  +-- 1. Imports             stdlib -> third-party (ollama, chromadb, pypdf) -> local
#  +-- 2. Optional deps       pymupdf4llm (PDF), CrossEncoder (reranking)
#  +-- 3. Global config
#  |      +-- 3.1 Terminal runtime (UTF-8)
#  |      +-- 3.2 Ollama models (RAG, CHAT, embedding, contextual, RECOMP)
#  |      +-- 3.3 Pipeline flags (per-stage toggle)
#  |      +-- 3.4 Paths and persistence (DB, history, debug)
#  |      +-- 3.5 Retrieval / generation parameters
#  |      +-- 3.6 Logging and environment
#  |
#  PUBLIC FACADE
#  +-- 4. System prompts      CHAT (identity + language); RAG prompt baked into Modelfile
#  +-- 5. Engine reexports    public API preserved from rag.engine.* modules
#  |
#  SPLIT IMPLEMENTATION (rag/engine/)
#  +-- history.py             CHAT history persistence
#  +-- chunking.py            Markdown chunking and neighbor IDs
#  +-- lexical.py             keywords, stopwords and exhaustive text search
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


from rag.cli.display import ui


# ─────────────────────────────────────────────
# SECTION 2: OPTIONAL DEPENDENCIES
# ─────────────────────────────────────────────


try:
    with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        import pymupdf4llm
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import fitz
    FITZ_DISPONIBLE = True
except ImportError:
    FITZ_DISPONIBLE = False

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False


# ─────────────────────────────────────────────
# SECTION 3: GLOBAL CONFIGURATION
# ─────────────────────────────────────────────


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


MODELO_RAG = os.getenv("OLLAMA_RAG_MODEL", "gemma4:e4b")
MODELO_CHAT = os.getenv("OLLAMA_CHAT_MODEL", "gemma4:e2b")
MODELO_EMBEDDING = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")
MODELO_CONTEXTUAL = os.getenv("OLLAMA_CONTEXTUAL_MODEL", "gemma4:e4b")
MODELO_RECOMP = os.getenv("OLLAMA_RECOMP_MODEL", "gemma4:e4b")
MODELO_OCR = os.getenv("OLLAMA_OCR_MODEL", "gemma4:e4b")


def _leer_env_bool(nombre_variable: str, default: bool) -> bool:
    """Parse a boolean environment variable with tolerant string values.

    Args:
        nombre_variable: Environment variable name to inspect.
        default: Fallback value when the variable is undefined or empty.

    Returns:
        Parsed boolean value.
    """
    valor = os.getenv(nombre_variable)
    if valor is None:
        return default

    valor_normalizado = valor.strip().lower()
    if valor_normalizado in {"1", "true", "yes", "y", "on", "si", "sí"}:
        return True
    if valor_normalizado in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _inferir_descripcion_modelo(nombre_modelo: str) -> str:
    """Extract the base model name by stripping the tag suffix.

    Args:
        nombre_modelo: Full model identifier (e.g. ``"gemma3:4b"``).

    Returns:
        Model name without the colon-separated tag.
    """
    return nombre_modelo.split(":")[0]


MODELO_DESC = os.getenv("MODELO_DESC", _inferir_descripcion_modelo(MODELO_RAG))


USAR_CONTEXTUAL_RETRIEVAL = True
USAR_LLM_QUERY_DECOMPOSITION = True
USAR_BUSQUEDA_HIBRIDA = True
USAR_BUSQUEDA_EXHAUSTIVA = True
USAR_RERANKER = RERANKER_AVAILABLE
EXPANDIR_CONTEXTO = True
USAR_OPTIMIZACION_CONTEXTO = True
USAR_RECOMP_SYNTHESIS = _leer_env_bool("USAR_RECOMP_SYNTHESIS", True)
USAR_EMBEDDINGS_IMAGEN = True
EVAL_RAGBENCH_RERANKER_LOW_SCORE_FALLBACK = False
LOGGING_METRICAS = True
GUARDAR_DEBUG_RAG = True

PIPELINE_RUNTIME_FLAGS = (
    "USAR_LLM_QUERY_DECOMPOSITION",
    "USAR_BUSQUEDA_HIBRIDA",
    "USAR_BUSQUEDA_EXHAUSTIVA",
    "USAR_RERANKER",
    "EXPANDIR_CONTEXTO",
    "USAR_OPTIMIZACION_CONTEXTO",
    "USAR_RECOMP_SYNTHESIS",
)


def set_ragbench_reranker_low_score_fallback(enabled: bool) -> bool:
    """Allow RagBench evals to generate from low-scored reranker candidates.

    This is intentionally not a general pipeline flag: normal RAG inference and
    non-RagBench evaluations keep the reranker threshold as a hard relevance
    gate. RagBench includes short factual questions where the cross-encoder can
    score useful retrieved evidence below the interactive threshold; for those
    runs we still use the reranker order, but fall back to the best candidates
    instead of returning an empty answer.
    """
    global EVAL_RAGBENCH_RERANKER_LOW_SCORE_FALLBACK
    previous = EVAL_RAGBENCH_RERANKER_LOW_SCORE_FALLBACK
    EVAL_RAGBENCH_RERANKER_LOW_SCORE_FALLBACK = bool(enabled)
    return previous


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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_DOCS = os.getenv("DOCS_FOLDER", os.path.join(BASE_DIR, "docs", "es"))

_carpeta_nombre = os.path.basename(os.path.abspath(CARPETA_DOCS))
_embed_slug = MODELO_EMBEDDING.split(":")[0].replace("/", "_")

PATH_DB = os.path.join(BASE_DIR, "vector_db", f"{_carpeta_nombre}_{_embed_slug}")
COLLECTION_NAME = f"docs_{_carpeta_nombre}"

_DEFAULT_CARPETA_DOCS = CARPETA_DOCS
_DEFAULT_PATH_DB = PATH_DB
_DEFAULT_COLLECTION_NAME = COLLECTION_NAME


def set_docs_folder_runtime(carpeta: str | None) -> tuple[str, str, str]:
    """Switch ``CARPETA_DOCS`` and derived Chroma paths (for evaluation scripts/tests).

    Restores module-level defaults when ``carpeta`` is ``None`` (values captured
    at import from ``DOCS_FOLDER`` / ``rag/pdfs``).

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
        abs_carp = os.path.abspath(carpeta)
        cn = os.path.basename(abs_carp)
        slug = MODELO_EMBEDDING.split(":")[0].replace("/", "_")
        CARPETA_DOCS = abs_carp
        PATH_DB = os.path.join(BASE_DIR, "vector_db", f"{cn}_{slug}")
        COLLECTION_NAME = f"docs_{cn}"
    return previous


HISTORIAL_PATH = os.path.join(BASE_DIR, "historial_chat.json")
MAX_HISTORIAL_MENSAJES = 40

CARPETA_DEBUG_RAG = os.path.join(BASE_DIR, "debug_rag")


_embed_name_lower = MODELO_EMBEDDING.lower().split(":")[0]
if "nomic" in _embed_name_lower:
    EMBED_PREFIX_QUERY = "search_query: "
    EMBED_PREFIX_DOC = "search_document: "
    _EMBED_PREFIX_DESC = "nomic prefixes (query/doc)"
else:
    EMBED_PREFIX_QUERY = ""
    EMBED_PREFIX_DOC = ""
    _EMBED_PREFIX_DESC = "no prefixes (native)"

MAX_CHARS_EMBED = 4000
CHUNK_SIZE = 2000          # raised from 1500: keeps full subsections (e.g. 3.2.3) in one chunk
CHUNK_OVERLAP = 400        # raised from 350: ~20% overlap, proportional to new chunk size
MIN_CHUNK_LENGTH = 150     # raised from 80: discards very short artefact chunks (copyright, author lists)
MAX_IMAGENES_POR_PAGINA = 5
MIN_IMAGEN_SIZE_PX = 100
CAPTION_MARGIN_PX = 80          # px below image bbox to search for figure caption text
_IMAGEN_CHUNK_OFFSET = 10_000

N_RESULTADOS_SEMANTICOS = 80
N_RESULTADOS_KEYWORD = 40
TOP_K_RERANK_CANDIDATES = 200
TOP_K_AFTER_RERANK = 15
TOP_K_FINAL = 8              # raised from 6: more fragments reach RECOMP, reducing split-list failures
N_TOP_PARA_EXPANSION = 3

RERANKER_MODEL_QUALITY = os.getenv("RERANKER_QUALITY", "quality")
# Relevance gate on the *top* fused score. After Cross-Encoder reranking, ``score_final``
# is replaced by the reranker score (same order as ``UMBRAL_SCORE_RERANKER``). With
# ``USAR_RERANKER`` off, ``score_final`` stays RRF-based (much smaller scale); callers
# must not compare that to this threshold — see CLI/web/eval paths.
UMBRAL_RELEVANCIA = 0.50
UMBRAL_SCORE_RERANKER = 0.55  # raised from 0.40: debug showed fragments at 0.47-0.52 are noise

RRF_K = 20                    # reciprocal rank fusion damping factor (was hardcoded 60)

MIN_LONGITUD_PREGUNTA_RAG = 10
MAX_CONTEXTO_CHARS = 12000    # raised from 8192 to accommodate expanded neighbor chunks


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


# ─────────────────────────────────────────────
# SECTION 4: SYSTEM PROMPTS
# ─────────────────────────────────────────────


SYSTEM_PROMPT_CHAT = f"""
You are MonkeyGrab, the conversational assistant for a local academic RAG system (TFG project).
Your purpose is to help users query indexed PDF documents and understand the system itself.

---

### SYSTEM OVERVIEW
- **Architecture:** Runs fully locally using Ollama (LLM inference) and ChromaDB (vector store).
- **Modes:**
  1. **CHAT**: General conversation, project guidance, and command help. Maintains local history.
  2. **RAG**: Document-grounded answers from indexed PDFs using hybrid retrieval.

---

### RAG PIPELINE ARCHITECTURE (Technical Knowledge Base)

Use this reference to explain how the system works or which parts are mandatory vs. configurable.

#### 1. INDEXING PHASE
* **CORE (Mandatory):**
    * **Extraction & Chunking:** Reading PDFs and splitting text (`dividir_en_chunks`).
    * **Embeddings:** Converting text to vectors and saving to ChromaDB.
* **OPTIONAL (Flag: `USAR_CONTEXTUAL_RETRIEVAL`):**
    * **Contextual Retrieval:** Uses an LLM to generate a summary/context for each chunk before indexing to improve retrieval accuracy.
* **OPTIONAL (Flag: `USAR_EMBEDDINGS_IMAGEN`):**
    * **Image Indexing:** Extracts raster images from each PDF page with PyMuPDF (fitz), describes them with `MODELO_OCR` (default: `gemma4:e4b`, override with `OLLAMA_OCR_MODEL`) using a structured OCR prompt that transcribes tables cell by cell, chart axes and legends, diagram components, and equations, then stores the result as a regular text chunk in ChromaDB.

#### 2. RETRIEVAL PHASE
Orchestrated by `realizar_busqueda_hibrida`. Core is semantic (vector) search; optional components extend it.
* **CORE (Mandatory):**
    * **Semantic Search:** Vector distance lookup; always performed.
* **OPTIONAL (execution order):**
    * **Query Decomposition** (`USAR_LLM_QUERY_DECOMPOSITION`): Uses an auxiliary LLM to generate sub-queries before semantic search; activates for long questions (>60 chars).
    * **Hybrid Search** (`USAR_BUSQUEDA_HIBRIDA`): Adds keyword/lexical search.
    * **Exhaustive Search** (`USAR_BUSQUEDA_EXHAUSTIVA`): Deep scan for critical terms (computationally expensive).

#### 3. RANKING & REFINEMENT
* **OPTIONAL:**
    * **Reranking** (`USAR_RERANKER`): Uses a Cross-Encoder (requires `sentence-transformers`) to re-score the top results for higher precision.

#### 4. CONTEXT & GENERATION
* **CORE (Mandatory):**
    * **Generation:** The RAG model (`MODELO_RAG`) generates the final answer based on the retrieved text.
* **OPTIONAL:**
    * **Context Optimization** (`USAR_OPTIMIZACION_CONTEXTO`): Cleans PDF artifacts (headers, footers, noise) before sending to the LLM.
    * **Neighbor Expansion** (`EXPANDIR_CONTEXTO`): Retrieves adjacent chunks to provide continuous context.
    * **RECOMP Synthesis** (`USAR_RECOMP_SYNTHESIS`): Uses `MODELO_RECOMP` (separate from `MODELO_RAG`) to summarize/synthesize the context instead of feeding raw chunks (Default: False).

---

### BEHAVIOR RULES
1. **Conciseness:** Be concise by default. Expand only when asked.
2. **Honesty:** Never fabricate system state or document contents. If you don't know, say so.
3. **Guidance:** If a user asks "what should I do?", provide concrete next steps (e.g., suggest switching to RAG mode to search their PDFs).
4. **Mode Enforcement:** If the user asks for information contained in the documents while in CHAT mode, redirect them to use RAG mode for document-grounded answers.
5. **Language:** Always respond in the exact same language the user uses. If they write in Spanish, respond in Spanish. If they write in Catalan, respond in Catalan. If they write in English, respond in English. Never switch languages mid-conversation.
6. **Tone:** Professional, academic, yet approachable.
"""

SYSTEM_PROMPT_RAG = """You are a professional document analysis assistant. Your role is to answer questions accurately based on the provided document context.

Guidelines:
- Base your answers strictly on the information within the <context> tags.
- Do not add information beyond what the context provides.
- Preserve technical terms, notation, formulas, and numbers exactly as they appear.
- Formulate clear, well-structured responses in complete sentences.
- For factual questions, be direct and precise.
- For analytical or complex questions, provide detailed explanations referencing specific information from the context.
- Always respond in the same language as the context (English, Spanish/Castellano, or Catalan/Català)."""


def _modelo_necesita_system_prompt(nombre_modelo: str) -> bool:
    """Return True if the model does not have a system prompt baked in its Modelfile.

    Fine-tuned models in this project include 'finetuned' in their Ollama name
    (e.g. phi4-finetuned:latest) and already carry the RAG system prompt via
    their Modelfile. Any other model receives the prompt explicitly via the API.
    """
    return "finetuned" not in nombre_modelo.lower()


# ─────────────────────────────────────────────

# -----------------------------------------------------------------------------
# BUSINESS LOGIC FACADE
# -----------------------------------------------------------------------------
# The implementation lives in rag.engine.*. Keep these imports explicit so all
# existing callers can continue to use rag.chat_pdfs as the public runtime API.

from rag.engine.history import cargar_historial, guardar_historial, limpiar_historial
from rag.engine.chunking import extraer_header_markdown, dividir_en_chunks, expandir_con_chunks_adyacentes
from rag.engine.lexical import (
    STOPWORDS,
    TERMINOS_EXPANSION,
    GENERIC_TERMS_BLACKLIST,
    extraer_keywords,
    busqueda_por_keywords,
    busqueda_exhaustiva_texto,
)
from rag.engine.reranking import (
    _detectar_dispositivo_reranker,
    obtener_modelo_reranker,
    rerank_resultados,
    generar_queries_con_llm,
    _validar_coherencia_query,
    _filtrar_terminos_criticos,
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
    _generar_respuesta_stream,
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
