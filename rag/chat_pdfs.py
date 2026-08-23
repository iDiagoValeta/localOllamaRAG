"""MonkeyGrab -- RAG engine for PDF document queries.

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
    points elsewhere. Generation model names use ``OLLAMA_RAG_MODEL`` and the
    related Ollama roles documented in the project README
    and in ``rag/README.md``.
"""

import logging
import os
import sys
import warnings
from typing import Dict


_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# rag.engine.* delegates parts of its logic to monkeygrab.application (the
# hexagonal-architecture layer under src/), so that package must be
# importable too -- mirrors pytest.ini's `pythonpath = . src`, which only
# covers the test process, not this module's own standalone/direct-execution
# bootstrap above.
_src_root = os.path.join(_project_root, "src")
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)


# Load configuration from a ``.env`` file at the project root, if present.
# ``override=False`` keeps the process environment authoritative over the file,
# matching the precedence documented in the README. ``python-dotenv`` is an
# optional convenience: when absent, variables must be exported in the shell.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_project_root, ".env"), override=False)
except ImportError:
    pass


# Compatibility constants, kept because the CLI and the web UI display them.
# They no longer gate anything: extraction, reranking and lexical search each
# import their own library inside the adapter that uses it, and each raises
# there if it is missing. Nothing in this module needs those libraries, so
# importing them here only slowed startup for callers that never index.

RERANKER_AVAILABLE = True
BM25_AVAILABLE = True


# GLOBAL CONFIGURATION


# Terminal runtime

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

# Environment readers


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


# Model roles and Ollama runtime

MODELO_RAG = os.getenv("OLLAMA_RAG_MODEL", "gemma4:e4b")
MODELO_CHAT = os.getenv("OLLAMA_CHAT_MODEL", "gemma4:e4b")
MODELO_CONTEXTUAL = os.getenv("OLLAMA_CONTEXTUAL_MODEL", "gemma4:e4b")
MODELO_RECOMP = os.getenv("OLLAMA_RECOMP_MODEL", "gemma4:e4b")
MODELO_DESC = os.getenv("MODELO_DESC", _inferir_descripcion_modelo(MODELO_RAG))

OLLAMA_RAG_NUM_CTX = _leer_env_int("OLLAMA_RAG_NUM_CTX", 16384)
OLLAMA_QUERY_NUM_CTX = _leer_env_int("OLLAMA_QUERY_NUM_CTX", 2048)
OLLAMA_RECOMP_NUM_CTX = _leer_env_int("OLLAMA_RECOMP_NUM_CTX", 8192)
OLLAMA_CONTEXTUAL_NUM_CTX = _leer_env_int("OLLAMA_CONTEXTUAL_NUM_CTX", 32768)
OLLAMA_REQUEST_TIMEOUT = _leer_env_int("OLLAMA_REQUEST_TIMEOUT", 900)
# Seconds to keep weights in VRAM after each Ollama call; 0 unloads immediately.
# Model load is 93-95% of query wall time (issue #25, 2026-07-29): at 120s,
# queries within that window reuse the resident weights instead of paying a
# ~170s cold load again. Costs VRAM residency for that long after every call
# -- lower it (or set 0) on a card that needs the headroom back sooner.
OLLAMA_KEEP_ALIVE = _leer_env_int("OLLAMA_KEEP_ALIVE", 120)
OLLAMA_GENERATE_RETRIES = _leer_env_int("OLLAMA_GENERATE_RETRIES", 2)
OLLAMA_GENERATE_RETRY_DELAY = _leer_env_int("OLLAMA_GENERATE_RETRY_DELAY", 3)


# Pipeline flags

USAR_CONTEXTUAL_RETRIEVAL = True
USAR_LLM_QUERY_DECOMPOSITION = True
USAR_BUSQUEDA_HIBRIDA = True
USAR_RERANKER = True
EXPANDIR_CONTEXTO = True
USAR_OPTIMIZACION_CONTEXTO = True
USAR_RECOMP_SYNTHESIS = True
USAR_EMBEDDINGS_IMAGEN = True
# Index-time; requires a vision-capable OLLAMA_CHAT_MODEL and a fresh index.
USAR_DESCRIPCION_IMAGEN = False
LOGGING_METRICAS = True
GUARDAR_DEBUG_RAG = True

# Runtime toggles are inference-time only; indexing flags require a fresh index.
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
    the FAISS index to be compared fairly.
    """
    invalid = sorted(set(overrides) - set(PIPELINE_RUNTIME_FLAGS))
    if invalid:
        valid = ", ".join(PIPELINE_RUNTIME_FLAGS)
        raise ValueError(f"Unsupported pipeline flag(s): {', '.join(invalid)}. Valid: {valid}")

    previous = get_pipeline_flags()
    for name, value in overrides.items():
        globals()[name] = bool(value)
    return previous


# Paths and persistence

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Writable data root. Defaults to the package dir for source/dev runs (paths stay
# identical to before), but the packaged desktop app points it at a per-user
# location via ``MONKEYGRAB_DATA_DIR`` (e.g. ``%LOCALAPPDATA%/MonkeyGrab``) so the
# vector DBs, history and debug dumps live outside the read-only application bundle.
DATA_DIR = os.path.abspath(os.getenv("MONKEYGRAB_DATA_DIR", BASE_DIR))

CARPETA_DOCS = os.getenv("DOCS_FOLDER", os.path.join(BASE_DIR, "docs", "en"))


def _derivar_paths_db(carpeta: str) -> tuple[str, str]:
    """Derive the fixed jina-clip/FAISS path for a docs folder.

    Args:
        carpeta: PDF directory (absolute or relative).
    Returns:
        Tuple ``(path_db, collection_name)``.
    """
    nombre = os.path.basename(os.path.abspath(carpeta))
    return (
        os.path.join(DATA_DIR, "vector_db", f"{nombre}_jina_clip"),
        f"docs_{nombre}",
    )


PATH_DB, COLLECTION_NAME = _derivar_paths_db(CARPETA_DOCS)

_DEFAULT_CARPETA_DOCS = CARPETA_DOCS
_DEFAULT_PATH_DB = PATH_DB
_DEFAULT_COLLECTION_NAME = COLLECTION_NAME


def set_docs_folder_runtime(carpeta: str | None) -> tuple[str, str, str]:
    """Switch ``CARPETA_DOCS`` and derived FAISS paths.

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
        PATH_DB, COLLECTION_NAME = _derivar_paths_db(CARPETA_DOCS)
    return previous


HISTORIAL_PATH = os.path.join(DATA_DIR, "historial_chat.json")
MAX_HISTORIAL_MENSAJES = 40

CARPETA_DEBUG_RAG = os.path.join(DATA_DIR, "debug_rag")


# Indexing, retrieval and ranking parameters

CONTEXTUAL_DOC_CHARS = _leer_env_int("CONTEXTUAL_DOC_CHARS", 24000)
CHUNK_SIZE = _leer_env_int("RAG_CHUNK_SIZE", 2000)
CHUNK_OVERLAP = _leer_env_int("RAG_CHUNK_OVERLAP", 400)
MIN_CHUNK_LENGTH = _leer_env_int("RAG_MIN_CHUNK_LENGTH", 150)
_IMAGEN_CHUNK_OFFSET = 10_000

N_RESULTADOS_SEMANTICOS = _leer_env_int("RAG_N_RESULTADOS_SEMANTICOS", 80)
N_RESULTADOS_KEYWORD = _leer_env_int("RAG_N_RESULTADOS_KEYWORD", 40)
TOP_K_RERANK_CANDIDATES = _leer_env_int("RAG_TOP_K_RERANK_CANDIDATES", 200)
TOP_K_FINAL = _leer_env_int("RAG_TOP_K_FINAL", 8)
N_TOP_PARA_EXPANSION = _leer_env_int("RAG_N_TOP_PARA_EXPANSION", 3)

UMBRAL_SCORE_RERANKER = _leer_env_float("RAG_UMBRAL_SCORE_RERANKER", 0.65)

RRF_K = _leer_env_int("RAG_RRF_K", 60)
BM25_K1 = _leer_env_float("RAG_BM25_K1", 1.5)
BM25_B = _leer_env_float("RAG_BM25_B", 0.75)
PESO_SEMANTICO_RRF = _leer_env_float("RAG_PESO_SEMANTICO_RRF", 0.55)
PESO_BM25_RRF = _leer_env_float("RAG_PESO_BM25_RRF", 0.45)

MIN_LONGITUD_PREGUNTA_RAG = _leer_env_int("RAG_MIN_LONGITUD_PREGUNTA", 10)
MAX_CONTEXTO_CHARS = _leer_env_int("MAX_CONTEXTO_CHARS", 24000)


# Logging and process environment

LOG_LEVEL = logging.ERROR

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s: %(message)s")

for _logger_name in (
    "httpx",
    "urllib3",
    "requests",
    "sentence_transformers",
    "transformers",
    "huggingface_hub",
    "tqdm",
    "filelock",
):
    logging.getLogger(_logger_name).setLevel(logging.CRITICAL)

warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*huggingface.*")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


# Model role runtime switching

# Maps the public role keys used by the web UI / API to the module-level model
# variables. Engine modules read these lazily through ``cfg``, so reassigning
# them takes effect on the next pipeline call without a restart.
MODEL_ROLE_VARS = {
    "rag": "MODELO_RAG",
    "chat": "MODELO_CHAT",
    "contextual": "MODELO_CONTEXTUAL",
    "recomp": "MODELO_RECOMP",
}

# The variable that pins each role, mirroring the ``os.getenv`` calls that read
# them above. A role listed here whose variable is set keeps its environment
# value: persisted UI choices describe an earlier run, the environment describes
# this one (see ``rag/engine/settings.py``).
MODEL_ROLE_ENV_VARS = {
    "rag": "OLLAMA_RAG_MODEL",
    "chat": "OLLAMA_CHAT_MODEL",
    "contextual": "OLLAMA_CONTEXTUAL_MODEL",
    "recomp": "OLLAMA_RECOMP_MODEL",
}


def get_model_roles() -> Dict[str, str]:
    """Return the Ollama model currently assigned to each pipeline role."""
    return {role: globals()[var] for role, var in MODEL_ROLE_VARS.items()}


def set_model_roles_runtime(overrides: Dict[str, str]) -> Dict[str, str]:
    """Reassign pipeline model roles for the current process.

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

    global MODELO_RAG, MODELO_CHAT, MODELO_CONTEXTUAL, MODELO_RECOMP, MODELO_DESC

    for role, modelo in overrides.items():
        modelo = (modelo or "").strip()
        if not modelo:
            continue
        globals()[MODEL_ROLE_VARS[role]] = modelo

    if (overrides.get("rag") or "").strip():
        MODELO_DESC = _inferir_descripcion_modelo(MODELO_RAG)
    return get_model_roles()


# SYSTEM PROMPTS


SYSTEM_PROMPT_CHAT = """
You are MonkeyGrab, the conversational assistant for a local academic RAG system (TFG project).
Your purpose is to help users query indexed PDF documents and understand the system itself.

---

### SYSTEM OVERVIEW
- **Architecture:** Runs fully locally using Ollama for generation and FAISS for retrieval.
- **Model configuration:** All model roles are configurable through environment variables. Explain roles and variables, not fixed model names.
- **Modes:**
  1. **CHAT**: General conversation, project guidance, and command help. Maintains local history.
  2. **RAG**: Document-grounded answers from indexed PDFs using hybrid retrieval.

---

### RAG PIPELINE ARCHITECTURE (Technical Knowledge Base)

Use this reference to explain how the system works or which parts are mandatory vs. configurable.

#### 1. INDEXING PHASE
* **CORE (Mandatory):**
    * **Extraction & Chunking:** MinerU preserves text, tables and figures.
    * **Embeddings:** jina-clip-v2 embeds text and images in one shared space and stores them in FAISS.
* **OPTIONAL (Flag: `USAR_CONTEXTUAL_RETRIEVAL`):**
    * **Contextual Retrieval:** Uses `MODELO_CONTEXTUAL` configured through `OLLAMA_CONTEXTUAL_MODEL` to generate summary/context for each chunk before indexing to improve retrieval accuracy.
* **OPTIONAL (Flag: `USAR_EMBEDDINGS_IMAGEN`):**
    * **Image Indexing:** MinerU extracts figures and jina-clip-v2 embeds them directly.

#### 2. RETRIEVAL PHASE
Orchestrated by `realizar_busqueda_hibrida`. Core is semantic (vector) search; optional components extend it.
* **CORE (Mandatory):**
    * **Semantic Search:** Cosine search in the shared jina-clip-v2 space; always performed.
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
5. **Configuration framing:** Distinguish the configurable Ollama generation roles from the fixed MinerU, Jina CLIP, FAISS and BGE retrieval stack.
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


from rag.engine.history import cargar_historial, guardar_historial, limpiar_historial
from rag.engine.chunking import dividir_en_chunks, expandir_con_chunks_adyacentes
from monkeygrab.adapters.reranking.cross_encoder_reranker import resolve_reranker_device
from monkeygrab.application.keywords import (
    GENERIC_TERMS_BLACKLIST,
    STOPWORDS,
    extract_keywords,
    is_coherent_query,
    tokenize_bm25,
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
    _preparar_mensaje_usuario_rag,
    generar_tokens_respuesta,
    _generar_respuesta_stream,
    preparar_fragmentos_para_generacion,
    generar_respuesta,
    generar_respuesta_silenciosa,
    evaluar_pregunta_rag,
)
from rag.engine.indexing import (
    index_fingerprint_mismatch,
    indexar_documentos,
    obtener_documentos_indexados,
)
from rag.engine.settings import (
    STORE_IDS,
    cargar_ajustes_persistidos,
    guardar_ajustes_persistidos,
    resolver_carpeta_store,
)
from rag.engine.wiring import (
    app_config_from_runtime,
    reset_vector_store_cache,
    vector_store,
)


def obtener_vector_store():
    """Return the FAISS store for the active corpus."""
    return vector_store(app_config_from_runtime())


def main():
    """Launch the MonkeyGrab CLI application."""
    from rag.cli import MonkeyGrabCLI

    # ``sys.modules[__name__]``, not ``import rag.chat_pdfs``: run as a script
    # this file is ``__main__``, and importing it by name would execute a second
    # copy whose globals nothing else reads -- the engine modules bind the
    # ``__main__`` one (see rag/engine/runtime.py), and so does the settings
    # loader below. Handing the CLI the other copy is how it would end up
    # displaying, listing and indexing one configuration while retrieval and
    # generation ran on another.
    rag_engine = sys.modules[__name__]
    # Same file the web control panel writes, applied here rather than inside
    # the CLI because this is where the engine is composed -- the CLI receives a
    # configured module, exactly as the Flask app does. Without it the CLI would
    # start from this module's defaults while the index on disk was built from
    # the UI's choices, which is what the fingerprint warning then reports with
    # nothing in the session to explain it.
    cargar_ajustes_persistidos()
    cli = MonkeyGrabCLI(rag_engine)
    cli.run()


if __name__ == "__main__":
    main()
