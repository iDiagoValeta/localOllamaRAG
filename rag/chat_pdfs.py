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
#  BUSINESS LOGIC
#  +-- 4. System prompts      CHAT (identity + language); RAG prompt baked into Modelfile
#  +-- 5. History persistence  local JSON for CHAT mode
#  +-- 6. Preprocessing and chunking  text -> Markdown chunks with overlap; expandir_con_chunks_adyacentes
#  +-- 7. Keywords and lexical search  acronyms, bigrams, where_document
#  +-- 8. Semantic reranking   CrossEncoder singleton; generar_queries_con_llm, _validar_coherencia_query, _filtrar_terminos_criticos
#  +-- 9. Hybrid retrieval pipeline  semantic + keywords + exhaustive -> RRF -> rerank
#  +--10. Context and generation
#  |      +-- 10.1 PDF text optimization (artifacts, footers, paragraphs)
#  |      +-- 10.2 Context construction (raw or RECOMP synthesis)
#  |      +-- 10.3 Debug         interaction dump in debug_rag/
#  |      +-- 10.4 Generation    Ollama streaming, <context> format
#  |      +-- 10.5 Evaluation    silent pipeline for RAGAS
#  +--11. Indexing and collection management
#  |      +-- 11.1 Contextual retrieval  chunk enrichment with LLM
#  |      |      +-- _detectar_idioma       heuristic language detector (Spanish/Catalan/English)
#  |      +-- 11.2 Image extraction      fitz raster images + positional caption extraction
#  |      |      +-- _es_descripcion_spam    degenerate OCR spam / repetition detector
#  |      |      +-- _es_prompt_echo         prompt-echo detector
#  |      |      +-- _es_solo_caption        caption-echo detector
#  |      |      +-- extraer_imagenes_pdf    extract images + captions per page
#  |      |      +-- describir_imagen_con_llm  glm-ocr vision description
#  |      +-- 11.3 Indexing              PDFs -> pymupdf4llm/pypdf + images -> ChromaDB
#  |      +-- 11.4 Collection mgmt      list indexed documents
#  |
#  ENTRY
#  +--12. Entry point   main() -> MonkeyGrabCLI.run()
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


def set_recomp_synthesis_enabled(enabled: bool) -> bool:
    """Override RECOMP synthesis usage for the current Python process.

    Args:
        enabled: Whether RECOMP synthesis should be enabled.

    Returns:
        Previous RECOMP flag value.
    """
    global USAR_RECOMP_SYNTHESIS
    anterior = USAR_RECOMP_SYNTHESIS
    USAR_RECOMP_SYNTHESIS = bool(enabled)
    return anterior


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


# ─────────────────────────────────────────────
# SECTION 5: HISTORY PERSISTENCE
# ─────────────────────────────────────────────


def cargar_historial() -> List[Dict[str, str]]:
    """Load CHAT history from disk.

    Returns:
        List of message dicts. Empty list if the file is missing or corrupt.
    """
    try:
        if os.path.exists(HISTORIAL_PATH):
            with open(HISTORIAL_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and "chat" in data:
                    return data["chat"]
                if isinstance(data, list):
                    return data
    except Exception as e:
        logging.warning(f"Error loading history: {e}")

    return []


def guardar_historial(historial: List[Dict[str, str]]) -> None:
    """Persist CHAT history to disk (last MAX_HISTORIAL_MENSAJES entries).

    Args:
        historial: Full list of message dicts to save.
    """
    try:
        historial_recortado = historial[-MAX_HISTORIAL_MENSAJES:]
        with open(HISTORIAL_PATH, 'w', encoding='utf-8') as f:
            json.dump(historial_recortado, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Error saving history: {e}")


def limpiar_historial(historial: List[Dict[str, str]]) -> None:
    """Clear history in-place and persist the empty state.

    Args:
        historial: The history list to clear.
    """
    historial.clear()
    guardar_historial(historial)


# ─────────────────────────────────────────────
# SECTION 6: PREPROCESSING AND CHUNKING
# ─────────────────────────────────────────────


def extraer_header_markdown(texto: str) -> str:
    """Extract the last Markdown header (``# ...``) from the text.

    Args:
        texto: Raw text that may contain Markdown headers.

    Returns:
        The last header string found, or empty string if none.
    """
    headers = re.findall(r'^(#{1,4}\s+.+)$', texto, re.MULTILINE)
    return headers[-1].strip() if headers else ""


def dividir_en_chunks(
    texto: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, str]]:
    """Split text into chunks by Markdown sections with overlap.

    Uses a recursive separator strategy to break content at natural
    boundaries (paragraphs, sentences, commas, spaces). Each chunk
    preserves its nearest Markdown header for context.

    Args:
        texto: Source text to split.
        chunk_size: Maximum character length per chunk.
        overlap: Number of trailing characters from the previous chunk
            to prepend to the next one.

    Returns:
        List of dicts with ``"text"`` and ``"header"`` keys.
    """
    if not texto or not texto.strip():
        return []

    texto = re.sub(r'~~`?[^~]*`?~~', '', texto)
    texto = re.sub(r'(?<!\*)\*{1,2}(?!\*)', '', texto)
    texto = re.sub(r'`([^`\n]{1,3})`', r'\1', texto)
    texto = re.sub(r'\n{3,}', '\n\n', texto)

    header_pattern = re.compile(
        r'^(?:#{1,4}\s+.+|\*\*(?:[A-Z0-9].+?)\*\*\s*)$',
        re.MULTILINE
    )

    secciones = []
    last_end = 0
    current_header = ""

    for match in header_pattern.finditer(texto):
        contenido_previo = texto[last_end:match.start()].strip()
        if contenido_previo:
            secciones.append({"header": current_header, "content": contenido_previo})

        raw_header = match.group(0).strip()
        current_header = re.sub(r'^\*\*(.+?)\*\*$', r'\1', raw_header).strip()
        last_end = match.end()

    contenido_final = texto[last_end:].strip()
    if contenido_final:
        secciones.append({"header": current_header, "content": contenido_final})

    if not secciones:
        secciones = [{"header": "", "content": texto.strip()}]

    separadores = ["\n\n", "\n", ". ", ".\n", "! ", "? ", "; ", ", ", " "]

    def _split_recursivo(text: str, max_size: int, depth: int = 0) -> List[str]:
        """Recursively split text using hierarchical separators."""
        if len(text) <= max_size:
            return [text] if text.strip() else []

        for sep_idx, separador in enumerate(separadores):
            if separador not in text:
                continue

            partes = text.split(separador)
            resultado = []
            chunk_actual = ""

            for i, parte in enumerate(partes):
                parte_con_sep = parte + separador if i < len(partes) - 1 else parte

                if len(chunk_actual) + len(parte_con_sep) <= max_size:
                    chunk_actual += parte_con_sep
                else:
                    if chunk_actual.strip():
                        resultado.append(chunk_actual.strip())

                    if len(parte_con_sep) > max_size and depth < len(separadores) - 1:
                        resultado.extend(_split_recursivo(parte_con_sep, max_size, depth + 1))
                        chunk_actual = ""
                    else:
                        while len(parte_con_sep) > max_size:
                            resultado.append(parte_con_sep[:max_size].strip())
                            parte_con_sep = parte_con_sep[max_size:]
                        chunk_actual = parte_con_sep

            if chunk_actual.strip():
                resultado.append(chunk_actual.strip())

            if resultado:
                return resultado

        resultado = []
        for i in range(0, len(text), max_size):
            fragmento = text[i:i + max_size].strip()
            if fragmento:
                resultado.append(fragmento)
        return resultado

    fragmentos_raw = []
    for seccion in secciones:
        header = seccion["header"]
        content = seccion["content"]

        header_prefix = f"{header}\n" if header else ""
        espacio_contenido = chunk_size - len(header_prefix)

        if espacio_contenido < MIN_CHUNK_LENGTH:
            espacio_contenido = chunk_size
            header_prefix = ""

        partes = _split_recursivo(content, espacio_contenido)

        for parte in partes:
            texto_chunk = (header_prefix + parte).strip()
            if len(texto_chunk) >= MIN_CHUNK_LENGTH:
                fragmentos_raw.append({"text": texto_chunk, "header": header})

    if not fragmentos_raw:
        if len(texto.strip()) >= MIN_CHUNK_LENGTH:
            return [{"text": texto.strip()[:chunk_size], "header": ""}]
        return []

    chunks_finales = []
    for i, frag in enumerate(fragmentos_raw):
        texto_chunk = frag["text"]

        if i > 0 and overlap > 0:
            prev_text = fragmentos_raw[i - 1]["text"]
            overlap_text = prev_text[-overlap:]
            space_idx = overlap_text.find(' ')
            if space_idx > 0:
                overlap_text = overlap_text[space_idx + 1:]
            if overlap_text and overlap_text not in texto_chunk[:overlap + 50]:
                texto_chunk = overlap_text + " " + texto_chunk

        chunks_finales.append({
            "text": texto_chunk.strip(),
            "header": frag["header"]
        })

    return chunks_finales


def expandir_con_chunks_adyacentes(
    chunk_id: str,
    metadata: Dict[str, Any],
    n_vecinos: int = 1
) -> List[str]:
    """Build IDs for neighboring chunks (same-page and cross-page) for context expansion.

    Args:
        chunk_id: Identifier of the anchor chunk.
        metadata: Metadata dict with ``source``, ``page``, ``chunk``,
            and optionally ``total_chunks_in_page``.
        n_vecinos: How many neighbors to include on each side.

    Returns:
        List of neighboring chunk IDs.
    """
    archivo = metadata['source']
    pagina = metadata['page']
    chunk_num = metadata.get('chunk', 0)
    total_in_page = metadata.get('total_chunks_in_page', None)

    ids_adyacentes = []

    for i in range(1, n_vecinos + 1):
        if chunk_num - i >= 0:
            ids_adyacentes.append(f"{archivo}_pag{pagina}_chunk{chunk_num - i}")

    if chunk_num == 0 and pagina > 0:
        for last_c in range(3):
            ids_adyacentes.append(f"{archivo}_pag{pagina - 1}_chunk{last_c}")

    if total_in_page is not None:
        for i in range(1, n_vecinos + 1):
            if chunk_num + i < total_in_page:
                ids_adyacentes.append(f"{archivo}_pag{pagina}_chunk{chunk_num + i}")

        if chunk_num >= total_in_page - 1:
            for first_c in range(min(2, n_vecinos + 1)):
                ids_adyacentes.append(f"{archivo}_pag{pagina + 1}_chunk{first_c}")
    else:
        ids_adyacentes.append(f"{archivo}_pag{pagina + 1}_chunk0")

    return ids_adyacentes


# ─────────────────────────────────────────────
# SECTION 7: KEYWORD AND LEXICAL SEARCH
# ─────────────────────────────────────────────


STOPWORDS = {
    # Castellano
    'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'de', 'del', 'en', 'a', 'al',
    'por', 'para', 'con', 'sin', 'sobre', 'entre', 'hacia', 'desde', 'hasta', 'durante', 'mediante',
    'según', 'contra', 'que', 'quien', 'cual', 'cuales', 'cuyo', 'cuya', 'cuyos', 'cuyas',
    'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas', 'aquel', 'aquella', 'aquellos', 'aquellas',
    'esto', 'eso', 'aquello', 'y', 'o', 'pero', 'sino', 'aunque', 'si', 'porque', 'cuando', 'donde',
    'como', 'más', 'menos', 'muy', 'poco', 'mucho', 'algo', 'nada', 'todo', 'toda', 'todos', 'todas',
    'cada', 'otro', 'otra', 'otros', 'otras', 'mismo', 'misma', 'mismos', 'mismas',
    'es', 'son', 'está', 'están', 'era', 'eran', 'fue', 'fueron', 'ser', 'estar',
    'hay', 'había', 'han', 'haber', 'tiene', 'tienen', 'tenía', 'tener', 'puede', 'pueden', 'poder',
    'se', 'me', 'te', 'nos', 'os', 'le', 'lo', 'les', 'su', 'sus', 'mi', 'tu', 'nuestro', 'vuestro',
    'aquí', 'ahí', 'allí', 'así', 'ya', 'también', 'solo', 'sólo', 'siempre', 'nunca', 'después', 'antes',
    'explica', 'explicar', 'describe', 'describir', 'detalla', 'detallar', 'indica', 'indicar',
    'respuesta', 'pregunta', 'preguntas', 'siguientes', 'siguiente', 'puntos', 'punto', 'ejemplo',
    'manera', 'forma', 'tipo', 'tipos', 'parte', 'partes', 'primer', 'primera', 'segundo', 'segunda',
    'tercer', 'tercera', 'uno', 'dos', 'tres', 'cuáles', 'cómo', 'qué', 'podrías', 'decirme', 'puedes',
    'principales', 'llaman', 'tanto', 'tan', 'fue', 'sido', 'siendo', 'hacer', 'ir',
    # English
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    'here', 'there', 'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now',
    'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'will', 'this', 'that',
    'it', 'its', 'they', 'them', 'their', 'we', 'our', 'you', 'your', 'he', 'she', 'him', 'her',
    'what', 'which', 'who', 'whom', 'whose',
    # Valencian
    'els', 'les', 'uns', 'unes', 'dels', 'als',
    'per', 'per a', 'amb', 'sense', 'des de', 'fins a', 'fins', 'dins', 'envers',
    'que', 'qui', 'qual', 'quals', 'què', 'aquest', 'aquesta', 'aquests', 'aquestes',
    'aquell', 'aquella', 'aquells', 'aquelles', 'això', 'allò', 'i', 'o', 'però', 'sinó',
    'perquè', 'quan', 'com', 'més', 'menys', 'molt', 'poc', 'res', 'tot', 'tota', 'tots', 'totes',
    'cada', 'altre', 'altra', 'altres', 'mateix', 'mateixa', 'mateixos', 'mateixes',
    'és', 'són', 'està', 'estan', 'era', 'eren', 'ha', 'han', 'hi ha', 'hi havia',
    'pot', 'poden', 'ser', 'estar', 'tenir', 'fer', 'anar', 'dir', 'veure',
    'aquí', 'allà', 'així', 'ja', 'també', 'només', 'sempre', 'mai', 'després', 'abans',
    'explica', 'explicar', 'descriu', 'detalla', 'detallar', 'indica', 'indicar',
    'resposta', 'pregunta', 'preguntes', 'següents', 'següent', 'punts', 'punt', 'exemple',
    'manera', 'forma', 'tipus', 'part', 'parts', 'primer', 'primera', 'segon', 'segona',
    'tercer', 'tercera', 'un', 'dos', 'tres', 'quins', 'quines', 'quin', 'quina',
}


TERMINOS_EXPANSION = {}

GENERIC_TERMS_BLACKLIST = {
    "paper", "according", "specific", "specifically", "terms", "allows",
    "allow", "achieve", "system", "model", "approach", "method", "results",
    "three", "two", "one", "first", "second", "following", "based",
    "using", "used", "show", "shows", "provide", "provides", "propose",
    "proposed", "models", "methods", "approaches", "direct",
    "training", "learning", "optimize", "scores", "phases", "primary",
    "compare", "evaluate", "section", "table", "figure", "described",
}


def extraer_keywords(texto: str) -> List[str]:
    """Extract acronyms, bigrams, parenthesized terms, and technical tokens.

    Filters stopwords, deduplicates case-insensitively, and removes
    single-word tokens already covered by multi-word n-grams.

    Args:
        texto: Input text (typically a user query).

    Returns:
        Deduplicated list of keywords sorted by specificity.
    """
    keywords = set()

    siglas = re.findall(r'\b[A-ZÁÉÍÓÚÑ]{2,}\b', texto)
    keywords.update(s for s in siglas if len(s) >= 2)

    parentesis = re.findall(r'\(([^)]+)\)', texto)
    for term in parentesis:
        term_clean = term.strip()
        if len(term_clean) > 2 and term_clean.lower() not in STOPWORDS:
            for sub in term_clean.split(','):
                sub_clean = sub.strip()
                if len(sub_clean) > 2 and sub_clean.lower() not in STOPWORDS and len(sub_clean) <= 50:
                    keywords.add(sub_clean)
            if len(term_clean) <= 50:
                keywords.add(term_clean)

    palabras = texto.split()
    for i in range(len(palabras) - 1):
        p1 = palabras[i].strip('¿?.,;:()[]{}"\'-')
        p2 = palabras[i + 1].strip('¿?.,;:()[]{}"\'-')

        if (len(p1) > 3 and len(p2) > 3 and
            p1.lower() not in STOPWORDS and p2.lower() not in STOPWORDS):
            bigrama = f"{p1} {p2}"
            keywords.add(bigrama.lower())

    for palabra in palabras:
        clean = palabra.strip('¿?.,;:()[]{}"\'-')
        if len(clean) > 3 and clean.lower() not in STOPWORDS:   # lowered from 4 to 3
            keywords.add(clean.lower())

    terminos_tecnicos = [
        palabra.strip('¿?.,;:()[]{}"\'-')
        for palabra in palabras
        if (any(c.isupper() for c in palabra[1:]) or
           any(c.isdigit() for c in palabra) or
           '-' in palabra) and
        len(palabra.strip('¿?.,;:()[]{}"\'-')) > 1
    ]
    keywords.update(t.lower() for t in terminos_tecnicos)
    keywords.update(terminos_tecnicos)

    keywords_expandidas = set(keywords)
    for kw in list(keywords):
        kw_lower = kw.lower()
        if kw_lower in TERMINOS_EXPANSION:
            keywords_expandidas.update(TERMINOS_EXPANSION[kw_lower])


    def _es_keyword_valida(kw: str) -> bool:
        if len(kw) > 50:
            return False
        if kw.strip().startswith('¿') or '?' in kw:
            return False
        return True

    keywords_filtradas = {k for k in keywords_expandidas if _es_keyword_valida(k)}

    seen_lower = set()
    deduped = []
    for kw in sorted(keywords_filtradas,
                     key=lambda x: (
                         0 if (x.isupper() or any(c.isupper() for c in x[1:]) or '-' in x) else 1,
                         len(x)
                     )):
        kw_lower = kw.lower()
        if kw_lower not in seen_lower:
            seen_lower.add(kw_lower)
            deduped.append(kw)

    ngrams = [kw for kw in deduped if len(kw.split()) >= 2]
    ngram_words = set()
    for ng in ngrams:
        for word in ng.lower().split():
            ngram_words.add(word)

    resultado = []
    for kw in deduped:
        if len(kw.split()) == 1 and kw.lower() in ngram_words:
            continue
        if len(kw.split()) == 1 and kw.lower() in GENERIC_TERMS_BLACKLIST:
            continue
        resultado.append(kw)

    return resultado


def busqueda_por_keywords(
    pregunta: str,
    collection: chromadb.Collection,
    n_results: int = N_RESULTADOS_KEYWORD
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Search by keyword containment using ChromaDB ``where_document``.

    Iterates over extracted keywords (with case variants and hyphen
    alternatives) and collects matching chunks via ``collection.get()``.

    Args:
        pregunta: User query to extract keywords from.
        collection: ChromaDB collection to search.
        n_results: Maximum results per keyword variant.

    Returns:
        Tuple of (matched chunks, metrics dict).
    """
    keywords = extraer_keywords(pregunta)

    metricas = {
        'keywords_totales': len(keywords),
        'keywords_encontradas': 0,
        'resultados_totales': 0,
        'errores': 0
    }

    if not keywords:
        return [], metricas

    resultados_keyword = []
    keywords_encontradas = set()
    ids_vistos = set()

    for keyword_base in keywords[:20]:
        variantes = list({
            keyword_base,
            keyword_base.lower(),
            keyword_base.capitalize(),
        })
        if '-' in keyword_base:
            variantes.append(keyword_base.replace('-', ' '))
            variantes.append(keyword_base.replace('-', ' ').lower())

        for keyword in variantes:
            if len(keyword) < 3:
                continue

            try:
                results = collection.get(
                    where_document={"$contains": keyword},
                    include=['documents', 'metadatas'],
                    limit=n_results
                )

                if results['documents']:
                    for doc, meta, doc_id in zip(
                        results['documents'],
                        results['metadatas'],
                        results['ids']
                    ):
                        if doc_id not in ids_vistos:
                            ids_vistos.add(doc_id)
                            resultados_keyword.append({
                                'doc': doc,
                                'metadata': meta,
                                'distancia': 0.5,
                                'keyword_match': keyword_base,
                                'id': doc_id
                            })
                            keywords_encontradas.add(keyword_base)
            except Exception as e:
                metricas['errores'] += 1
                logging.warning(f"Error searching keyword '{keyword}': {e}")

    metricas['keywords_encontradas'] = len(keywords_encontradas)
    metricas['resultados_totales'] = len(resultados_keyword)

    if keywords_encontradas:
        ui.debug(f"keywords: {', '.join(list(keywords_encontradas)[:10])}")
    else:
        ui.debug("no direct keyword matches")

    if LOGGING_METRICAS:
        logging.info(f"Keyword search: {metricas['keywords_encontradas']}/{metricas['keywords_totales']} found, {metricas['resultados_totales']} results")

    return resultados_keyword, metricas


def busqueda_exhaustiva_texto(
    terminos_criticos: List[str],
    collection: chromadb.Collection,
    max_results: int = 20
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Batch-scan all documents for critical terms.

    Iterates over the entire collection in batches and returns chunks
    containing the specified terms, sorted by match count.

    Args:
        terminos_criticos: Domain-specific terms to search for.
        collection: ChromaDB collection to scan.
        max_results: Maximum number of results to return.

    Returns:
        Tuple of (matched chunks sorted by match count, metrics dict).
    """
    resultados = []
    total_docs = collection.count()
    batch_size = 100

    metricas = {
        'documentos_escaneados': 0,
        'documentos_con_matches': 0,
        'errores': 0
    }

    for offset in range(0, total_docs, batch_size):
        try:
            batch = collection.get(
                limit=batch_size,
                offset=offset,
                include=['documents', 'metadatas']
            )

            metricas['documentos_escaneados'] += len(batch['documents'])

            for doc, meta, doc_id in zip(
                batch['documents'],
                batch['metadatas'],
                batch['ids']
            ):
                doc_lower = doc.lower()
                matches_encontrados = []

                for termino in terminos_criticos:
                    termino_lower = termino.lower()
                    if termino_lower in doc_lower:
                        matches_encontrados.append(termino)

                if matches_encontrados:
                    metricas['documentos_con_matches'] += 1
                    resultados.append({
                        'doc': doc,
                        'metadata': meta,
                        'id': doc_id,
                        'matches': matches_encontrados,
                        'num_matches': len(matches_encontrados)
                    })
        except Exception as e:
            metricas['errores'] += 1
            logging.warning(f"Error in exhaustive search (offset {offset}): {e}")

    resultados.sort(key=lambda x: x['num_matches'], reverse=True)

    if LOGGING_METRICAS:
        logging.info(f"Exhaustive search: {metricas['documentos_con_matches']} docs with matches out of {metricas['documentos_escaneados']} scanned")

    return resultados[:max_results], metricas


# ─────────────────────────────────────────────
# SECTION 8: SEMANTIC RERANKING
# ─────────────────────────────────────────────


_reranker_model = None


def _detectar_dispositivo_reranker() -> str:
    """Detect the best available device for the reranker.

    Returns:
        ``"cuda"`` if a CUDA GPU is available, otherwise ``"cpu"``.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def obtener_modelo_reranker():
    """Return the Cross-Encoder singleton, loading it lazily on first call.

    Uses FP16 on CUDA when a GPU is available. The model variant is
    controlled by ``RERANKER_MODEL_QUALITY`` (``"quality"`` or fast).

    Returns:
        The loaded ``CrossEncoder`` instance, or ``None`` on failure.
    """
    global _reranker_model

    if not USAR_RERANKER:
        return None

    if _reranker_model is None:
        try:
            if RERANKER_MODEL_QUALITY == "quality":
                modelo_nombre = "BAAI/bge-reranker-v2-m3"
            else:
                modelo_nombre = "cross-encoder/ms-marco-MiniLM-L-6-v2"

            device = _detectar_dispositivo_reranker()

            ui.debug(f"Loading reranker: {modelo_nombre}")
            ui.debug(f"device: {device.upper()}" + (" (FP16)" if device == "cuda" else ""))

            model_kwargs = {"torch_dtype": "float16"} if device == "cuda" else {}
            import io, contextlib
            with contextlib.redirect_stderr(io.StringIO()):
                _reranker_model = CrossEncoder(
                    modelo_nombre,
                    device=device,
                    model_kwargs=model_kwargs,
                )

            ui.debug(f"reranker loaded on {device.upper()}")
        except Exception as e:
            logging.error(f"Error loading reranker model: {e}")
            return None

    return _reranker_model


def rerank_resultados(
    pregunta: str,
    documentos_recuperados: List[Dict[str, Any]],
    top_k: int = TOP_K_AFTER_RERANK
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Reorder candidates using a Cross-Encoder for finer relevance scoring.

    Args:
        pregunta: The user query.
        documentos_recuperados: Candidate chunks from prior retrieval stages.
        top_k: Number of top results to keep after reranking.

    Returns:
        Tuple of (top-k documents with ``score_reranker``, metrics dict).
    """
    metricas = {
        'candidatos_entrada': len(documentos_recuperados),
        'resultados_salida': 0,
        'tiempo_reranking': 0,
        'modelo_usado': RERANKER_MODEL_QUALITY,
        'dispositivo': _detectar_dispositivo_reranker()
    }

    if not USAR_RERANKER or not documentos_recuperados:
        metricas['resultados_salida'] = len(documentos_recuperados)
        return documentos_recuperados, metricas

    reranker = obtener_modelo_reranker()
    if reranker is None:
        metricas['resultados_salida'] = len(documentos_recuperados)
        return documentos_recuperados, metricas

    try:
        import time
        inicio = time.time()

        textos_documentos = []
        for doc in documentos_recuperados:
            texto = doc['doc']
            if '\\n\\n' in texto:
                partes = texto.split('\\n\\n', 1)
                texto = partes[-1]
            textos_documentos.append(texto)

        import io, contextlib
        with contextlib.redirect_stderr(io.StringIO()):
            ranks = reranker.rank(
                pregunta,
                textos_documentos,
                top_k=min(top_k, len(textos_documentos)),
                return_documents=False
            )

        documentos_reordenados = []
        for rank_info in ranks:
            idx = rank_info['corpus_id']
            score = rank_info['score']

            doc_original = documentos_recuperados[idx].copy()
            doc_original['score_reranker'] = float(score)
            doc_original['score_final'] = float(score)
            documentos_reordenados.append(doc_original)

        metricas['tiempo_reranking'] = time.time() - inicio
        metricas['resultados_salida'] = len(documentos_reordenados)

        if LOGGING_METRICAS:
            dev = metricas['dispositivo'].upper()
            logging.info(f"Reranking ({dev}): {metricas['candidatos_entrada']} -> {metricas['resultados_salida']} in {metricas['tiempo_reranking']:.2f}s")

        return documentos_reordenados, metricas

    except Exception as e:
        logging.error(f"Error in reranking: {e}")
        metricas['resultados_salida'] = len(documentos_recuperados)
        return documentos_recuperados, metricas


def generar_queries_con_llm(pregunta: str) -> List[str]:
    """Generate 3 search sub-queries via an auxiliary LLM.

    Each sub-query targets a different semantic aspect of the original
    question and is written in the same language.

    Ollama ``think=False`` disables the reasoning trace for thinking-capable
    models (Gemma 4, Qwen3, etc.); see ``scripts/tests/test_gemma4_aux_nothink.py``.
    With ``think=True``, ``num_predict`` can be consumed by the trace alone, leaving
    an empty answer — production therefore keeps ``think=False`` and ``num_predict`` 400.
    To print raw model output in the terminal, run ``scripts/tests/debug_aux_subqueries.py``.

    Args:
        pregunta: The original user question.

    Returns:
        Up to 3 generated queries, or empty list on failure.
    """
    try:
        prompt = (
            "Generate exactly 3 search queries to retrieve relevant content "
            "from academic documents about the question below.\n\n"
            "Requirements:\n"
            "- Each query must target a DIFFERENT semantic aspect of the question\n"
            "- Write every query in the EXACT SAME LANGUAGE as the question\n"
            "- Output ONLY the 3 queries, one per line\n"
            "- No numbering, no bullets, no labels, no explanations\n\n"
            f"Question: {pregunta}\n\n"
            "Queries:\n"
        )

        response = ollama.generate(
            model=MODELO_CHAT,
            prompt=prompt,
            think=False,
            options={
                "temperature": 0.5,
                "num_predict": 400,
                "stop": ["\n\n\n"],
            },
        )

        queries = [
            q.strip().lstrip("0123456789.-) ")
            for q in response["response"].strip().split("\n")
            if q.strip() and len(q.strip()) > 20
        ]

        return queries[:3]

    except Exception as e:
        logging.warning(f"Error generating queries with LLM ({MODELO_CHAT}): {e}")
        return []


def _validar_coherencia_query(query: str) -> bool:
    """Detect whether a query is an incoherent bag-of-words.

    Checks unique-word ratio, word repetition, and presence of
    connectors to ensure the query reads as a natural sentence.

    Args:
        query: The candidate query string.

    Returns:
        ``True`` if the query appears coherent, ``False`` otherwise.
    """
    words = query.lower().split()
    if len(words) < 2:
        return True

    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.7:
        return False

    word_freq = Counter(words)
    if word_freq.most_common(1)[0][1] >= 3:
        return False

    connectors = {
    # English
    "the", "a", "an", "is", "are", "how", "what", "why",
    "when", "where", "which", "does", "do", "to", "in", "of",
    "that", "for", "and", "with", "by", "on", "as",
    # Spanish
    "cómo", "qué", "cuál", "cuáles", "cuándo", "dónde", "por",
    "para", "que", "son", "está", "entre", "con", "los", "las",
    # Valencian
    "com", "quins", "quines", "quan", "quin", "quina", "per", "que",
    }

    has_connectors = any(w in connectors for w in words)
    if len(words) > 8 and not has_connectors:
        return False

    return True


def _filtrar_terminos_criticos(terminos: List[str]) -> List[str]:
    """Keep only high-discrimination domain-specific terms for exhaustive search.

    Filters out generic blacklisted single words and retains multi-word
    terms, capitalized terms, and acronyms.

    Args:
        terminos: Candidate critical terms.

    Returns:
        Filtered list of domain-specific terms.
    """
    filtered = []
    for term in terminos:
        words = term.lower().split()

        if len(words) == 1 and term.lower() in GENERIC_TERMS_BLACKLIST:
            continue

        if len(words) >= 2:
            filtered.append(term)
            continue

        if term[0].isupper() or term.isupper():
            filtered.append(term)

    return filtered


# ─────────────────────────────────────────────
# SECTION 9: HYBRID RETRIEVAL PIPELINE
# ─────────────────────────────────────────────


def realizar_busqueda_hibrida(
    pregunta: str,
    collection: chromadb.Collection
) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    """Orchestrate the full hybrid retrieval pipeline.

    Combines multi-query semantic search, keyword search, exhaustive
    deep scan, RRF fusion, and optional Cross-Encoder reranking into
    a single ranked result set.

    Args:
        pregunta: User query.
        collection: ChromaDB collection to search.

    Returns:
        Tuple of (ranked fragments, best score, full metrics dict).
    """
    ui.debug("Starting hybrid search...")

    metricas_totales = {
        'fase_semantica': {},
        'fase_keywords': {},
        'fase_exhaustiva': {},
        'fase_reranking': {},
        'candidatos_fusion': 0,
        'resultados_finales': 0
    }

    llm_queries = []
    if USAR_LLM_QUERY_DECOMPOSITION and len(pregunta) > 60:
        ui.debug("decomposing query...")
        llm_queries = generar_queries_con_llm(pregunta)
        if llm_queries:
            ui.debug(f"{len(llm_queries)} sub-queries generated")

    ui.debug("semantic search...")

    queries = [pregunta]

    _KEEP_SHORT = {"to", "of", "in", "on", "by", "for", "as", "is", "are", "was",
                    "and", "or", "the", "its", "how", "not", "no", "what", "does"}
    palabras_clave_pregunta = [
        p for p in pregunta.split()
        if len(p) > 4 or p.lower().strip('"?,()') in _KEEP_SHORT
    ]
    query_corta = ' '.join(palabras_clave_pregunta[:20]).strip()
    if query_corta and query_corta != pregunta:
        queries.append(query_corta)

    keywords_expandidas = extraer_keywords(pregunta)

    if llm_queries:
        fallback_q = llm_queries[0]
        if _validar_coherencia_query(fallback_q) and fallback_q not in queries:
            queries.append(fallback_q)
    elif keywords_expandidas:
        query_kw = ' '.join(keywords_expandidas[:6]).strip()
        if query_kw and _validar_coherencia_query(query_kw) and query_kw not in queries:
            queries.append(query_kw)

    for lq in llm_queries:
        if lq not in queries:
            queries.append(lq)

    ui.debug(f"{len(queries)} query variant(s)")

    all_semantic_results = {}

    for q_idx, query in enumerate(queries):
        query_con_prefijo = f"{EMBED_PREFIX_QUERY}{query}"
        response_emb = ollama.embeddings(model=MODELO_EMBEDDING, prompt=query_con_prefijo)

        results_semantic = collection.query(
            query_embeddings=[response_emb["embedding"]],
            n_results=N_RESULTADOS_SEMANTICOS,
            include=['documents', 'distances', 'metadatas']
        )

        for idx, (doc, distancia, metadata) in enumerate(zip(
            results_semantic['documents'][0],
            results_semantic['distances'][0],
            results_semantic['metadatas'][0]
        ), 1):
            chunk_id = f"{metadata['source']}_pag{metadata['page']}_chunk{metadata.get('chunk', 0)}"

            if chunk_id not in all_semantic_results:
                all_semantic_results[chunk_id] = {
                    'doc': doc,
                    'metadata': metadata,
                    'distancia': distancia,
                    'id': chunk_id,
                    'score_semantic': 0.0,
                    'score_keyword': 0.0,
                    'matches': [],
                    'query_matches': []
                }

            all_semantic_results[chunk_id]['score_semantic'] += 1.0 / (idx + RRF_K)
            all_semantic_results[chunk_id]['query_matches'].append(q_idx + 1)
            if distancia < all_semantic_results[chunk_id]['distancia']:
                all_semantic_results[chunk_id]['distancia'] = distancia

    metricas_totales['fase_semantica'] = {
        'queries_generadas': len(queries),
        'fragmentos_unicos': len(all_semantic_results)
    }

    ui.debug(f"{len(all_semantic_results)} unique fragments")

    results_keyword = []
    metricas_keywords = {}
    if USAR_BUSQUEDA_HIBRIDA:
        ui.debug("keyword search...")
        keywords = extraer_keywords(pregunta)
        if keywords:
            ui.debug(f"detected: {', '.join(keywords[:8])}")
        results_keyword, metricas_keywords = busqueda_por_keywords(pregunta, collection)
        metricas_totales['fase_keywords'] = metricas_keywords

    ui.debug("fusing results...")

    fragmentos_data = all_semantic_results.copy()

    for idx, result in enumerate(results_keyword, 1):
        chunk_id = result['id']

        if chunk_id in fragmentos_data:
            fragmentos_data[chunk_id]['score_keyword'] += 1.0 / (idx + RRF_K)
            if result['keyword_match'] not in fragmentos_data[chunk_id]['matches']:
                fragmentos_data[chunk_id]['matches'].append(result['keyword_match'])
        else:
            fragmentos_data[chunk_id] = {
                'doc': result['doc'],
                'metadata': result['metadata'],
                'distancia': result['distancia'],
                'id': chunk_id,
                'score_semantic': 0.0,
                'score_keyword': 1.0 / (idx + RRF_K),
                'matches': [result['keyword_match']],
                'query_matches': []
            }

    for frag in fragmentos_data.values():
        frag['score_final'] = (frag['score_semantic'] * 0.55 + frag['score_keyword'] * 0.45)

    terminos_criticos = _filtrar_terminos_criticos(
        [k for k in keywords_expandidas[:12] if len(k) > 3]
    )

    metricas_exhaustiva = {}
    if USAR_BUSQUEDA_EXHAUSTIVA and terminos_criticos:
        ui.debug(f"deep search: {', '.join(terminos_criticos[:6])}")
        resultados_exhaustivos, metricas_exhaustiva = busqueda_exhaustiva_texto(
            terminos_criticos, collection, max_results=30
        )
        metricas_totales['fase_exhaustiva'] = metricas_exhaustiva

        for idx, result in enumerate(resultados_exhaustivos):
            chunk_id = result['id']

            if chunk_id in fragmentos_data:
                fragmentos_data[chunk_id]['score_keyword'] += 0.3 * result['num_matches']
                fragmentos_data[chunk_id]['matches'].extend(
                    m for m in result['matches'] if m not in fragmentos_data[chunk_id]['matches']
                )
            else:
                fragmentos_data[chunk_id] = {
                    'doc': result['doc'],
                    'metadata': result['metadata'],
                    'distancia': float('inf'),
                    'id': chunk_id,
                    'score_semantic': 0.0,
                    'score_keyword': 0.3 * result['num_matches'],
                    'matches': result['matches'],
                    'query_matches': []
                }

        for frag in fragmentos_data.values():
            frag['score_final'] = (frag['score_semantic'] * 0.55 + frag['score_keyword'] * 0.45)

    fragmentos_ranked = sorted(
        fragmentos_data.values(),
        key=lambda x: x['score_final'],
        reverse=True
    )

    metricas_totales['candidatos_fusion'] = len(fragmentos_ranked)

    if USAR_RERANKER and fragmentos_ranked:
        n_candidatos = min(TOP_K_RERANK_CANDIDATES, len(fragmentos_ranked))
        ui.debug(f"reranking top {n_candidatos} candidates...")

        candidatos_rerank = fragmentos_ranked[:TOP_K_RERANK_CANDIDATES]
        fragmentos_ranked, metricas_rerank = rerank_resultados(
            pregunta,
            candidatos_rerank,
            top_k=TOP_K_AFTER_RERANK
        )
        metricas_totales['fase_reranking'] = metricas_rerank
        ui.debug(f"top {len(fragmentos_ranked)} after reranking")

    mejor_score = fragmentos_ranked[0]['score_final'] if fragmentos_ranked else 0
    metricas_totales['resultados_finales'] = len(fragmentos_ranked)

    metricas_totales['sub_queries'] = llm_queries
    metricas_totales['queries_semanticas'] = queries
    metricas_totales['keywords'] = list(keywords_expandidas)
    metricas_totales['terminos_criticos'] = terminos_criticos

    if LOGGING_METRICAS:
        sem_unicos = metricas_totales['fase_semantica'].get('fragmentos_unicos', 0)
        kw_total = metricas_keywords.get('resultados_totales', 0)
        logging.info(
            f"Full pipeline: Semantic({sem_unicos}) + "
            f"Keywords({kw_total}) -> "
            f"Fusion({metricas_totales['candidatos_fusion']}) -> "
            f"Reranking({metricas_totales['resultados_finales']})"
        )

    return fragmentos_ranked, mejor_score, metricas_totales


# ─────────────────────────────────────────────
# SECTION 10: CONTEXT AND GENERATION
# ─────────────────────────────────────────────


def _es_continuacion_parrafo(linea_previa: str, linea_actual: str) -> bool:
    """Heuristic to detect whether the current line continues a paragraph.

    Avoids false paragraph breaks caused by PDF extraction double-spacing.

    Args:
        linea_previa: The preceding line (right-stripped).
        linea_actual: The current line (stripped).

    Returns:
        ``True`` if the current line likely continues the paragraph.
    """
    if not linea_previa or not linea_actual:
        return False

    prev_stripped = linea_previa.rstrip()
    curr_stripped = linea_actual.strip()

    if not prev_stripped or not curr_stripped:
        return False

    prev_end = prev_stripped[-1]

    if prev_end in '.?!':
        return False

    if re.match(r'^\d+[.\)]', curr_stripped):
        return False
    if curr_stripped.startswith('#'):
        return False
    if re.match(r'^[-*•]\s', curr_stripped):
        return False
    if curr_stripped.startswith('**'):
        return False

    if prev_end in ',;(':
        return True
    if curr_stripped[0].islower():
        return True

    if prev_end not in '.?!:':
        return True

    return False


def _reunir_parrafos(texto: str) -> str:
    """Re-join lines split by PDF extraction using look-ahead for continuations.

    Args:
        texto: Text with potentially broken paragraph lines.

    Returns:
        Text with properly joined paragraphs.
    """
    lines = texto.split('\n')
    result = []
    buffer = ""

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if not stripped:
            if buffer:
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1

                if j < len(lines) and _es_continuacion_parrafo(buffer, lines[j].strip()):
                    buffer += ' ' + lines[j].strip()
                    i = j + 1
                    continue
                else:
                    result.append(buffer)
                    buffer = ""
                    result.append("")
                    i = j if j < len(lines) else i + 1
                    continue
            else:
                result.append("")
                i += 1
                continue

        if not buffer:
            buffer = stripped
        elif _es_continuacion_parrafo(buffer, stripped):
            buffer += ' ' + stripped
        else:
            result.append(buffer)
            buffer = stripped

        i += 1

    if buffer:
        result.append(buffer)

    return '\n'.join(result)


def optimizar_texto_contexto(texto: str) -> str:
    """Remove PDF noise (box artifacts, footers, double spacing).

    Typical savings are 30-50% of characters.

    Args:
        texto: Raw text extracted from a PDF chunk.

    Returns:
        Cleaned text ready for LLM context.
    """
    texto = re.sub(r'#{1,6}\s*□', '', texto)
    texto = re.sub(r'^□\s*$', '', texto, flags=re.MULTILINE)

    texto = re.sub(r'^#{1,6}\s+', '', texto, flags=re.MULTILINE)

    texto = re.sub(
        r'^[A-ZÀ-Ú][a-zà-ú]+ [A-ZÀ-Ú][a-zà-ú]+,\s+[A-ZÀ-Ú].*?\d+\s*/\s*\d+\s+\w+.*$',
        '', texto, flags=re.MULTILINE
    )

    texto = re.sub(r'\*{0,2}Solución:?\*{0,2}\s*La\s+\d+\s*', '', texto)

    texto = re.sub(r'\*\*(.+?)\*\*', r'\1', texto)

    texto = re.sub(r'[ \t]{2,}', ' ', texto)

    texto = re.sub(r'[ \t]+$', '', texto, flags=re.MULTILINE)

    texto = _reunir_parrafos(texto)

    texto = re.sub(r'^\s+$', '', texto, flags=re.MULTILINE)

    texto = re.sub(r'(?m)^\s*\d{1,3}\s*$', '', texto)   # standalone PDF page numbers

    texto = re.sub(r'\n{3,}', '\n\n', texto)

    return texto.strip()


def _marcar_fragmento_incompleto(texto: str) -> str:
    """Append ``[incomplete fragment]`` if text lacks closing punctuation.

    Args:
        texto: Fragment text to inspect.

    Returns:
        Original text, possibly with an appended incompleteness marker.
    """
    stripped = texto.rstrip()
    if not stripped:
        return texto
    _CLOSING = frozenset('.?!:")]')
    if stripped[-1] not in _CLOSING and not stripped[-1].isdigit():
        return texto + '\n[excerpt ends mid-sentence]'
    return texto


def _texto_fuente_fragmento(doc: str) -> str:
    """Return chunk body without the contextual-retrieval summary prefix.

    Indexed chunks store ``<summary>\\n\\n<body>`` using the *literal* 6-char
    sequence ``\\n\\n`` (backslash-n-backslash-n) as separator, intentionally
    distinct from real paragraph breaks (``\n\n``).  Do NOT replace with real
    newlines: PDF body text naturally contains ``\n\n``, so a real-newline
    separator would produce false splits on the first paragraph break.

    Args:
        doc: Raw ``fragment['doc']`` string.

    Returns:
        Source body text, or full ``doc`` if no separator is present.
    """
    if "\\n\\n" in doc:
        _, cuerpo = doc.split("\\n\\n", 1)
        return cuerpo.strip()
    return doc.strip()


def _strip_ollama_think_blocks(text: str) -> str:
    """Remove ``...</think>`` wrappers often emitted by Qwen-style Ollama models."""
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


_RECOMP_FACTS_HEADER = "## Facts relevant to the question"


def _normalizar_salida_recomp(texto: str) -> str:
    """Ensure RECOMP briefing has the expected markdown header when possible."""
    t = texto.strip()
    if not t:
        return t
    if _RECOMP_FACTS_HEADER.lower() in t.lower():
        return t
    if re.search(r"^-\s+\S", t, flags=re.MULTILINE):
        return f"{_RECOMP_FACTS_HEADER}\n{t}"
    return t


def construir_contexto_para_modelo(fragmentos: List[Dict[str, Any]]) -> str:
    """Build the context string for the RAG model from retrieved fragments.

    Output format per fragment::

        --- [Fragment N] ---
        [Fragment Context]            <- only if Contextual Retrieval summary exists
        ...
        [Source Text]
        ...
        [incomplete fragment]         <- only if the chunk is truncated

    Fragments are separated by double newlines. PDF text optimization
    is applied when ``USAR_OPTIMIZACION_CONTEXTO`` is enabled.

    Args:
        fragmentos: List of retrieved chunk dicts with ``doc`` and ``metadata``.

    Returns:
        Formatted context string ready for the ``<context>`` tag.
    """
    fragmentos_ordenados = sorted(
        fragmentos,
        key=lambda f: (f['metadata']['source'], f['metadata']['page'], f['metadata'].get('chunk', 0))
    )

    textos_originales = [frag['doc'] for frag in fragmentos_ordenados]
    chars_original = sum(len(t) for t in textos_originales)

    fragmentos_formateados = []
    for i, frag in enumerate(fragmentos_ordenados, 1):
        texto = optimizar_texto_contexto(frag['doc']) if USAR_OPTIMIZACION_CONTEXTO else frag['doc']
        if not texto:
            continue

        if '\\n\\n' in texto:
            ctx_summary, raw_content = texto.split('\\n\\n', 1)
            raw_content = _marcar_fragmento_incompleto(raw_content.strip())
            texto = f"[Fragment Context]\n{ctx_summary.strip()}\n\n[Source Text]\n{raw_content}"
        else:
            texto = _marcar_fragmento_incompleto(texto.strip())

        fragmentos_formateados.append(f"--- [Fragment {i}] ---\n{texto}")

    resultado = "\n\n".join(fragmentos_formateados)

    chars_optimizado = len(resultado)
    if chars_original > 0 and LOGGING_METRICAS:
        ahorro = chars_original - chars_optimizado
        pct = (ahorro / chars_original) * 100 if ahorro > 0 else 0
        logging.info(
            f"Optimized context: {chars_original} -> {chars_optimizado} chars "
            f"({ahorro} saved, {pct:.1f}%)"
        )

    return resultado


def sintetizar_contexto_recomp(fragmentos: List[Dict[str, Any]], query_usuario: str = "") -> str:
    """Synthesize context using MODELO_RECOMP instead of raw chunks.

    Uses the original user question and a fixed markdown outline (``## Facts
    relevant to the question`` + bullets). Evidence is taken from chunk
    *body* only, omitting contextual-retrieval summaries, to avoid
    meta-descriptive prose in the briefing.

    Falls back to ``construir_contexto_para_modelo`` if synthesis is
    disabled, fails, or produces too little output.

    Args:
        fragmentos: Retrieved chunk dicts.
        query_usuario: Original user question (required for focused synthesis).

    Returns:
        Synthesized context string or raw formatted context on fallback.
    """
    if not USAR_RECOMP_SYNTHESIS or not fragmentos:
        return construir_contexto_para_modelo(fragmentos)

    textos_preparados = []
    for f in fragmentos:
        cuerpo = _texto_fuente_fragmento(f.get("doc", "") or "")
        content = cuerpo.replace("\n", " ").strip()
        content = re.sub(r'\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]', '', content)  # strip citation markers [38, 2, 9]
        if content:
            n = len(textos_preparados) + 1
            textos_preparados.append(f"Fragment {n}:\n{content}")

    contexto_raw = "\n\n".join(textos_preparados)
    if not contexto_raw.strip():
        return construir_contexto_para_modelo(fragmentos)

    q = (query_usuario or "").strip()
    bloque_pregunta = (
        f"## User question\n{q}\n"
        if q
        else "## User question\n(No question provided; extract the main technical facts from the excerpts.)\n"
    )

    system_prompt = (
        "You compress retrieved document excerpts into a brief briefing for a downstream "
        "answer model.\n"
        "GROUNDING:\n"
        "- Use ONLY information stated in the evidence excerpts. No outside knowledge.\n"
        "- Preserve technical terms, notation, formulas, and numbers exactly as written.\n"
        "ENUMERATION (critical):\n"
        "- If the question asks for a list or a count (e.g. 'three types', 'two ways'), "
        "search ALL fragments and enumerate EVERY item you find, even if items are spread "
        "across different fragments. Never say an item 'is not mentioned' if it appears "
        "anywhere in the excerpts.\n"
        "- When an excerpt ends with [excerpt ends mid-sentence], the list may continue in "
        "another fragment — collect items from ALL fragments before writing your bullets.\n"
        "STYLE:\n"
        "- Write ONLY facts that help answer the user question. Do NOT describe the documents, "
        "the paper, or the excerpts (forbidden openers: \"This paper\", \"The excerpt\", "
        "\"This section\", \"The document\", \"The text\", \"The fragment\").\n"
        "- Do NOT restate meta-summaries; every bullet must be substantive content from the excerpts.\n"
        "- Do NOT cite fragment numbers, sources, or page numbers.\n"
        "OUTPUT FORMAT (exactly this structure, markdown):\n"
        "## Facts relevant to the question\n"
        "- (first grounded fact)\n"
        "- (second grounded fact)\n"
        "Use one bullet per distinct fact; merge duplicates. If nothing in the excerpts bears "
        "on the question, output exactly one bullet: "
        "\"Insufficient evidence in the excerpts to answer the question.\"\n"
        "Language: same as the evidence excerpts (or the user question if excerpts mix languages)."
    )

    user_prompt = (
        f"{bloque_pregunta}"
        "## Evidence excerpts (verbatim from retrieval; may be partial)\n"
        f"{contexto_raw}\n\n"
        "Produce the briefing using the required OUTPUT FORMAT. No text before "
        "## Facts relevant to the question."
    )

    try:
        payload = {
            "model": MODELO_RECOMP,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 10000,
                "top_p": 0.9,
                "repeat_penalty": 1.15,
                "num_ctx": 16384,
            },
        }
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=60)
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "")

        sintesis = _strip_ollama_think_blocks(raw.strip())
        sintesis = _normalizar_salida_recomp(sintesis)

        if len(sintesis) < 20:
            logging.info(
                "RECOMP: falling back to raw chunks (synthesis too short after "
                "stripping think blocks; check %s / OLLAMA_RECOMP_MODEL)",
                MODELO_RECOMP,
            )
            return construir_contexto_para_modelo(fragmentos)

        if _RECOMP_FACTS_HEADER.lower() not in sintesis.lower():
            logging.info(
                "RECOMP: falling back to raw chunks (missing '%s' in model output)",
                _RECOMP_FACTS_HEADER,
            )
            return construir_contexto_para_modelo(fragmentos)

        return sintesis

    except Exception as e:
        logging.warning(f"Critical error in RECOMP synthesis ({MODELO_RECOMP}): {e}")
        return construir_contexto_para_modelo(fragmentos)


def guardar_debug_rag(
    pregunta: str,
    mensaje_usuario: str = "",
    respuesta: str = "",
    fragmentos: Optional[List[Dict[str, Any]]] = None,
    motivo_interrupcion: Optional[str] = None,
    metricas: Optional[Dict[str, Any]] = None
) -> None:
    """Dump a full RAG interaction to ``debug_rag/`` for inspection.

    The output file (timestamped + slug) includes sub-queries, keywords,
    pipeline configuration, the full prompt, the model response, and all
    retrieved fragments with their scores.

    Args:
        pregunta: Original user question.
        mensaje_usuario: Complete user message (with ``<context>``).
        respuesta: Model response text.
        fragmentos: Retrieved fragments used for context.
        motivo_interrupcion: Reason for early interruption, if any.
        metricas: Full pipeline metrics dict.
    """
    fragmentos = fragmentos or []

    if not GUARDAR_DEBUG_RAG:
        return

    try:
        os.makedirs(CARPETA_DEBUG_RAG, exist_ok=True)

        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r'[^\w\s]', '', pregunta)[:40].strip().replace(' ', '_')
        nombre_archivo = f"{timestamp}_{slug}.txt"
        ruta = os.path.join(CARPETA_DEBUG_RAG, nombre_archivo)

        with open(ruta, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"  DEBUG RAG - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            if metricas and (metricas.get('sub_queries') or metricas.get('queries_semanticas') or metricas.get('keywords') or metricas.get('terminos_criticos') or metricas.get('fase_semantica') or metricas.get('fase_keywords') or metricas.get('fase_exhaustiva') or metricas.get('fase_reranking')):
                f.write("─" * 80 + "\n")
                f.write("  RETRIEVAL PIPELINE (sub-queries, keywords, terms, metrics)\n")
                f.write("─" * 80 + "\n")
                sub_q = metricas.get('sub_queries', [])
                if sub_q:
                    f.write("\nSub-queries (Query Decomposition):\n")
                    for i, sq in enumerate(sub_q, 1):
                        f.write(f"  {i}. {sq}\n")
                queries_sem = metricas.get('queries_semanticas', [])
                if queries_sem:
                    f.write("\nQueries used in semantic search:\n")
                    for i, q in enumerate(queries_sem, 1):
                        f.write(f"  {i}. {q}\n")
                keywords = metricas.get('keywords', [])
                if keywords:
                    f.write(f"\nExtracted keywords ({len(keywords)}):\n  {', '.join(keywords[:30])}\n")
                    if len(keywords) > 30:
                        f.write(f"  ... and {len(keywords) - 30} more\n")
                terminos = metricas.get('terminos_criticos', [])
                if terminos:
                    f.write(f"\nCritical terms (exhaustive search):\n  {', '.join(terminos)}\n")
                fase_kw = metricas.get('fase_keywords', {})
                if fase_kw:
                    f.write(f"\nKeyword metrics: {fase_kw.get('keywords_encontradas', 0)}/{fase_kw.get('keywords_totales', 0)} found, {fase_kw.get('resultados_totales', 0)} results\n")
                f.write(f"\nFull metrics:\n{json.dumps(metricas, indent=2, ensure_ascii=False, default=str)}\n\n")

            if motivo_interrupcion:
                f.write("─" * 80 + "\n")
                f.write("  EARLY INTERRUPTION\n")
                f.write("─" * 80 + "\n")
                f.write(f"{motivo_interrupcion}\n\n")
                if metricas:
                    f.write("Search metrics:\n")
                    f.write(json.dumps(metricas, indent=2, ensure_ascii=False) + "\n\n")

            f.write("─" * 80 + "\n")
            f.write("  PIPELINE CONFIGURATION\n")
            f.write("─" * 80 + "\n")
            f.write(f"RAG Model: {_inferir_descripcion_modelo(MODELO_RAG)}\n")
            f.write(f"Contextual Retrieval (Indexing): {'YES' if USAR_CONTEXTUAL_RETRIEVAL else 'NO'}\n")
            f.write(f"Query Decomposition: {'YES' if USAR_LLM_QUERY_DECOMPOSITION else 'NO'}\n")
            f.write(f"Hybrid Search (keywords): {'YES' if USAR_BUSQUEDA_HIBRIDA else 'NO'}\n")
            f.write(f"Exhaustive Search: {'YES' if USAR_BUSQUEDA_EXHAUSTIVA else 'NO'}\n")
            f.write(f"Reranker: {'YES' if USAR_RERANKER else 'NO'}\n")
            f.write(f"Expand Context: {'YES' if EXPANDIR_CONTEXTO else 'NO'}\n")
            f.write(f"Optimize Context: {'YES' if USAR_OPTIMIZACION_CONTEXTO else 'NO'}\n")
            f.write(f"RECOMP Synthesis: {'YES' if USAR_RECOMP_SYNTHESIS else 'NO'}\n\n")

            f.write("─" * 80 + "\n")
            f.write("  ORIGINAL QUESTION\n")
            f.write("─" * 80 + "\n")
            f.write(f"{pregunta}\n\n")

            f.write("─" * 80 + "\n")
            f.write("  SYSTEM PROMPT\n")
            f.write("─" * 80 + "\n")
            f.write("(baked into Modelfile — not sent via API)\n\n")

            context_match = re.search(r'<context>(.*?)</context>', mensaje_usuario, re.DOTALL)
            contexto_enviado = context_match.group(1).strip() if context_match else "(empty)"

            f.write("─" * 80 + "\n")
            if USAR_RECOMP_SYNTHESIS:
                f.write("  RECOMP SYNTHESIS SENT TO FINAL MODEL (instead of raw chunks)\n")
            else:
                f.write("  RAW CONTEXT SENT TO FINAL MODEL\n")
            f.write("─" * 80 + "\n")
            f.write(f"{contexto_enviado}\n\n")

            f.write("─" * 80 + "\n")
            f.write("  USER MESSAGE (full actual prompt)\n")
            f.write("─" * 80 + "\n")
            f.write(f"{mensaje_usuario or '(not sent to model)'}\n\n")

            f.write("─" * 80 + "\n")
            f.write("  MODEL RESPONSE\n")
            f.write("─" * 80 + "\n")
            f.write(f"{respuesta or '(not generated)'}\n\n")

            f.write("─" * 80 + "\n")
            f.write(f"  RETRIEVED FRAGMENTS ({len(fragmentos)})\n")
            f.write("─" * 80 + "\n")
            for i, frag in enumerate(fragmentos, 1):
                meta = frag.get('metadata', {})
                score = frag.get('score_final', 'N/A')
                score_rr = frag.get('score_reranker', 'N/A')
                f.write(f"\n--- Fragment {i} ---\n")
                pag = meta.get('page', 0)
                f.write(f"Source: {meta.get('source', '?')}, page {pag + 1 if isinstance(pag, int) else pag}\n")
                f.write(f"Final score: {score}  |  Reranker score: {score_rr}\n")
                matches = frag.get('matches', [])
                if matches:
                    f.write(f"Matched keywords: {', '.join(matches)}\n")
                query_matches = frag.get('query_matches', [])
                if query_matches:
                    f.write(f"Matched query(s): {query_matches}\n")
                f.write(f"Section: {meta.get('section_header', '(no header)')}\n")
                doc_text = frag.get('doc', '')
                if '\\n\\n' in doc_text:
                    ctx_part, orig_part = doc_text.split('\\n\\n', 1)
                    f.write(f"[Contextual Retrieval]:\n{ctx_part}\n\n")
                    f.write(f"[Document text]:\n{orig_part}\n")
                else:
                    f.write(f"[Document text]:\n{doc_text}\n")

        logging.info(f"Debug RAG saved: {ruta}")

    except Exception as e:
        logging.warning(f"Error saving debug RAG: {e}")


OLLAMA_BASE_URL = "http://localhost:11434"


def _ollama_generate_stream(model: str, prompt: str, options: dict):
    """Stream tokens from the Ollama ``/api/generate`` endpoint.

    The system prompt is intentionally omitted here: it is baked directly
    into the model's Modelfile, so re-sending it via the API would be
    redundant and would override the Modelfile default unnecessarily.

    ``think=False`` avoids spending ``num_predict`` on a reasoning trace when
    ``MODELO_RAG`` (or any substitute) is a thinking model.

    Args:
        model: Ollama model name.
        prompt: User prompt.
        options: Generation options (temperature, top_p, etc.).

    Yields:
        Parsed JSON objects from each streamed line.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "think": False,
        "options": options,
    }
    with requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, stream=True) as resp:
        for line in resp.iter_lines():
            if line:
                yield json.loads(line)


def generar_respuesta(
    pregunta: str,
    fragmentos: List[Dict[str, Any]],
    metricas: Optional[Dict[str, Any]] = None,
    on_token=None,
) -> str:
    """Generate a RAG response and optionally stream tokens through a callback.

    Builds context from fragments, streams the response via Ollama,
    optionally emits visible tokens through ``on_token``, and saves a
    debug dump.

    Args:
        pregunta: User question.
        fragmentos: Retrieved context fragments.
        metricas: Pipeline metrics for debug logging.
        on_token: Optional callable receiving each streamed token.

    Returns:
        Complete response text.
    """
    if USAR_RECOMP_SYNTHESIS:
        contexto_str = sintetizar_contexto_recomp(fragmentos, query_usuario=pregunta)
    else:
        contexto_str = construir_contexto_para_modelo(fragmentos)

    mensaje_usuario = f"{pregunta}\n\n<context>{contexto_str}</context>"

    respuesta_completa = ""
    for chunk in _ollama_generate_stream(
        model=MODELO_RAG,
        prompt=mensaje_usuario,
        options={"temperature": 0.15, "top_p": 0.9, "repeat_penalty": 1.15, "num_ctx": 16384},
    ):
        content = chunk.get("response", "")
        if content:
            respuesta_completa += content
            if on_token is not None:
                on_token(content)

    guardar_debug_rag(pregunta, mensaje_usuario, respuesta_completa, fragmentos, metricas=metricas)
    return respuesta_completa


def generar_respuesta_silenciosa(pregunta: str, fragmentos: List[Dict[str, Any]], metricas: Optional[Dict[str, Any]] = None) -> str:
    """Generate a RAG response silently (no terminal output).

    Same as ``generar_respuesta`` but without printing or debug dump.

    Args:
        pregunta: User question.
        fragmentos: Retrieved context fragments.
        metricas: Pipeline metrics (unused here, kept for API compat).

    Returns:
        Complete response text.
    """
    if USAR_RECOMP_SYNTHESIS:
        contexto_str = sintetizar_contexto_recomp(fragmentos, query_usuario=pregunta)
    else:
        contexto_str = construir_contexto_para_modelo(fragmentos)

    mensaje_usuario = f"{pregunta}\n\n<context>{contexto_str}</context>"

    respuesta_completa = ""
    for chunk in _ollama_generate_stream(
        model=MODELO_RAG,
        prompt=mensaje_usuario,
        options={"temperature": 0.15, "top_p": 0.9, "repeat_penalty": 1.15, "num_ctx": 16384},
    ):
        content = chunk.get("response", "")
        if content:
            respuesta_completa += content
    return respuesta_completa


def evaluar_pregunta_rag(
    pregunta: str,
    collection: chromadb.Collection
) -> Tuple[str, List[str]]:
    """Run the full RAG pipeline silently for evaluation (RAGAS).

    Performs search, filtering, neighbor expansion, and generation
    without any terminal output.

    Args:
        pregunta: The evaluation question.
        collection: ChromaDB collection to search.

    Returns:
        Tuple of (response text, list of context strings).
    """
    import io
    import contextlib
    if len(pregunta.strip()) < MIN_LONGITUD_PREGUNTA_RAG:
        return ("", [])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        fragmentos_ranked, mejor_score, _ = realizar_busqueda_hibrida(pregunta, collection)
        if not fragmentos_ranked:
            return ("", [])
        # UMBRAL_RELEVANCIA is calibrated for reranker-scale scores, not RRF fusion scores.
        if (
            USAR_RERANKER
            and mejor_score < UMBRAL_RELEVANCIA
            and not EVAL_RAGBENCH_RERANKER_LOW_SCORE_FALLBACK
        ):
            return ("", [])
        if USAR_RERANKER:
            fragmentos_filtrados = [
                f for f in fragmentos_ranked
                if f.get('score_reranker', f.get('score_final', 0)) >= UMBRAL_SCORE_RERANKER
            ]
            if fragmentos_filtrados:
                fragmentos_ranked = fragmentos_filtrados
            elif not EVAL_RAGBENCH_RERANKER_LOW_SCORE_FALLBACK:
                return ("", [])
        fragmentos_finales = fragmentos_ranked[:TOP_K_FINAL]
        ids_usados = {f['id'] for f in fragmentos_finales}
        if EXPANDIR_CONTEXTO and fragmentos_finales and 'chunk' in fragmentos_finales[0]['metadata']:
            for frag in list(fragmentos_finales):  # snapshot: don't expand neighbors of neighbors
                ids_vecinos = expandir_con_chunks_adyacentes(frag['id'], frag['metadata'], n_vecinos=1)
                if ids_vecinos:
                    try:
                        vecinos = collection.get(ids=ids_vecinos, include=['documents', 'metadatas'])
                        for v_doc, v_meta in zip(vecinos['documents'], vecinos['metadatas']):
                            v_id = f"{v_meta['source']}_pag{v_meta['page']}_chunk{v_meta.get('chunk', 0)}"
                            if v_id not in ids_usados:
                                fragmentos_finales.append({
                                    'doc': v_doc, 'metadata': v_meta, 'distancia': float('inf'),
                                    'score_final': 0.0, 'id': v_id
                                })
                                ids_usados.add(v_id)
                    except Exception:
                        pass
        contexto_total = sum(len(f['doc']) for f in fragmentos_finales)
        if contexto_total > MAX_CONTEXTO_CHARS:
            fragmentos_truncados = []
            chars_acum = 0
            for f in fragmentos_finales:
                if chars_acum + len(f['doc']) > MAX_CONTEXTO_CHARS:
                    break
                fragmentos_truncados.append(f)
                chars_acum += len(f['doc'])
            fragmentos_finales = fragmentos_truncados
        respuesta = generar_respuesta_silenciosa(pregunta, fragmentos_finales)
        contexts = [f['doc'] for f in fragmentos_finales]
        return (respuesta, contexts)


# ─────────────────────────────────────────────
# SECTION 11: INDEXING AND COLLECTION MANAGEMENT
# ─────────────────────────────────────────────


# --- 11.1 Contextual retrieval ---


def _detectar_idioma(texto: str) -> str:
    """Heuristic language detector from a document text sample.

    Counts distinctive function-word occurrences to distinguish Spanish,
    Catalan, and English. Designed for documents where the first ~4000
    characters are available; accuracy degrades on very short samples.

    Args:
        texto: Representative text sample from the document (ideally ≥500 chars).

    Returns:
        Full language name suitable for prompt injection: 'Spanish',
        'Catalan', or 'English'.
    """
    t = texto.lower()
    # Catalan-distinctive markers (absent or rare in Spanish)
    ca = (t.count("però ") + t.count("també ") + t.count("molt ")
          + t.count(" amb ") + t.count(" va ") + t.count("els ")
          + t.count("l'") + t.count("d'") + t.count("s'")
          + t.count("n'") + t.count("m'"))
    # Spanish-distinctive markers (absent or rare in Catalan)
    es = (t.count("también ") + t.count("además ") + t.count("pero ")
          + t.count("muy ") + t.count(" con ") + t.count("los ")
          + t.count("las ") + t.count("así ") + t.count("sin ")
          + t.count("después"))
    # English-distinctive markers
    en = (t.count(" the ") + t.count(" is ") + t.count(" are ")
          + t.count(" was ") + t.count(" were ") + t.count(" have ")
          + t.count(" this ") + t.count(" that ") + t.count(" from ")
          + t.count(" with "))
    scores = {"Spanish": es, "Catalan": ca, "English": en}
    return max(scores, key=scores.get)


def generar_contexto_situacional(
    chunk_text: str,
    texto_base: str,
    idioma_doc: str = "",
) -> str:
    """Generate 2-3 sentences of situational context for a chunk using an LLM.

    Produces a brief document summary plus a note on how the chunk fits
    within the larger document, to improve retrieval accuracy. The output
    is always written in the document's own language.

    Args:
        chunk_text: The text of the chunk to contextualize.
        texto_base: A representative excerpt of the full document.
        idioma_doc: Document language ('Spanish', 'Catalan', 'English').
            When empty, falls back to heuristic detection from ``texto_base``.

    Returns:
        Situational context string (with trailing ``\\n\\n``), or empty
        string if disabled or on failure.
    """
    if not USAR_CONTEXTUAL_RETRIEVAL:
        return ""

    idioma = idioma_doc or _detectar_idioma(texto_base)

    system_prompt = (
        f"You are an expert at analyzing academic documents. "
        f"MANDATORY: Write your entire response in {idioma} — the same language as the document. "
        f"Do NOT translate. Do NOT switch to any other language, including English. "
        f"When given a full document and an excerpt from it, produce exactly 2-3 sentences: "
        f"first a brief summary of what the document is about, then how the excerpt fits within it. "
        f"No introductions, no labels, no meta-commentary. "
        f"Do NOT include bibliographic citation markers such as [1], [38], or similar."
    )

    user_prompt = (
        f"<document>\\n{texto_base}\\n</document>\\n\\n"
        f"<excerpt>\\n{chunk_text}\\n</excerpt>\\n\\n"
        f"Write the 2-3 sentence situational context in {idioma}."
    )

    try:
        response = ollama.chat(
            model=MODELO_CONTEXTUAL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            think=False,
            options={"temperature": 0.1, "num_predict": 250},
        )
        contexto = response['message']['content'].strip()
        if contexto:
            return f"{contexto}\\n\\n"
    except Exception as e:
        logging.warning(f"Error generating situational context: {e}")
    return ""


# --- 11.2 Image extraction ---


_PROMPT_ECHO_MARKERS = (
    # Prompt template fragments — update if the prompt wording changes
    "Using this caption as context, describe the visual content",
    "describe the visual content of this image in detail",
    "Focus on what is depicted, not on any text overlaid",
    "describe the arrows between blocks",
    "name the inputs and outputs",
    "lists every labeled block",
    "The figure caption reads",     # model echoing the injected caption prefix
)


def _es_descripcion_spam(texto: str) -> bool:
    """Detect degenerate OCR output: repeated phrases or 'no text' spam.

    Two complementary checks:
    - Low lexical diversity: if unique words / total words < 0.35, the text
      is almost certainly a repetitive loop (any pattern, not only 'no text').
    - 'no text' specific: ratio of 'no' + 'text' tokens > 20%.

    Args:
        texto: Raw description returned by the vision model.

    Returns:
        True if the description is considered degenerate spam.
    """
    palabras = texto.lower().split()
    if len(palabras) < 10:
        return False
    # Low lexical diversity catches "the image is a row, and the image is a row..."
    diversidad = len(set(palabras)) / len(palabras)
    if diversidad < 0.35:
        return True
    # Specific check for "no text, no text, no text..." pattern
    ratio_no_text = (palabras.count("no") + palabras.count("text")) / len(palabras)
    return ratio_no_text > 0.20


def _es_prompt_echo(descripcion: str) -> bool:
    """Detect when the vision model echoes back the prompt instead of describing the image.

    Some vision models (e.g. glm-ocr) occasionally treat the text content of the
    prompt as image text to transcribe, returning the prompt verbatim.

    Args:
        descripcion: Raw description returned by the vision model.

    Returns:
        True if the description contains a literal fragment of the system prompt.
    """
    desc_lower = descripcion.lower()
    return any(marker.lower() in desc_lower for marker in _PROMPT_ECHO_MARKERS)


def _es_solo_caption(descripcion: str, caption: str) -> bool:
    """Check if the description merely echoes the caption with no new visual content.

    Args:
        descripcion: Description returned by the vision model.
        caption: Caption text used as context in the prompt.

    Returns:
        True if >85% of caption tokens appear in the description and the
        description is not substantially longer than the caption.
    """
    if not caption:
        return False
    tokens_desc = set(descripcion.lower().split())
    tokens_cap  = set(caption.lower().split())
    if not tokens_cap:
        return False
    overlap = len(tokens_desc & tokens_cap) / len(tokens_cap)
    return overlap > 0.85 and len(tokens_desc) < len(tokens_cap) * 1.3


def extraer_imagenes_pdf(
    ruta_pdf: str,
    max_por_pagina: int = MAX_IMAGENES_POR_PAGINA,
    min_size_px: int = MIN_IMAGEN_SIZE_PX,
) -> Dict[int, List[Dict[str, Any]]]:
    """Extract all raster images from a PDF, grouped by zero-based page number.

    Opens the PDF once with PyMuPDF (fitz) and iterates over every page.
    Images below ``min_size_px`` on either side (icons, decorations) are
    discarded.  At most ``max_por_pagina`` images are kept per page,
    in document order.

    Args:
        ruta_pdf: Absolute path to the PDF file.
        max_por_pagina: Maximum number of images to keep per page.
        min_size_px: Minimum width **and** height in pixels; smaller images
            are skipped.

    Returns:
        Dict mapping zero-based page number to a list of image dicts.
        Each dict has keys ``"image_bytes"`` (raw bytes), ``"width"``,
        ``"height"``, ``"ext"`` (format string, e.g. ``"png"``), and
        ``"caption"`` (text found immediately below the image bbox on the
        same page; empty string if not found).
        Pages with no qualifying images are omitted from the dict.
    """
    if not FITZ_DISPONIBLE:
        return {}

    imagenes_por_pagina: Dict[int, List[Dict[str, Any]]] = {}

    try:
        doc = fitz.open(ruta_pdf)

        for num_pag in range(len(doc)):
            page = doc[num_pag]
            image_list = page.get_images(full=True)
            imagenes_pagina: List[Dict[str, Any]] = []

            for img_info in image_list:
                if len(imagenes_pagina) >= max_por_pagina:
                    break

                xref = img_info[0]
                try:
                    img_data = doc.extract_image(xref)
                    width = img_data.get("width", 0)
                    height = img_data.get("height", 0)

                    if width < min_size_px or height < min_size_px:
                        continue

                    caption_text = ""
                    try:
                        img_rects = page.get_image_rects(xref)
                        if img_rects:
                            img_rect = img_rects[0]
                            below_rect = fitz.Rect(
                                img_rect.x0,
                                img_rect.y1,
                                img_rect.x1,
                                img_rect.y1 + CAPTION_MARGIN_PX,
                            )
                            candidate = page.get_text("text", clip=below_rect).strip()
                            # Normalize: collapse newlines and extra spaces
                            candidate = " ".join(candidate.split())
                            if candidate and len(candidate) <= 300:
                                caption_text = candidate
                    except Exception:
                        pass  # caption is optional; never interrupt image extraction

                    imagenes_pagina.append({
                        "image_bytes": img_data["image"],
                        "width": width,
                        "height": height,
                        "ext": img_data.get("ext", "png"),
                        "caption": caption_text,
                    })
                except Exception as e:
                    logging.warning(
                        f"Error extracting image xref={xref} from {ruta_pdf} page {num_pag}: {e}"
                    )

            if imagenes_pagina:
                imagenes_por_pagina[num_pag] = imagenes_pagina

        doc.close()

    except Exception as e:
        logging.warning(f"Error reading {ruta_pdf} with fitz for image extraction: {e}")

    return imagenes_por_pagina


def describir_imagen_con_llm(
    image_bytes: bytes,
    caption: str = "",
    idioma_doc: str = "English",
) -> str:
    """Generate a textual description of an academic image using ``MODELO_OCR``.

    Sends the raw image bytes to ``gemma4:e4b`` (or the model set in
    ``OLLAMA_OCR_MODEL``) via Ollama and returns a description focused on
    the visual content: architecture components and data flow for diagrams,
    column structure for tables, axis labels and trends for charts.

    If a ``caption`` is provided, it is injected as context at the start of
    the prompt so the model can anchor its description to the figure's topic
    without simply echoing the caption text.

    Degenerate outputs are filtered before returning:
    - OCR spam (>20% of words are 'no'/'text') → returns ``""``
    - Description that merely echoes the caption → returns ``""``

    The returned string is plain text embedded with ``MODELO_EMBEDDING``
    and stored as a regular chunk in ChromaDB — no extra infrastructure
    is required for cross-modal retrieval.

    The function is a no-op (returns ``""``) when ``USAR_EMBEDDINGS_IMAGEN``
    is ``False`` or when the Ollama call fails.

    Args:
        image_bytes: Raw image bytes in any format supported by Ollama
            (PNG, JPEG, WebP, …).
        caption: Optional caption text extracted from the PDF near the image.
            Used as context to orient the description; not transcribed literally.
        idioma_doc: Language of the surrounding document ('Spanish', 'Catalan',
            'English'). The description will be written in this language so it
            is semantically consistent with the document's text chunks.

    Returns:
        Non-empty description string on success, or ``""`` on failure,
        degenerate output, or when image embeddings are disabled.
    """
    if not USAR_EMBEDDINGS_IMAGEN:
        return ""

    try:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        idioma_instruccion = (
            f"Write your entire description in {idioma_doc}. "
            if idioma_doc and idioma_doc != "English"
            else ""
        )

        response = ollama.chat(
            model=MODELO_OCR,
            messages=[{
                "role": "user",
                "content": (
                    idioma_instruccion
                    + (f"The figure caption reads: '{caption}'. " if caption else "")
                    + "Describe the visual content of this image. "
                    "If it is a flowchart or architecture diagram: list every labeled block "
                    "(e.g. MatMul, SoftMax, Encoder, Linear), name the inputs and outputs "
                    "(e.g. Q, K, V), describe the arrows between blocks, and state the "
                    "architecture family (e.g. Transformer, CNN). "
                    "If it is a table: state the number of rows and columns, list the column "
                    "headers, and describe the type of data in each column. "
                    "Only describe what you can actually see. Do not invent numbers, metrics, "
                    "or data that are not visible in the image. "
                    "End with one sentence summarizing what this figure illustrates."
                ),
                "images": [image_b64],
            }],
            think=False,
            options={"temperature": 0.1, "num_predict": 2000},
        )
        descripcion = response["message"]["content"].strip()

        if _es_descripcion_spam(descripcion):
            logging.warning(f"Discarding degenerate OCR output (spam) from {MODELO_OCR}")
            return ""
        if _es_prompt_echo(descripcion):
            logging.warning(f"Discarding prompt echo from {MODELO_OCR}")
            return ""
        if caption and _es_solo_caption(descripcion, caption):
            logging.warning(f"Discarding description that merely echoes the caption from {MODELO_OCR}")
            return ""
        return descripcion if len(descripcion) > 10 else ""

    except Exception as e:
        logging.warning(f"Error describing image with {MODELO_OCR}: {e}")
        return ""


# --- 11.3 Indexing ---


def indexar_documentos(
    carpeta: str,
    collection: chromadb.Collection,
    solo_archivos: Optional[List[str]] = None,
    silent: bool = False,
    progress_callback=None,
) -> int:
    """Index PDFs from a folder into ChromaDB.

    Uses pymupdf4llm for Markdown extraction (preferred) with pypdf as
    fallback. Each page is split into chunks, optionally enriched with
    contextual retrieval, embedded via Ollama, and stored in ChromaDB.

    Args:
        carpeta: Path to the folder containing PDF files.
        collection: ChromaDB collection to index into.
        solo_archivos: If set, only index these specific filenames
            (for incremental adds without full re-index).
        silent: Suppress all terminal output (for background/web use).
        progress_callback: Called with ``{"file", "file_index",
            "total_files"}`` at the start of each file.

    Returns:
        Total number of chunks successfully indexed.
    """
    global PYMUPDF_AVAILABLE

    os.makedirs(carpeta, exist_ok=True)
    archivos_pdf = [f for f in os.listdir(carpeta) if f.endswith('.pdf')]
    if solo_archivos is not None:
        archivos_pdf = [f for f in archivos_pdf if f in solo_archivos]

    if not archivos_pdf:
        if not silent:
            ui.warning("No PDF files found in folder")
        return 0

    if not silent:
        ui.pipeline_start("Indexing documents...")

    total_chunks = 0

    def _indexar_chunk(id_doc: str, chunk_text: str, chunk_doc_text: str,
                       metadata: Dict, collection_ref: chromadb.Collection) -> bool:
        """Embed a chunk and add it to ChromaDB. Retries with truncation on length errors."""
        text_to_embed = f"{EMBED_PREFIX_DOC}{chunk_text[:MAX_CHARS_EMBED]}"

        try:
            response = ollama.embeddings(model=MODELO_EMBEDDING, prompt=text_to_embed)
            embedding = response["embedding"]

            collection_ref.add(
                ids=[id_doc],
                embeddings=[embedding],
                documents=[chunk_doc_text],
                metadatas=[metadata]
            )
            return True
        except Exception as e:
            if "context length" in str(e).lower() or "500" in str(e):
                logging.warning(f"Long chunk at {id_doc}, truncating to 1000 chars")
                text_to_embed = f"{EMBED_PREFIX_DOC}{chunk_text[:1000]}"
                try:
                    response = ollama.embeddings(model=MODELO_EMBEDDING, prompt=text_to_embed)
                    collection_ref.add(
                        ids=[id_doc],
                        embeddings=[response["embedding"]],
                        documents=[chunk_doc_text],
                        metadatas=[metadata]
                    )
                    return True
                except Exception as e2:
                    logging.error(f"Persistent embedding error for {id_doc}: {e2}")
            else:
                logging.error(f"Error embedding {id_doc}: {e}")
            return False

    for idx, archivo in enumerate(archivos_pdf):
        if progress_callback:
            try:
                progress_callback({"file": archivo, "file_index": idx + 1, "total_files": len(archivos_pdf)})
            except Exception:
                pass
        if not silent:
            ui.pipeline_update(f"Processing: {archivo}")
        usar_pypdf_fallback = False

        try:
            ruta_pdf = os.path.join(carpeta, archivo)

            imagenes_pdf: Dict[int, List[Dict[str, Any]]] = {}
            if USAR_EMBEDDINGS_IMAGEN and FITZ_DISPONIBLE:
                imagenes_pdf = extraer_imagenes_pdf(ruta_pdf)
                n_imgs_total = sum(len(v) for v in imagenes_pdf.values())
                if n_imgs_total > 0 and not silent:
                    ui.debug(f"  {n_imgs_total} images found across {len(imagenes_pdf)} page(s)")

            if PYMUPDF_AVAILABLE:
                try:
                    page_chunks = pymupdf4llm.to_markdown(ruta_pdf, page_chunks=True)

                    _textos_paginas = [p['text'][:500] for p in page_chunks[:10]]
                    texto_base_doc = "\n\n".join(_textos_paginas)[:4000]
                    idioma_doc = _detectar_idioma(texto_base_doc)

                    for page_info in page_chunks:
                        i = page_info['metadata']['page']
                        texto = page_info['text']

                        if not texto or len(texto) < MIN_CHUNK_LENGTH:
                            continue

                        chunks = dividir_en_chunks(texto)

                        for chunk_idx, chunk_info in enumerate(chunks):
                            chunk_text = chunk_info['text'] if isinstance(chunk_info, dict) else chunk_info
                            chunk_header = chunk_info.get('header', '') if isinstance(chunk_info, dict) else ''

                            id_doc = f"{archivo}_pag{i}_chunk{chunk_idx}"

                            metadata = {
                                "source": archivo,
                                "page": i,
                                "chunk": chunk_idx,
                                "total_chunks_in_page": len(chunks),
                                "format": "markdown",
                                "section_header": chunk_header
                            }

                            if USAR_CONTEXTUAL_RETRIEVAL:
                                contexto_sit = generar_contexto_situacional(chunk_text, texto_base_doc, idioma_doc)
                                chunk_text_con_contexto = (contexto_sit + chunk_text).strip()
                            else:
                                chunk_text_con_contexto = chunk_text

                            if _indexar_chunk(id_doc, chunk_text_con_contexto, chunk_text_con_contexto, metadata, collection):
                                total_chunks += 1

                except Exception as e:
                    logging.error(f"Error with pymupdf4llm on {archivo}: {e}, using pypdf fallback")
                    usar_pypdf_fallback = True

            if not PYMUPDF_AVAILABLE or usar_pypdf_fallback:
                reader = PdfReader(ruta_pdf)

                _textos_paginas = [p.extract_text()[:500] for p in reader.pages[:10]]
                texto_base_doc = "\n\n".join(_textos_paginas)[:4000]
                idioma_doc = _detectar_idioma(texto_base_doc)

                for i, page in enumerate(reader.pages):
                    texto = page.extract_text()

                    if not texto or len(texto) < MIN_CHUNK_LENGTH:
                        continue

                    chunks = dividir_en_chunks(texto)

                    for chunk_idx, chunk_info in enumerate(chunks):
                        chunk_text = chunk_info['text'] if isinstance(chunk_info, dict) else chunk_info
                        chunk_header = chunk_info.get('header', '') if isinstance(chunk_info, dict) else ''

                        id_doc = f"{archivo}_pag{i}_chunk{chunk_idx}"

                        metadata = {
                            "source": archivo,
                            "page": i,
                            "chunk": chunk_idx,
                            "total_chunks_in_page": len(chunks),
                            "format": "plain_text",
                            "section_header": chunk_header
                        }

                        if USAR_CONTEXTUAL_RETRIEVAL:
                            contexto_sit = generar_contexto_situacional(chunk_text, texto_base_doc, idioma_doc)
                            chunk_text_con_contexto = (contexto_sit + chunk_text).strip()
                        else:
                            chunk_text_con_contexto = chunk_text

                        if _indexar_chunk(id_doc, chunk_text_con_contexto, chunk_text_con_contexto, metadata, collection):
                            total_chunks += 1

            if imagenes_pdf:
                if not silent:
                    ui.debug("  describing and indexing images...")
                for num_pag, pagina_imagenes in imagenes_pdf.items():
                    for img_idx, img_info in enumerate(pagina_imagenes):
                        caption = img_info.get("caption", "")
                        descripcion = describir_imagen_con_llm(img_info["image_bytes"], caption=caption, idioma_doc=idioma_doc)
                        if not descripcion:
                            continue

                        contexto_img = generar_contexto_situacional(descripcion, texto_base_doc, idioma_doc)
                        descripcion_enriquecida = (contexto_img + descripcion).strip()

                        img_chunk_idx = _IMAGEN_CHUNK_OFFSET + img_idx
                        id_img = f"{archivo}_pag{num_pag}_chunk{img_chunk_idx}"

                        metadata_img: Dict[str, Any] = {
                            "source": archivo,
                            "page": num_pag,
                            "chunk": img_chunk_idx,
                            "format": "image",
                            "section_header": "",
                            "image_width": img_info["width"],
                            "image_height": img_info["height"],
                        }

                        if _indexar_chunk(id_img, descripcion_enriquecida, descripcion_enriquecida, metadata_img, collection):
                            total_chunks += 1

        except Exception as e:
            logging.error(f"Error processing {archivo}: {e}")
            if not silent:
                ui.error(f"error in {archivo}: {e}")

    if not silent:
        ui.pipeline_stop()
    return total_chunks


def obtener_documentos_indexados(collection: chromadb.Collection) -> List[str]:
    """List unique document names (``source``) in the collection.

    Args:
        collection: ChromaDB collection to inspect.

    Returns:
        Sorted list of document filenames.
    """
    try:
        all_metadata = collection.get(include=['metadatas'])
        documentos = set()
        for meta in all_metadata['metadatas']:
            if 'source' in meta:
                documentos.add(meta['source'])
        return sorted(list(documentos))
    except Exception:
        return []


# ─────────────────────────────────────────────
# SECTION 12: ENTRY POINT
# ─────────────────────────────────────────────

def main():
    """Launch the MonkeyGrab CLI application."""
    import rag.chat_pdfs as rag_engine
    from rag.cli import MonkeyGrabCLI
    cli = MonkeyGrabCLI(rag_engine)
    cli.run()


if __name__ == "__main__":
    main()
