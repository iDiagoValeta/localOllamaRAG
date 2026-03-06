"""
MonkeyGrab — Sistema RAG para consulta de PDFs
==============================================

Aplicación interactiva con dos modos:

    CHAT  Conversación libre con modelo base, historial persistente e identidad del proyecto.
    RAG   Consulta documental sobre PDFs indexados, recuperación híbrida y respuestas con fuentes.

Pipeline RAG (etapas activables por flags):
    1. Indexación      chunking + embeddings + contextual retrieval (opc.)
    2. Recuperación    semántica + descomposición (opc.) + keywords
    3. Profundización  búsqueda exhaustiva por términos críticos
    4. Ranking         fusión RRF + reranking Cross-Encoder (opc.)
    5. Contexto        expansión de adyacentes + optimización
    6. Generación      síntesis RECOMP (opc.) + streaming
    7. Observabilidad  métricas y dumps de debug

Uso:
    python chat_pdfs.py
"""

# =============================================================================
# MAPA DEL MÓDULO — Índice de secciones
# =============================================================================
#
#  CONFIGURACIÓN (arranque)
#  ├── 1. Importaciones     stdlib → terceros (ollama, chromadb, pypdf) → locales
#  ├── 2. Dependencias opc. pymupdf4llm (PDF), CrossEncoder (reranking)
#  └── 3. Configuración global
#       ├── 3.1 Runtime terminal (UTF-8)
#       ├── 3.2 Modelos Ollama (RAG, CHAT, embedding, contextual, RECOMP)
#       ├── 3.3 Flags del pipeline (toggle por etapa)
#       ├── 3.4 Rutas y persistencia (DB, historial, debug)
#       ├── 3.5 Parámetros recuperación/generación
#       └── 3.6 Logging y entorno
#
#  LÓGICA DE NEGOCIO
#  ├── 4. Prompts del sistema    RAG (vacío), CHAT (identidad + idioma)
#  ├── 5. Persistencia historial  JSON local modo CHAT
#  ├── 6. Preprocesado y chunking texto → chunks Markdown con overlap
#  ├── 7. Keywords y búsqueda léxica  siglas, bigramas, where_document
#  ├── 8. Reranking semántico    CrossEncoder singleton, CUDA/CPU
#  ├── 9. Pipeline recuperación  semántica + keywords + exhaustiva → RRF → rerank
#  ├──10. Contexto y generación
#  │      ├── 10.1 Optimización texto PDF (artefactos, footers, párrafos)
#  │      ├── 10.2 Construcción contexto (raw o síntesis RECOMP)
#  │      ├── 10.3 Debug         volcado interacción en debug_rag/
#  │      ├── 10.4 Generación    streaming Ollama, formato <context>
#  │      └── 10.5 Evaluación    pipeline silencioso para RAGAS
#  └──11. Indexación y colección
#       ├── 11.1 Contextual retrieval  enriquecimiento chunk con LLM
#       ├── 11.2 Indexación           PDFs → pymupdf4llm/pypdf → ChromaDB
#       └── 11.3 Gestión colección    listar documentos indexados
#
#  ENTRADA
#  └──12. Punto de entrada   main() → MonkeyGrabCLI.run()
#
# =============================================================================


# =============================================================================
# SECCIÓN 1: IMPORTACIONES
# =============================================================================
# Organizadas por responsabilidad: stdlib → terceros → locales.
# =============================================================================


# --- 1.1 Librería estándar ---

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

# --- 1.2 Terceros (Ollama, ChromaDB, pypdf) ---

import chromadb
import ollama
from pypdf import PdfReader

# --- 1.3 Bootstrap: path del proyecto ---

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# --- 1.4 Locales ---

from rag.cli.display import ui


# =============================================================================
# SECCIÓN 2: DEPENDENCIAS OPCIONALES
# =============================================================================
# pymupdf4llm (extracción PDF) y CrossEncoder (reranking).
# =============================================================================


try:
    with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        import pymupdf4llm
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False


# =============================================================================
# SECCIÓN 3: CONFIGURACIÓN GLOBAL
# =============================================================================


# --- 3.1 Runtime terminal (encoding UTF-8) ---

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

# --- 3.2 Modelos Ollama ---

MODELO_RAG = os.getenv("OLLAMA_RAG_MODEL", "Qwen3-FineTuned:latest")
MODELO_CHAT = os.getenv("OLLAMA_CHAT_MODEL", "gemma3:4b")
MODELO_EMBEDDING = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")
MODELO_CONTEXTUAL = os.getenv("OLLAMA_CONTEXTUAL_MODEL", "gemma3:4b")
MODELO_RECOMP = os.getenv("OLLAMA_RECOMP_MODEL", "gemma3:4b")


def _inferir_descripcion_modelo(nombre_modelo: str) -> str:
    return nombre_modelo.split(":")[0]


MODELO_DESC = os.getenv("MODELO_DESC", _inferir_descripcion_modelo(MODELO_RAG))

# --- 3.3 Flags del pipeline (toggle por etapa) ---

USAR_CONTEXTUAL_RETRIEVAL = True
USAR_LLM_QUERY_DECOMPOSITION = True
USAR_BUSQUEDA_HIBRIDA = True
USAR_BUSQUEDA_EXHAUSTIVA = True
USAR_RERANKER = RERANKER_AVAILABLE
EXPANDIR_CONTEXTO = True
USAR_OPTIMIZACION_CONTEXTO = True
USAR_RECOMP_SYNTHESIS = False
LOGGING_METRICAS = True
GUARDAR_DEBUG_RAG = True

# --- 3.4 Rutas y persistencia ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_DOCS = os.getenv("DOCS_FOLDER", os.path.join(BASE_DIR, "pdfs"))

_carpeta_nombre = os.path.basename(os.path.abspath(CARPETA_DOCS))
_embed_slug = MODELO_EMBEDDING.split(":")[0].replace("/", "_")

PATH_DB = os.path.join(BASE_DIR, "mi_vector_db", f"{_carpeta_nombre}_{_embed_slug}")
COLLECTION_NAME = f"docs_{_carpeta_nombre}"

HISTORIAL_PATH = os.path.join(BASE_DIR, "historial_chat.json")
MAX_HISTORIAL_MENSAJES = 40

CARPETA_DEBUG_RAG = os.path.join(BASE_DIR, "debug_rag")

# --- 3.5 Parámetros de recuperación y generación ---

_embed_name_lower = MODELO_EMBEDDING.lower().split(":")[0]
if "nomic" in _embed_name_lower:
    EMBED_PREFIX_QUERY = "search_query: "
    EMBED_PREFIX_DOC = "search_document: "
    _EMBED_PREFIX_DESC = "prefijos nomic (query/doc)"
else:
    EMBED_PREFIX_QUERY = ""
    EMBED_PREFIX_DOC = ""
    _EMBED_PREFIX_DESC = "sin prefijos (nativo)"

MAX_CHARS_EMBED = 4000
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 350
MIN_CHUNK_LENGTH = 80

N_RESULTADOS_SEMANTICOS = 80
N_RESULTADOS_KEYWORD = 40
TOP_K_RERANK_CANDIDATES = 200
TOP_K_AFTER_RERANK = 15
TOP_K_FINAL = 6
N_TOP_PARA_EXPANSION = 3

RERANKER_MODEL_QUALITY = os.getenv("RERANKER_QUALITY", "quality")
UMBRAL_RELEVANCIA = 0.50
UMBRAL_SCORE_RERANKER = 0.40

MIN_LONGITUD_PREGUNTA_RAG = 10
MAX_CONTEXTO_CHARS = 8192

# --- 3.6 Logging y entorno ---

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


# =============================================================================
# SECCIÓN 4: PROMPTS DEL SISTEMA
# =============================================================================
# Define la identidad y comportamiento del modelo en cada modo.
# =============================================================================


SYSTEM_PROMPT_RAG = """You are a professional document analysis assistant. Your role is to answer questions accurately based on the provided document context.

Guidelines:
- Base your answers strictly on the information within the <context> tags.
- Do not add information beyond what the context provides.
- Formulate clear, well-structured responses in complete sentences.
- For factual questions, be direct and precise.
- For analytical or complex questions, provide detailed explanations referencing specific information from the context.
- Always respond in the same language as the context (English, Spanish/Castellano, or Catalan/Català).
- Synthesize information naturally rather than copying text verbatim."""

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


# =============================================================================
# SECCIÓN 5: PERSISTENCIA DEL HISTORIAL
# =============================================================================
# JSON local para modo CHAT. RAG no persiste (consultas independientes).
# =============================================================================


def cargar_historial() -> List[Dict[str, str]]:
    """Carga historial CHAT desde disco. Retorna [] si no existe o falla."""
    try:
        if os.path.exists(HISTORIAL_PATH):
            with open(HISTORIAL_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and "chat" in data:
                    return data["chat"]
                if isinstance(data, list):
                    return data
    except Exception as e:
        logging.warning(f"Error cargando historial: {e}")
    
    return []


def guardar_historial(historial: List[Dict[str, str]]) -> None:
    """Guarda historial CHAT a disco (últimos MAX_HISTORIAL_MENSAJES)."""
    try:
        historial_recortado = historial[-MAX_HISTORIAL_MENSAJES:]
        with open(HISTORIAL_PATH, 'w', encoding='utf-8') as f:
            json.dump(historial_recortado, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Error guardando historial: {e}")


def limpiar_historial(historial: List[Dict[str, str]]) -> None:
    """Vacía el historial in-place y persiste."""
    historial.clear()
    guardar_historial(historial)


# =============================================================================
# SECCIÓN 6: PREPROCESADO Y CHUNKING
# =============================================================================
# Texto → chunks semánticos con headers Markdown, overlap y expansión adyacente.
# =============================================================================


def extraer_header_markdown(texto: str) -> str:
    """Extrae el último header Markdown (#...#) del texto. Vacío si no hay."""
    headers = re.findall(r'^(#{1,4}\s+.+)$', texto, re.MULTILINE)
    return headers[-1].strip() if headers else ""


def dividir_en_chunks(
    texto: str, 
    chunk_size: int = CHUNK_SIZE, 
    overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, str]]:
    """Divide texto en chunks por secciones Markdown, con overlap. Retorna [{"text", "header"}]. """
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
        """División recursiva usando separadores jerárquicos."""
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
    """IDs de chunks vecinos (misma página y cross-page) para contexto expandido."""
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


# =============================================================================
# SECCIÓN 7: KEYWORDS Y BÚSQUEDA LÉXICA
# =============================================================================
# Complementa la semántica con búsqueda por términos (siglas, bigramas, where_document).
# =============================================================================


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
    # Valencià
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
    """Extrae siglas, bigramas, términos entre paréntesis y tokens técnicos. Filtra stopwords."""
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
        if len(clean) > 4 and clean.lower() not in STOPWORDS:
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
    """Búsqueda por where_document (get, no query). Retorna (resultados, métricas)."""
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
                logging.warning(f"Error buscando keyword '{keyword}': {e}")
    
    metricas['keywords_encontradas'] = len(keywords_encontradas)
    metricas['resultados_totales'] = len(resultados_keyword)
    
    if keywords_encontradas:
        ui.debug(f"keywords: {', '.join(list(keywords_encontradas)[:10])}")
    else:
        ui.debug("sin coincidencias directas por keywords")
    
    if LOGGING_METRICAS:
        logging.info(f"Búsqueda keywords: {metricas['keywords_encontradas']}/{metricas['keywords_totales']} encontradas, {metricas['resultados_totales']} resultados")
    
    return resultados_keyword, metricas


def busqueda_exhaustiva_texto(
    terminos_criticos: List[str], 
    collection: chromadb.Collection,
    max_results: int = 20
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Escaneo por lotes de todos los docs. Retorna los que contienen los términos, ordenados por matches."""
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
            logging.warning(f"Error en búsqueda exhaustiva (offset {offset}): {e}")
    
    resultados.sort(key=lambda x: x['num_matches'], reverse=True)
    
    if LOGGING_METRICAS:
        logging.info(f"Búsqueda exhaustiva: {metricas['documentos_con_matches']} docs con matches de {metricas['documentos_escaneados']} escaneados")
    
    return resultados[:max_results], metricas


# =============================================================================
# SECCIÓN 8: RERANKING SEMÁNTICO
# =============================================================================
# CrossEncoder (singleton, CUDA/CPU). Reordena candidatos por relevancia pregunta-doc.
# =============================================================================


_reranker_model = None


def _detectar_dispositivo_reranker() -> str:
    """Devuelve 'cuda' si está disponible, si no 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def obtener_modelo_reranker():
    """Singleton del Cross-Encoder. Carga lazy, FP16 en CUDA si hay GPU."""
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
            
            ui.debug(f"Cargando reranker: {modelo_nombre}")
            ui.debug(f"dispositivo: {device.upper()}" + (" (FP16)" if device == "cuda" else ""))

            model_kwargs = {"torch_dtype": "float16"} if device == "cuda" else {}
            import io, contextlib
            with contextlib.redirect_stderr(io.StringIO()):
                _reranker_model = CrossEncoder(
                    modelo_nombre,
                    device=device,
                    model_kwargs=model_kwargs,
                )

            ui.debug(f"reranker cargado en {device.upper()}")
        except Exception as e:
            logging.error(f"Error cargando modelo de reranking: {e}")
            return None
    
    return _reranker_model


def rerank_resultados(
    pregunta: str, 
    documentos_recuperados: List[Dict[str, Any]], 
    top_k: int = TOP_K_AFTER_RERANK
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Reordena candidatos con Cross-Encoder. Retorna (top_k docs con score_reranker, métricas)."""
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
            logging.info(f"Reranking ({dev}): {metricas['candidatos_entrada']} → {metricas['resultados_salida']} en {metricas['tiempo_reranking']:.2f}s")
        
        return documentos_reordenados, metricas
        
    except Exception as e:
        logging.error(f"Error en reranking: {e}")
        metricas['resultados_salida'] = len(documentos_recuperados)
        return documentos_recuperados, metricas


def generar_queries_con_llm(pregunta: str) -> List[str]:
    """Genera 3 queries de búsqueda con un modelo auxiliar. Mismo idioma que la pregunta."""
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
            options={
                "temperature": 0.5,
                "num_predict": 400,
                "stop": ["\n\n\n"],
            }
        )

        queries = [
            q.strip().lstrip("0123456789.-) ")
            for q in response["response"].strip().split("\n")
            if q.strip() and len(q.strip()) > 20
        ]

        return queries[:3]

    except Exception as e:
        logging.warning(f"Error generando queries con LLM ({MODELO_CHAT}): {e}")
        return []


def _validar_coherencia_query(query: str) -> bool:
    """Detecta si una query es un bag-of-words incoherente."""
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
    # Español
    "cómo", "qué", "cuál", "cuáles", "cuándo", "dónde", "por", 
    "para", "que", "son", "está", "entre", "con", "los", "las",
    # Valencià
    "com", "quins", "quines", "quan", "quin", "quina", "per", "que",
    }

    has_connectors = any(w in connectors for w in words)
    if len(words) > 8 and not has_connectors:
        return False

    return True


def _filtrar_terminos_criticos(terminos: List[str]) -> List[str]:
    """Mantiene solo términos de dominio específico con alta discriminación para fase exhaustiva."""
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


# =============================================================================
# SECCIÓN 9: PIPELINE DE RECUPERACIÓN HÍBRIDA
# =============================================================================
# Semántica multi-query + keywords + exhaustiva → fusión RRF → reranking.
# =============================================================================


def realizar_busqueda_hibrida(
    pregunta: str,
    collection: chromadb.Collection
) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    """Orquesta semántica, keywords, exhaustiva, fusión y reranking. Retorna (fragmentos, mejor_score, métricas)."""
    ui.debug("Iniciando búsqueda híbrida...")
    
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
        ui.debug("descomponiendo pregunta...")
        llm_queries = generar_queries_con_llm(pregunta)
        if llm_queries:
            ui.debug(f"{len(llm_queries)} sub-queries generadas")

    ui.debug("busqueda semántica...")

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
    
    ui.debug(f"{len(queries)} variante(s) de la pregunta")

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

            all_semantic_results[chunk_id]['score_semantic'] += 1.0 / (idx + 60)
            all_semantic_results[chunk_id]['query_matches'].append(q_idx + 1)
            if distancia < all_semantic_results[chunk_id]['distancia']:
                all_semantic_results[chunk_id]['distancia'] = distancia
    
    metricas_totales['fase_semantica'] = {
        'queries_generadas': len(queries),
        'fragmentos_unicos': len(all_semantic_results)
    }
    
    ui.debug(f"{len(all_semantic_results)} fragmentos únicos")

    results_keyword = []
    metricas_keywords = {}
    if USAR_BUSQUEDA_HIBRIDA:
        ui.debug("busqueda por keywords...")
        keywords = extraer_keywords(pregunta)
        if keywords:
            ui.debug(f"detectadas: {', '.join(keywords[:8])}")
        results_keyword, metricas_keywords = busqueda_por_keywords(pregunta, collection)
        metricas_totales['fase_keywords'] = metricas_keywords

    ui.debug("fusionando resultados...")
    
    fragmentos_data = all_semantic_results.copy()
    
    for idx, result in enumerate(results_keyword, 1):
        chunk_id = result['id']
        
        if chunk_id in fragmentos_data:
            fragmentos_data[chunk_id]['score_keyword'] += 1.0 / (idx + 60)
            if result['keyword_match'] not in fragmentos_data[chunk_id]['matches']:
                fragmentos_data[chunk_id]['matches'].append(result['keyword_match'])
        else:
            fragmentos_data[chunk_id] = {
                'doc': result['doc'],
                'metadata': result['metadata'],
                'distancia': result['distancia'],
                'id': chunk_id,
                'score_semantic': 0.0,
                'score_keyword': 1.0 / (idx + 60),
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
        ui.debug(f"busqueda profunda: {', '.join(terminos_criticos[:6])}")
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
        ui.debug(f"reranking top {n_candidatos} candidatos...")

        candidatos_rerank = fragmentos_ranked[:TOP_K_RERANK_CANDIDATES]
        fragmentos_ranked, metricas_rerank = rerank_resultados(
            pregunta,
            candidatos_rerank,
            top_k=TOP_K_AFTER_RERANK
        )
        metricas_totales['fase_reranking'] = metricas_rerank
        ui.debug(f"top {len(fragmentos_ranked)} tras reranking")
    
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
            f"Pipeline completo: Semántica({sem_unicos}) + "
            f"Keywords({kw_total}) → "
            f"Fusión({metricas_totales['candidatos_fusion']}) → "
            f"Reranking({metricas_totales['resultados_finales']})"
        )
    
    return fragmentos_ranked, mejor_score, metricas_totales


# =============================================================================
# SECCIÓN 10: CONTEXTO Y GENERACIÓN
# =============================================================================
# Fragmentos → prompt <context>...</context> → streaming Ollama. Formato alineado con train.py.
# =============================================================================


# --- 10.1 Optimización de texto PDF ---

def _es_continuacion_parrafo(linea_previa: str, linea_actual: str) -> bool:
    """Heurística: detecta si la línea actual continúa el párrafo (evita roturas por doble espaciado PDF)."""
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
    """Re-une líneas separadas por extracción PDF (look-ahead para continuaciones)."""
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
    """Elimina ruido PDF (artefactos □, footers, doble espaciado). Ahorro típico 30-50% chars."""
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
    
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    
    return texto.strip()


# --- 10.2 Construcción de contexto ---


def _marcar_fragmento_incompleto(texto: str) -> str:
    """Añade [incomplete fragment] si el texto no termina en puntuación de cierre."""
    stripped = texto.rstrip()
    if not stripped:
        return texto
    if stripped[-1] not in '.?!:':
        return texto + '\n[incomplete fragment]'
    return texto


def construir_contexto_para_modelo(fragmentos: List[Dict[str, Any]]) -> str:
    """
    Construye el contexto para el modelo RAG a partir de fragmentos recuperados.

    Formato de salida por fragmento:
        --- [Fragment N] ---
        [Fragment Context]            ← solo si hay resumen de Contextual Retrieval
        ...
        [Source Text]
        ...
        [incomplete fragment]         ← solo si el chunk está truncado

    Separador entre fragmentos: doble salto de línea.
    Optimización de texto PDF opcional (flag USAR_OPTIMIZACION_CONTEXTO).
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

        # Separar resumen de Contextual Retrieval del texto fuente
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
            f"Contexto optimizado: {chars_original} → {chars_optimizado} chars "
            f"({ahorro} ahorrados, {pct:.1f}%)"
        )

    return resultado


def sintetizar_contexto_recomp(fragmentos: List[Dict[str, Any]], query_usuario: str = "") -> str:
    """Síntesis con MODELO_RECOMP. Si falla o está desactivado, usa construir_contexto_para_modelo."""
    if not USAR_RECOMP_SYNTHESIS or not fragmentos:
        return construir_contexto_para_modelo(fragmentos)
        
    textos_preparados = []
    for i, f in enumerate(fragmentos):
        content = f['doc'].replace("\n", " ").strip()
        textos_preparados.append(f"Fragment {i+1}:\n{content}")

    contexto_raw = "\n\n".join(textos_preparados)

    system_prompt = (
        "You are a context synthesizer. Your task is to distill the key concepts "
        "from text fragments into a clear, concise summary.\n"
        "RULES:\n"
        "1. Extract ONLY the crucial concepts, definitions, and technical details "
        "that directly answer the user's question.\n"
        "2. ONLY use information EXPLICITLY written in the fragments. "
        "NEVER add external knowledge.\n"
        "3. Do NOT cite or reference fragment numbers, sources, pages, "
        "or any document metadata. Write as continuous prose.\n"
        "4. Preserve technical terms, formulas, and numerical values exactly.\n"
        "5. Be concise: include only what is essential to understand the concepts.\n"
        "6. Output in the same language as the input fragments."
    )

    focus_instruction = (
        f"Focus specifically on information answering: '{query_usuario}'. "
        "Include only the essential details."
        if query_usuario else 
        "Summarize the key technical definitions and comparisons."
    )

    user_prompt = (
        f"{focus_instruction}\n\n"
        f"--- INPUT FRAGMENTS ---\n{contexto_raw}\n-----------------------\n\n"
        "Concise synthesis:"
    )
    
    try:
        response = ollama.chat(
            model=MODELO_RECOMP,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": 0.1,
                "num_predict": 2048,
                "top_p": 0.9,
                "repeat_penalty": 1.15,
                "num_ctx": 8192
            }
        )
        
        sintesis = response['message']['content'].strip()
        
        if len(sintesis) < 20: 
            return construir_contexto_para_modelo(fragmentos)
            
        return sintesis

    except Exception as e:
        logging.warning(f"Error crítico en síntesis RECOMP ({MODELO_RECOMP}): {e}")
        return construir_contexto_para_modelo(fragmentos)


# --- 10.3 Debug ---

def guardar_debug_rag(
    pregunta: str,
    system_prompt: str = "",
    mensaje_usuario: str = "",
    respuesta: str = "",
    fragmentos: Optional[List[Dict[str, Any]]] = None,
    motivo_interrupcion: Optional[str] = None,
    metricas: Optional[Dict[str, Any]] = None
) -> None:
    """Volcado de interacción RAG en debug_rag/ (timestamp + slug). Incluye sub-queries, keywords, métricas."""
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
                f.write("  PIPELINE DE RECUPERACIÓN (sub-queries, keywords, términos, métricas)\n")
                f.write("─" * 80 + "\n")
                sub_q = metricas.get('sub_queries', [])
                if sub_q:
                    f.write("\nSub-queries (Query Decomposition):\n")
                    for i, sq in enumerate(sub_q, 1):
                        f.write(f"  {i}. {sq}\n")
                queries_sem = metricas.get('queries_semanticas', [])
                if queries_sem:
                    f.write("\nQueries usadas en búsqueda semántica:\n")
                    for i, q in enumerate(queries_sem, 1):
                        f.write(f"  {i}. {q}\n")
                keywords = metricas.get('keywords', [])
                if keywords:
                    f.write(f"\nKeywords extraídas ({len(keywords)}):\n  {', '.join(keywords[:30])}\n")
                    if len(keywords) > 30:
                        f.write(f"  ... y {len(keywords) - 30} más\n")
                terminos = metricas.get('terminos_criticos', [])
                if terminos:
                    f.write(f"\nTérminos críticos (búsqueda exhaustiva):\n  {', '.join(terminos)}\n")
                fase_kw = metricas.get('fase_keywords', {})
                if fase_kw:
                    f.write(f"\nMétricas keywords: {fase_kw.get('keywords_encontradas', 0)}/{fase_kw.get('keywords_totales', 0)} encontradas, {fase_kw.get('resultados_totales', 0)} resultados\n")
                f.write(f"\nMétricas completas:\n{json.dumps(metricas, indent=2, ensure_ascii=False, default=str)}\n\n")

            if motivo_interrupcion:
                f.write("─" * 80 + "\n")
                f.write("  ⚠ INTERRUPCIÓN TEMPRANA\n")
                f.write("─" * 80 + "\n")
                f.write(f"{motivo_interrupcion}\n\n")
                if metricas:
                    f.write("Métricas de búsqueda:\n")
                    f.write(json.dumps(metricas, indent=2, ensure_ascii=False) + "\n\n")

            f.write("─" * 80 + "\n")
            f.write("  CONFIGURACIÓN DEL PIPELINE\n")
            f.write("─" * 80 + "\n")
            f.write(f"Modelo RAG: {_inferir_descripcion_modelo(MODELO_RAG)}\n")
            f.write(f"Contextual Retrieval (Indexación): {'SÍ' if USAR_CONTEXTUAL_RETRIEVAL else 'NO'}\n")
            f.write(f"Query Decomposition: {'SÍ' if USAR_LLM_QUERY_DECOMPOSITION else 'NO'}\n")
            f.write(f"Búsqueda Híbrida (keywords): {'SÍ' if USAR_BUSQUEDA_HIBRIDA else 'NO'}\n")
            f.write(f"Búsqueda Exhaustiva: {'SÍ' if USAR_BUSQUEDA_EXHAUSTIVA else 'NO'}\n")
            f.write(f"Reranker: {'SÍ' if USAR_RERANKER else 'NO'}\n")
            f.write(f"Expandir Contexto: {'SÍ' if EXPANDIR_CONTEXTO else 'NO'}\n")
            f.write(f"Optimizar Contexto: {'SÍ' if USAR_OPTIMIZACION_CONTEXTO else 'NO'}\n")
            f.write(f"RECOMP Synthesis: {'SÍ' if USAR_RECOMP_SYNTHESIS else 'NO'}\n\n")
            
            f.write("─" * 80 + "\n")
            f.write("  PREGUNTA ORIGINAL\n")
            f.write("─" * 80 + "\n")
            f.write(f"{pregunta}\n\n")
            
            f.write("─" * 80 + "\n")
            f.write("  SYSTEM PROMPT\n")
            f.write("─" * 80 + "\n")
            f.write(f"{system_prompt or '(no enviado)'}\n\n")
            
            context_match = re.search(r'<context>(.*?)</context>', mensaje_usuario, re.DOTALL)
            contexto_enviado = context_match.group(1).strip() if context_match else "(vacío)"
            
            f.write("─" * 80 + "\n")
            if USAR_RECOMP_SYNTHESIS:
                f.write("  SÍNTESIS RECOMP ENVIADA AL MODELO FINAL (en vez de raw chunks)\n")
            else:
                f.write("  CONTEXTO RAW ENVIADO AL MODELO FINAL\n")
            f.write("─" * 80 + "\n")
            f.write(f"{contexto_enviado}\n\n")
            
            f.write("─" * 80 + "\n")
            f.write("  MENSAJE DE USUARIO (Prompt real completo)\n")
            f.write("─" * 80 + "\n")
            f.write(f"{mensaje_usuario or '(no enviado al modelo)'}\n\n")
            
            f.write("─" * 80 + "\n")
            f.write("  RESPUESTA DEL MODELO\n")
            f.write("─" * 80 + "\n")
            f.write(f"{respuesta or '(no generada)'}\n\n")

            f.write("─" * 80 + "\n")
            f.write(f"  FRAGMENTOS RECUPERADOS ({len(fragmentos)})\n")
            f.write("─" * 80 + "\n")
            for i, frag in enumerate(fragmentos, 1):
                meta = frag.get('metadata', {})
                score = frag.get('score_final', 'N/A')
                score_rr = frag.get('score_reranker', 'N/A')
                f.write(f"\n--- Fragmento {i} ---\n")
                pag = meta.get('page', 0)
                f.write(f"Fuente: {meta.get('source', '?')}, pág. {pag + 1 if isinstance(pag, int) else pag}\n")
                f.write(f"Score final: {score}  |  Score reranker: {score_rr}\n")
                matches = frag.get('matches', [])
                if matches:
                    f.write(f"Keywords que coincidieron: {', '.join(matches)}\n")
                query_matches = frag.get('query_matches', [])
                if query_matches:
                    f.write(f"Coincidió con query(s): {query_matches}\n")
                f.write(f"Sección: {meta.get('section_header', '(sin header)')}\n")
                doc_text = frag.get('doc', '')
                if '\\n\\n' in doc_text:
                    ctx_part, orig_part = doc_text.split('\\n\\n', 1)
                    f.write(f"[Contextual Retrieval]:\n{ctx_part}\n\n")
                    f.write(f"[Texto del documento]:\n{orig_part}\n")
                else:
                    f.write(f"[Texto del documento]:\n{doc_text}\n")
        
        logging.info(f"Debug RAG guardado: {ruta}")
        
    except Exception as e:
        logging.warning(f"Error guardando debug RAG: {e}")


# --- 10.4 Generación de respuesta ---

OLLAMA_BASE_URL = "http://localhost:11434"


def _ollama_generate_stream(model: str, system: str, prompt: str, options: dict):
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": True,
        "options": options,
    }
    with requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, stream=True) as resp:
        for line in resp.iter_lines():
            if line:
                yield json.loads(line)


def generar_respuesta(pregunta: str, fragmentos: List[Dict[str, Any]], metricas: Optional[Dict[str, Any]] = None) -> str:
    if USAR_RECOMP_SYNTHESIS:
        contexto_str = sintetizar_contexto_recomp(fragmentos, query_usuario=pregunta)
    else:
        contexto_str = construir_contexto_para_modelo(fragmentos)

    mensaje_usuario = f"{pregunta}\n\n<context>{contexto_str}</context>"

    respuesta_completa = ""
    for chunk in _ollama_generate_stream(
        model=MODELO_RAG,
        system=SYSTEM_PROMPT_RAG,
        prompt=mensaje_usuario,
        options={"temperature": 0.15, "top_p": 0.9, "repeat_penalty": 1.15, "num_ctx": 8192},
    ):
        content = chunk.get("response", "")
        if content:
            respuesta_completa += content

    print()
    ui.stream_token(respuesta_completa)
    print()

    guardar_debug_rag(pregunta, SYSTEM_PROMPT_RAG, mensaje_usuario, respuesta_completa, fragmentos, metricas=metricas)
    return respuesta_completa


def generar_respuesta_silenciosa(pregunta: str, fragmentos: List[Dict[str, Any]], metricas: Optional[Dict[str, Any]] = None) -> str:
    if USAR_RECOMP_SYNTHESIS:
        contexto_str = sintetizar_contexto_recomp(fragmentos, query_usuario=pregunta)
    else:
        contexto_str = construir_contexto_para_modelo(fragmentos)

    mensaje_usuario = f"{pregunta}\n\n<context>{contexto_str}</context>"

    respuesta_completa = ""
    for chunk in _ollama_generate_stream(
        model=MODELO_RAG,
        system=SYSTEM_PROMPT_RAG,
        prompt=mensaje_usuario,
        options={"temperature": 0.15, "top_p": 0.9, "repeat_penalty": 1.15, "num_ctx": 8192},
    ):
        content = chunk.get("response", "")
        if content:
            respuesta_completa += content
    return respuesta_completa


# --- 10.5 Evaluación (RAGAS) ---

def evaluar_pregunta_rag(
    pregunta: str,
    collection: chromadb.Collection
) -> Tuple[str, List[str]]:
    """Pipeline completo silencioso: búsqueda → filtrado → expansión → generación. Retorna (respuesta, contextos)."""
    import io
    import contextlib
    if len(pregunta.strip()) < MIN_LONGITUD_PREGUNTA_RAG:
        return ("", [])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        fragmentos_ranked, mejor_score, _ = realizar_busqueda_hibrida(pregunta, collection)
        if not fragmentos_ranked or mejor_score < UMBRAL_RELEVANCIA:
            return ("", [])
        if USAR_RERANKER:
            fragmentos_filtrados = [
                f for f in fragmentos_ranked
                if f.get('score_reranker', f.get('score_final', 0)) >= UMBRAL_SCORE_RERANKER
            ]
            if not fragmentos_filtrados:
                return ("", [])
            fragmentos_ranked = fragmentos_filtrados
        fragmentos_finales = fragmentos_ranked[:TOP_K_FINAL]
        ids_usados = {f['id'] for f in fragmentos_finales}
        if EXPANDIR_CONTEXTO and fragmentos_finales and 'chunk' in fragmentos_finales[0]['metadata']:
            for frag in fragmentos_finales[:N_TOP_PARA_EXPANSION]:
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


# =============================================================================
# SECCIÓN 11: INDEXACIÓN Y COLECCIÓN
# =============================================================================
# PDFs → pymupdf4llm/pypdf → chunks → embeddings Ollama → ChromaDB.
# =============================================================================


# --- 11.1 Contextual retrieval ---

def generar_contexto_situacional(chunk_text: str, texto_base: str) -> str:
    """2-3 frases de resumen global + contexto situacional. Usa MODELO_CONTEXTUAL."""
    if not USAR_CONTEXTUAL_RETRIEVAL:
        return ""
        
    system_prompt = (
        "You are an expert at analyzing academic documents. "
        "When given a full document and an excerpt from it, produce exactly 2-3 sentences: "
        "first a brief summary of what the document is about, then how the excerpt fits within it. "
        "No introductions, no labels, no meta-commentary. "
        "Respond in the same language as the input document."
    )

    user_prompt = (
        "Generate the situational context for the following excerpt:\\n\\n"
        f"<document>\\n{texto_base}\\n</document>\\n\\n"
        f"<excerpt>\\n{chunk_text}\\n</excerpt>\\n"
    )
    
    try:
        response = ollama.chat(
            model=MODELO_CONTEXTUAL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            options={"temperature": 0.1, "num_predict": 250}
        )
        contexto = response['message']['content'].strip()
        if contexto:
            return f"{contexto}\\n\\n"
    except Exception as e:
        logging.warning(f"Error generando contexto situacional: {e}")
    return ""


# --- 11.2 Indexación ---

def indexar_documentos(
    carpeta: str, 
    collection: chromadb.Collection,
    solo_archivos: Optional[List[str]] = None,
    silent: bool = False,
    progress_callback=None,
) -> int:
    """Indexa PDFs de carpeta. pymupdf4llm preferente, pypdf fallback. Retorna total de chunks.
    Si solo_archivos está definido, solo indexa esos archivos (para añadir sin reindexar todo).
    silent=True suprime toda salida por pantalla (uso en background/web).
    progress_callback(info) se llama al iniciar cada archivo con {"file", "file_index", "total_files"}.
    """
    global PYMUPDF_AVAILABLE

    os.makedirs(carpeta, exist_ok=True)
    archivos_pdf = [f for f in os.listdir(carpeta) if f.endswith('.pdf')]
    if solo_archivos is not None:
        archivos_pdf = [f for f in archivos_pdf if f in solo_archivos]
    
    if not archivos_pdf:
        if not silent:
            ui.warning("No se encontraron archivos PDF en la carpeta")
        return 0

    if not silent:
        ui.pipeline_start("Indexando documentos...")
    
    total_chunks = 0
    
    def _indexar_chunk(id_doc: str, chunk_text: str, chunk_doc_text: str, 
                       metadata: Dict, collection_ref: chromadb.Collection) -> bool:
        """Embedding + add a ChromaDB. Reintento con truncado si falla por longitud."""
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
                logging.warning(f"Chunk largo en {id_doc}, truncando a 1000 chars")
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
                    logging.error(f"Error persistente embeddeando {id_doc}: {e2}")
            else:
                logging.error(f"Error embeddeando {id_doc}: {e}")
            return False
    
    for idx, archivo in enumerate(archivos_pdf):
        if progress_callback:
            try:
                progress_callback({"file": archivo, "file_index": idx + 1, "total_files": len(archivos_pdf)})
            except Exception:
                pass
        if not silent:
            ui.pipeline_update(f"Procesando: {archivo}")
        usar_pypdf_fallback = False

        try:
            ruta_pdf = os.path.join(carpeta, archivo)

            if PYMUPDF_AVAILABLE:
                try:
                    page_chunks = pymupdf4llm.to_markdown(ruta_pdf, page_chunks=True)

                    _textos_paginas = [p['text'][:500] for p in page_chunks[:10]]
                    texto_base_doc = "\n\n".join(_textos_paginas)[:4000]

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
                                contexto_sit = generar_contexto_situacional(chunk_text, texto_base_doc)
                                chunk_text_con_contexto = (contexto_sit + chunk_text).strip()
                            else:
                                chunk_text_con_contexto = chunk_text

                            if _indexar_chunk(id_doc, chunk_text_con_contexto, chunk_text_con_contexto, metadata, collection):
                                total_chunks += 1

                except Exception as e:
                    logging.error(f"Error con pymupdf4llm en {archivo}: {e}, usando pypdf fallback")
                    usar_pypdf_fallback = True

            if not PYMUPDF_AVAILABLE or usar_pypdf_fallback:
                reader = PdfReader(ruta_pdf)

                _textos_paginas = [p.extract_text()[:500] for p in reader.pages[:10]]
                texto_base_doc = "\n\n".join(_textos_paginas)[:4000]

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
                            contexto_sit = generar_contexto_situacional(chunk_text, texto_base_doc)
                            chunk_text_con_contexto = (contexto_sit + chunk_text).strip()
                        else:
                            chunk_text_con_contexto = chunk_text

                        if _indexar_chunk(id_doc, chunk_text_con_contexto, chunk_text_con_contexto, metadata, collection):
                            total_chunks += 1

        except Exception as e:
            logging.error(f"Error procesando {archivo}: {e}")
            if not silent:
                ui.error(f"error en {archivo}: {e}")
    
    if not silent:
        ui.pipeline_stop()
    return total_chunks


# --- 11.3 Gestión de colección ---

def obtener_documentos_indexados(collection: chromadb.Collection) -> List[str]:
    """Lista de nombres únicos (source) en la colección."""
    try:
        all_metadata = collection.get(include=['metadatas'])
        documentos = set()
        for meta in all_metadata['metadatas']:
            if 'source' in meta:
                documentos.add(meta['source'])
        return sorted(list(documentos))
    except Exception:
        return []


# =============================================================================
# SECCIÓN 12: PUNTO DE ENTRADA
# =============================================================================
# main() → MonkeyGrabCLI.run()
# =============================================================================

def main():
    """Arranca MonkeyGrabCLI."""
    import rag.chat_pdfs as rag_engine
    from rag.cli import MonkeyGrabCLI
    cli = MonkeyGrabCLI(rag_engine)
    cli.run()


if __name__ == "__main__":
    main()
