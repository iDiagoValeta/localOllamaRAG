"""
MonkeyGrab - Sistema RAG para consulta de PDFs
======================================================

Aplicación interactiva con dos modos de operación:

  - Modo CHAT: Conversación libre con el modelo base (configurable).
    Mantiene historial multi-turno persistente y conoce la identidad del
    proyecto MonkeyGrab. Ideal para preguntas generales y charla.

  - Modo RAG: Retrieval-Augmented Generation con el modelo fine-tuneado.
    Consulta documentos PDF mediante búsqueda híbrida y genera respuestas 
    verificables con citas de fuentes.

Pipeline de recuperación (modo RAG):
  1. Indexación enriquecida: chunking Markdown, embeddings y Contextual Retrieval.
  2. Descomposición de consulta con modelo base (configurable).
  3. Búsqueda híbrida: semántica multi-query + keywords + exhaustiva.
  4. Fusión RRF (Reciprocal Rank Fusion) + reranking con Cross-Encoder.
  5. Síntesis de contexto interactiva mediante RECOMP.
  6. Generación de respuesta con streaming, usando el formato de prompt
     alineado con el fine-tuning del modelo Teacher (ver train.py).

Características principales:
  - Modo dual: conversación libre (chat) + consulta documental (RAG).
  - Persistencia de historial entre sesiones (JSON).
  - Separación de modelos: base para chat/queries, contextual, recomp y teacher.
  - Generación de Contexto Situacional (Anthropic-style) en la indexación.
  - Síntesis de contexto densa (RECOMP) antes de la generación.
  - Búsqueda híbrida (semántica + palabras clave) con reranking.
  - Chunking con solapamiento para mejor contexto documental.
  - Citas precisas con documento y página exacta.
  - Base de datos vectorial persistente con ChromaDB.

Uso:
    python chat_pdfs.py
    DOCS_FOLDER=/ruta/pdfs python chat_pdfs.py
"""

# =============================================================================
# SECCIÓN 1: IMPORTACIONES Y DEPENDENCIAS
# =============================================================================
# Propósito:
# - Declarar de forma explícita todas las dependencias del sistema RAG para
#   asegurar trazabilidad técnica y reproducibilidad en el entorno del TFG.
# Tecnologías implicadas:
# - Librerías estándar de Python: gestión de rutas, regex, tipado y logging.
# - Ollama: inferencia local para generación de respuesta y embeddings.
# - ChromaDB: base de datos vectorial persistente para recuperación semántica.
# - pypdf: extracción de texto base como fallback robusto para PDFs.
# Resultado esperado:
# - Entorno de ejecución con imports organizados por responsabilidad funcional.
# =============================================================================

import os
import re
import sys
import json
import logging
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

import ollama
import chromadb

from pypdf import PdfReader

# =============================================================================
# SECCIÓN 1B: CLI — Interfaz de usuario
# CLI — Interfaz de usuario (importada desde rag.cli)
# =============================================================================
# Asegurar que el directorio raíz del proyecto esté en sys.path
# para que 'rag' sea importable cuando se ejecuta directamente.
# =============================================================================

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rag.cli.display import ui


# =============================================================================
# SECCIÓN 2: CARGA CONDICIONAL DE COMPONENTES OPCIONALES
# =============================================================================
# Propósito:
# - Activar capacidades avanzadas de forma progresiva sin romper la ejecución
#   cuando ciertas dependencias no están instaladas en el sistema.
# Tecnologías implicadas:
# - pymupdf4llm: extracción en formato Markdown con mejor preservación de
#   estructura documental (encabezados, tablas y segmentación por página).
# - sentence-transformers (CrossEncoder): reranking neural para mejorar
#   precisión del top final frente a ranking solo por embedding.
# Resultado esperado:
# - Detección de disponibilidad en runtime y degradación controlada (fallback)
#   con mensajes informativos para diagnóstico del usuario.
# =============================================================================

try:
    import io
    import contextlib
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
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
# SECCIÓN 3: CONFIGURACIÓN DE RUNTIME Y PARÁMETROS GLOBALES
# =============================================================================
# Propósito:
# - Centralizar la configuración experimental y operativa del asistente para
#   facilitar ajustes, comparativas y repetibilidad de resultados.
# Tecnologías implicadas:
# - Variables de entorno para selección dinámica de modelos Ollama.
# - Configuración UTF-8 para salida robusta en terminal Windows/Linux/macOS.
# - Parámetros del pipeline: chunking, top-k, umbrales, estrategia híbrida,
#   descomposición de consulta y activación de métricas.
# Resultado esperado:
# - Un único bloque de hiperparámetros que controla comportamiento, coste y
#   calidad de recuperación/generación del sistema.
# =============================================================================

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

MODELO_CHAT = os.getenv("OLLAMA_CHAT_MODEL", "Qwen-2.5-FineTuned:latest")
MODELO_AUXILIAR = os.getenv("OLLAMA_AUX_MODEL", "gemma3:4b")
MODELO_EMBEDDING = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")
MODELO_CONTEXTUAL = os.getenv("OLLAMA_CONTEXTUAL_MODEL", "gemma3:4b")
MODELO_RECOMP = os.getenv("OLLAMA_RECOMP_MODEL", "gemma3:4b")

USAR_CONTEXTUAL_RETRIEVAL = True
USAR_RECOMP_SYNTHESIS = True

_MODELO_FAMILIAS = {
    "qwen":    "Qwen 2.5",
    "llama":   "Llama 3.1",
    "gemma":   "Gemma 3"
}

def _inferir_descripcion_modelo(nombre_modelo: str) -> str:
    """Devuelve un nombre legible a partir del slug de Ollama."""
    nombre_lower = nombre_modelo.lower()
    slug = nombre_lower.split(":")[0]
    for clave, desc in _MODELO_FAMILIAS.items():
        if clave in slug:
            match = re.search(r'(\d+\.?\d*b)', nombre_lower)
            size = f" {match.group(1).upper()}" if match else ""
            return f"{desc}{size}"
    return nombre_modelo.split(":")[0]

MODELO_DESC = os.getenv(
    "MODELO_DESC",
    _inferir_descripcion_modelo(MODELO_AUXILIAR)
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CARPETA_DOCS = os.getenv("DOCS_FOLDER", os.path.join(BASE_DIR, "pdfs"))

_carpeta_nombre = os.path.basename(os.path.abspath(CARPETA_DOCS))
_embed_slug = MODELO_EMBEDDING.split(":")[0].replace("/", "_")

_DB_VERSION = "v4"

PATH_DB = os.path.join(BASE_DIR, "mi_vector_db", f"{_carpeta_nombre}_{_embed_slug}_{_DB_VERSION}")
COLLECTION_NAME = f"docs_{_carpeta_nombre}"

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
EXPANDIR_CONTEXTO = True
N_TOP_PARA_EXPANSION = 3
USAR_BUSQUEDA_HIBRIDA = True
USAR_RERANKER = RERANKER_AVAILABLE

RERANKER_MODEL_QUALITY = os.getenv("RERANKER_QUALITY", "quality")

UMBRAL_RELEVANCIA = 0.10
UMBRAL_SCORE_RERANKER = 0.15
MIN_LONGITUD_PREGUNTA_RAG = 10

MAX_CONTEXTO_CHARS = 8000

USAR_LLM_QUERY_DECOMPOSITION = True

HISTORIAL_PATH = os.path.join(BASE_DIR, "historial_chat.json")
MAX_HISTORIAL_MENSAJES = 40

CARPETA_DEBUG_RAG = os.path.join(BASE_DIR, "debug_rag")

LOGGING_METRICAS = True
LOG_LEVEL = logging.ERROR

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(levelname)s: %(message)s'
)

for _logger_name in (
    "httpx", "chromadb", "chromadb.telemetry", "urllib3", "requests",
    "sentence_transformers", "transformers", "huggingface_hub",
    "tqdm", "filelock",
):
    logging.getLogger(_logger_name).setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*huggingface.*")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


# =============================================================================
# System prompt para modo RAG (modelo teacher fine-tuneado)
# CRÍTICO: Debe ser IDÉNTICO al system_prompt de train.py para evitar
# distribution shift. Cualquier cambio aquí degrada la calidad del modelo.
# =============================================================================
SYSTEM_PROMPT_RAG = """Eres un asistente que responde preguntas basándote EXCLUSIVAMENTE en el contexto proporcionado.

REGLAS ESTRICTAS:
1. Responde SOLO con información que esté contenida en el contexto dentro de <contexto>...</contexto>.
2. No inventes, no uses conocimiento externo. Si la respuesta no está en el contexto, indícalo claramente.
3. Cita o parafrasea el material del contexto cuando sea útil.
4. Responde en el mismo idioma que use el usuario (español, catalán o inglés).
5. Sé claro, conciso y estructurado."""

# =============================================================================
# System prompt para modo CHAT (modelo auxiliar configurable)
# Proporciona identidad del proyecto y comportamiento conversacional.
# =============================================================================
SYSTEM_PROMPT_CHAT = f"""Eres MonkeyGrab, un asistente inteligente desarrollado como parte de un Trabajo de Fin de Grado (TFG) sobre Inteligencia Artificial aplicada a la educación.

SOBRE TI:
- Tu nombre es MonkeyGrab y estás basado en el modelo {MODELO_DESC}, fine-tuneado con LoRA para responder preguntas fundamentadas en documentos académicos.
- Formas parte de un sistema RAG (Retrieval-Augmented Generation) que permite consultar documentos PDF de forma inteligente.
- Has sido entrenado con el dataset RAG_Multilingual (42.303 ejemplos en español, catalán e inglés).
- Tu despliegue es completamente local mediante Ollama y ChromaDB como base de datos vectorial.

SOBRE EL PROYECTO:
- Este TFG demuestra cómo los modelos de lenguaje pueden asistir en el estudio de material académico universitario.
- El sistema indexa PDFs, realiza búsqueda híbrida (semántica + keywords) con reranking neural, y genera respuestas verificables con citas de fuentes.
- El pipeline técnico incluye: chunking Markdown, embeddings con modelo vectorial (embeddinggemma/nomic), fusión RRF (Reciprocal Rank Fusion) y Cross-Encoder multilingüe.

TU COMPORTAMIENTO EN MODO CHAT:
- Mantén una conversación natural, amable y útil con el usuario.
- Puedes responder preguntas generales, explicar conceptos, ayudar con dudas y ser conversacional.
- Si el usuario quiere consultar los documentos académicos indexados, indícale que escriba '/rag' para activar el modo de recuperación de documentos.
- Responde en el mismo idioma que use el usuario (español, catalán o inglés).
- Sé conciso pero completo en tus respuestas."""


# =============================================================================
# SECCIÓN 5b: PERSISTENCIA DEL HISTORIAL DE CONVERSACIÓN
# =============================================================================
# Propósito:
# - Mantener el historial de conversación entre sesiones para continuidad UX.
# Tecnologías implicadas:
# - Almacenamiento en JSON local con separación por modo (chat / rag).
# Resultado esperado:
# - El usuario retoma la conversación donde la dejó al reiniciar el programa.
# =============================================================================


def cargar_historial() -> List[Dict[str, str]]:
    """
    Carga el historial de conversación del modo CHAT desde disco.
    
    Solo el modo CHAT tiene persistencia. El modo RAG no guarda historial
    porque cada consulta es independiente (el teacher fue entrenado para
    respuestas puntuales con contexto, no para conversación multi-turno).
    
    Returns:
        Lista de mensajes [{"role": "user"|"assistant", "content": "..."}]
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
        logging.warning(f"Error cargando historial: {e}")
    
    return []


def guardar_historial(historial: List[Dict[str, str]]) -> None:
    """
    Guarda el historial de conversación del modo CHAT a disco.
    
    Mantiene solo los últimos MAX_HISTORIAL_MENSAJES mensajes para
    evitar crecimiento descontrolado del archivo.
    
    Args:
        historial: Lista de mensajes del chat
    """
    try:
        historial_recortado = historial[-MAX_HISTORIAL_MENSAJES:]
        with open(HISTORIAL_PATH, 'w', encoding='utf-8') as f:
            json.dump(historial_recortado, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Error guardando historial: {e}")


def limpiar_historial(historial: List[Dict[str, str]]) -> None:
    """
    Limpia el historial de conversación del chat y lo persiste vacío.
    
    Args:
        historial: Lista de mensajes del chat (se modifica in-place)
    """
    historial.clear()
    guardar_historial(historial)


# =============================================================================
# SECCIÓN 6: PREPROCESADO DOCUMENTAL Y CHUNKING
# =============================================================================
# Propósito:
# - Transformar texto bruto en unidades semánticas aptas para embeddings,
#   conservando contexto estructural de documentos académicos extensos.
# Tecnologías implicadas:
# - Regex para detección de headers Markdown y jerarquía de secciones.
# - Chunking recursivo con separadores jerárquicos y overlap controlado.
# - Expansión por vecindad (chunks adyacentes) para aumentar coherencia local
#   del contexto enviado al modelo generativo.
# Resultado esperado:
# - Fragmentos balanceados en tamaño/relevancia, con trazabilidad de sección y
#   menor pérdida de continuidad temática entre bloques.
# =============================================================================


def extraer_header_markdown(texto: str) -> str:
    """
    Extrae el último encabezado Markdown visible antes del contenido.
    Útil para dar contexto a cada chunk (ej: "## 3.2 Optimización local").
    
    Args:
        texto: Texto en formato Markdown
    
    Returns:
        Último header encontrado, o cadena vacía
    """
    headers = re.findall(r'^(#{1,4}\s+.+)$', texto, re.MULTILINE)
    return headers[-1].strip() if headers else ""


def dividir_en_chunks(
    texto: str, 
    chunk_size: int = CHUNK_SIZE, 
    overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, str]]:
    """
    Divide el texto en fragmentos (chunks) con solapamiento correcto.
    
    Estrategia mejorada para documentos académicos en Markdown:
    1. Respeta encabezados Markdown (##, ###) como límites de sección
    2. Dentro de cada sección, divide por párrafos y frases
    3. Aplica solapamiento CORRECTO entre chunks consecutivos
    4. Prepende el encabezado de sección a cada chunk para contexto
    
    Args:
        texto: Texto completo a dividir (puede ser Markdown)
        chunk_size: Tamaño máximo de cada fragmento en caracteres
        overlap: Cantidad de caracteres de solapamiento entre fragmentos
    
    Returns:
        Lista de dicts con 'text' y 'header' por cada chunk
    """
    if not texto or not texto.strip():
        return []
    
    header_pattern = re.compile(r'^(#{1,4}\s+.+)$', re.MULTILINE)
    
    secciones = []
    last_end = 0
    current_header = ""
    
    for match in header_pattern.finditer(texto):
        contenido_previo = texto[last_end:match.start()].strip()
        if contenido_previo:
            secciones.append({"header": current_header, "content": contenido_previo})
        
        current_header = match.group(1).strip()
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
    """
    Genera IDs de chunks adyacentes para proporcionar más contexto.
    
    Incluye expansión CROSS-PAGE: si el chunk es el último de su página,
    también incluye chunk 0 de la página siguiente. Si es el primero,
    incluye el último chunk de la página anterior (si se conoce).
    
    Args:
        chunk_id: ID del chunk actual
        metadata: Metadata del chunk con información de página y posición
        n_vecinos: Número de vecinos a cada lado
    
    Returns:
        Lista de IDs de chunks adyacentes
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


# =============================================================================
# SECCIÓN 7: EXTRACCIÓN DE KEYWORDS Y BÚSQUEDA LEXICAL
# =============================================================================
# Propósito:
# - Complementar la recuperación semántica con señales léxicas explícitas para
#   mejorar recall en términos técnicos, siglas y expresiones específicas.
# Tecnologías implicadas:
# - Stopwords multilingües (ES/EN) para filtrado de ruido lingüístico.
# - Heurísticas de extracción: siglas, bigramas, términos entre paréntesis,
#   tokens técnicos (mayúsculas internas, guiones y dígitos).
# - Búsqueda textual en ChromaDB mediante where_document y escaneo exhaustivo
#   por lotes cuando la señal semántica es insuficiente.
# Resultado esperado:
# - Recuperación más robusta ante consultas especializadas o altamente técnicas.
# =============================================================================

STOPWORDS = {
    'el', 'la', 'de', 'en', 'y', 'a', 'los', 'las', 'un', 'una', 'por', 'para',
    'con', 'del', 'que', 'es', 'son', 'se', 'al', 'como', 'más', 'su', 'me',
    'está', 'hay', 'tiene', 'puede', 'ser', 'sobre', 'entre', 'también',
    'podrías', 'decirme', 'cuáles', 'cómo', 'qué', 'indica', 'indicar', 'puedes',
    'tres', 'dos', 'estas', 'estos', 'principales', 'llaman', 'partes',
    'pero', 'sino', 'desde', 'hasta', 'cuando', 'donde', 'este', 'esta',
    'ese', 'esa', 'aquí', 'ahí', 'allí', 'así', 'cada', 'todo', 'toda',
    'todos', 'todas', 'otro', 'otra', 'otros', 'otras', 'mismo', 'misma',
    'cual', 'quien', 'cuyo', 'cuya', 'muy', 'poco', 'mucho', 'algo', 'nada',
    'uno', 'dos', 'tres', 'siempre', 'nunca', 'después', 'antes', 'durante',
    'mediante', 'según', 'hacia', 'tanto', 'tan', 'sin', 'contra', 'ya',
    'fue', 'sido', 'siendo', 'han', 'haber', 'hacer', 'tener', 'ir',
    'explica', 'explicar', 'describe', 'describir', 'detalla', 'detallar',
    'respuesta', 'pregunta', 'siguientes', 'siguiente', 'puntos', 'punto',
    'ejemplo', 'manera', 'forma', 'tipo', 'tipos', 'parte', 'primer', 'primera',
    'segundo', 'segunda', 'tercer', 'tercera',
    'the', 'in', 'and', 'of', 'to', 'a', 'is', 'for', 'on', 'with', 'as', 'are',
    'this', 'that', 'it', 'be', 'or', 'an', 'by', 'from', 'at', 'which',
    'how', 'what', 'when', 'where', 'who', 'why', 'does', 'do', 'did',
}

TERMINOS_EXPANSION = {}


def extraer_keywords(texto: str) -> List[str]:
    """
    Extrae keywords importantes de la pregunta para búsqueda híbrida.
    
    Mejoras:
    - Detecta términos multi-palabra entre paréntesis o mayúsculas consecutivas
    - Identifica siglas (GDA, RISC, CISC, etc.)
    - Filtra stopwords y palabras genéricas con mayor precisión
    - Extrae N-gramas técnicos (ej: "bloque básico", "código intermedio")
    
    Args:
        texto: Texto del cual extraer keywords
    
    Returns:
        Lista de keywords extraídas (términos técnicos primero)
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

    resultado = sorted(keywords_filtradas, 
                      key=lambda x: (
                          0 if (x.isupper() or any(c.isupper() for c in x[1:]) or '-' in x) else 1,
                          len(x)
                      ))
    
    return resultado


def busqueda_por_keywords(
    pregunta: str, 
    collection: chromadb.Collection,
    n_results: int = N_RESULTADOS_KEYWORD
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Realiza búsqueda por palabras clave usando collection.get() con where_document.
    
    NOTA: Usa get() en lugar de query() para evitar el problema de dimensiones
    de embeddings. get() filtra por contenido textual sin necesidad de embeddings.
    
    Args:
        pregunta: Pregunta del usuario
        collection: Colección de ChromaDB
        n_results: Número máximo de resultados por keyword
    
    Returns:
        Tupla de (lista de resultados, métricas de búsqueda)
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
    """
    Búsqueda exhaustiva en todos los documentos por términos críticos.
    
    Útil cuando la búsqueda semántica falla para términos técnicos específicos.
    
    Args:
        terminos_criticos: Lista de términos a buscar
        collection: Colección de ChromaDB
        max_results: Número máximo de resultados
    
    Returns:
        Tupla de (lista de documentos que contienen los términos, métricas)
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
            logging.warning(f"Error en búsqueda exhaustiva (offset {offset}): {e}")
    
    resultados.sort(key=lambda x: x['num_matches'], reverse=True)
    
    if LOGGING_METRICAS:
        logging.info(f"Búsqueda exhaustiva: {metricas['documentos_con_matches']} docs con matches de {metricas['documentos_escaneados']} escaneados")
    
    return resultados[:max_results], metricas


# =============================================================================
# SECCIÓN 8: RERANKING SEMÁNTICO CON CROSS-ENCODER
# =============================================================================
# Propósito:
# - Refinar el orden de candidatos recuperados para priorizar evidencia más
#   relevante antes de la fase de generación de respuesta.
# Tecnologías implicadas:
# - CrossEncoder de sentence-transformers con carga lazy (singleton).
# - Detección de hardware (CUDA/CPU) para optimizar latencia de inferencia.
# - Estrategia configurable de modelos (quality vs fast) según coste-tiempo.
# Resultado esperado:
# - Mayor precisión en top-k final y reducción de contexto irrelevante.
# =============================================================================
_reranker_model = None

def _detectar_dispositivo_reranker() -> str:
    """
    Detecta el mejor dispositivo disponible para el Cross-Encoder.

    Prioridad: CUDA > CPU. Se importa torch de forma segura ya que
    sentence-transformers lo incluye como dependencia.

    Returns:
        String con el dispositivo: 'cuda' o 'cpu'.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def obtener_modelo_reranker():
    """
    Obtiene el modelo de reranking (singleton pattern).

    Carga el Cross-Encoder en GPU con FP16 cuando CUDA está disponible,
    lo que reduce el tiempo de reranking de ~2-3 minutos a segundos.
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
    """
    Reordena los resultados usando un Cross-Encoder.
    
    El Cross-Encoder evalúa cada par (pregunta, documento) de forma precisa,
    asignando un score de relevancia. Esto mejora significativamente la calidad
    de los resultados finales comparado con solo usar embeddings.
    
    Args:
        pregunta: Pregunta del usuario
        documentos_recuperados: Lista de documentos candidatos
        top_k: Número de documentos a retornar tras reranking
    
    Returns:
        Tupla de (lista reordenada de top_k documentos, métricas)
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

        textos_documentos = [doc['doc'] for doc in documentos_recuperados]

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
    """
    Usa el modelo auxiliar (base) para generar consultas de búsqueda diversas.
    
    NOTA: Se usa MODELO_AUXILIAR ({MODELO_DESC}) en lugar del modelo fine-tuneado
    (teacher) porque el teacher fue entrenado para responder con contexto, no para
    generar queries de búsqueda. El modelo base es más adecuado para esta tarea
    de generación libre de consultas diversas.
    
    Args:
        pregunta: Pregunta original del usuario
    
    Returns:
        Lista de 3 queries de búsqueda generadas por el LLM
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
            model=MODELO_AUXILIAR,
            prompt=prompt,
            options={
                "temperature": 0.3,
                "num_predict": 200,
                "stop": ["\n\n\n"],
            }
        )

        queries = [
            q.strip().lstrip("0123456789.-) ")
            for q in response["response"].strip().split("\n")
            if q.strip() and len(q.strip()) > 10
        ]

        return queries[:3]

    except Exception as e:
        logging.warning(f"Error generando queries con LLM ({MODELO_AUXILIAR}): {e}")
        return []


# =============================================================================
# SECCIÓN 9: PIPELINE DE RECUPERACIÓN HÍBRIDA
# =============================================================================
# Propósito:
# - Coordinar de extremo a extremo la recuperación de evidencia documental
#   combinando múltiples estrategias complementarias.
# Tecnologías implicadas:
# - Ollama embeddings para búsqueda semántica multi-query.
# - Descomposición de pregunta con LLM para aumentar cobertura (recall).
# - Fusión de scores semánticos y léxicos + búsqueda exhaustiva por términos
#   críticos + reranking neural sobre candidatos finales.
# Resultado esperado:
# - Conjunto de fragmentos ordenados por relevancia con métricas de cada fase,
#   listo para construir contexto de generación.
# =============================================================================

def realizar_busqueda_hibrida(
    pregunta: str,
    collection: chromadb.Collection
) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    """
    Ejecuta búsqueda híbrida combinando semántica, keywords, exhaustiva y reranking.
    
    Pipeline mejorado:
    1. (Opcional) Descomposición de pregunta con LLM → sub-queries
    2. Búsqueda semántica multi-query con prefijos de embedding (recupera candidatos)
    3. Búsqueda por keywords con collection.get() (aumenta recall)
    4. Búsqueda exhaustiva por términos críticos (máximo recall)
    5. Fusión híbrida con RRF (Reciprocal Rank Fusion)
    6. Reranking con Cross-Encoder multilingual (precisión final)
    
    Args:
        pregunta: Pregunta del usuario
        collection: Colección de ChromaDB
    
    Returns:
        Tupla de (fragmentos_rankeados, mejor_score, métricas_totales)
    """
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

    palabras_clave_pregunta = [p for p in pregunta.split() if len(p) > 4]
    query_corta = ' '.join(palabras_clave_pregunta[:10]).strip()
    if query_corta and query_corta != pregunta:
        queries.append(query_corta)

    keywords_expandidas = extraer_keywords(pregunta)
    if keywords_expandidas:
        query_kw = ' '.join(keywords_expandidas[:10]).strip()
        if query_kw and query_kw not in queries:
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

    terminos_criticos = [k for k in keywords_expandidas[:12] if len(k) > 3]
    
    metricas_exhaustiva = {}
    if terminos_criticos:
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
# SECCIÓN 10: CONSTRUCCIÓN DE CONTEXTO Y GENERACIÓN DE RESPUESTA
# =============================================================================
# Propósito:
# - Convertir fragmentos recuperados en un prompt alineado con el formato del
#   fine-tuning para maximizar fidelidad al contexto documental.
# Tecnologías implicadas:
# - Plantilla con etiqueta <contexto>...</contexto> consistente con train.py.
# - Llamada de chat en streaming con Ollama para salida incremental en consola.
# - Mensaje de sistema restrictivo para minimizar alucinaciones y forzar
#   respuestas basadas únicamente en evidencia recuperada.
# Resultado esperado:
# - Respuesta trazable, en tiempo real y con fuentes explícitas al usuario.
# =============================================================================

def _es_continuacion_parrafo(linea_previa: str, linea_actual: str) -> bool:
    """
    Detecta si linea_actual es continuación del párrafo de linea_previa.
    
    Los PDFs extraídos a Markdown insertan líneas en blanco entre cada
    línea de texto (doble espaciado), rompiendo párrafos continuos en
    fragmentos inconexos. Esta heurística detecta esas roturas para
    poder re-unir el texto original.
    
    Criterios de continuación:
    - La línea previa NO termina en finalizador de frase (. ? !)
    - La línea actual NO inicia un bloque nuevo (lista, header, bold)
    - La línea actual empieza en minúscula (continuación clara) O
      la previa termina en , ; ) ] (continuación fuerte) O
      la previa acaba con una palabra cortada (típico de line wrap del PDF)
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
    """
    Re-une líneas de párrafos que fueron separadas por la extracción del PDF.
    
    El problema: pymupdf4llm y otros extractores convierten el doble
    espaciado del PDF en líneas en blanco entre cada línea de texto,
    haciendo que un párrafo de 3 líneas ocupe 5-7 líneas. Esto
    desperdicia ~30-50% de los caracteres del contexto.
    
    Solución: al encontrar una línea en blanco, mirar hacia adelante
    (look-ahead) para ver si la siguiente línea con contenido es
    continuación del párrafo actual. Si lo es, saltar las líneas vacías
    y unir. Si no, cerrar el párrafo y empezar uno nuevo.
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
    """
    Optimiza el texto extraído de PDFs eliminando ruido y desperdicio de caracteres.
    
    Los PDFs convertidos a Markdown suelen tener:
    - Líneas en blanco entre cada línea de texto (doble espaciado del PDF)
    - Múltiples saltos de línea consecutivos sin contenido
    - Artefactos de Markdown vacíos como '## □' o '□'
    - Footers de página (autor, universidad, número de página)
    - Marcadores de solución de exámenes tipo '**Solución:** La X'
    - Espacios múltiples dentro de las líneas
    
    El ahorro típico es del 30-50%, permitiendo que el mismo límite de
    MAX_CONTEXTO_CHARS contenga significativamente más información
    relevante para el modelo.
    
    Args:
        texto: Texto crudo extraído del PDF
    
    Returns:
        Texto compactado conservando la estructura semántica
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
    
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    
    return texto.strip()

def construir_contexto_para_modelo(fragmentos: List[Dict[str, Any]]) -> str:
    """
    Construye el contexto formateado para enviar al modelo LLM.
    
    CRÍTICO: El formato debe ser texto puro sin metadatos inyectados, alineado
    con el dataset RAG_Multilingual usado en el fine-tuning (train.py).
    
    Durante el entrenamiento, el contexto era simplemente el texto del campo
    "context" del dataset, sin cabeceras como "[Fragmento X | archivo.pdf...]".
    Inyectar metadatos causa distribution shift y degrada la calidad.
    
    Las fuentes se muestran al usuario de forma separada (formatear_fuentes_respuesta),
    no dentro del contexto que recibe el modelo.
    
    Cada fragmento se optimiza con optimizar_texto_contexto() para eliminar
    ruido del PDF (líneas vacías, artefactos, footers) y aprovechar mejor
    el límite de MAX_CONTEXTO_CHARS.
    
    Args:
        fragmentos: Lista de fragmentos seleccionados
    
    Returns:
        Contexto formateado como string (texto puro optimizado)
    """
    fragmentos_ordenados = sorted(
        fragmentos,
        key=lambda f: (f['metadata']['source'], f['metadata']['page'], f['metadata'].get('chunk', 0))
    )
    
    textos_originales = [frag['doc'] for frag in fragmentos_ordenados]
    chars_original = sum(len(t) for t in textos_originales)
    
    contextos_texto = [
        optimizar_texto_contexto(frag['doc'])
        for frag in fragmentos_ordenados
    ]
    
    contextos_texto = [t for t in contextos_texto if t]
    
    resultado = "\n\n...\n\n".join(contextos_texto)
    
    chars_optimizado = len(resultado)
    if chars_original > 0 and LOGGING_METRICAS:
        ahorro = chars_original - chars_optimizado
        pct = (ahorro / chars_original) * 100 if ahorro > 0 else 0
        logging.info(
            f"Contexto optimizado: {chars_original} → {chars_optimizado} chars "
            f"({ahorro} ahorrados, {pct:.1f}%)"
        )
    
    return resultado

def guardar_debug_rag(
    pregunta: str,
    system_prompt: str = "",
    mensaje_usuario: str = "",
    respuesta: str = "",
    fragmentos: Optional[List[Dict[str, Any]]] = None,
    motivo_interrupcion: Optional[str] = None,
    metricas: Optional[Dict[str, Any]] = None
) -> None:
    """
    Guarda un volcado completo de la interacción RAG para análisis y depuración.
    
    Crea un archivo de texto legible por cada consulta con timestamp, que incluye:
    - La pregunta original del usuario
    - El system prompt enviado al modelo
    - El mensaje de usuario completo (pregunta + instrucción idioma + contexto)
    - La respuesta generada por el modelo
    - Metadatos de los fragmentos recuperados
    
    Los archivos se guardan en la carpeta debug_rag/ con nombre basado en timestamp.
    
    En casos de interrupción temprana (sin resultados, fuera de alcance, etc.),
    se pueden pasar motivo_interrupcion y metricas para registrar el fallo.
    
    Args:
        pregunta: Pregunta original del usuario
        system_prompt: System prompt enviado al modelo (vacío si no se llegó a generar)
        mensaje_usuario: Mensaje de usuario completo enviado al modelo (vacío si no)
        respuesta: Respuesta generada por el modelo (vacía si no se generó)
        fragmentos: Fragmentos de contexto recuperados (puede ser lista vacía)
        motivo_interrupcion: Motivo por el que no se generó respuesta (opcional)
        metricas: Métricas de búsqueda para casos de fallo (opcional)
    """
    fragmentos = fragmentos or []
    
    try:
        os.makedirs(CARPETA_DEBUG_RAG, exist_ok=True)
        
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Primeras palabras de la pregunta como parte del nombre
        slug = re.sub(r'[^\w\s]', '', pregunta)[:40].strip().replace(' ', '_')
        nombre_archivo = f"{timestamp}_{slug}.txt"
        ruta = os.path.join(CARPETA_DEBUG_RAG, nombre_archivo)
        
        with open(ruta, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"  DEBUG RAG - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

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
            f.write(f"Modelo Chat (Teacher): {MODELO_CHAT}\n")
            f.write(f"Contextual Retrieval (Indexación): {'SÍ (qwen3-14b)' if USAR_CONTEXTUAL_RETRIEVAL else 'NO'}\n")
            f.write(f"RECOMP Synthesis (Generación): {'SÍ (qwen3-4b)' if USAR_RECOMP_SYNTHESIS else 'NO'}\n\n")
            
            f.write("─" * 80 + "\n")
            f.write("  PREGUNTA ORIGINAL\n")
            f.write("─" * 80 + "\n")
            f.write(f"{pregunta}\n\n")
            
            f.write("─" * 80 + "\n")
            f.write("  SYSTEM PROMPT\n")
            f.write("─" * 80 + "\n")
            f.write(f"{system_prompt or '(no enviado)'}\n\n")
            
            # Extraer el contexto real enviado
            context_match = re.search(r'<contexto>(.*?)</contexto>', mensaje_usuario, re.DOTALL)
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
            f.write(f"  FRAGMENTOS RECUPERADOS ({len(fragmentos)}) - Pueden incluir Contextual Retrieval prependeado\n")
            f.write("─" * 80 + "\n")
            for i, frag in enumerate(fragmentos, 1):
                meta = frag.get('metadata', {})
                score = frag.get('score_final', 'N/A')
                score_rr = frag.get('score_reranker', 'N/A')
                f.write(f"\n--- Fragmento {i} ---\n")
                pag = meta.get('page', 0)
                f.write(f"Fuente: {meta.get('source', '?')}, pág. {pag + 1 if isinstance(pag, int) else pag}\n")
                f.write(f"Score final: {score}  |  Score reranker: {score_rr}\n")
                f.write(f"Sección: {meta.get('section_header', '(sin header)')}\n")
                f.write(f"Texto original (con Contextual Retrieval si estaba activo al indexar):\n{frag.get('doc', '')}\n")
        
        logging.info(f"Debug RAG guardado: {ruta}")
        
    except Exception as e:
        logging.warning(f"Error guardando debug RAG: {e}")


def sintetizar_contexto_recomp(fragmentos: List[Dict[str, Any]], query_usuario: str = "") -> str:
    """
    Sintetiza fragmentos usando RECOMP (Qwen 2.5 3B), preservando citas 
    y precisión técnica (fórmulas O(n), etc).
    """
    if not USAR_RECOMP_SYNTHESIS or not fragmentos:
        return construir_contexto_para_modelo(fragmentos)
        
    textos_preparados = []
    for i, f in enumerate(fragmentos):
        content = f['doc'].replace("\n", " ").strip()
        textos_preparados.append(f"Fragment {i+1}:\n{content}")

    contexto_raw = "\n\n".join(textos_preparados)

    system_prompt = (
        "You are a precise technical editor. Your task is to consolidate text fragments "
        "into a comprehensive summary that PRESERVES ALL specific information.\n"
        "ABSOLUTE RULES:\n"
        "1. Extract and include EVERY named concept, technique, method, term, and "
        "definition from EVERY fragment. Do NOT skip any fragment.\n"
        "2. ONLY use information EXPLICITLY written in the fragments. "
        "NEVER add information from your own knowledge.\n"
        "3. Preserve all citations, references, numbers and formulas exactly.\n"
        "4. If fragments contain DIFFERENT information, include ALL of it. "
        "Fragments may come from different pages of the same document.\n"
        "5. Output in the same language as the input fragments."
    )

    focus_instruction = (
        f"Focus specifically on information answering: '{query_usuario}'. "
        "Include ALL relevant details from ALL fragments."
        if query_usuario else 
        "Summarize the key technical definitions and comparisons."
    )

    user_prompt = (
        f"{focus_instruction}\n\n"
        f"--- INPUT FRAGMENTS ---\n{contexto_raw}\n-----------------------\n\n"
        "Detailed Summary (with citations):"
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

def generar_respuesta(
    pregunta: str, 
    fragmentos: List[Dict[str, Any]]
) -> str:
    """
    Genera y muestra la respuesta del asistente RAG con streaming.
    
    IMPORTANTE: Usa el formato exacto del entrenamiento:
    - System prompt alineado con el Modelfile del teacher.
    - Pregunta del usuario seguida de: "\\n\\n<contexto>{contenido}</contexto>"
    - Esto coincide con el dataset RAG_Multilingual usado en el fine-tuning.
    
    Args:
        pregunta: Pregunta del usuario
        fragmentos: Fragmentos de contexto relevantes del RAG
    
    Returns:
        Texto completo de la respuesta generada (para persistencia)
    """
    if USAR_RECOMP_SYNTHESIS:
        ui.debug("sintetizando contexto con RECOMP...")
        contexto_str = sintetizar_contexto_recomp(fragmentos, query_usuario=pregunta)
    else:
        contexto_str = construir_contexto_para_modelo(fragmentos)

    mensaje_usuario = f"{pregunta}\n\n<contexto>{contexto_str}</contexto>"

    stream = ollama.chat(
        model=MODELO_CHAT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_RAG},
            {"role": "user", "content": mensaje_usuario}
        ],
        stream=True,
        options={"temperature": 0.15, "top_p": 0.85, "repeat_penalty": 1.15, "num_ctx": 8192}
    )

    respuesta_completa = ""
    print()
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "") or chunk.get("content", "")
        if content:
            ui.stream_token(content)
            respuesta_completa += content
    print()
    
    guardar_debug_rag(pregunta, SYSTEM_PROMPT_RAG, mensaje_usuario, respuesta_completa, fragmentos)
    
    return respuesta_completa


def generar_respuesta_silenciosa(
    pregunta: str,
    fragmentos: List[Dict[str, Any]]
) -> str:
    """
    Genera la respuesta RAG sin streaming ni salida visual.
    
    Variante silenciosa de generar_respuesta() para evaluación
    automatizada (RAGAS) y tests sin interacción de consola.
    
    Args:
        pregunta: Pregunta del usuario
        fragmentos: Fragmentos de contexto relevantes
    
    Returns:
        Texto completo de la respuesta generada
    """
    if USAR_RECOMP_SYNTHESIS:
        contexto_str = sintetizar_contexto_recomp(fragmentos, query_usuario=pregunta)
    else:
        contexto_str = construir_contexto_para_modelo(fragmentos)
    mensaje_usuario = f"{pregunta}\n\n<contexto>{contexto_str}</contexto>"
    stream = ollama.chat(
        model=MODELO_CHAT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_RAG},
            {"role": "user", "content": mensaje_usuario}
        ],
        stream=True,
        options={"temperature": 0.15, "top_p": 0.85, "repeat_penalty": 1.15, "num_ctx": 8192}
    )
    respuesta_completa = ""
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "") or chunk.get("content", "")
        if content:
            respuesta_completa += content
    return respuesta_completa


def evaluar_pregunta_rag(
    pregunta: str,
    collection: chromadb.Collection
) -> Tuple[str, List[str]]:
    """
    Ejecuta el pipeline RAG completo sin imprimir. Para evaluación con RAGAS.
    Returns:
        (respuesta, lista de contextos usados)
    """
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
# SECCIÓN 11: INDEXACIÓN DE PDFS Y OPERACIONES DE COLECCIÓN
# =============================================================================
# Propósito:
# - Construir y mantener el índice vectorial a partir de PDFs para habilitar
#   recuperación semántica persistente entre ejecuciones.
# Tecnologías implicadas:
# - Extracción dual: pymupdf4llm (preferente) y pypdf (fallback).
# - Embeddings locales con Ollama + prefijos dinámicos para doc/query alignment.
# - Persistencia en ChromaDB con metadatos enriquecidos (source, page, chunk,
#   formato, encabezado de sección) y control de errores/reintentos.
# Resultado esperado:
# - Colección vectorial consistente, auditable y utilizable por el chat RAG.
# =============================================================================

def generar_contexto_situacional(chunk_text: str, texto_base: str) -> str:
    """
    Usa un LLM para generar contexto situacional (Contextual Retrieval) para un chunk.
    Anthropic (2024): enriquece el chunk con 2-3 frases de resumen global + contexto situacional.
    """
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
            options={"temperature": 0.2, "num_predict": 200}
        )
        contexto = response['message']['content'].strip()
        if contexto:
            return f"{contexto}\\n\\n"
    except Exception as e:
        logging.warning(f"Error generando contexto situacional: {e}")
    return ""

def indexar_documentos(
    carpeta: str, 
    collection: chromadb.Collection
) -> int:
    """
    Indexa todos los PDFs de una carpeta en la colección de ChromaDB.
    
    Usa pymupdf4llm para mejor extracción (preserva tablas, estructura, formato).
    Si no está disponible, usa pypdf como fallback.
    
    Args:
        carpeta: Ruta a la carpeta con PDFs
        collection: Colección de ChromaDB
    
    Returns:
        Número total de fragmentos indexados
    """
    global PYMUPDF_AVAILABLE
    
    archivos_pdf = [f for f in os.listdir(carpeta) if f.endswith('.pdf')]
    
    if not archivos_pdf:
        ui.warning("No se encontraron archivos PDF en la carpeta")
        return 0

    ui.pipeline_start("Indexando documentos...")
    
    total_chunks = 0
    
    def _indexar_chunk(id_doc: str, chunk_text: str, chunk_doc_text: str, 
                       metadata: Dict, collection_ref: chromadb.Collection) -> bool:
        """
        Embeddea e indexa un chunk individual con prefijo de embedding y reintentos.
        
        Args:
            id_doc: ID único del chunk
            chunk_text: Texto a embeddear (puede ser truncado)
            chunk_doc_text: Texto completo a almacenar en la DB
            metadata: Metadatos del chunk
            collection_ref: Colección ChromaDB
        
        Returns:
            True si se indexó correctamente
        """
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
    
    for archivo in archivos_pdf:
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
            ui.error(f"error en {archivo}: {e}")
    
    ui.pipeline_stop()
    return total_chunks

def obtener_documentos_indexados(collection: chromadb.Collection) -> List[str]:
    """
    Obtiene la lista de documentos únicos indexados.
    
    Args:
        collection: Colección de ChromaDB
    
    Returns:
        Lista de nombres de documentos únicos
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




# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

def main():
    """Arranca la CLI profesional de MonkeyGrab."""
    import rag.chat_pdfs as rag_engine
    from rag.cli import MonkeyGrabCLI
    cli = MonkeyGrabCLI(rag_engine)
    cli.run()


if __name__ == "__main__":
    main()
