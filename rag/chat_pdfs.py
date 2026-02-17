"""
Teacher - Sistema RAG Dual para consulta de PDFs (TFG)
======================================================

Aplicación interactiva con dos modos de operación:

  - **Modo CHAT**: Conversación libre con el modelo base (llama3.1:8b).
    Mantiene historial multi-turno persistente y conoce la identidad del
    proyecto Teacher. Ideal para preguntas generales y charla.

  - **Modo RAG**: Retrieval-Augmented Generation con el modelo fine-tuneado
    (teacher-q4km). Consulta documentos PDF mediante búsqueda híbrida y
    genera respuestas verificables con citas de fuentes.

Pipeline de recuperación (modo RAG):
  1. Indexación de PDFs con chunking Markdown y embeddings (configurable).
  2. Descomposición de consulta con modelo base (llama3.1:8b).
  3. Búsqueda híbrida: semántica multi-query + keywords + exhaustiva.
  4. Fusión RRF (Reciprocal Rank Fusion) + reranking con Cross-Encoder.
  5. Generación de respuesta con streaming, usando el formato de prompt
     alineado con el fine-tuning del modelo Teacher (ver train.py).

Características principales:
  - Modo dual: conversación libre (chat) + consulta documental (RAG).
  - Persistencia de historial entre sesiones (JSON).
  - Separación de modelos: base para chat/queries, teacher para RAG.
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
    print("pymupdf4llm no disponible, usando pypdf (menor calidad)")

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    print("sentence-transformers no disponible, reranking desactivado")

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

MODELO_CHAT = os.getenv("OLLAMA_CHAT_MODEL", "teacher-q4km:latest")
MODELO_AUXILIAR = os.getenv("OLLAMA_AUX_MODEL", "qwen2.5:14b")
MODELO_EMBEDDING = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CARPETA_DOCS = os.getenv("DOCS_FOLDER", os.path.join(BASE_DIR, "pdfs"))

_carpeta_nombre = os.path.basename(os.path.abspath(CARPETA_DOCS))
_embed_slug = MODELO_EMBEDDING.split(":")[0].replace("/", "_")

_DB_VERSION = "v3"

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
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
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

MAX_CONTEXTO_CHARS = 6000

USAR_LLM_QUERY_DECOMPOSITION = True

HISTORIAL_PATH = os.path.join(BASE_DIR, "historial_chat.json")
MAX_HISTORIAL_MENSAJES = 40

CARPETA_DEBUG_RAG = os.path.join(BASE_DIR, "debug_rag")

LOGGING_METRICAS = True
LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(levelname)s: %(message)s'
)


# =============================================================================
# SECCIÓN 4: INTERFAZ Y MENSAJERÍA DEL ASISTENTE
# =============================================================================
# Propósito:
# - Definir una capa de presentación homogénea para mejorar UX y legibilidad
#   durante sesiones de consulta documental en terminal.
# Tecnologías implicadas:
# - ANSI escape codes para formato visual (negrita, reset, separación).
# - Diccionario de mensajes para respuestas de estado, ayuda y control de flujo.
# - Iconografía Unicode para identificar fases del pipeline de forma intuitiva.
# Resultado esperado:
# - Interacción consistente, clara y mantenible sin duplicación de textos UI.
# =============================================================================

class EstiloUI:
    """Configuración de estilos y mensajes para la interfaz de usuario."""
    
    LINEA_DOBLE = "═"
    LINEA_SIMPLE = "─"
    ANCHO = 70
    
    RESET = "\033[0m"
    NEGRITA = "\033[1m"
    
    ICONO_DOCUMENTO = "📄"
    ICONO_BUSQUEDA = "🔍"
    ICONO_EXITO = "✅"
    ICONO_ADVERTENCIA = "⚠️"
    ICONO_ERROR = "❌"
    ICONO_INFO = "💡"
    ICONO_CHAT = "💬"
    ICONO_ROBOT = "🤖"
    ICONO_LIBRO = "📚"
    ICONO_CARPETA = "📁"
    ICONO_ENGRANAJE = "⚙️"
    ICONO_ESTADISTICA = "📊"
    ICONO_CITA = "📌"
    ICONO_PAGINA = "📃"

MENSAJES = {
    "info_no_encontrada": (
        "Lo siento, no he encontrado información específica sobre tu pregunta en los "
        "documentos disponibles. Esto puede deberse a que:\n\n"
        "  • La información no está contenida en los documentos indexados\n"
        "  • La pregunta podría formularse de manera diferente\n"
        "  • El tema está fuera del alcance de los documentos\n\n"
        "💡 **Sugerencia**: Intenta reformular tu pregunta o pregunta sobre los "
        "temas principales de los documentos."
    ),
    
    "fuera_de_ambito": (
        "Esta pregunta parece estar fuera del ámbito de los documentos disponibles.\n\n"
        "💡 **Sugerencia**: Escribe '/temas' para ver un resumen de los contenidos "
        "disponibles, o '/docs' para ver la lista de documentos."
    ),
    
    "bienvenida": (
        "¡Bienvenido a Teacher, tu asistente inteligente de documentos!\n\n"
        "🤖 Tienes dos modos de interacción:\n\n"
        "  💬 **Modo CHAT** (activo por defecto):\n"
        "     Conversación libre conmigo. Puedo responder preguntas generales,\n"
        "     explicarte quién soy, o simplemente charlar.\n\n"
        "  📚 **Modo RAG** (recuperación de documentos):\n"
        "     Respuestas precisas basadas en los documentos académicos indexados,\n"
        "     con citas y fuentes verificables.\n\n"
        "📝 **Comandos disponibles** (todos empiezan por /):\n"
        "  • /rag      - Activar modo RAG (consulta de documentos)\n"
        "  • /chat     - Activar modo CHAT (conversación libre)\n"
        "  • /limpiar  - Limpiar historial de conversación\n"
        "  • /stats    - Ver estadísticas de la base de datos\n"
        "  • /docs     - Ver lista de documentos indexados\n"
        "  • /temas    - Ver resumen de contenidos disponibles\n"
        "  • /reindex  - Forzar re-indexación de documentos\n"
        "  • /ayuda    - Mostrar esta ayuda\n"
        "  • /salir    - Terminar la sesión\n\n"
        f"🚀 **Mejoras activas**: Chunks {CHUNK_SIZE} chars, {_EMBED_PREFIX_DESC} ✓, "
        f"{'pymupdf4llm ✓' if PYMUPDF_AVAILABLE else 'pypdf ⚠'}, "
        f"{'Reranker quality ✓' if USAR_RERANKER else 'Sin reranker ⚠'}, "
        f"{'Híbrida ✓' if USAR_BUSQUEDA_HIBRIDA else 'Solo semántica'}, "
        f"{'LLM decomp ✓' if USAR_LLM_QUERY_DECOMPOSITION else 'Sin decomp'}"
    ),
    
    "despedida": "¡Hasta luego! Gracias por usar Teacher. 👋"
}

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
# System prompt para modo CHAT (modelo base llama3.1)
# Proporciona identidad del proyecto y comportamiento conversacional.
# =============================================================================
SYSTEM_PROMPT_CHAT = """Eres Teacher, un asistente inteligente desarrollado como parte de un Trabajo de Fin de Grado (TFG) sobre Inteligencia Artificial aplicada a la educación.

SOBRE TI:
- Tu nombre es Teacher y estás basado en el modelo Llama 3.1 8B, fine-tuneado con LoRA para responder preguntas fundamentadas en documentos académicos.
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
# SECCIÓN 5: UTILIDADES DE PRESENTACIÓN Y FORMATEO
# =============================================================================
# Propósito:
# - Proveer utilidades reutilizables para renderizar salidas estructuradas en
#   consola y estandarizar la representación de evidencias documentales.
# Tecnologías implicadas:
# - Funciones helper de impresión para banners/separadores orientados a CLI.
# - Formateo de citas con metadatos (documento, página, fragmento) para
#   transparencia y verificabilidad de respuestas RAG.
# Resultado esperado:
# - Presentación uniforme de fuentes y bloques informativos en todo el sistema.
# =============================================================================

def mostrar_banner(titulo: str, estilo: str = "doble") -> None:
    """
    Muestra un banner visual para separar secciones.
    
    Args:
        titulo: Texto a mostrar en el banner
        estilo: 'doble' para líneas dobles, 'simple' para líneas simples
    """
    char = EstiloUI.LINEA_DOBLE if estilo == "doble" else EstiloUI.LINEA_SIMPLE
    ancho = EstiloUI.ANCHO
    
    print(f"\n{char * ancho}")
    print(f"  {titulo}")
    print(f"{char * ancho}")


def mostrar_separador(estilo: str = "simple") -> None:
    """Muestra una línea separadora."""
    char = EstiloUI.LINEA_DOBLE if estilo == "doble" else EstiloUI.LINEA_SIMPLE
    print(char * EstiloUI.ANCHO)


def formatear_cita(documento: str, pagina: int, fragmento: Optional[int] = None) -> str:
    """
    Formatea una cita de manera consistente.
    
    Args:
        documento: Nombre del documento fuente
        pagina: Número de página (0-indexed, se mostrará +1)
        fragmento: Número de fragmento opcional
    
    Returns:
        Cita formateada
    """
    cita = f"📄 {documento} | Página {pagina + 1}"
    if fragmento is not None:
        cita += f" | Fragmento {fragmento + 1}"
    return cita


def formatear_fuentes_respuesta(fragmentos: List[Dict]) -> str:
    """
    Genera una lista formateada de fuentes para mostrar al usuario.
    
    Args:
        fragmentos: Lista de fragmentos con metadata
    
    Returns:
        Texto formateado con las fuentes
    """
    fuentes_unicas = {}
    
    for frag in fragmentos:
        meta = frag['metadata']
        doc = meta['source']
        pagina = meta['page'] + 1
        
        if doc not in fuentes_unicas:
            fuentes_unicas[doc] = set()
        fuentes_unicas[doc].add(pagina)
    
    lineas = []
    for doc, paginas in sorted(fuentes_unicas.items()):
        paginas_str = ", ".join(str(p) for p in sorted(paginas))
        lineas.append(f"  {EstiloUI.ICONO_DOCUMENTO} {doc}")
        lineas.append(f"     Páginas consultadas: {paginas_str}")
    
    return "\n".join(lineas)


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
                # Migración: si viene del formato antiguo {"chat": [...], "rag": [...]},
                # extraer solo la parte de chat
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
    
    ids_adyacentes = []

    for i in range(1, n_vecinos + 1):
        if chunk_num - i >= 0:
            ids_adyacentes.append(f"{archivo}_pag{pagina}_chunk{chunk_num - i}")

    if 'total_chunks_in_page' in metadata:
        for i in range(1, n_vecinos + 1):
            if chunk_num + i < metadata['total_chunks_in_page']:
                ids_adyacentes.append(f"{archivo}_pag{pagina}_chunk{chunk_num + i}")
    
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
        print(f"   {EstiloUI.ICONO_EXITO} Keywords encontradas: {', '.join(list(keywords_encontradas)[:10])}")
    else:
        print(f"   {EstiloUI.ICONO_INFO} No se encontraron coincidencias directas por keywords")
    
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

_reranker_model = None


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
            
            print(f"\n{EstiloUI.ICONO_ENGRANAJE} Cargando modelo de reranking: {modelo_nombre}...")
            print(f"   Dispositivo: {device.upper()}" + (" (FP16)" if device == "cuda" else ""))

            model_kwargs = {"torch_dtype": "float16"} if device == "cuda" else {}
            _reranker_model = CrossEncoder(
                modelo_nombre,
                device=device,
                model_kwargs=model_kwargs,
            )
            
            print(f"   {EstiloUI.ICONO_EXITO} Modelo cargado correctamente en {device.upper()}")
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
    
    NOTA: Se usa MODELO_AUXILIAR (llama3.1:8b) en lugar del modelo fine-tuneado
    (teacher) porque el teacher fue entrenado para responder con contexto, no para
    generar queries de búsqueda. El modelo base es más adecuado para esta tarea
    de generación libre de consultas diversas.
    
    Args:
        pregunta: Pregunta original del usuario
    
    Returns:
        Lista de 2-3 queries de búsqueda generadas por el LLM
    """
    try:
        prompt = (
            "Genera exactamente 3 consultas de búsqueda DIFERENTES para encontrar "
            "información relevante en documentos académicos sobre esta pregunta.\n"
            "Cada consulta debe cubrir un aspecto diferente de la pregunta.\n"
            "Responde SOLO con las 3 consultas, una por línea, sin numeración ni explicaciones.\n\n"
            f"Pregunta: {pregunta}"
        )
        
        response = ollama.generate(
            model=MODELO_AUXILIAR,
            prompt=prompt,
            options={"temperature": 0.3, "num_predict": 150}
        )
        
        queries = [
            q.strip().lstrip('0123456789.-) ') 
            for q in response['response'].strip().split('\n') 
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
    mostrar_banner("FASE 1: BÚSQUEDA INTELIGENTE", "simple")
    
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
        print(f"\n{EstiloUI.ICONO_BUSQUEDA} [0/4] Descomponiendo pregunta con LLM...")
        llm_queries = generar_queries_con_llm(pregunta)
        if llm_queries:
            print(f"   {EstiloUI.ICONO_EXITO} {len(llm_queries)} sub-queries generadas")
    
    print(f"\n{EstiloUI.ICONO_BUSQUEDA} [1/4] Búsqueda semántica...")

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
    
    print(f"   Analizando {len(queries)} variantes de la pregunta")

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
    
    print(f"   {EstiloUI.ICONO_EXITO} {len(all_semantic_results)} fragmentos únicos encontrados")
    
    results_keyword = []
    metricas_keywords = {}
    if USAR_BUSQUEDA_HIBRIDA:
        print(f"\n{EstiloUI.ICONO_BUSQUEDA} [2/4] Búsqueda por palabras clave...")
        keywords = extraer_keywords(pregunta)
        print(f"   Keywords detectadas: {', '.join(keywords[:10])}...")
        results_keyword, metricas_keywords = busqueda_por_keywords(pregunta, collection)
        metricas_totales['fase_keywords'] = metricas_keywords

    print(f"\n{EstiloUI.ICONO_BUSQUEDA} [3/4] Combinando resultados...")
    
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
        print(f"\n{EstiloUI.ICONO_BUSQUEDA} Búsqueda profunda para: {', '.join(terminos_criticos[:6])}")
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
        print(f"\n{EstiloUI.ICONO_BUSQUEDA} [4/4] [RERANKING] Refinando los mejores {n_candidatos} candidatos...")
        
        candidatos_rerank = fragmentos_ranked[:TOP_K_RERANK_CANDIDATES]
        fragmentos_ranked, metricas_rerank = rerank_resultados(
            pregunta, 
            candidatos_rerank, 
            top_k=TOP_K_AFTER_RERANK
        )
        metricas_totales['fase_reranking'] = metricas_rerank
        print(f"   {EstiloUI.ICONO_EXITO} Top {len(fragmentos_ranked)} resultados tras reranking")
    
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
    system_prompt: str,
    mensaje_usuario: str,
    respuesta: str,
    fragmentos: List[Dict[str, Any]]
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
    
    Args:
        pregunta: Pregunta original del usuario
        system_prompt: System prompt enviado al modelo
        mensaje_usuario: Mensaje de usuario completo enviado al modelo
        respuesta: Respuesta generada por el modelo
        fragmentos: Fragmentos de contexto recuperados
    """
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
            
            f.write("─" * 80 + "\n")
            f.write("  PREGUNTA ORIGINAL\n")
            f.write("─" * 80 + "\n")
            f.write(f"{pregunta}\n\n")
            
            f.write("─" * 80 + "\n")
            f.write(f"  MODELO: {MODELO_CHAT}\n")
            f.write("─" * 80 + "\n\n")
            
            f.write("─" * 80 + "\n")
            f.write("  SYSTEM PROMPT\n")
            f.write("─" * 80 + "\n")
            f.write(f"{system_prompt}\n\n")
            
            f.write("─" * 80 + "\n")
            f.write("  MENSAJE DE USUARIO (enviado al modelo)\n")
            f.write("─" * 80 + "\n")
            f.write(f"{mensaje_usuario}\n\n")
            
            f.write("─" * 80 + "\n")
            f.write("  RESPUESTA DEL MODELO\n")
            f.write("─" * 80 + "\n")
            f.write(f"{respuesta}\n\n")
            
            f.write("─" * 80 + "\n")
            f.write(f"  FRAGMENTOS RECUPERADOS ({len(fragmentos)})\n")
            f.write("─" * 80 + "\n")
            for i, frag in enumerate(fragmentos, 1):
                meta = frag['metadata']
                score = frag.get('score_final', 'N/A')
                score_rr = frag.get('score_reranker', 'N/A')
                f.write(f"\n--- Fragmento {i} ---\n")
                f.write(f"Fuente: {meta.get('source', '?')}, pág. {meta.get('page', '?') + 1}\n")
                f.write(f"Score final: {score}  |  Score reranker: {score_rr}\n")
                f.write(f"Sección: {meta.get('section_header', '(sin header)')}\n")
                f.write(f"Texto:\n{frag['doc']}\n")
        
        logging.info(f"Debug RAG guardado: {ruta}")
        
    except Exception as e:
        logging.warning(f"Error guardando debug RAG: {e}")


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
    mostrar_banner("RESPUESTA DEL ASISTENTE", "doble")

    contexto_str = construir_contexto_para_modelo(fragmentos)

    mensaje_usuario = f"{pregunta}\n\n<contexto>{contexto_str}</contexto>"
    
    print(f"\n{EstiloUI.ICONO_ROBOT} Analizando {len(fragmentos)} fragmentos relevantes...\n")
    mostrar_separador()
    
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
            print(content, end='', flush=True)
            respuesta_completa += content
    print()
    
    mostrar_separador()
    print(f"\n{EstiloUI.ICONO_LIBRO} **FUENTES CONSULTADAS**:\n")
    print(formatear_fuentes_respuesta(fragmentos))
    print()
    
    # Guardar volcado de depuración para análisis
    guardar_debug_rag(pregunta, SYSTEM_PROMPT_RAG, mensaje_usuario, respuesta_completa, fragmentos)
    
    return respuesta_completa


def generar_respuesta_silenciosa(
    pregunta: str,
    fragmentos: List[Dict[str, Any]]
) -> str:
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


def responder_en_modo_chat(
    pregunta: str,
    historial_chat: List[Dict[str, str]]
) -> str:
    """
    Genera respuesta en modo conversacional usando el modelo base (auxiliar).
    
    Envía el historial completo de la conversación para mantener coherencia
    multi-turno. El modelo base (llama3.1:8b) conoce la identidad del proyecto
    y puede responder preguntas generales, explicar conceptos o mantener
    una conversación natural.
    
    Args:
        pregunta: Mensaje del usuario
        historial_chat: Historial previo de mensajes [{"role": ..., "content": ...}]
    
    Returns:
        Texto completo de la respuesta generada
    """
    mostrar_banner("TEACHER - MODO CONVERSACIÓN", "doble")
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]

    mensajes_recientes = historial_chat[-(MAX_HISTORIAL_MENSAJES):]
    messages.extend(mensajes_recientes)
    
    messages.append({"role": "user", "content": pregunta})
    
    print()
    mostrar_separador()
    
    stream = ollama.chat(
        model=MODELO_AUXILIAR,
        messages=messages,
        stream=True,
        options={"temperature": 0.7, "top_p": 0.9, "num_ctx": 8192}
    )
    
    respuesta_completa = ""
    print()
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "") or chunk.get("content", "")
        if content:
            print(content, end='', flush=True)
            respuesta_completa += content
    print()
    
    mostrar_separador()
    print()
    
    return respuesta_completa


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
        print(f"{EstiloUI.ICONO_ADVERTENCIA} No se encontraron archivos PDF en la carpeta")
        return 0
    
    mostrar_banner("PROCESANDO DOCUMENTOS", "doble")
    print(f"\n{EstiloUI.ICONO_ENGRANAJE} Configuración de indexación:")
    print(f"   • Extractor: {'pymupdf4llm (Markdown)' if PYMUPDF_AVAILABLE else 'pypdf (básico)'}")
    print(f"   • Tamaño de fragmento: {CHUNK_SIZE} caracteres")
    print(f"   • Solapamiento: {CHUNK_OVERLAP} caracteres")
    print(f"   • Máx. para embedding: {MAX_CHARS_EMBED} caracteres")
    print(f"   • Longitud mínima: {MIN_CHUNK_LENGTH} caracteres\n")
    
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
        print(f"\n{EstiloUI.ICONO_DOCUMENTO} Procesando: {archivo}")
        usar_pypdf_fallback = False
        
        try:
            ruta_pdf = os.path.join(carpeta, archivo)
            
            if PYMUPDF_AVAILABLE:
                try:
                    page_chunks = pymupdf4llm.to_markdown(ruta_pdf, page_chunks=True)
                    print(f"   Páginas: {len(page_chunks)}")
                    
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
                            
                            if _indexar_chunk(id_doc, chunk_text, chunk_text, metadata, collection):
                                total_chunks += 1
                        
                        print(f"   ✓ Página {i + 1}: {len(chunks)} fragmentos")
                    
                except Exception as e:
                    logging.error(f"Error con pymupdf4llm en {archivo}: {e}, usando pypdf fallback")
                    usar_pypdf_fallback = True

            if not PYMUPDF_AVAILABLE or usar_pypdf_fallback:
                reader = PdfReader(ruta_pdf)
                print(f"   Páginas: {len(reader.pages)}")
                
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
                        
                        if _indexar_chunk(id_doc, chunk_text, chunk_text, metadata, collection):
                            total_chunks += 1
                    
                    print(f"   ✓ Página {i + 1}: {len(chunks)} fragmentos")
                        
        except Exception as e:
            logging.error(f"Error procesando {archivo}: {e}")
            print(f"   {EstiloUI.ICONO_ERROR} Error: {e}")
    
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
# SECCIÓN 12: INTERFAZ DUAL, COMANDOS Y EXPERIENCIA DE CHAT
# =============================================================================
# Propósito:
# - Implementar el flujo conversacional interactivo con soporte dual:
#   modo CHAT (conversación libre con modelo base) y modo RAG (consulta
#   documental con modelo teacher fine-tuneado).
# Tecnologías implicadas:
# - Bucle CLI con parsing de comandos (/rag, /chat, /limpiar, stats, docs...).
# - Integración del pipeline híbrido en modo RAG con enriquecimiento contextual.
# - Conversación multi-turno con historial persistente en modo CHAT.
# - Cambio dinámico de modo durante la sesión.
# Resultado esperado:
# - Sesión dual estable: charla libre + recuperación documental verificable.
# =============================================================================

def mostrar_estadisticas(collection: chromadb.Collection) -> None:
    """Muestra estadísticas de la base de datos."""
    mostrar_banner("ESTADÍSTICAS DEL SISTEMA", "doble")
    
    docs = obtener_documentos_indexados(collection)
    
    print(f"\n{EstiloUI.ICONO_ESTADISTICA} **Base de datos vectorial**:")
    print(f"   • Fragmentos totales indexados: {collection.count()}")
    print(f"   • Documentos únicos: {len(docs)}")
    
    if docs:
        print(f"\n{EstiloUI.ICONO_DOCUMENTO} **Documentos indexados**:")
        for doc in docs:
            print(f"   • {doc}")
    
    print()

def mostrar_ayuda() -> None:
    """Muestra la ayuda del sistema."""
    mostrar_banner("AYUDA DEL SISTEMA", "doble")
    print(MENSAJES['bienvenida'])

def mostrar_documentos(collection: chromadb.Collection) -> None:
    """Muestra la lista de documentos indexados."""
    mostrar_banner("DOCUMENTOS DISPONIBLES", "simple")
    
    docs = obtener_documentos_indexados(collection)
    
    if docs:
        print(f"\n{EstiloUI.ICONO_CARPETA} Se encontraron {len(docs)} documento(s):\n")
        for i, doc in enumerate(docs, 1):
            print(f"   {i}. {doc}")
    else:
        print(f"\n{EstiloUI.ICONO_ADVERTENCIA} No hay documentos indexados.")
    
    print()

def mostrar_temas(collection: chromadb.Collection) -> None:
    """
    Muestra un resumen de los temas/contenidos de los documentos indexados.
    Extrae muestras representativas de cada documento.
    """
    mostrar_banner("RESUMEN DE CONTENIDOS DISPONIBLES", "doble")
    
    docs = obtener_documentos_indexados(collection)
    
    if not docs:
        print(f"\n{EstiloUI.ICONO_ADVERTENCIA} No hay documentos indexados.")
        print()
        return
    
    print(f"\n{EstiloUI.ICONO_LIBRO} **Documentos indexados**: {len(docs)}\n")
    
    for doc_name in docs:
        print(f"{EstiloUI.LINEA_SIMPLE * EstiloUI.ANCHO}")
        print(f"{EstiloUI.ICONO_DOCUMENTO} **{doc_name}**\n")
        
        try:
            all_data = collection.get(
                where={"source": doc_name},
                include=['documents', 'metadatas'],
                limit=100
            )
            
            if all_data['documents']:
                paginas_unicas = {meta['page'] for meta in all_data['metadatas']}
                print(f"   📃 Páginas indexadas: {len(paginas_unicas)}")
                print(f"   📊 Fragmentos totales: {len(all_data['documents'])}")
                
                texto_completo = " ".join(all_data['documents'][:20])
                palabras = texto_completo.split()
                
                palabras_significativas = [
                    p.strip('.,;:()[]{}"\'-').lower() 
                    for p in palabras 
                    if len(p) > 5 and p.strip('.,;:()[]{}"\'-').lower() not in STOPWORDS
                ]
                
                frecuencias = Counter(palabras_significativas)
                top_palabras = [palabra for palabra, _ in frecuencias.most_common(10)]
                
                if top_palabras:
                    print(f"\n   🏷️  **Términos frecuentes**: {', '.join(top_palabras)}")
                
                primer_fragmento = all_data['documents'][0][:300]
                print(f"\n   📝 **Muestra de contenido**:")
                print(f"      \"{primer_fragmento}...\"")
                
        except Exception as e:
            print(f"   {EstiloUI.ICONO_ERROR} Error al obtener información: {e}")
        
        print()
    
    print(f"{EstiloUI.LINEA_SIMPLE * EstiloUI.ANCHO}")
    print(f"\n{EstiloUI.ICONO_INFO} Escribe tu pregunta sobre cualquiera de estos temas.\n")

def _procesar_pregunta_rag(
    pregunta: str, 
    collection: chromadb.Collection
) -> None:
    """
    Procesa una pregunta en modo RAG: búsqueda híbrida + generación con contexto.
    
    Ejecuta el pipeline completo de recuperación y genera respuesta con el modelo
    teacher fine-tuneado. No persiste historial (cada consulta es independiente).
    
    Args:
        pregunta: Pregunta del usuario
        collection: Colección de ChromaDB
    """
    if len(pregunta.strip()) < MIN_LONGITUD_PREGUNTA_RAG:
        print(f"\n{EstiloUI.ICONO_INFO} Tu mensaje es demasiado corto para una consulta documental.")
        print(f"   En modo RAG necesito una pregunta concreta sobre el contenido de los documentos.")
        print(f"   Si quieres conversar libremente, escribe '/chat'.")
        return
    
    fragmentos_ranked, mejor_score, metricas = realizar_busqueda_hibrida(pregunta, collection)

    if not fragmentos_ranked:
        print(f"\n{EstiloUI.ICONO_INFO} {MENSAJES['info_no_encontrada']}")
        return
    
    if mejor_score < UMBRAL_RELEVANCIA:
        print(f"\n{EstiloUI.ICONO_INFO} {MENSAJES['fuera_de_ambito']}")
        print(f"   Mejor score: {mejor_score:.4f} (umbral: {UMBRAL_RELEVANCIA})")
        return

    if USAR_RERANKER:
        fragmentos_filtrados = [
            f for f in fragmentos_ranked
            if f.get('score_reranker', f.get('score_final', 0)) >= UMBRAL_SCORE_RERANKER
        ]
        n_descartados = len(fragmentos_ranked) - len(fragmentos_filtrados)
        if n_descartados > 0 and LOGGING_METRICAS:
            logging.info(f"Filtro reranker: {n_descartados} fragmentos descartados (score < {UMBRAL_SCORE_RERANKER})")
        
        if not fragmentos_filtrados:
            print(f"\n{EstiloUI.ICONO_INFO} {MENSAJES['info_no_encontrada']}")
            print(f"   Ningún fragmento superó el umbral de relevancia del reranker ({UMBRAL_SCORE_RERANKER}).")
            return
        
        fragmentos_ranked = fragmentos_filtrados

    fragmentos_finales = fragmentos_ranked[:TOP_K_FINAL]
    ids_usados = {f['id'] for f in fragmentos_finales}

    if EXPANDIR_CONTEXTO and fragmentos_finales and 'chunk' in fragmentos_finales[0]['metadata']:
        chunks_adicionales = []
        errores_expansion = 0

        for frag in fragmentos_finales[:N_TOP_PARA_EXPANSION]:
            ids_vecinos = expandir_con_chunks_adyacentes(
                frag['id'], 
                frag['metadata'], 
                n_vecinos=1
            )
            
            if ids_vecinos:
                try:
                    vecinos = collection.get(
                        ids=ids_vecinos,
                        include=['documents', 'metadatas']
                    )
                    
                    for v_doc, v_meta in zip(vecinos['documents'], vecinos['metadatas']):
                        v_id = f"{v_meta['source']}_pag{v_meta['page']}_chunk{v_meta.get('chunk', 0)}"
                        if v_id not in ids_usados:
                            chunks_adicionales.append({
                                'doc': v_doc,
                                'metadata': v_meta,
                                'distancia': float('inf'),
                                'score_final': 0.0,
                                'id': v_id
                            })
                            ids_usados.add(v_id)
                except Exception as e:
                    errores_expansion += 1
                    logging.warning(f"Error expandiendo contexto para {frag['id']}: {e}")
        
        if chunks_adicionales:
            fragmentos_finales.extend(chunks_adicionales)
            if LOGGING_METRICAS:
                logging.info(f"Contexto expandido: +{len(chunks_adicionales)} chunks adyacentes")
        
        if errores_expansion > 0:
            logging.warning(f"Errores en expansión de contexto: {errores_expansion}")
    
    contexto_total = sum(len(f['doc']) for f in fragmentos_finales)
    if contexto_total > MAX_CONTEXTO_CHARS:
        fragmentos_truncados = []
        chars_acum = 0
        for f in fragmentos_finales:
            if chars_acum + len(f['doc']) > MAX_CONTEXTO_CHARS:
                break
            fragmentos_truncados.append(f)
            chars_acum += len(f['doc'])
        
        n_eliminados = len(fragmentos_finales) - len(fragmentos_truncados)
        if n_eliminados > 0 and LOGGING_METRICAS:
            logging.info(f"Contexto truncado: {n_eliminados} fragmentos eliminados para respetar {MAX_CONTEXTO_CHARS} chars")
        fragmentos_finales = fragmentos_truncados
    
    print(f"\n{EstiloUI.ICONO_EXITO} Contexto preparado: {len(fragmentos_finales)} fragmentos relevantes")

    respuesta = generar_respuesta(pregunta, fragmentos_finales)

    # RAG no persiste historial: cada consulta es independiente


def ejecutar_chat(collection: chromadb.Collection) -> None:
    """
    Ejecuta el bucle principal del chat interactivo con soporte dual:
    
    - **Modo CHAT** (por defecto): Conversación libre con el modelo base
      (llama3.1:8b). Mantiene historial multi-turno y conoce la identidad
      del proyecto Teacher.
    - **Modo RAG**: Consulta de documentos académicos con el modelo teacher
      fine-tuneado. Cada pregunta es independiente (sin historial multi-turno)
      ya que el teacher fue entrenado para respuestas puntuales con contexto.
    
    Comandos (todos con prefijo /):
      /rag, /chat, /limpiar  → Cambio de modo e historial
      /stats, /docs, /temas  → Información del corpus
      /reindex, /ayuda, /salir → Operativos
    
    Args:
        collection: Colección de ChromaDB con documentos indexados
    """
    mostrar_banner("TEACHER - ASISTENTE INTELIGENTE", "doble")
    print(MENSAJES['bienvenida'])

    historial_chat = cargar_historial()
    
    if historial_chat:
        n_chat = len(historial_chat)
        print(f"\n{EstiloUI.ICONO_INFO} Historial cargado: {n_chat} mensajes de conversación previos.")

    modo_actual = "chat"
    print(f"\n{EstiloUI.ICONO_CHAT} Modo activo: **CHAT** (conversación libre con Teacher)")
    print(f"   Escribe '/rag' para consultar documentos académicos.")
    
    while True:
        print(f"\n{EstiloUI.LINEA_SIMPLE * EstiloUI.ANCHO}")
        
        if modo_actual == "chat":
            icono_prompt = EstiloUI.ICONO_CHAT
            etiqueta = "CHAT"
        else:
            icono_prompt = EstiloUI.ICONO_LIBRO
            etiqueta = "RAG"
        
        try:
            pregunta = input(f"{icono_prompt} [{etiqueta}] Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{MENSAJES['despedida']}")
            guardar_historial(historial_chat)
            break

        if pregunta.lower() in ['/salir', '/exit']:
            guardar_historial(historial_chat)
            print(f"\n{MENSAJES['despedida']}")
            break

        if pregunta.lower() == '/rag':
            modo_actual = "rag"
            print(f"\n{EstiloUI.ICONO_LIBRO} Modo cambiado a **RAG** (consulta de documentos)")
            print(f"   Las preguntas se responderán con fragmentos de los documentos indexados.")
            print(f"   Modelo: {MODELO_CHAT}")
            print(f"   Escribe '/chat' para volver a conversación libre.")
            continue
        
        if pregunta.lower() == '/chat':
            modo_actual = "chat"
            print(f"\n{EstiloUI.ICONO_CHAT} Modo cambiado a **CHAT** (conversación libre)")
            print(f"   Puedes preguntarme lo que quieras. Mantendré el contexto de la conversación.")
            print(f"   Modelo: {MODELO_AUXILIAR}")
            print(f"   Escribe '/rag' para consultar documentos académicos.")
            continue
        
        if pregunta.lower() in ['/limpiar', '/clear']:
            limpiar_historial(historial_chat)
            print(f"\n{EstiloUI.ICONO_EXITO} Historial de conversación limpiado.")
            continue

        if pregunta.lower() == '/stats':
            mostrar_estadisticas(collection)
            continue
        
        if pregunta.lower() in ['/ayuda', '/help']:
            mostrar_ayuda()
            continue
        
        if pregunta.lower() == '/docs':
            mostrar_documentos(collection)
            continue
        
        if pregunta.lower() == '/temas':
            mostrar_temas(collection)
            continue
        
        if pregunta.lower() == '/reindex':
            print(f"\n{EstiloUI.ICONO_ENGRANAJE} Forzando re-indexación...")
            print(f"   Esto eliminará la base de datos actual y la reconstruirá.")
            print(f"   Por favor, reinicia el programa después de la re-indexación.\n")
            
            import shutil
            try:
                if os.path.exists(PATH_DB):
                    shutil.rmtree(PATH_DB)
                    print(f"   {EstiloUI.ICONO_EXITO} Base de datos anterior eliminada")
                
                client_new = chromadb.PersistentClient(path=PATH_DB)
                collection_new = client_new.get_or_create_collection(name=COLLECTION_NAME)
                
                total = indexar_documentos(CARPETA_DOCS, collection_new)
                print(f"\n{EstiloUI.ICONO_EXITO} Re-indexación completada: {total} fragmentos")
                print(f"\n{EstiloUI.ICONO_ADVERTENCIA} Reinicia el programa para usar la nueva base de datos.")
                guardar_historial(historial_chat)
                return
            except Exception as e:
                print(f"   {EstiloUI.ICONO_ERROR} Error durante re-indexación: {e}")
            continue
        
        if not pregunta:
            continue

        if pregunta.startswith('/'):
            print(f"\n{EstiloUI.ICONO_ADVERTENCIA} Comando no reconocido: {pregunta}")
            print(f"   Escribe '/ayuda' para ver los comandos disponibles.")
            continue

        if modo_actual == "rag":
            _procesar_pregunta_rag(pregunta, collection)
        else:
            respuesta = responder_en_modo_chat(pregunta, historial_chat)
            
            historial_chat.append({"role": "user", "content": pregunta})
            historial_chat.append({"role": "assistant", "content": respuesta})
            guardar_historial(historial_chat)


# =============================================================================
# SECCIÓN 13: ARRANQUE, INICIALIZACIÓN Y CICLO DE VIDA
# =============================================================================
# Propósito:
# - Orquestar la secuencia de inicio del sistema desde configuración hasta
#   disponibilidad completa del asistente para consulta.
# Tecnologías implicadas:
# - Inicialización de cliente persistente de ChromaDB y colección activa.
# - Descubrimiento de PDFs en carpeta objetivo y indexación inicial condicional.
# - Reporte de estado de componentes críticos (modelos, extractor, reranker,
#   parámetros de chunking y métricas) para observabilidad operativa.
# Resultado esperado:
# - Arranque reproducible con diagnóstico claro antes de entrar al chat.
# =============================================================================

def main():
    """Función principal del programa."""
    mostrar_banner("INICIALIZANDO TEACHER - SISTEMA RAG", "doble")
    print(f"\n{EstiloUI.ICONO_CARPETA} Carpeta de ejecución (cwd): {os.getcwd()}")
    print(f"{EstiloUI.ICONO_CARPETA} Carpeta PDFs: {CARPETA_DOCS}")
    print(f"{EstiloUI.ICONO_CARPETA} DB vectorial: {PATH_DB}")
    print(f"{EstiloUI.ICONO_CARPETA} Historial: {HISTORIAL_PATH}")
    print(f"\n{EstiloUI.ICONO_ENGRANAJE} **MODELOS**:")
    print(f"   • RAG (teacher fine-tuneado): {MODELO_CHAT}")
    print(f"   • Chat / Auxiliar (base):     {MODELO_AUXILIAR}")
    print(f"   • Embeddings:                 {MODELO_EMBEDDING}")
    print(f"\n{EstiloUI.ICONO_EXITO} **MEJORAS ACTIVAS**:")
    print(f"   • Modelo embeddings: {MODELO_EMBEDDING.split(':')[0]} ({_EMBED_PREFIX_DESC} ✓)")
    print(f"   • Chunking: {CHUNK_SIZE} chars (estructura Markdown, overlap {CHUNK_OVERLAP})")
    print(f"   • Embedding capacity: {MAX_CHARS_EMBED} chars")
    print(f"   • Extracción PDF: {'pymupdf4llm (Markdown) ✓' if PYMUPDF_AVAILABLE else 'pypdf (básico) ⚠️'}")
    print(f"   • Búsqueda: {'Híbrida (semántica + keywords) ✓' if USAR_BUSQUEDA_HIBRIDA else 'Solo semántica'}")
    print(f"   • LLM query decomposition: {'Activado ✓ (modelo: ' + MODELO_AUXILIAR + ')' if USAR_LLM_QUERY_DECOMPOSITION else 'Desactivado'}")
    print(f"   • Reranker: {'CrossEncoder ✓' if USAR_RERANKER else 'Desactivado ⚠️'}")
    if USAR_RERANKER:
        _reranker_device = _detectar_dispositivo_reranker()
        print(f"     └─ Modelo: {'BAAI/bge-reranker-v2-m3 (multilingual quality)' if RERANKER_MODEL_QUALITY == 'quality' else 'ms-marco-MiniLM-L-6-v2 (fast)'}")
        print(f"     └─ Dispositivo: {_reranker_device.upper()}" + (" (FP16)" if _reranker_device == "cuda" else " (considerar GPU para acelerar)"))
    print(f"   • Persistencia de historial: Activada ✓")
    print(f"   • Modo dual: Chat (base) + RAG (teacher) ✓")
    print(f"   • DB versión: {_DB_VERSION}")
    print(f"   • Logging métricas: {'Activado' if LOGGING_METRICAS else 'Desactivado'}")
    
    client = chromadb.PersistentClient(path=PATH_DB)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    archivos_pdf = []
    try:
        archivos_pdf = [f for f in os.listdir(CARPETA_DOCS) if f.lower().endswith('.pdf')]
    except FileNotFoundError:
        print(f"{EstiloUI.ICONO_ADVERTENCIA} No existe la carpeta de PDFs: {CARPETA_DOCS}")
    print(f"{EstiloUI.ICONO_DOCUMENTO} PDFs detectados: {len(archivos_pdf)}")

    if collection.count() == 0:
        total_chunks = indexar_documentos(CARPETA_DOCS, collection)
        
        if total_chunks > 0:
            mostrar_banner("INDEXACIÓN COMPLETADA", "doble")
            print(f"\n{EstiloUI.ICONO_EXITO} Total de fragmentos indexados: {total_chunks}")
            print(f"{EstiloUI.ICONO_ESTADISTICA} Documentos en la colección: {collection.count()}")
        else:
            print(f"\n{EstiloUI.ICONO_ADVERTENCIA} No se indexaron documentos.")
            return
    else:
        print(f"\n{EstiloUI.ICONO_EXITO} Base de datos cargada: {collection.count()} fragmentos indexados")

    ejecutar_chat(collection)


# =============================================================================
# SECCIÓN 14: EJECUCIÓN COMO SCRIPT
# =============================================================================
# Propósito:
# - Habilitar ejecución directa del módulo como aplicación CLI autónoma.
# Tecnologías implicadas:
# - Convención estándar de Python __main__ para punto de entrada ejecutable.
# Resultado esperado:
# - Inicio del sistema con `python chat_pdfs.py` sin dependencias externas de
#   orquestación adicional.
# =============================================================================

if __name__ == "__main__":
    main()
