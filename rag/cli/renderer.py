"""
MonkeyGrab CLI — Renderer
===========================

Funciones de renderizado para la interfaz de terminal:
banners, separadores, tablas, spinners, streaming y
panels de información.
"""

import sys
import threading
import time
from typing import List, Dict, Any, Optional

from rag.cli.theme import Theme, MESSAGES


# =====================================================================
# SPINNER — Indicador de progreso animado
# =====================================================================

class Spinner:
    """
    Spinner animado para operaciones de larga duración.

    Uso:
        with Spinner("buscando semánticamente..."):
            resultado = operacion_lenta()
    """

    def __init__(self, message: str, color: str = None):
        self.message = message
        self.color = color or Theme.BLUE
        self._stop = threading.Event()
        self._thread = None

    def _animate(self):
        frames = Theme.SPINNER_FRAMES
        i = 0
        while not self._stop.is_set():
            frame = frames[i % len(frames)]
            text = f"\r  {self.color}{frame}{Theme.RESET} {Theme.TEXT_MUTED}{self.message}{Theme.RESET}"
            sys.stdout.write(text)
            sys.stdout.flush()
            i += 1
            self._stop.wait(0.08)

    def __enter__(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        # Limpiar línea
        width = Theme.terminal_width()
        sys.stdout.write(f"\r{' ' * width}\r")
        sys.stdout.flush()

    def finish(self, message: str, success: bool = True):
        """Reemplaza el spinner con un mensaje de finalización."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        width = Theme.terminal_width()
        sys.stdout.write(f"\r{' ' * width}\r")
        icon = Theme.ICON_OK if success else Theme.ICON_FAIL
        color = Theme.GREEN if success else Theme.RED
        sys.stdout.write(f"  {color}{icon}{Theme.RESET} {Theme.TEXT_MUTED}{message}{Theme.RESET}\n")
        sys.stdout.flush()


# =====================================================================
# FUNCIONES DE RENDERIZADO
# =====================================================================

def render_banner(title: str, style: str = "simple", color: str = None) -> None:
    """
    Muestra un banner con bordes y título.

    Args:
        title: Texto del banner
        style: 'simple' o 'doble'
        color: Color ANSI override
    """
    col = color or Theme.BORDER
    width = Theme.terminal_width()
    char = Theme.BOX_DOUBLE_H if style == "doble" else Theme.BOX_H

    print(f"\n{col}{char * width}{Theme.RESET}")
    print(f"  {col}{Theme.BOLD}{title}{Theme.RESET}")
    print(f"{col}{char * width}{Theme.RESET}")


def render_separator(color: str = None) -> None:
    """Muestra una línea separadora sutil."""
    col = color or Theme.BG_DARK
    width = Theme.terminal_width()
    print(f"{col}{Theme.BOX_H * width}{Theme.RESET}")


def render_step(step_num: int, total: int, message: str, color: str = None) -> None:
    """
    Muestra un paso del pipeline con formato [n/total].

    Args:
        step_num: Número del paso actual
        total: Total de pasos
        message: Descripción del paso
        color: Color override
    """
    col = color or Theme.BLUE
    print(f"\n  {col}[{step_num}/{total}]{Theme.RESET} {Theme.TEXT_MUTED}{message}{Theme.RESET}")


def render_detail(message: str, icon: str = None, color: str = None) -> None:
    """Muestra un detalle/sub-paso indentado."""
    col = color or Theme.TEXT_DIM
    ic = icon or Theme.ICON_INFO
    print(f"      {col}{ic}{Theme.RESET} {col}{message}{Theme.RESET}")


def render_success(message: str) -> None:
    """Muestra un mensaje de éxito."""
    print(f"      {Theme.GREEN}{Theme.ICON_OK}{Theme.RESET} {Theme.TEXT_MUTED}{message}{Theme.RESET}")


def render_warning(message: str) -> None:
    """Muestra un mensaje de advertencia."""
    print(f"      {Theme.YELLOW}{Theme.ICON_WARN}{Theme.RESET} {Theme.TEXT_MUTED}{message}{Theme.RESET}")


def render_error(message: str) -> None:
    """Muestra un mensaje de error."""
    print(f"      {Theme.RED}{Theme.ICON_FAIL}{Theme.RESET} {Theme.TEXT_MUTED}{message}{Theme.RESET}")


# ─────────────────────────────────────────────────────────────────────
# PROMPT DE ENTRADA
# ─────────────────────────────────────────────────────────────────────

def build_prompt(mode: str, model: str = "") -> str:
    """
    Construye el string del prompt con estilo profesional.

    Formato:
        ╭─ monkeygrab ── chat ── modelo ──╮
        ╰─ ›

    Args:
        mode: 'chat' o 'rag'
        model: Nombre del modelo activo
    """
    T = Theme
    width = T.terminal_width()

    # Colores según modo
    if mode == "chat":
        mode_color = T.PURPLE
        mode_label = "chat"
    else:
        mode_color = T.CYAN
        mode_label = "rag"

    # Línea superior del prompt
    model_short = model.split(":")[0] if model else ""
    header_content = f" monkeygrab {T.BORDER}{T.BOX_H}{T.BOX_H} {mode_color}{mode_label}{T.BORDER} {T.BOX_H}{T.BOX_H} {T.TEXT_DIM}{model_short}"

    # Calcular relleno (aprox, sin contar escapes ANSI)
    visible_len = len(f" monkeygrab -- {mode_label} -- {model_short} ")
    padding = max(0, width - visible_len - 4)

    top_line = (
        f"{T.BORDER}{T.BOX_TL}{T.BOX_H}{T.RESET}"
        f"{T.BRAND} monkeygrab{T.RESET} "
        f"{T.BORDER}{T.BOX_H}{T.BOX_H}{T.RESET} "
        f"{mode_color}{mode_label}{T.RESET} "
        f"{T.BORDER}{T.BOX_H}{T.BOX_H}{T.RESET} "
        f"{T.TEXT_DIM}{model_short}{T.RESET} "
        f"{T.BORDER}{T.BOX_H * max(0, padding)}{T.BOX_TR}{T.RESET}"
    )

    # Línea del cursor
    bottom_line = f"{T.BORDER}{T.BOX_BL}{T.BOX_H}{T.RESET} {mode_color}{T.ICON_ARROW}{T.RESET} "

    print(f"\n{top_line}")
    return bottom_line


# ─────────────────────────────────────────────────────────────────────
# PANTALLA DE BIENVENIDA
# ─────────────────────────────────────────────────────────────────────

def render_welcome(config: Dict[str, Any]) -> None:
    """
    Muestra la pantalla de bienvenida con comandos disponibles.

    Args:
        config: Dict con claves chunk_size, chunk_overlap, extractor,
                reranker, hybrid, llm_decomp
    """
    T = Theme

    print(f"\n  {T.TEXT}Dos modos de interacción disponibles:{T.RESET}\n")

    # Modo CHAT
    print(f"  {T.PURPLE}{T.ICON_CHAT}{T.RESET}  {T.BOLD}{T.TEXT}Modo CHAT{T.RESET} {T.TEXT_DIM}(activo por defecto){T.RESET}")
    print(f"      {T.TEXT_DIM}Conversación libre. Preguntas generales, explicaciones.{T.RESET}\n")

    # Modo RAG
    print(f"  {T.CYAN}{T.ICON_RAG}{T.RESET}  {T.BOLD}{T.TEXT}Modo RAG{T.RESET} {T.TEXT_DIM}(recuperación de documentos){T.RESET}")
    print(f"      {T.TEXT_DIM}Respuestas con citas de los documentos académicos indexados.{T.RESET}\n")

    # Tabla de comandos
    commands = [
        ("/rag",     "Activar modo RAG"),
        ("/chat",    "Activar modo CHAT"),
        ("/limpiar", "Limpiar historial"),
        ("/stats",   "Estadísticas de la base de datos"),
        ("/docs",    "Lista de documentos indexados"),
        ("/temas",   "Resumen de contenidos"),
        ("/reindex", "Forzar re-indexación"),
        ("/ayuda",   "Mostrar esta ayuda"),
        ("/salir",   "Terminar la sesión"),
    ]

    print(f"  {T.TEXT}Comandos:{T.RESET}\n")
    for cmd, desc in commands:
        print(f"    {T.BRAND}{cmd:<10}{T.RESET} {T.TEXT_DIM}{desc}{T.RESET}")

    # Config footer
    print(f"\n  {T.BORDER}{Theme.BOX_H * (Theme.terminal_width() - 4)}{T.RESET}")
    cfg_parts = [
        f"chunks={config.get('chunk_size', '?')}c",
        f"overlap={config.get('chunk_overlap', '?')}c",
        f"extractor={config.get('extractor', '?')}",
        f"reranker={'on' if config.get('reranker') else 'off'}",
        f"hybrid={'on' if config.get('hybrid') else 'off'}",
        f"llm-decomp={'on' if config.get('llm_decomp') else 'off'}",
    ]
    print(f"  {T.TEXT_DIM}{' · '.join(cfg_parts)}{T.RESET}")


# ─────────────────────────────────────────────────────────────────────
# PANTALLA DE AYUDA
# ─────────────────────────────────────────────────────────────────────

def render_help() -> None:
    """Muestra la ayuda completa del sistema."""
    T = Theme
    render_banner("AYUDA", "simple", color=T.BRAND_DIM)
    render_welcome({})  # Se rellenará en runtime


# ─────────────────────────────────────────────────────────────────────
# INICIALIZACIÓN
# ─────────────────────────────────────────────────────────────────────

def render_init_info(info: Dict[str, Any]) -> None:
    """
    Muestra la información de inicialización del sistema.

    Args:
        info: Dict con cwd, pdfs_path, db_path, historial_path,
              modelo_chat, modelo_auxiliar, modelo_embedding,
              extractor, busqueda, llm_decomp, reranker, reranker_model,
              reranker_device, chunk_size, chunk_overlap, embed_max,
              embed_prefix_desc, db_version
    """
    T = Theme

    render_banner("INICIALIZANDO", "simple", color=T.BORDER)

    print(f"\n  {T.TEXT_DIM}cwd{T.RESET}       {T.TEXT_MUTED}{info.get('cwd', '')}{T.RESET}")
    print(f"  {T.TEXT_DIM}pdfs{T.RESET}      {T.TEXT_MUTED}{info.get('pdfs_path', '')}{T.RESET}")
    print(f"  {T.TEXT_DIM}db{T.RESET}        {T.TEXT_MUTED}{info.get('db_path', '')}{T.RESET}")
    print(f"  {T.TEXT_DIM}historial{T.RESET} {T.TEXT_MUTED}{info.get('historial_path', '')}{T.RESET}")

    print(f"\n  {T.TEXT}modelos:{T.RESET}")
    print(f"    {T.CYAN}rag / finetuned{T.RESET}  {T.TEXT_MUTED}{info.get('modelo_chat', '')}{T.RESET}")
    print(f"    {T.PURPLE}chat / auxiliar{T.RESET}  {T.TEXT_MUTED}{info.get('modelo_auxiliar', '')}{T.RESET}")
    print(f"    {T.TEXT_DIM}embeddings{T.RESET}      {T.TEXT_MUTED}{info.get('modelo_embedding', '')}{T.RESET}")

    print(f"\n  {T.TEXT}pipeline:{T.RESET}")
    print(f"    {T.TEXT_DIM}extractor{T.RESET}   {T.TEXT_MUTED}{info.get('extractor', '')}{T.RESET}")
    print(f"    {T.TEXT_DIM}búsqueda{T.RESET}    {T.TEXT_MUTED}{info.get('busqueda', '')}{T.RESET}")
    print(f"    {T.TEXT_DIM}llm-decomp{T.RESET}  {T.TEXT_MUTED}{info.get('llm_decomp', '')}{T.RESET}")
    print(f"    {T.TEXT_DIM}reranker{T.RESET}    {T.TEXT_MUTED}{info.get('reranker', '')}{T.RESET}")

    if info.get('reranker_model'):
        print(f"      {T.TEXT_DIM}modelo{T.RESET}    {T.TEXT_MUTED}{info['reranker_model']}{T.RESET}")
        print(f"      {T.TEXT_DIM}device{T.RESET}    {T.TEXT_MUTED}{info.get('reranker_device', '')}{T.RESET}")

    print(f"    {T.TEXT_DIM}chunks{T.RESET}      {T.TEXT_MUTED}{info.get('chunk_size', '')}c  "
          f"overlap: {info.get('chunk_overlap', '')}c  "
          f"embed-max: {info.get('embed_max', '')}c{T.RESET}")
    print(f"    {T.TEXT_DIM}embed-pfx{T.RESET}   {T.TEXT_MUTED}{info.get('embed_prefix_desc', '')}{T.RESET}")
    print(f"    {T.TEXT_DIM}db-version{T.RESET}  {T.TEXT_MUTED}{info.get('db_version', '')}{T.RESET}")


# ─────────────────────────────────────────────────────────────────────
# ESTADÍSTICAS, DOCUMENTOS, TEMAS
# ─────────────────────────────────────────────────────────────────────

def render_stats(total_fragments: int, docs: List[str]) -> None:
    """Muestra estadísticas de la base de datos."""
    T = Theme
    render_banner("ESTADÍSTICAS", "simple", color=T.BRAND_DIM)

    print(f"\n  {T.TEXT_DIM}fragmentos indexados{T.RESET}  {T.TEXT}{total_fragments}{T.RESET}")
    print(f"  {T.TEXT_DIM}documentos únicos{T.RESET}    {T.TEXT}{len(docs)}{T.RESET}")

    if docs:
        print(f"\n  {T.TEXT}documentos:{T.RESET}")
        for doc in docs:
            print(f"    {T.BORDER}{T.ICON_DOT}{T.RESET} {T.TEXT_MUTED}{doc}{T.RESET}")

    print()


def render_docs(docs: List[str]) -> None:
    """Muestra la lista de documentos indexados."""
    T = Theme
    render_banner("DOCUMENTOS INDEXADOS", "simple", color=T.BRAND_DIM)

    if docs:
        print(f"\n  {T.TEXT}{len(docs)} documento(s):{T.RESET}\n")
        for i, doc in enumerate(docs, 1):
            print(f"    {T.TEXT_DIM}{i:>2}.{T.RESET} {T.TEXT_MUTED}{doc}{T.RESET}")
    else:
        print(f"\n  {T.YELLOW}{T.ICON_WARN}{T.RESET} {T.TEXT_DIM}No hay documentos indexados.{T.RESET}")

    print()


def render_topics(docs_data: List[Dict[str, Any]]) -> None:
    """
    Muestra resumen de contenidos/temas.

    Args:
        docs_data: Lista de dicts con 'name', 'pages', 'fragments',
                   'terms', 'sample'
    """
    T = Theme
    render_banner("CONTENIDOS DISPONIBLES", "simple", color=T.BRAND_DIM)

    if not docs_data:
        print(f"\n  {T.YELLOW}{T.ICON_WARN}{T.RESET} {T.TEXT_DIM}No hay documentos indexados.{T.RESET}\n")
        return

    print(f"\n  {T.TEXT}{len(docs_data)} documento(s) indexado(s){T.RESET}\n")

    for doc_info in docs_data:
        print(f"  {T.BORDER}{Theme.BOX_H * (Theme.terminal_width() - 4)}{T.RESET}")
        print(f"  {T.BRAND}{doc_info['name']}{T.RESET}\n")

        if doc_info.get('pages') is not None:
            print(f"    {T.TEXT_DIM}páginas{T.RESET}      {T.TEXT_MUTED}{doc_info['pages']}{T.RESET}")
        if doc_info.get('fragments') is not None:
            print(f"    {T.TEXT_DIM}fragmentos{T.RESET}   {T.TEXT_MUTED}{doc_info['fragments']}{T.RESET}")
        if doc_info.get('terms'):
            print(f"    {T.TEXT_DIM}términos{T.RESET}     {T.TEXT_MUTED}{doc_info['terms']}{T.RESET}")
        if doc_info.get('sample'):
            sample = doc_info['sample'][:300]
            print(f"\n    {T.TEXT_DIM}muestra{T.RESET}      {T.TEXT_DIM}\"{sample}...\"{T.RESET}")

        print()

    print(f"  {T.BORDER}{Theme.BOX_H * (Theme.terminal_width() - 4)}{T.RESET}")
    print(f"\n  {T.TEXT_DIM}Escribe tu pregunta sobre cualquiera de estos temas.{T.RESET}\n")


# ─────────────────────────────────────────────────────────────────────
# STREAMING DE RESPUESTA
# ─────────────────────────────────────────────────────────────────────

def render_response_header(mode: str, model: str = "", n_fragments: int = 0) -> None:
    """Muestra el encabezado de la respuesta."""
    T = Theme

    if mode == "rag":
        render_banner("RESPUESTA", "doble", color=T.CYAN_DIM)
        if n_fragments > 0:
            print(f"  {T.TEXT_DIM}{n_fragments} fragmento(s) de contexto  {T.BORDER}│{T.RESET}  {T.TEXT_DIM}modelo: {model}{T.RESET}")
    else:
        render_banner("CHAT", "doble", color=T.PURPLE_DIM)

    render_separator()


def stream_token(token: str) -> None:
    """Imprime un token de respuesta en streaming."""
    print(token, end='', flush=True)


def render_response_footer(sources: Optional[str] = None) -> None:
    """Muestra el pie de respuesta con fuentes opcionales."""
    T = Theme
    print()
    render_separator()

    if sources:
        print(f"\n  {T.TEXT}FUENTES:{T.RESET}\n")
        print(sources)

    print()


# ─────────────────────────────────────────────────────────────────────
# FORMATEO DE FUENTES Y CITAS
# ─────────────────────────────────────────────────────────────────────

def format_citation(document: str, page: int, fragment: Optional[int] = None) -> str:
    """
    Formatea una cita de forma consistente.

    Args:
        document: Nombre del documento fuente
        page: Número de página (0-indexed, se mostrará +1)
        fragment: Número de fragmento opcional
    """
    cita = f"  [{document} | p.{page + 1}]"
    if fragment is not None:
        cita += f" frag.{fragment + 1}"
    return cita


def format_sources(fragments: List[Dict]) -> str:
    """
    Genera lista formateada de fuentes para la respuesta RAG.

    Args:
        fragments: Lista de fragmentos con metadata
    """
    T = Theme
    sources_map = {}

    for frag in fragments:
        meta = frag['metadata']
        doc = meta['source']
        page = meta['page'] + 1

        if doc not in sources_map:
            sources_map[doc] = set()
        sources_map[doc].add(page)

    lines = []
    for doc, pages in sorted(sources_map.items()):
        pages_str = ", ".join(str(p) for p in sorted(pages))
        lines.append(f"    {T.BORDER}{T.ICON_DOT}{T.RESET} {T.TEXT_MUTED}{doc}{T.RESET}")
        lines.append(f"      {T.TEXT_DIM}páginas: {pages_str}{T.RESET}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# MENSAJES DE MODO
# ─────────────────────────────────────────────────────────────────────

def render_mode_change(mode: str, model: str = "") -> None:
    """Muestra mensaje de cambio de modo."""
    T = Theme

    if mode == "rag":
        print(f"  {T.CYAN}{T.ICON_RAG}{T.RESET}  {T.TEXT}modo: rag{T.RESET}  {T.BORDER}│{T.RESET}  {T.TEXT_DIM}modelo: {model}{T.RESET}")
        print(f"      {T.TEXT_DIM}usa {T.PURPLE}/chat{T.TEXT_DIM} para volver a conversación libre{T.RESET}")
    else:
        print(f"  {T.PURPLE}{T.ICON_CHAT}{T.RESET}  {T.TEXT}modo: chat{T.RESET}  {T.BORDER}│{T.RESET}  {T.TEXT_DIM}modelo: {model}{T.RESET}")
        print(f"      {T.TEXT_DIM}usa {T.CYAN}/rag{T.TEXT_DIM} para consultar documentos{T.RESET}")


def render_history_loaded(n_messages: int) -> None:
    """Muestra cuántos mensajes del historial se cargaron."""
    T = Theme
    print(f"  {T.TEXT_DIM}{T.ICON_INFO} historial: {n_messages} mensaje(s) de sesión anterior{T.RESET}")


def render_history_cleared() -> None:
    """Muestra confirmación de historial limpiado."""
    T = Theme
    print(f"  {T.GREEN}{T.ICON_OK}{T.RESET} {T.TEXT_MUTED}historial limpiado{T.RESET}")


def render_unknown_command(cmd: str) -> None:
    """Muestra error de comando no reconocido."""
    T = Theme
    print(f"  {T.YELLOW}{T.ICON_WARN}{T.RESET} {T.TEXT_DIM}comando no reconocido: {T.TEXT_MUTED}{cmd}{T.RESET}")
    print(f"      {T.TEXT_DIM}usa {T.BRAND}/ayuda{T.TEXT_DIM} para ver los comandos disponibles{T.RESET}")


def render_reindex_start() -> None:
    """Muestra inicio de re-indexación."""
    T = Theme
    print(f"\n  {T.YELLOW}{T.ICON_GEAR}{T.RESET} {T.TEXT}Reindexando documentos...{T.RESET}")
    print(f"      {T.TEXT_DIM}Se eliminará la base de datos actual y se reconstruirá{T.RESET}")
    print(f"      {T.TEXT_DIM}Reinicia el programa tras completar{T.RESET}\n")


def render_reindex_complete(total: int) -> None:
    """Muestra finalización de re-indexación."""
    T = Theme
    print(f"\n  {T.GREEN}{T.ICON_OK}{T.RESET} {T.TEXT}Reindexación completada: {total} fragmentos{T.RESET}")
    print(f"\n  {T.YELLOW}{T.ICON_WARN}{T.RESET} {T.TEXT_DIM}Reinicia el programa para usar la nueva base de datos{T.RESET}")
