"""
MonkeyGrab CLI Renderer.

Rendering functions for the terminal interface: banners, separators,
tables, spinners, streaming output, and information panels. This module
provides the visual layer that formats and displays all CLI output using
ANSI escape codes and the centralized Theme configuration.

Usage:
    from rag.cli.renderer import render_banner, Spinner
    render_banner("TITLE")
    with Spinner("processing..."):
        do_work()

Dependencies:
    - rag.cli.theme (Theme, MESSAGES)
"""

import sys
import threading
import time
from typing import List, Dict, Any, Optional

from rag.cli.theme import Theme, MESSAGES

# ─────────────────────────────────────────────
# SPINNER CLASS
# ─────────────────────────────────────────────

class Spinner:
    """
    Animated spinner for long-running operations.

    Displays a cycling animation on the terminal while a background
    operation is in progress. Designed to be used as a context manager.

    Usage:
        with Spinner("searching semantically..."):
            result = slow_operation()
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
        width = Theme.terminal_width()
        sys.stdout.write(f"\r{' ' * width}\r")
        sys.stdout.flush()

    def finish(self, message: str, success: bool = True):
        """Replace the spinner with a completion message.

        Args:
            message: Text to display after the spinner stops.
            success: If True, shows a success icon; otherwise shows a failure icon.
        """
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        width = Theme.terminal_width()
        sys.stdout.write(f"\r{' ' * width}\r")
        icon = Theme.ICON_OK if success else Theme.ICON_FAIL
        color = Theme.GREEN if success else Theme.RED
        sys.stdout.write(f"  {color}{icon}{Theme.RESET} {Theme.TEXT_MUTED}{message}{Theme.RESET}\n")
        sys.stdout.flush()

# ─────────────────────────────────────────────
# RENDERING FUNCTIONS
# ─────────────────────────────────────────────

def render_banner(title: str, style: str = "simple", color: str = None) -> None:
    """
    Display a banner with borders and a title.

    Args:
        title: Banner text to display.
        style: 'simple' for single-line borders or 'doble' for double-line borders.
        color: ANSI color override for the border.
    """
    col = color or Theme.BORDER
    width = Theme.terminal_width()
    char = Theme.BOX_DOUBLE_H if style == "doble" else Theme.BOX_H

    print(f"\n{col}{char * width}{Theme.RESET}")
    print(f"  {col}{Theme.BOLD}{title}{Theme.RESET}")
    print(f"{col}{char * width}{Theme.RESET}")


def render_separator(color: str = None) -> None:
    """Display a subtle separator line."""
    col = color or Theme.BG_DARK
    width = Theme.terminal_width()
    print(f"{col}{Theme.BOX_H * width}{Theme.RESET}")


def render_step(step_num: int, total: int, message: str, color: str = None) -> None:
    """
    Display a pipeline step with [n/total] format.

    Args:
        step_num: Current step number.
        total: Total number of steps.
        message: Description of the step.
        color: ANSI color override.
    """
    col = color or Theme.BLUE
    print(f"\n  {col}[{step_num}/{total}]{Theme.RESET} {Theme.TEXT_MUTED}{message}{Theme.RESET}")


def render_detail(message: str, icon: str = None, color: str = None) -> None:
    """Display an indented detail or sub-step."""
    col = color or Theme.TEXT_DIM
    ic = icon or Theme.ICON_INFO
    print(f"      {col}{ic}{Theme.RESET} {col}{message}{Theme.RESET}")


def render_success(message: str) -> None:
    """Display a success message."""
    print(f"      {Theme.GREEN}{Theme.ICON_OK}{Theme.RESET} {Theme.TEXT_MUTED}{message}{Theme.RESET}")


def render_warning(message: str) -> None:
    """Display a warning message."""
    print(f"      {Theme.YELLOW}{Theme.ICON_WARN}{Theme.RESET} {Theme.TEXT_MUTED}{message}{Theme.RESET}")


def render_error(message: str) -> None:
    """Display an error message."""
    print(f"      {Theme.RED}{Theme.ICON_FAIL}{Theme.RESET} {Theme.TEXT_MUTED}{message}{Theme.RESET}")

# ─────────────────────────────────────────────
# INPUT PROMPT
# ─────────────────────────────────────────────

def build_prompt(mode: str, model: str = "") -> str:
    """
    Build the styled input prompt string.

    Renders a two-line prompt box showing the application name,
    current mode, and active model.

    Args:
        mode: Current interaction mode ('chat' or 'rag').
        model: Name of the active model.

    Returns:
        The bottom-line prompt string ready for user input.
    """
    T = Theme
    width = T.terminal_width()

    if mode == "chat":
        mode_color = T.PURPLE
        mode_label = "chat"
    else:
        mode_color = T.CYAN
        mode_label = "rag"

    model_short = model.split(":")[0] if model else ""
    header_content = f" monkeygrab {T.BORDER}{T.BOX_H}{T.BOX_H} {mode_color}{mode_label}{T.BORDER} {T.BOX_H}{T.BOX_H} {T.TEXT_DIM}{model_short}"

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

    bottom_line = f"{T.BORDER}{T.BOX_BL}{T.BOX_H}{T.RESET} {mode_color}{T.ICON_ARROW}{T.RESET} "

    print(f"\n{top_line}")
    return bottom_line

# ─────────────────────────────────────────────
# WELCOME SCREEN
# ─────────────────────────────────────────────

def render_welcome() -> None:
    """Display the welcome screen with available commands and modes."""
    T = Theme

    print(f"\n  {T.TEXT}Dos modos de interacción disponibles:{T.RESET}\n")

    print(f"  {T.PURPLE}{T.ICON_CHAT}{T.RESET}  {T.BOLD}{T.TEXT}Modo CHAT{T.RESET} {T.TEXT_DIM}(activo por defecto){T.RESET}")
    print(f"      {T.TEXT_DIM}Conversación libre. Preguntas generales, explicaciones.{T.RESET}\n")

    print(f"  {T.CYAN}{T.ICON_RAG}{T.RESET}  {T.BOLD}{T.TEXT}Modo RAG{T.RESET} {T.TEXT_DIM}(recuperación de documentos){T.RESET}")
    print(f"      {T.TEXT_DIM}Respuestas con citas de los documentos académicos indexados.{T.RESET}\n")

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

    print()

# ─────────────────────────────────────────────
# HELP SCREEN
# ─────────────────────────────────────────────

def render_help() -> None:
    """Display the full system help screen."""
    T = Theme
    render_banner("AYUDA", "simple", color=T.BRAND_DIM)
    render_welcome()

# ─────────────────────────────────────────────
# INITIALIZATION
# ─────────────────────────────────────────────

def render_init_info(info: Dict[str, Any]) -> None:
    """
    Display system initialization information.

    Shows model configuration, pipeline settings, and current database
    status including document and fragment counts.

    Args:
        info: Dictionary containing initialization data with keys such as
              cwd, pdfs_path, db_path, historial_path, modelo_chat,
              modelo_rag, modelo_embedding, extractor, busqueda,
              llm_decomp, reranker, reranker_model, reranker_device,
              chunk_size, chunk_overlap, embed_max, embed_prefix_desc,
              total_documentos, and total_fragmentos.
    """
    T = Theme
    L_WIDTH = 18

    render_banner("INICIALIZANDO", "simple", color=T.BORDER)

    print(f"\n  {T.TEXT}modelos:{T.RESET}")
    print(f"    {T.CYAN}{'rag / finetuned':<{L_WIDTH-2}}{T.RESET}{T.TEXT_MUTED}{info.get('modelo_rag', '')}{T.RESET}")
    print(f"    {T.PURPLE}{'chat / base':<{L_WIDTH-2}}{T.RESET}{T.TEXT_MUTED}{info.get('modelo_chat', '')}{T.RESET}")
    print(f"    {T.TEXT_DIM}{'embeddings':<{L_WIDTH-2}}{T.RESET}{T.TEXT_MUTED}{info.get('modelo_embedding', '')}{T.RESET}")

    print(f"\n  {T.TEXT}pipeline:{T.RESET}")
    print(f"    {T.TEXT_DIM}{'extractor':<{L_WIDTH-2}}{T.RESET}{T.TEXT_MUTED}{info.get('extractor', '')}{T.RESET}")
    print(f"    {T.TEXT_DIM}{'búsqueda':<{L_WIDTH-2}}{T.RESET}{T.TEXT_MUTED}{info.get('busqueda', '')}{T.RESET}")
    print(f"    {T.TEXT_DIM}{'llm-decomp':<{L_WIDTH-2}}{T.RESET}{T.TEXT_MUTED}{info.get('llm_decomp', '')}{T.RESET}")

    if info.get('reranker') == 'on':
        rr_model = info.get('reranker_model', '')
        rr_device = info.get('reranker_device', '')
        print(f"    {T.TEXT_DIM}{'reranker':<{L_WIDTH-2}}{T.RESET}{T.TEXT_MUTED}on ({rr_model} - {rr_device}){T.RESET}")
    else:
        print(f"    {T.TEXT_DIM}{'reranker':<{L_WIDTH-2}}{T.RESET}{T.TEXT_MUTED}off{T.RESET}")

    chunk_str = f"{info.get('chunk_size', '')}c (overlap: {info.get('chunk_overlap', '')}c)"
    print(f"    {T.TEXT_DIM}{'chunks':<{L_WIDTH-2}}{T.RESET}{T.TEXT_MUTED}{chunk_str}{T.RESET}")
    print(f"    {T.TEXT_DIM}{'embed-max':<{L_WIDTH-2}}{T.RESET}{T.TEXT_MUTED}{info.get('embed_max', '')}c{T.RESET}")
    print(f"    {T.TEXT_DIM}{'embed-pfx':<{L_WIDTH-2}}{T.RESET}{T.TEXT_MUTED}{info.get('embed_prefix_desc', '')}{T.RESET}")

    print(f"\n  {T.TEXT}estado:{T.RESET}")
    doc_count = info.get('total_documentos', 0)
    frag_count = info.get('total_fragmentos', 0)
    print(f"    {T.TEXT_DIM}{'documentos':<{L_WIDTH-2}}{T.RESET}{T.TEXT_MUTED}{doc_count} PDF(s) detectados{T.RESET}")
    print(f"    {T.TEXT_DIM}{'fragmentos index.':<{L_WIDTH-2}}{T.RESET}{T.TEXT_MUTED}{frag_count} en base de datos{T.RESET}")

# ─────────────────────────────────────────────
# STATISTICS, DOCUMENTS, TOPICS
# ─────────────────────────────────────────────

def render_stats(total_fragments: int, docs: List[str]) -> None:
    """Display database statistics including fragment and document counts.

    Args:
        total_fragments: Total number of indexed fragments.
        docs: List of unique document names.
    """
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
    """Display the list of indexed documents.

    Args:
        docs: List of document names to display.
    """
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
    Display a summary of available content and topics.

    Args:
        docs_data: List of dicts, each containing 'name', 'pages',
                   'fragments', 'terms', and 'sample' keys.
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

# ─────────────────────────────────────────────
# RESPONSE STREAMING
# ─────────────────────────────────────────────

def render_response_header(mode: str, model: str = "", n_fragments: int = 0) -> None:
    """Display the response header with mode and model information.

    Args:
        mode: Current mode ('rag' or 'chat').
        model: Name of the model generating the response.
        n_fragments: Number of context fragments (RAG mode only).
    """
    T = Theme

    if mode == "rag":
        render_banner("RESPUESTA", "doble", color=T.CYAN_DIM)
        if n_fragments > 0:
            print(f"  {T.TEXT_DIM}{n_fragments} fragmento(s) de contexto  {T.BORDER}│{T.RESET}  {T.TEXT_DIM}modelo: {model}{T.RESET}")
    else:
        render_banner("CHAT", "doble", color=T.PURPLE_DIM)

    render_separator()


def stream_token(token: str) -> None:
    """Print a single streaming response token to stdout."""
    print(token, end='', flush=True)


def render_response_footer(sources: Optional[str] = None) -> None:
    """Display the response footer with optional source citations.

    Args:
        sources: Pre-formatted sources string to display, or None.
    """
    T = Theme
    print()
    render_separator()

    if sources:
        print(f"\n  {T.TEXT}FUENTES:{T.RESET}\n")
        print(sources)

    print()

# ─────────────────────────────────────────────
# SOURCE AND CITATION FORMATTING
# ─────────────────────────────────────────────

def format_citation(document: str, page: int, fragment: Optional[int] = None) -> str:
    """
    Format a citation string in a consistent style.

    Args:
        document: Name of the source document.
        page: Page number (0-indexed; displayed as page + 1).
        fragment: Optional fragment number (0-indexed; displayed as fragment + 1).

    Returns:
        Formatted citation string.
    """
    cita = f"  [{document} | p.{page + 1}]"
    if fragment is not None:
        cita += f" frag.{fragment + 1}"
    return cita


def format_sources(fragments: List[Dict]) -> str:
    """
    Generate a formatted list of sources for a RAG response.

    Aggregates fragments by document and collects unique page numbers,
    then formats each document with its referenced pages.

    Args:
        fragments: List of fragment dicts containing metadata with
                   'source' and 'page' keys.

    Returns:
        Multi-line formatted string listing all sources and pages.
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

# ─────────────────────────────────────────────
# MODE MESSAGES
# ─────────────────────────────────────────────

def render_mode_change(mode: str, model: str = "") -> None:
    """Display a mode change notification.

    Args:
        mode: The new mode ('rag' or 'chat').
        model: Name of the model for the new mode.
    """
    T = Theme

    if mode == "rag":
        print(f"  {T.CYAN}{T.ICON_RAG}{T.RESET}  {T.TEXT}modo: rag{T.RESET}  {T.BORDER}│{T.RESET}  {T.TEXT_DIM}modelo: {model}{T.RESET}")
        print(f"      {T.TEXT_DIM}usa {T.PURPLE}/chat{T.TEXT_DIM} para volver a conversación libre{T.RESET}")
    else:
        print(f"  {T.PURPLE}{T.ICON_CHAT}{T.RESET}  {T.TEXT}modo: chat{T.RESET}  {T.BORDER}│{T.RESET}  {T.TEXT_DIM}modelo: {model}{T.RESET}")
        print(f"      {T.TEXT_DIM}usa {T.CYAN}/rag{T.TEXT_DIM} para consultar documentos{T.RESET}")


def render_history_loaded(n_messages: int) -> None:
    """Display how many history messages were loaded from a previous session.

    Args:
        n_messages: Number of messages restored.
    """
    T = Theme
    print(f"  {T.TEXT_DIM}{T.ICON_INFO} historial: {n_messages} mensaje(s) de sesión anterior{T.RESET}")


def render_history_cleared() -> None:
    """Display confirmation that chat history has been cleared."""
    T = Theme
    print(f"  {T.GREEN}{T.ICON_OK}{T.RESET} {T.TEXT_MUTED}historial limpiado{T.RESET}")


def render_unknown_command(cmd: str) -> None:
    """Display an unrecognized command error.

    Args:
        cmd: The command string that was not recognized.
    """
    T = Theme
    print(f"  {T.YELLOW}{T.ICON_WARN}{T.RESET} {T.TEXT_DIM}comando no reconocido: {T.TEXT_MUTED}{cmd}{T.RESET}")
    print(f"      {T.TEXT_DIM}usa {T.BRAND}/ayuda{T.TEXT_DIM} para ver los comandos disponibles{T.RESET}")


def render_reindex_start() -> None:
    """Display the start of a re-indexing operation."""
    T = Theme
    print(f"\n  {T.YELLOW}{T.ICON_GEAR}{T.RESET} {T.TEXT}Reindexando documentos...{T.RESET}")
    print(f"      {T.TEXT_DIM}Se eliminará la base de datos actual y se reconstruirá{T.RESET}")
    print(f"      {T.TEXT_DIM}Reinicia el programa tras completar{T.RESET}\n")


def render_reindex_complete(total: int) -> None:
    """Display completion of a re-indexing operation.

    Args:
        total: Total number of fragments indexed.
    """
    T = Theme
    print(f"\n  {T.GREEN}{T.ICON_OK}{T.RESET} {T.TEXT}Reindexación completada: {total} fragmentos{T.RESET}")
    print(f"\n  {T.YELLOW}{T.ICON_WARN}{T.RESET} {T.TEXT_DIM}Reinicia el programa para usar la nueva base de datos{T.RESET}")
