"""
MonkeyGrab CLI
===========================================

Clase centralizada que encapsula toda la salida visual del sistema
usando la librería `rich`. Reemplaza los prints ANSI manuales por
componentes semánticos: paneles, tablas, spinners, markdown y progreso.

"""

import os
import sys
import time
from typing import List, Dict, Any, Optional

if os.name == 'nt':
    os.system('')

from rich.console import Console
from rich.theme import Theme as RichTheme
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.status import Status
from rich import box


# ─────────────────────────────────────────────────────────────
# TEMA GLOBAL — Paleta MonkeyGrab
# ─────────────────────────────────────────────────────────────

MONKEYGRAB_THEME = RichTheme({
    # Marca
    "brand":        "bold #d7875f",
    "brand.dim":    "#af5f00",

    # Modos
    "mode.chat":    "#af87d7",
    "mode.rag":     "#5fafaf",

    # Funcional
    "info":         "#5f87af",
    "success":      "#5faf5f",
    "warning":      "#d7af00",
    "error":        "#d75f5f",

    # Texto
    "text":         "#dadada",
    "muted":        "#808080",
    "dim":          "#585858",

    # Pipeline
    "step":         "bold #5f87af",
    "step.detail":  "#6c6c6c",
})


# ─────────────────────────────────────────────────────────────
# LOGO COMPACTO
# ─────────────────────────────────────────────────────────────

_MONKEY_MINI = r'''
               .-"""-.
             _/-=-.   \
            (_|a a/   |_
             / "  \   ,_)
        _    \`=' /__/
       / \_  .;--'  `-.
       \___)//      ,  \
        \ \/;        \  \
         \_.|         | |
          .-\ '     _/_/
        .'  _;.    (_  \
       /  .'   `\   \_/
      |_ /       |  |\\
     /  _)       /  / ||
jgs /  /       _/  /  //
    \_/       ( `-/  ||
              /  /   \\ .-.
              \_/     \'-'/
                       `"`'''


# ─────────────────────────────────────────────────────────────
# CLASE DISPLAY — Singleton de salida visual
# ─────────────────────────────────────────────────────────────

class Display:
    """
    Gestor centralizado de la interfaz de terminal de MonkeyGrab.

    Todos los métodos de salida pasan por `self.console` (rich.Console),
    asegurando cohesión visual y tema consistente.
    """

    def __init__(self):
        self.console = Console(
            theme=MONKEYGRAB_THEME,
            highlight=False,
        )
        self._status: Optional[Status] = None
        self._debug_mode = os.getenv("MONKEYGRAB_DEBUG", "").lower() in ("1", "true", "yes")

    # ─────────────────────────────────────────────────────────────
    # LOGO Y PANTALLA DE INICIO
    # ─────────────────────────────────────────────────────────────

    def logo(self) -> None:
        """Muestra el logo ASCII del mono de MonkeyGrab."""
        self.console.print(_MONKEY_MINI, style="brand")

        title = Text("              M O N K E Y G R A B", style="brand")
        self.console.print(title)
        self.console.print()

    def init_panel(self, info: Dict[str, Any]) -> None:
        """
        Muestra panel de inicialización con información del sistema
        en dos columnas: Modelos y Pipeline.
        """
        models = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        models.add_column("key", style="dim", width=16)
        models.add_column("val", style="muted")

        models.add_row("rag / finetuned", info.get("modelo_chat", ""))
        models.add_row("chat / base", info.get("modelo_auxiliar", ""))
        models.add_row("embeddings", info.get("modelo_embedding", ""))

        pipeline = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        pipeline.add_column("key", style="dim", width=16)
        pipeline.add_column("val", style="muted")

        pipeline.add_row("extractor", info.get("extractor", ""))
        pipeline.add_row("búsqueda", info.get("busqueda", ""))

        if info.get("reranker") == "on":
            rr = f"{info.get('reranker_model', '')} · {info.get('reranker_device', '')}"
            pipeline.add_row("reranker", rr)
        else:
            pipeline.add_row("reranker", "off")

        chunk_str = f"{info.get('chunk_size', '')}c (overlap {info.get('chunk_overlap', '')}c)"
        pipeline.add_row("chunks", chunk_str)
        pipeline.add_row("db-version", info.get("db_version", ""))

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(
            Panel(models, title="[info]Modelos[/]", border_style="dim", box=box.ROUNDED),
            Panel(pipeline, title="[info]Pipeline[/]", border_style="dim", box=box.ROUNDED),
        )

        doc_count = info.get("total_documentos", 0)
        frag_count = info.get("total_fragmentos", 0)
        status_line = f"[dim]{doc_count} PDF(s) detectados · {frag_count} fragmentos indexados[/]"

        self.console.print(grid)
        self.console.print(f"  {status_line}")
        self.console.print()

    # ─────────────────────────────────────────────────────────────
    # BIENVENIDA Y AYUDA
    # ─────────────────────────────────────────────────────────────

    def welcome(self) -> None:
        """Muestra la pantalla de bienvenida con modos y comandos."""
        self.console.print()
        self.console.print("  [mode.chat]◆[/]  [bold]Modo CHAT[/]  [dim](activo por defecto)[/]")
        self.console.print("     [dim]Conversación libre. Preguntas generales, explicaciones.[/]")
        self.console.print()
        self.console.print("  [mode.rag]◇[/]  [bold]Modo RAG[/]  [dim](recuperación de documentos)[/]")
        self.console.print("     [dim]Respuestas con citas de los documentos académicos indexados.[/]")
        self.console.print()

        cmd_table = Table(
            box=box.ROUNDED,
            border_style="dim",
            title="[info]Comandos[/]",
            title_style="info",
            show_header=False,
            padding=(0, 2),
        )
        cmd_table.add_column("cmd", style="brand", width=12)
        cmd_table.add_column("desc", style="muted")

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
        for cmd, desc in commands:
            cmd_table.add_row(cmd, desc)

        self.console.print(cmd_table)
        self.console.print()

    # ─────────────────────────────────────────────────────────────
    # MENSAJES DE ESTADO
    # ─────────────────────────────────────────────────────────────

    def success(self, msg: str) -> None:
        self.console.print(f"  [success]✓[/] [muted]{msg}[/]")

    def warning(self, msg: str) -> None:
        self.console.print(f"  [warning]![/] [muted]{msg}[/]")

    def error(self, msg: str) -> None:
        self.console.print(f"  [error]✗[/] [muted]{msg}[/]")

    def info(self, msg: str) -> None:
        self.console.print(f"  [info]·[/] [dim]{msg}[/]")

    def debug(self, msg: str) -> None:
        """Solo imprime si MONKEYGRAB_DEBUG está activo."""
        if self._debug_mode:
            self.console.print(f"  [dim]⊡ {msg}[/]")

    # ─────────────────────────────────────────────────────────────
    # PIPELINE RAG
    # ─────────────────────────────────────────────────────────────

    def pipeline_start(self, message: str = "Buscando en documentos...") -> Status:
        """
        Inicia un spinner de pipeline. Devuelve el Status para actualizarlo.

        Uso:
            status = ui.pipeline_start("Buscando...")
            # ... operación ...
            status.update("Re-ordenando resultados...")
            # ... operación ...
            status.stop()
            ui.success("4 fragmentos recuperados")
        """
        self._status = self.console.status(
            f"[info]{message}[/]",
            spinner="dots",
            spinner_style="info",
        )
        self._status.start()
        return self._status

    def pipeline_update(self, message: str) -> None:
        """Actualiza el mensaje del spinner de pipeline activo."""
        if self._status:
            self._status.update(f"[info]{message}[/]")

    def pipeline_stop(self) -> None:
        """Detiene el spinner sin mostrar resumen."""
        if self._status:
            self._status.stop()
            self._status = None

    # ─────────────────────────────────────────────────────────────
    # PROMPT DE ENTRADA
    # ─────────────────────────────────────────────────────────────

    def prompt(self, mode: str, model: str = "") -> str:
        """
        Construye y muestra el prompt Powerline.
        Devuelve la cadena del input prompt (línea inferior).
        """
        width = self.console.width
        model_short = model.split(":")[0] if model else ""

        if mode == "chat":
            mode_color = "mode.chat"
            mode_label = "chat"
        else:
            mode_color = "mode.rag"
            mode_label = "rag"

        visible_len = len(f" monkeygrab -- {mode_label} -- {model_short} ")
        padding = max(0, width - visible_len - 4)

        top = Text()
        top.append("╭─", style="dim")
        top.append(" monkeygrab ", style="brand")
        top.append("── ", style="dim")
        top.append(mode_label, style=mode_color)
        top.append(" ── ", style="dim")
        top.append(model_short, style="dim")
        top.append(" " + "─" * padding + "╮", style="dim")

        self.console.print()
        self.console.print(top)

        mode_ansi = "\033[38;5;141m" if mode == "chat" else "\033[38;5;73m"
        reset = "\033[0m"
        dim = "\033[38;5;240m"
        return f"{dim}╰─{reset} {mode_ansi}›{reset} "

    # ─────────────────────────────────────────────────────────────
    # RESPUESTA — Streaming + Markdown
    # ─────────────────────────────────────────────────────────────

    def response_header(self, mode: str, model: str = "") -> None:
        """Muestra header de la respuesta."""
        if mode == "rag":
            title = f"RAG [dim]── {model.split(':')[0]}[/]"
            style = "mode.rag"
        else:
            title = f"Chat [dim]── {model.split(':')[0]}[/]"
            style = "mode.chat"

        self.console.print()
        self.console.rule(title, style=style)

    def stream_token(self, token: str) -> None:
        """Imprime un token de streaming (sin rich, directo a stdout)."""
        sys.stdout.write(token)
        sys.stdout.flush()

    def render_markdown(self, text: str) -> None:
        """Renderiza texto como Markdown con rich."""
        self.console.print()
        md = Markdown(text)
        self.console.print(md, width=min(self.console.width - 4, 100))
        self.console.print()

    def sources_panel(self, fragments: List[Dict[str, Any]]) -> None:
        """
        Muestra las fuentes de la respuesta RAG en un panel discreto.
        """
        if not fragments:
            return

        sources_map: Dict[str, set] = {}
        for frag in fragments:
            meta = frag.get("metadata", {})
            doc = meta.get("source", "?")
            page = meta.get("page", 0) + 1
            if doc not in sources_map:
                sources_map[doc] = set()
            sources_map[doc].add(page)

        lines = []
        for doc, pages in sorted(sources_map.items()):
            pages_str = ", ".join(str(p) for p in sorted(pages))
            lines.append(f"[muted]📄 {doc}[/]  [dim]páginas: {pages_str}[/]")

        content = "\n".join(lines)
        panel = Panel(
            content,
            title="[dim]Fuentes[/]",
            border_style="dim",
            box=box.ROUNDED,
            expand=False,
            padding=(0, 2),
        )
        self.console.print(panel)

    def response_footer(self) -> None:
        """Separador final de respuesta."""
        self.console.rule(style="dim")
        self.console.print()

    # ─────────────────────────────────────────────────────────────
    # STATS, DOCS, TEMAS
    # ─────────────────────────────────────────────────────────────

    def stats_table(self, total_fragments: int, docs: List[str]) -> None:
        """Muestra estadísticas en tabla con bordes redondeados."""
        table = Table(
            title="[info]Estadísticas[/]",
            box=box.ROUNDED,
            border_style="dim",
            show_header=True,
            header_style="info",
        )
        table.add_column("Métrica", style="muted")
        table.add_column("Valor", style="text", justify="right")

        table.add_row("Fragmentos indexados", str(total_fragments))
        table.add_row("Documentos únicos", str(len(docs)))

        self.console.print()
        self.console.print(table)

        if docs:
            self.console.print()
            for doc in docs:
                self.console.print(f"  [dim]•[/] [muted]{doc}[/]")

        self.console.print()

    def docs_table(self, docs: List[str]) -> None:
        """Muestra lista de documentos indexados."""
        if not docs:
            self.warning("No hay documentos indexados.")
            return

        table = Table(
            title="[info]Documentos Indexados[/]",
            box=box.ROUNDED,
            border_style="dim",
            show_header=True,
            header_style="info",
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Documento", style="muted")

        for i, doc in enumerate(docs, 1):
            table.add_row(str(i), doc)

        self.console.print()
        self.console.print(table)
        self.console.print()

    def topics_display(self, docs_data: List[Dict[str, Any]]) -> None:
        """Muestra resumen de contenidos/temas."""
        if not docs_data:
            self.warning("No hay documentos indexados.")
            return

        self.console.print()
        self.console.rule("[info]Contenidos Disponibles[/]", style="dim")
        self.console.print()

        for doc_info in docs_data:
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("key", style="dim", width=14)
            table.add_column("val", style="muted")

            if doc_info.get("pages") is not None:
                table.add_row("páginas", str(doc_info["pages"]))
            if doc_info.get("fragments") is not None:
                table.add_row("fragmentos", str(doc_info["fragments"]))
            if doc_info.get("terms"):
                table.add_row("términos", doc_info["terms"])

            panel = Panel(
                table,
                title=f"[brand]{doc_info['name']}[/]",
                border_style="dim",
                box=box.ROUNDED,
                expand=False,
            )
            self.console.print(panel)

        self.console.print()
        self.console.print("  [dim]Escribe tu pregunta sobre cualquiera de estos temas.[/]")
        self.console.print()

    # ─────────────────────────────────────────────────────────────
    # CAMBIOS DE MODO / HISTORIAL / MISC
    # ─────────────────────────────────────────────────────────────

    def mode_change(self, mode: str, model: str = "") -> None:
        """Silenciado a petición del usuario para evitar ruido visual."""
        pass

    def history_loaded(self, n: int) -> None:
        pass

    def history_cleared(self) -> None:
        self.success("historial limpiado")

    def unknown_command(self, cmd: str) -> None:
        self.warning(f"Comando no reconocido: {cmd} [dim](usa /ayuda para ver los comandos)[/]")

    def reindex_start(self) -> None:
        self.console.print()
        self.console.print("  [warning]⚙[/] [text]Reindexando documentos...[/]")
        self.console.print("     [dim]Se eliminará la base de datos actual y se reconstruirá.[/]")
        self.console.print("     [dim]Reinicia el programa tras completar.[/]")
        self.console.print()

    def reindex_complete(self, total: int) -> None:
        self.success(f"Reindexación completada: {total} fragmentos")
        self.console.print()
        self.warning("Reinicia el programa para usar la nueva base de datos")

    def farewell(self) -> None:
        self.console.print()
        self.console.print("  [dim]Hasta luego. Sesión finalizada.[/]")
        self.console.print()

    def no_results(self) -> None:
        panel = Panel(
            "[muted]No se encontró información relevante en los documentos.[/]\n\n"
            "[dim]• La información no está en los documentos indexados\n"
            "• La pregunta podría formularse de otra manera\n"
            "• El tema está fuera del alcance del corpus[/]\n\n"
            "[dim]Sugerencia: reformula la pregunta o usa [brand]/temas[/brand] para explorar.[/]",
            border_style="warning",
            box=box.ROUNDED,
            expand=False,
            padding=(0, 2),
        )
        self.console.print(panel)

    def out_of_scope(self, score: float, threshold: float) -> None:
        self.console.print(
            f"  [warning]![/] [muted]Pregunta fuera de ámbito[/]  "
            f"[dim](score: {score:.4f} < umbral: {threshold})[/]"
        )
        self.console.print(f"     [dim]Usa [brand]/temas[/brand] para ver contenidos.[/]")

    def question_too_short(self) -> None:
        self.console.print(
            f"  [dim]Pregunta demasiado corta. Formula una pregunta concreta "
            f"o usa [brand]/chat[/brand] para conversar.[/]"
        )

    # ─────────────────────────────────────────────────────────────
    # MISC
    # ─────────────────────────────────────────────────────────────

    def no_pdfs(self, folder: str) -> None:
        self.warning(f"No existe la carpeta de PDFs o está vacía: {folder}")

# ─────────────────────────────────────────────────────────────
# SINGLETON — Instancia global
# ─────────────────────────────────────────────────────────────

ui = Display()
