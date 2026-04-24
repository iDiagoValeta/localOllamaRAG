"""
MonkeyGrab CLI Display.

Rich-based terminal presentation layer for the interactive CLI. The module
keeps layout decisions centralized so the app loop can focus on RAG behavior.
"""

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  CONFIGURATION
#  +-- 1. Imports and terminal setup
#  +-- 2. Theme and constants
#
#  COMPONENTS
#  +-- 3. Formatting helpers
#  +-- 4. Display class
#  +-- 5. Singleton
#
# ─────────────────────────────────────────────

import os
import sys
from typing import Any, Dict, Iterable, List, Optional

if os.name == "nt":
    os.system("")

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text
from rich.theme import Theme as RichTheme


# ─────────────────────────────────────────────
# SECTION 2: THEME AND CONSTANTS
# ─────────────────────────────────────────────

MONKEYGRAB_THEME = RichTheme({
    "brand": "bold #d7875f",
    "brand.dim": "#af5f00",
    "mode.chat": "#af87d7",
    "mode.rag": "#5fafaf",
    "info": "#5f87af",
    "success": "#5faf5f",
    "warning": "#d7af00",
    "error": "#d75f5f",
    "text": "#dadada",
    "muted": "#a8a8a8",
    "dim": "#6c6c6c",
    "off": "#606060",
})

COMMANDS = [
    ("/rag", "Activar modo RAG"),
    ("/chat", "Activar modo CHAT"),
    ("/limpiar", "Limpiar historial"),
    ("/stats", "Estado, modelos y base vectorial"),
    ("/docs", "Documentos indexados"),
    ("/temas", "Resumen de contenidos"),
    ("/reindex", "Reconstruir el índice"),
    ("/ayuda", "Mostrar esta ayuda"),
    ("/salir", "Terminar la sesión"),
]


# ─────────────────────────────────────────────
# SECTION 3: FORMATTING HELPERS
# ─────────────────────────────────────────────

def _coerce_bool(value: Any) -> bool:
    """Return a tolerant boolean for mixed config values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "si", "sí"}


def _short_model(model: str, max_len: int = 34, keep_tag: bool = False) -> str:
    """Shorten a model identifier without hiding the useful tail."""
    if not model:
        return "no configurado"

    model = str(model)
    if not keep_tag and ":" in model:
        model = model.split(":", 1)[0]

    if len(model) <= max_len:
        return model

    if "/" in model:
        head, tail = model.rsplit("/", 1)
        candidate = f"{head.split('/', 1)[0]}/.../{tail}"
        if len(candidate) <= max_len:
            return candidate

    head_len = max(8, (max_len - 3) // 2)
    tail_len = max(8, max_len - head_len - 3)
    return f"{model[:head_len]}...{model[-tail_len:]}"


def _model_tag(model: str) -> str:
    """Return the Ollama tag/version suffix when present."""
    if not model or ":" not in model:
        return ""
    return model.split(":", 1)[1]


def _state_label(enabled: Any, on_label: str = "on") -> Text:
    """Build a small styled state label."""
    text = Text()
    if _coerce_bool(enabled):
        text.append(on_label, style="success")
    else:
        text.append("off", style="off")
    return text


def _safe_pages(pages: Iterable[Any]) -> str:
    """Format page numbers compactly."""
    values = sorted({p for p in pages if isinstance(p, int)})
    if not values:
        return "-"
    shown = ", ".join(str(v + 1) for v in values[:12])
    if len(values) > 12:
        shown += f" +{len(values) - 12}"
    return shown


def _safe_tty_enabled() -> bool:
    """Return whether the CLI should prefer plain, stable terminal output."""
    env = os.getenv("MONKEYGRAB_SAFE_TTY")
    if env is not None:
        return _coerce_bool(env)
    return os.name == "nt"


# ─────────────────────────────────────────────
# SECTION 4: DISPLAY CLASS
# ─────────────────────────────────────────────

class Display:
    """Centralized terminal interface manager for MonkeyGrab."""

    def __init__(self):
        self.console = Console(
            theme=MONKEYGRAB_THEME,
            highlight=False,
            safe_box=True,
        )
        self._status: Optional[Status] = None
        self.safe_tty = _safe_tty_enabled()
        self._debug_mode = os.getenv("MONKEYGRAB_DEBUG", "").lower() in (
            "1", "true", "yes", "on"
        )

    def _print_lines(self, lines: Iterable[str]) -> None:
        for line in lines:
            self._plain(line)

    def _plain(self, text: str = "") -> None:
        print(text, flush=True)

    def _plain_rule(self, title: str = "", char: str = "-") -> None:
        width = 78
        if not title:
            self._plain(char * width)
            return
        label = f" {title} "
        side = max(2, (width - len(label)) // 2)
        rule = f"{char * side}{label}{char * max(2, width - len(label) - side)}"
        self._plain(rule[:width])

    # ─────────────────────────────────────────────
    # STARTUP
    # ─────────────────────────────────────────────

    def logo(self) -> None:
        """Display a compact brand mark."""
        if self.safe_tty:
            self._plain_rule("MonkeyGrab local PDF RAG", "=")
            return
        title = Text()
        title.append("Monkey", style="brand")
        title.append("Grab", style="brand.dim")
        title.append("  local PDF RAG", style="dim")
        self.console.rule(title, style="brand.dim")

    def init_panel(self, info: Dict[str, Any]) -> None:
        """Display the compact startup dashboard."""
        mode = info.get("mode", "chat")
        mode_style = "mode.rag" if mode == "rag" else "mode.chat"
        if self.safe_tty:
            self._plain(f"Modo inicial: {mode}  |  usa /ayuda para comandos")
            self._plain_rule()
            self._plain(f"  PDFs: {info.get('total_documentos', 0)}")
            self._plain(f"  Fragmentos: {info.get('total_fragmentos', 0)}")
            self._plain(f"  Carpeta: {info.get('docs_folder', '-')}")
            self._plain(f"  Coleccion: {info.get('collection_name', '-')}")
            self._plain(f"  Extractor: {info.get('extractor', '-')}")
            self._plain(f"  Busqueda: {info.get('busqueda', '-')}")
            rr = "off"
            if info.get("reranker") == "on":
                rr = f"{info.get('reranker_model', '-')}, {info.get('reranker_device', '-')}"
            self._plain(f"  Reranker: {rr}")
            self._plain(
                f"  Flags: hybrid={_coerce_bool(info.get('hybrid'))} "
                f"exhaustive={_coerce_bool(info.get('exhaustive'))} "
                f"rerank={info.get('reranker') == 'on'} "
                f"contextual={_coerce_bool(info.get('contextual'))} "
                f"recomp={_coerce_bool(info.get('recomp'))} "
                f"images={_coerce_bool(info.get('images'))} "
                f"expand={_coerce_bool(info.get('expand'))}"
            )
            self._plain_rule()
            self._plain()
            return

        corpus = Table.grid(padding=(0, 2))
        corpus.add_column(style="dim", no_wrap=True)
        corpus.add_column(style="text")
        corpus.add_row("PDFs", str(info.get("total_documentos", 0)))
        corpus.add_row("Fragmentos", str(info.get("total_fragmentos", 0)))
        corpus.add_row("Carpeta", str(info.get("docs_folder", "-")))
        corpus.add_row("Colección", str(info.get("collection_name", "-")))

        models = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="info",
            expand=True,
            pad_edge=False,
        )
        models.add_column("Rol", style="dim", width=12, no_wrap=True)
        models.add_column("Modelo", style="text", overflow="ellipsis")
        models.add_column("Tag", style="muted", width=12, overflow="ellipsis")
        models.add_column("Uso", style="muted", width=12, no_wrap=True)

        model_rows = [
            ("RAG", info.get("modelo_rag", ""), "respuestas"),
            ("Chat", info.get("modelo_chat", ""), "general"),
            ("Embeddings", info.get("modelo_embedding", ""), "índice"),
            ("Contextual", info.get("modelo_contextual", ""), "indexación" if info.get("contextual") else "off"),
            ("RECOMP", info.get("modelo_recomp", ""), "síntesis" if info.get("recomp") else "off"),
            ("OCR", info.get("modelo_ocr", ""), "imágenes" if info.get("images") else "off"),
        ]
        model_width = 44 if self.console.width >= 110 else 28
        for role, model, use in model_rows:
            use_style = "off" if use == "off" else "muted"
            models.add_row(
                role,
                _short_model(model, max_len=model_width, keep_tag=False),
                _model_tag(model) or "-",
                Text(str(use), style=use_style),
            )

        flags = Table.grid(padding=(0, 2))
        flags.add_column(no_wrap=True)
        flags.add_column(no_wrap=True)
        flags.add_column(no_wrap=True)
        flag_items = [
            ("hybrid", info.get("hybrid")),
            ("exhaustive", info.get("exhaustive")),
            ("rerank", info.get("reranker") == "on"),
            ("contextual", info.get("contextual")),
            ("recomp", info.get("recomp")),
            ("images", info.get("images")),
            ("expand", info.get("expand")),
        ]
        flag_cells: List[Text] = []
        for name, enabled in flag_items:
            cell = Text()
            cell.append(name, style="dim")
            cell.append("=")
            cell.append_text(_state_label(enabled))
            flag_cells.append(cell)
        for idx in range(0, len(flag_cells), 3):
            flags.add_row(*flag_cells[idx:idx + 3])

        details = Table.grid(padding=(0, 2))
        details.add_column(style="dim", no_wrap=True)
        details.add_column(style="muted")
        details.add_row("Extractor", str(info.get("extractor", "-")))
        details.add_row("Búsqueda", str(info.get("busqueda", "-")))
        if info.get("reranker") == "on":
            rr = f"{info.get('reranker_model', '-')}, {info.get('reranker_device', '-')}"
        else:
            rr = "off"
        details.add_row("Reranker", rr)
        details.add_row(
            "Chunks",
            f"{info.get('chunk_size', '-')}c, overlap {info.get('chunk_overlap', '-')}c",
        )

        top = Table.grid(expand=True)
        top.add_column(ratio=1)
        top.add_column(ratio=2)
        top.add_row(
            Panel(corpus, title="[info]Corpus[/]", border_style="dim", box=box.ROUNDED),
            Panel(details, title="[info]Pipeline[/]", border_style="dim", box=box.ROUNDED),
        )

        header = Text()
        header.append("Modo inicial ", style="dim")
        header.append(mode, style=mode_style)
        header.append("  ·  ")
        header.append("usa /ayuda para comandos", style="dim")

        self.console.print(header)
        self.console.print(top)
        self.console.print(models)
        self.console.print(Panel(flags, title="[info]Flags[/]", border_style="dim", box=box.ROUNDED))
        self.console.print()

    # ─────────────────────────────────────────────
    # HELP
    # ─────────────────────────────────────────────

    def welcome(self) -> None:
        """Display the complete command help."""
        if self.safe_tty:
            self._plain()
            self._plain_rule("Modos")
            self._plain("  CHAT  Conversacion libre con historial.")
            self._plain("  RAG   Consulta los PDFs indexados y muestra fuentes.")
            self._plain()
            self._plain_rule("Comandos")
            for cmd, desc in COMMANDS:
                self._plain(f"  {cmd:<12} {desc}")
            self._plain_rule()
            self._plain()
            return

        self.console.print()

        modes = Table.grid(padding=(0, 2))
        modes.add_column(no_wrap=True)
        modes.add_column(style="muted")
        modes.add_row("[mode.chat]CHAT[/]", "Conversación libre con historial.")
        modes.add_row("[mode.rag]RAG[/]", "Consulta los PDFs indexados y muestra fuentes.")

        commands = Table(
            title="[info]Comandos[/]",
            box=box.ROUNDED,
            border_style="dim",
            show_header=False,
            pad_edge=False,
        )
        commands.add_column("Comando", style="brand", width=12, no_wrap=True)
        commands.add_column("Descripción", style="muted")
        for cmd, desc in COMMANDS:
            commands.add_row(cmd, desc)

        self.console.print(Panel(modes, title="[info]Modos[/]", border_style="dim", box=box.ROUNDED))
        self.console.print(commands)
        self.console.print()

    # ─────────────────────────────────────────────
    # STATUS MESSAGES
    # ─────────────────────────────────────────────

    def success(self, msg: str) -> None:
        if self.safe_tty:
            self._plain(f"  OK {msg}")
            return
        self.console.print(f"  [success]✓[/] [muted]{msg}[/]")

    def warning(self, msg: str) -> None:
        if self.safe_tty:
            self._plain(f"  ! {msg}")
            return
        self.console.print(f"  [warning]![/] [muted]{msg}[/]")

    def error(self, msg: str) -> None:
        if self.safe_tty:
            self._plain(f"  x {msg}")
            return
        self.console.print(f"  [error]✗[/] [muted]{msg}[/]")

    def info(self, msg: str) -> None:
        if self.safe_tty:
            self._plain(f"  - {msg}")
            return
        self.console.print(f"  [info]·[/] [dim]{msg}[/]")

    def debug(self, msg: str) -> None:
        if self._debug_mode:
            if self.safe_tty:
                self._plain(f"  debug {msg}")
                return
            self.console.print(f"  [dim]⊡ {msg}[/]")

    def exception(self, title: str, exc: Exception) -> None:
        """Show a concise recoverable exception message."""
        detail = str(exc).strip() or exc.__class__.__name__
        if self.safe_tty:
            self._plain(f"  Error: {title}")
            self._plain(f"  {detail}")
            self._plain("  Comprueba que Ollama este activo y que el modelo exista localmente.")
            return
        panel = Panel(
            f"[muted]{detail}[/]\n\n[dim]Comprueba que Ollama esté activo y que el modelo exista localmente.[/]",
            title=f"[error]{title}[/]",
            border_style="error",
            box=box.ROUNDED,
            expand=False,
        )
        self.console.print(panel)

    # ─────────────────────────────────────────────
    # PIPELINE
    # ─────────────────────────────────────────────

    def pipeline_start(self, message: str = "Buscando en documentos...") -> Status:
        self.pipeline_stop()
        if self.safe_tty:
            self.info(message)
            return None  # type: ignore[return-value]
        self._status = self.console.status(
            f"[info]{message}[/]",
            spinner="dots",
            spinner_style="info",
        )
        self._status.start()
        return self._status

    def pipeline_update(self, message: str) -> None:
        if self.safe_tty:
            self.info(message)
            return
        if self._status:
            self._status.update(f"[info]{message}[/]")

    def pipeline_stop(self) -> None:
        if self._status:
            self._status.stop()
            self._status = None

    # ─────────────────────────────────────────────
    # INPUT PROMPT
    # ─────────────────────────────────────────────

    def prompt(self, mode: str, model: str = "") -> str:
        """Return a plain fallback prompt for code paths still using input()."""
        model_short = _short_model(model, max_len=22, keep_tag=False)
        return f"monkeygrab {mode} {model_short} > "

    def read_input(self, mode: str, model: str = "") -> str:
        """Read user input with a one-line Rich prompt."""
        # Stop any active status renderer before switching back to raw input.
        # Mixing a manual ``print(..., end="")`` prompt with ``input()`` has
        # proven fragile on Windows terminals after wider Rich layouts such as
        # help tables/panels. Let Rich own the prompt write and final flush.
        self.pipeline_stop()
        if self.safe_tty:
            return input(self.prompt(mode, model))
        self.console.file.flush()
        mode_style = "mode.rag" if mode == "rag" else "mode.chat"
        model_short = _short_model(model, max_len=max(18, min(34, self.console.width - 28)))
        prompt = Text()
        prompt.append("monkeygrab", style="brand")
        prompt.append(" ")
        prompt.append(mode, style=mode_style)
        prompt.append(" ")
        prompt.append(model_short, style="dim")
        prompt.append(" > ", style=mode_style)
        return self.console.input(prompt)

    # ─────────────────────────────────────────────
    # RESPONSE
    # ─────────────────────────────────────────────

    def response_header(self, mode: str, model: str = "") -> None:
        model_short = _short_model(model, max_len=36)
        label = "RAG" if mode == "rag" else "Chat"
        if self.safe_tty:
            self._plain()
            self._plain_rule(f"{label} {model_short}")
            self._plain()
            return
        self.console.print()
        style = "mode.rag" if mode == "rag" else "mode.chat"
        self.console.rule(f"[{style}]{label}[/] [dim]{model_short}[/]", style=style)

    def stream_token(self, token: str) -> None:
        if self.safe_tty:
            print(token, end="", flush=True)
            return
        self.console.print(token, end="")
        self.console.file.flush()

    def render_response(self, text: str) -> None:
        if self.safe_tty:
            body = (text or "(sin respuesta)").strip()
            for line in body.splitlines() or ["(sin respuesta)"]:
                self._plain(f"  {line}".rstrip())
            self._plain()
        else:
            self.console.print()
            self.console.print(Markdown(text), width=min(self.console.width - 4, 100))
            self.console.print()

    def render_markdown(self, text: str) -> None:
        self.render_response(text)

    def sources_panel(self, fragments: List[Dict[str, Any]]) -> None:
        if not fragments:
            return

        sources_map: Dict[str, set] = {}
        for frag in fragments:
            meta = frag.get("metadata", {})
            doc = meta.get("source", "?")
            page = meta.get("page", None)
            sources_map.setdefault(doc, set()).add(page)

        if self.safe_tty:
            self._plain_rule("Fuentes")
            for doc, pages in sorted(sources_map.items()):
                self._plain(f"  {_short_model(doc, max_len=56, keep_tag=True)} (paginas: {_safe_pages(pages)})")
            return

        table = Table(box=None, show_header=False, pad_edge=False)
        table.add_column("Documento", style="muted", overflow="fold")
        table.add_column("Páginas", style="dim", no_wrap=True)
        for doc, pages in sorted(sources_map.items()):
            table.add_row(_short_model(doc, max_len=56, keep_tag=True), _safe_pages(pages))

        self.console.print(Panel(table, title="[dim]Fuentes[/]", border_style="dim", box=box.ROUNDED))

    def response_footer(self) -> None:
        if self.safe_tty:
            self._plain_rule()
            self._plain()
            return
        self.console.rule(style="dim")
        self.console.print()

    # ─────────────────────────────────────────────
    # STATS, DOCS, TOPICS
    # ─────────────────────────────────────────────

    def stats_table(
        self,
        total_fragments: int,
        docs: List[str],
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        info = info or {}
        if self.safe_tty:
            self._plain()
            self._plain_rule("Estado")
            self._plain(f"  PDFs: {info.get('total_documentos', len(docs))}")
            self._plain(f"  Documentos indexados: {len(docs)}")
            self._plain(f"  Fragmentos: {total_fragments}")
            self._plain(f"  Carpeta: {info.get('docs_folder', '-')}")
            self._plain(f"  Base vectorial: {info.get('path_db', '-')}")
            self._plain(f"  Coleccion: {info.get('collection_name', '-')}")
            self._plain_rule()
            self._plain()
            return
        table = Table(title="[info]Estado[/]", box=box.ROUNDED, border_style="dim")
        table.add_column("Métrica", style="dim", no_wrap=True)
        table.add_column("Valor", style="text", overflow="fold")
        table.add_row("PDFs", str(info.get("total_documentos", len(docs))))
        table.add_row("Documentos indexados", str(len(docs)))
        table.add_row("Fragmentos", str(total_fragments))
        table.add_row("Carpeta", str(info.get("docs_folder", "-")))
        table.add_row("Base vectorial", str(info.get("path_db", "-")))
        table.add_row("Colección", str(info.get("collection_name", "-")))

        self.console.print()
        self.console.print(table)
        if info:
            self.init_panel(info)

    def docs_table(self, docs: List[Any]) -> None:
        if not docs:
            self.warning("No hay documentos indexados.")
            return

        if self.safe_tty:
            self._plain()
            self._plain_rule("Documentos")
            for idx, item in enumerate(docs, 1):
                if isinstance(item, dict):
                    self._plain(
                        f"  {idx}. {item.get('name', '-')} "
                        f"(pags: {item.get('pages', '-')}, frag: {item.get('fragments', '-')}, tipos: {item.get('formats', '-')})"
                    )
                else:
                    self._plain(f"  {idx}. {item}")
            self._plain_rule()
            self._plain()
            return

        table = Table(
            title="[info]Documentos[/]",
            box=box.ROUNDED,
            border_style="dim",
            show_header=True,
            header_style="info",
            pad_edge=False,
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Documento", style="text", overflow="fold")
        table.add_column("Págs.", style="muted", justify="right", no_wrap=True)
        table.add_column("Frag.", style="muted", justify="right", no_wrap=True)
        table.add_column("Tipos", style="dim", no_wrap=True)

        for idx, item in enumerate(docs, 1):
            if isinstance(item, dict):
                table.add_row(
                    str(idx),
                    item.get("name", "-"),
                    str(item.get("pages", "-")),
                    str(item.get("fragments", "-")),
                    item.get("formats", "-"),
                )
            else:
                table.add_row(str(idx), str(item), "-", "-", "-")

        self.console.print()
        self.console.print(table)
        self.console.print()

    def topics_display(self, docs_data: List[Dict[str, Any]]) -> None:
        if not docs_data:
            self.warning("No hay documentos indexados.")
            return

        if self.safe_tty:
            self._plain()
            self._plain_rule("Contenidos")
            for doc_info in docs_data:
                self._plain(
                    f"  {doc_info.get('name', '-')} "
                    f"(pags: {doc_info.get('pages', '-')}, frag: {doc_info.get('fragments', '-')})"
                )
                self._plain(f"    {doc_info.get('terms') or '-'}")
            self._plain("  Escribe una pregunta concreta o usa /docs para revisar el corpus.")
            self._plain_rule()
            self._plain()
            return

        table = Table(
            title="[info]Contenidos[/]",
            box=box.ROUNDED,
            border_style="dim",
            show_header=True,
            header_style="info",
            pad_edge=False,
        )
        table.add_column("Documento", style="text", overflow="fold")
        table.add_column("Págs.", style="muted", justify="right", no_wrap=True)
        table.add_column("Frag.", style="muted", justify="right", no_wrap=True)
        table.add_column("Términos frecuentes", style="dim", overflow="fold")

        for doc_info in docs_data:
            table.add_row(
                doc_info.get("name", "-"),
                str(doc_info.get("pages", "-")),
                str(doc_info.get("fragments", "-")),
                doc_info.get("terms") or "-",
            )

        self.console.print()
        self.console.print(table)
        self.console.print("  [dim]Escribe una pregunta concreta o usa /docs para revisar el corpus.[/]")
        self.console.print()

    # ─────────────────────────────────────────────
    # MODE CHANGES / HISTORY / MISC
    # ─────────────────────────────────────────────

    def mode_change(self, mode: str, model: str = "") -> None:
        if self.safe_tty:
            purpose = "consulta documental" if mode == "rag" else "conversacion libre"
            self._plain_rule()
            self._plain(f"  modo: {mode}")
            self._plain(f"  uso: {purpose}")
            self._plain(f"  modelo: {_short_model(model, 36)}")
            self._plain_rule()
            return
        style = "mode.rag" if mode == "rag" else "mode.chat"
        purpose = "consulta documental" if mode == "rag" else "conversación libre"
        self.console.print(
            f"  [{style}]modo {mode}[/] [dim]{purpose} · {_short_model(model, 36)}[/]"
        )

    def history_loaded(self, n: int) -> None:
        self.info(f"historial restaurado: {n} mensajes")

    def history_cleared(self) -> None:
        self.success("historial limpiado")

    def unknown_command(self, cmd: str) -> None:
        self.warning(f"Comando no reconocido: {cmd} [dim](usa /ayuda)[/]")

    def reindex_start(self) -> None:
        if self.safe_tty:
            self._plain()
            self._plain_rule("Reindex")
            self.warning("Reindexando documentos: se reconstruira la base vectorial.")
            return
        self.console.print()
        self.warning("Reindexando documentos: se reconstruirá la base vectorial.")

    def reindex_complete(self, total: int) -> None:
        self.success(f"Reindexación completada: {total} fragmentos")
        self.warning("Reinicia el programa para usar la nueva base de datos")

    def farewell(self) -> None:
        if self.safe_tty:
            self._plain()
            self._plain_rule()
            self._plain("  Sesion finalizada.")
            self._plain()
            return
        self.console.print()
        self.console.print("  [dim]Sesión finalizada.[/]")
        self.console.print()

    def no_results(self) -> None:
        if self.safe_tty:
            self._plain_rule("Sin Resultados")
            self._plain("  ! No se encontro informacion relevante en los documentos.")
            self._print_lines([
                "  - La informacion puede no estar indexada",
                "  - La pregunta puede necesitar mas detalle",
                "  - El tema puede estar fuera del corpus",
                "  Prueba con /temas o reformula la consulta.",
            ])
            self._plain_rule()
            return
        panel = Panel(
            "[muted]No se encontró información relevante en los documentos.[/]\n\n"
            "[dim]- La información puede no estar indexada\n"
            "- La pregunta puede necesitar más detalle\n"
            "- El tema puede estar fuera del corpus[/]\n\n"
            "[dim]Prueba con /temas o reformula la consulta.[/]",
            border_style="warning",
            box=box.ROUNDED,
            expand=False,
        )
        self.console.print(panel)

    def out_of_scope(self, score: float, threshold: float) -> None:
        if self.safe_tty:
            self._plain_rule("Fuera de Ambito")
            self.warning(f"Pregunta fuera de ambito (score {score:.4f} < {threshold})")
            self._plain("     Usa /temas para explorar el corpus.")
            self._plain_rule()
            return
        self.warning(f"Pregunta fuera de ámbito (score {score:.4f} < {threshold})")
        self.console.print("     [dim]Usa /temas para explorar el corpus.[/]")

    def question_too_short(self) -> None:
        if self.safe_tty:
            self._plain("  Pregunta demasiado corta. Formula una pregunta concreta o usa /chat.")
            return
        self.console.print(
            "  [dim]Pregunta demasiado corta. Formula una pregunta concreta o usa /chat.[/]"
        )

    def no_pdfs(self, folder: str) -> None:
        self.warning(f"No existe la carpeta de PDFs o está vacía: {folder}")


# ─────────────────────────────────────────────
# SECTION 5: SINGLETON
# ─────────────────────────────────────────────

ui = Display()
