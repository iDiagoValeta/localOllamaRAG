"""
MonkeyGrab CLI display backends.

Rich remains the default renderer outside Windows. On Windows the interactive
path uses prompt_toolkit + ANSI output to keep the prompt stable while still
providing a professional, colored CLI.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from typing import Any, Dict, Iterable, List, Literal, Optional

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text
from rich.theme import Theme as RichTheme

try:
    from colorama import just_fix_windows_console
except ImportError:  # pragma: no cover - dependency expected at runtime
    just_fix_windows_console = None

try:
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.shortcuts import print_formatted_text
    from prompt_toolkit.shortcuts import prompt as ptk_prompt

    PTK_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency expected at runtime
    ANSI = None
    print_formatted_text = None
    ptk_prompt = None
    PTK_AVAILABLE = False


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

if os.name == "nt" and just_fix_windows_console:
    just_fix_windows_console()


MONKEYGRAB_THEME = RichTheme(
    {
        "brand": "bold #d7af5f",
        "brand.dim": "#af875f",
        "mode.chat": "#87afd7",
        "mode.rag": "#5fafaf",
        "info": "#87afd7",
        "success": "#87af5f",
        "warning": "#d7af5f",
        "error": "#d75f5f",
        "text": "#dadada",
        "muted": "#a8a8a8",
        "dim": "#6c6c6c",
        "off": "#606060",
    }
)

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

BackendName = Literal["rich", "prompt_toolkit", "plain"]


class AnsiTheme:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    BRAND = "\033[38;5;179m"
    BRAND_DIM = "\033[38;5;137m"
    CHAT = "\033[38;5;110m"
    RAG = "\033[38;5;79m"
    INFO = "\033[38;5;110m"
    SUCCESS = "\033[38;5;71m"
    WARNING = "\033[38;5;179m"
    ERROR = "\033[38;5;167m"
    TEXT = "\033[38;5;252m"
    MUTED = "\033[38;5;246m"
    DIM_TEXT = "\033[38;5;240m"


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "si", "sí"}


def _short_model(model: str, max_len: int = 34, keep_tag: bool = False) -> str:
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
    if not model or ":" not in model:
        return ""
    return model.split(":", 1)[1]


def _state_label(enabled: Any, on_label: str = "on") -> Text:
    text = Text()
    if _coerce_bool(enabled):
        text.append(on_label, style="success")
    else:
        text.append("off", style="off")
    return text


def _safe_pages(pages: Iterable[Any]) -> str:
    values = sorted({p for p in pages if isinstance(p, int)})
    if not values:
        return "-"
    shown = ", ".join(str(v + 1) for v in values[:12])
    if len(values) > 12:
        shown += f" +{len(values) - 12}"
    return shown


def _select_backend() -> BackendName:
    env = os.getenv("MONKEYGRAB_CLI_BACKEND", "").strip().lower()
    if env in {"rich", "prompt_toolkit", "plain"}:
        if env == "prompt_toolkit" and not PTK_AVAILABLE:
            return "plain"
        return env  # type: ignore[return-value]
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return "plain"
    if os.name == "nt":
        return "prompt_toolkit" if PTK_AVAILABLE else "plain"
    return "rich"


class Display:
    """Centralized terminal interface manager for MonkeyGrab."""

    def __init__(self):
        self.console = Console(theme=MONKEYGRAB_THEME, highlight=False, safe_box=True)
        self._status: Optional[Status] = None
        self.backend: BackendName = _select_backend()
        self.safe_tty = self.backend != "rich"
        self._debug_mode = os.getenv("MONKEYGRAB_DEBUG", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._stream_bold = False
        self._stream_line_start = True
        self._pending_stars = 0

    # shared helpers
    def _ansi(self, text: str, *styles: str) -> str:
        return f"{''.join(styles)}{text}{AnsiTheme.RESET}"

    def _print_line(self, text: str = "") -> None:
        if self.backend == "prompt_toolkit":
            try:
                print_formatted_text(ANSI(text))
            except Exception:
                self.backend = "plain"
                self.safe_tty = True
                print(text, flush=True)
        else:
            print(text, flush=True)

    def _rule(self, title: str = "", char: str = "─", color: str = "") -> None:
        width = max(40, shutil.get_terminal_size(fallback=(88, 24)).columns)
        if not title:
            self._print_line(self._ansi(char * width, color or AnsiTheme.DIM_TEXT))
            return
        label = f" {title} "
        side = max(2, (width - len(label)) // 2)
        rule = f"{char * side}{label}{char * max(2, width - len(label) - side)}"
        self._print_line(self._ansi(rule[:width], color or AnsiTheme.DIM_TEXT))

    def _status_prefix(self, color: str, label: str, msg: str) -> None:
        self._print_line(
            f"  {self._ansi(label, color, AnsiTheme.BOLD)} {self._ansi(msg, AnsiTheme.MUTED)}"
        )

    def _mode_color(self, mode: str) -> str:
        return AnsiTheme.RAG if mode == "rag" else AnsiTheme.CHAT

    def _write_stream_text(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def _plain_markdown(self, text: str) -> str:
        lines = []
        for raw in text.splitlines():
            line = raw.strip()
            if line.startswith(("* ", "- ")):
                line = f"• {line[2:].lstrip()}"
            line = re.sub(r"^\s{0,4}\*\s+", "• ", line)
            line = line.replace("**", "").replace("__", "").replace("`", "")
            lines.append(line)
        return "\n".join(lines)

    def begin_stream(self) -> None:
        self._stream_bold = False
        self._stream_line_start = True
        self._pending_stars = 0

    def _flush_pending_stars(self, next_char: str = "") -> bool:
        consumed_space = False
        while self._pending_stars >= 2:
            self._stream_bold = not self._stream_bold
            self._write_stream_text(AnsiTheme.BOLD if self._stream_bold else AnsiTheme.RESET + AnsiTheme.TEXT)
            self._pending_stars -= 2
        if self._pending_stars == 1:
            if self._stream_line_start and next_char == " ":
                self._write_stream_text(f"{AnsiTheme.MUTED}• {AnsiTheme.TEXT}")
                self._stream_line_start = False
                consumed_space = True
            else:
                self._write_stream_text("*")
            self._pending_stars = 0
        return consumed_space

    # startup
    def logo(self) -> None:
        if self.backend == "rich":
            title = Text()
            title.append("Monkey", style="brand")
            title.append("Grab", style="brand.dim")
            title.append("  local PDF RAG", style="dim")
            self.console.rule(title, style="brand.dim")
            return
        self._rule("MonkeyGrab local PDF RAG", "═", AnsiTheme.BRAND)

    def init_panel(self, info: Dict[str, Any]) -> None:
        mode = info.get("mode", "chat")
        mode_style = "mode.rag" if mode == "rag" else "mode.chat"
        if self.backend != "rich":
            mode_color = self._mode_color(mode)
            self._print_line(
                f"{self._ansi('Modo inicial:', AnsiTheme.DIM_TEXT)} "
                f"{self._ansi(mode, mode_color, AnsiTheme.BOLD)}"
                f"{self._ansi('  |  usa /ayuda para comandos', AnsiTheme.DIM_TEXT)}"
            )
            self._rule(color=AnsiTheme.DIM_TEXT)
            rows = [
                ("PDFs", info.get("total_documentos", 0)),
                ("Fragmentos", info.get("total_fragmentos", 0)),
                ("Carpeta", info.get("docs_folder", "-")),
                ("Colección", info.get("collection_name", "-")),
                ("Extractor", info.get("extractor", "-")),
                ("Búsqueda", info.get("busqueda", "-")),
            ]
            rr = "off"
            if info.get("reranker") == "on":
                rr = f"{info.get('reranker_model', '-')}, {info.get('reranker_device', '-')}"
            rows.append(("Reranker", rr))
            for key, value in rows:
                self._print_line(
                    f"  {self._ansi(f'{key}:', AnsiTheme.DIM_TEXT)} {self._ansi(str(value), AnsiTheme.TEXT)}"
                )
            flags = (
                f"hybrid={_coerce_bool(info.get('hybrid'))} "
                f"exhaustive={_coerce_bool(info.get('exhaustive'))} "
                f"rerank={info.get('reranker') == 'on'} "
                f"contextual={_coerce_bool(info.get('contextual'))} "
                f"recomp={_coerce_bool(info.get('recomp'))} "
                f"images={_coerce_bool(info.get('images'))} "
                f"expand={_coerce_bool(info.get('expand'))}"
            )
            self._print_line(f"  {self._ansi('Flags:', AnsiTheme.DIM_TEXT)} {self._ansi(flags, AnsiTheme.MUTED)}")
            self._rule(color=AnsiTheme.DIM_TEXT)
            self._print_line()
            return

        corpus = Table.grid(padding=(0, 2))
        corpus.add_column(style="dim", no_wrap=True)
        corpus.add_column(style="text")
        corpus.add_row("PDFs", str(info.get("total_documentos", 0)))
        corpus.add_row("Fragmentos", str(info.get("total_fragmentos", 0)))
        corpus.add_row("Carpeta", str(info.get("docs_folder", "-")))
        corpus.add_row("Colección", str(info.get("collection_name", "-")))

        models = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="info", expand=True, pad_edge=False)
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
            models.add_row(role, _short_model(model, max_len=model_width, keep_tag=False), _model_tag(model) or "-", Text(str(use), style=use_style))

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
        cells: List[Text] = []
        for name, enabled in flag_items:
            cell = Text()
            cell.append(name, style="dim")
            cell.append("=")
            cell.append_text(_state_label(enabled))
            cells.append(cell)
        for idx in range(0, len(cells), 3):
            flags.add_row(*cells[idx:idx + 3])

        details = Table.grid(padding=(0, 2))
        details.add_column(style="dim", no_wrap=True)
        details.add_column(style="muted")
        details.add_row("Extractor", str(info.get("extractor", "-")))
        details.add_row("Búsqueda", str(info.get("busqueda", "-")))
        rr = "off"
        if info.get("reranker") == "on":
            rr = f"{info.get('reranker_model', '-')}, {info.get('reranker_device', '-')}"
        details.add_row("Reranker", rr)
        details.add_row("Chunks", f"{info.get('chunk_size', '-')}c, overlap {info.get('chunk_overlap', '-')}c")

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

    # help
    def welcome(self) -> None:
        if self.backend != "rich":
            self._print_line()
            self._rule("Modos", color=AnsiTheme.INFO)
            self._print_line(f"  {self._ansi('CHAT', AnsiTheme.CHAT, AnsiTheme.BOLD)} {self._ansi('Conversación libre con historial.', AnsiTheme.MUTED)}")
            self._print_line(f"  {self._ansi('RAG', AnsiTheme.RAG, AnsiTheme.BOLD)}  {self._ansi('Consulta los PDFs indexados y muestra fuentes.', AnsiTheme.MUTED)}")
            self._print_line()
            self._rule("Comandos", color=AnsiTheme.BRAND)
            for cmd, desc in COMMANDS:
                self._print_line(f"  {self._ansi(f'{cmd:<12}', AnsiTheme.BRAND, AnsiTheme.BOLD)} {self._ansi(desc, AnsiTheme.TEXT)}")
            self._rule(color=AnsiTheme.DIM_TEXT)
            self._print_line()
            return

        self.console.print()
        modes = Table.grid(padding=(0, 2))
        modes.add_column(no_wrap=True)
        modes.add_column(style="muted")
        modes.add_row("[mode.chat]CHAT[/]", "Conversación libre con historial.")
        modes.add_row("[mode.rag]RAG[/]", "Consulta los PDFs indexados y muestra fuentes.")
        commands = Table(title="[info]Comandos[/]", box=box.ROUNDED, border_style="dim", show_header=False, pad_edge=False)
        commands.add_column("Comando", style="brand", width=12, no_wrap=True)
        commands.add_column("Descripción", style="muted")
        for cmd, desc in COMMANDS:
            commands.add_row(cmd, desc)
        self.console.print(Panel(modes, title="[info]Modos[/]", border_style="dim", box=box.ROUNDED))
        self.console.print(commands)
        self.console.print()

    # status
    def success(self, msg: str) -> None:
        if self.backend != "rich":
            self._status_prefix(AnsiTheme.SUCCESS, "OK", msg)
            return
        self.console.print(f"  [success]✓[/] [muted]{msg}[/]")

    def warning(self, msg: str) -> None:
        if self.backend != "rich":
            self._status_prefix(AnsiTheme.WARNING, "!", msg)
            return
        self.console.print(f"  [warning]![/] [muted]{msg}[/]")

    def error(self, msg: str) -> None:
        if self.backend != "rich":
            self._status_prefix(AnsiTheme.ERROR, "x", msg)
            return
        self.console.print(f"  [error]✗[/] [muted]{msg}[/]")

    def info(self, msg: str) -> None:
        if self.backend != "rich":
            self._status_prefix(AnsiTheme.INFO, "·", msg)
            return
        self.console.print(f"  [info]·[/] [dim]{msg}[/]")

    def debug(self, msg: str) -> None:
        if not self._debug_mode:
            return
        if self.backend != "rich":
            self._status_prefix(AnsiTheme.DIM_TEXT, "debug", msg)
            return
        self.console.print(f"  [dim]⊡ {msg}[/]")

    def exception(self, title: str, exc: Exception) -> None:
        detail = str(exc).strip() or exc.__class__.__name__
        if self.backend != "rich":
            self._print_line()
            self._rule(title, color=AnsiTheme.ERROR)
            self._print_line(f"  {self._ansi(detail, AnsiTheme.TEXT)}")
            self._print_line(f"  {self._ansi('Comprueba que Ollama esté activo y que el modelo exista localmente.', AnsiTheme.DIM_TEXT)}")
            self._rule(color=AnsiTheme.DIM_TEXT)
            return
        panel = Panel(
            f"[muted]{detail}[/]\n\n[dim]Comprueba que Ollama esté activo y que el modelo exista localmente.[/]",
            title=f"[error]{title}[/]",
            border_style="error",
            box=box.ROUNDED,
            expand=False,
        )
        self.console.print(panel)

    # pipeline
    def pipeline_start(self, message: str = "Buscando en documentos...") -> Optional[Status]:
        self.pipeline_stop()
        if self.backend != "rich":
            self.info(message)
            return None
        self._status = self.console.status(f"[info]{message}[/]", spinner="dots", spinner_style="info")
        self._status.start()
        return self._status

    def pipeline_update(self, message: str) -> None:
        if self.backend != "rich":
            self.info(message)
            return
        if self._status:
            self._status.update(f"[info]{message}[/]")

    def pipeline_stop(self) -> None:
        if self._status:
            self._status.stop()
            self._status = None

    # input
    def prompt(self, mode: str, model: str = "") -> str:
        model_short = _short_model(model, max_len=22, keep_tag=False)
        return f"monkeygrab {mode} {model_short} > "

    def _ansi_prompt(self, mode: str, model: str = "") -> str:
        model_short = _short_model(model, max_len=22, keep_tag=False)
        mode_color = self._mode_color(mode)
        return (
            f"{AnsiTheme.BRAND}{AnsiTheme.BOLD}monkeygrab{AnsiTheme.RESET} "
            f"{mode_color}{AnsiTheme.BOLD}{mode}{AnsiTheme.RESET} "
            f"{AnsiTheme.DIM_TEXT}{model_short}{AnsiTheme.RESET} "
            f"{mode_color}{AnsiTheme.BOLD}> {AnsiTheme.RESET}"
        )

    def read_input(self, mode: str, model: str = "") -> str:
        self.pipeline_stop()
        if self.backend == "prompt_toolkit":
            try:
                return ptk_prompt(ANSI(self._ansi_prompt(mode, model)))
            except Exception:
                self.backend = "plain"
                self.safe_tty = True
                return input(self.prompt(mode, model))
        if self.backend == "plain":
            return input(self.prompt(mode, model))
        self.console.file.flush()
        mode_style = "mode.rag" if mode == "rag" else "mode.chat"
        model_short = _short_model(model, max_len=max(18, min(34, self.console.width - 28)))
        prompt_text = Text()
        prompt_text.append("monkeygrab", style="brand")
        prompt_text.append(" ")
        prompt_text.append(mode, style=mode_style)
        prompt_text.append(" ")
        prompt_text.append(model_short, style="dim")
        prompt_text.append(" > ", style=mode_style)
        return self.console.input(prompt_text)

    # response
    def response_header(self, mode: str, model: str = "") -> None:
        model_short = _short_model(model, max_len=36)
        label = "RAG" if mode == "rag" else "Chat"
        if self.backend != "rich":
            self._print_line()
            self._rule(f"{label} {model_short}", color=self._mode_color(mode))
            self._print_line()
            return
        self.console.print()
        style = "mode.rag" if mode == "rag" else "mode.chat"
        self.console.rule(f"[{style}]{label}[/] [dim]{model_short}[/]", style=style)

    def stream_token(self, token: str) -> None:
        if self.backend != "rich":
            for char in token:
                if char == "*":
                    self._pending_stars += 1
                    continue
                if self._pending_stars:
                    consumed_space = self._flush_pending_stars(char)
                    if consumed_space and char == " ":
                        continue
                self._write_stream_text(char)
                self._stream_line_start = char == "\n"
            return
        self.console.print(token, end="")
        self.console.file.flush()

    def can_stream_responses(self) -> bool:
        return self.backend != "plain"

    def end_stream(self) -> None:
        if self.backend != "rich":
            if self._pending_stars:
                self._flush_pending_stars()
            if self._stream_bold:
                self._write_stream_text(AnsiTheme.RESET + AnsiTheme.TEXT)
                self._stream_bold = False
            self._print_line()
            return
        self.console.print()

    def render_response(self, text: str) -> None:
        if self.backend != "rich":
            body = self._plain_markdown((text or "(sin respuesta)").strip())
            for line in body.splitlines() or ["(sin respuesta)"]:
                self._print_line(f"  {self._ansi(line.rstrip(), AnsiTheme.TEXT)}")
            self._print_line()
            return
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

        if self.backend != "rich":
            self._print_line()
            self._rule("Fuentes", color=AnsiTheme.BRAND)
            for doc, pages in sorted(sources_map.items()):
                self._print_line(
                    f"  {self._ansi(_short_model(doc, max_len=56, keep_tag=True), AnsiTheme.TEXT)} "
                    f"{self._ansi(f'(páginas: {_safe_pages(pages)})', AnsiTheme.MUTED)}"
                )
            return

        table = Table(box=None, show_header=False, pad_edge=False)
        table.add_column("Documento", style="muted", overflow="fold")
        table.add_column("Páginas", style="dim", no_wrap=True)
        for doc, pages in sorted(sources_map.items()):
            table.add_row(_short_model(doc, max_len=56, keep_tag=True), _safe_pages(pages))
        self.console.print(Panel(table, title="[dim]Fuentes[/]", border_style="dim", box=box.ROUNDED))

    def response_footer(self) -> None:
        if self.backend != "rich":
            self._rule(color=AnsiTheme.DIM_TEXT)
            self._print_line()
            return
        self.console.rule(style="dim")
        self.console.print()

    # stats/docs/topics
    def stats_table(self, total_fragments: int, docs: List[str], info: Optional[Dict[str, Any]] = None) -> None:
        info = info or {}
        if self.backend != "rich":
            self._print_line()
            self._rule("Estado", color=AnsiTheme.INFO)
            rows = [
                ("PDFs", info.get("total_documentos", len(docs))),
                ("Documentos indexados", len(docs)),
                ("Fragmentos", total_fragments),
                ("Carpeta", info.get("docs_folder", "-")),
                ("Base vectorial", info.get("path_db", "-")),
                ("Colección", info.get("collection_name", "-")),
            ]
            for key, value in rows:
                self._print_line(f"  {self._ansi(f'{key}:', AnsiTheme.DIM_TEXT)} {self._ansi(str(value), AnsiTheme.TEXT)}")
            self._rule(color=AnsiTheme.DIM_TEXT)
            self._print_line()
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
        if self.backend != "rich":
            self._print_line()
            self._rule("Documentos", color=AnsiTheme.BRAND)
            for idx, item in enumerate(docs, 1):
                if isinstance(item, dict):
                    self._print_line(
                        f"  {self._ansi(f'{idx}.', AnsiTheme.DIM_TEXT)} "
                        f"{self._ansi(item.get('name', '-'), AnsiTheme.TEXT)} "
                        f"{self._ansi(f'(págs: {item.get('pages', '-')}, frag: {item.get('fragments', '-')}, tipos: {item.get('formats', '-')})', AnsiTheme.MUTED)}"
                    )
                else:
                    self._print_line(f"  {self._ansi(f'{idx}.', AnsiTheme.DIM_TEXT)} {self._ansi(str(item), AnsiTheme.TEXT)}")
            self._rule(color=AnsiTheme.DIM_TEXT)
            self._print_line()
            return

        table = Table(title="[info]Documentos[/]", box=box.ROUNDED, border_style="dim", show_header=True, header_style="info", pad_edge=False)
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Documento", style="text", overflow="fold")
        table.add_column("Págs.", style="muted", justify="right", no_wrap=True)
        table.add_column("Frag.", style="muted", justify="right", no_wrap=True)
        table.add_column("Tipos", style="dim", no_wrap=True)
        for idx, item in enumerate(docs, 1):
            if isinstance(item, dict):
                table.add_row(str(idx), item.get("name", "-"), str(item.get("pages", "-")), str(item.get("fragments", "-")), item.get("formats", "-"))
            else:
                table.add_row(str(idx), str(item), "-", "-", "-")
        self.console.print()
        self.console.print(table)
        self.console.print()

    def topics_display(self, docs_data: List[Dict[str, Any]]) -> None:
        if not docs_data:
            self.warning("No hay documentos indexados.")
            return
        if self.backend != "rich":
            self._print_line()
            self._rule("Contenidos", color=AnsiTheme.INFO)
            for doc_info in docs_data:
                self._print_line(
                    f"  {self._ansi(doc_info.get('name', '-'), AnsiTheme.TEXT)} "
                    f"{self._ansi(f'(págs: {doc_info.get('pages', '-')}, frag: {doc_info.get('fragments', '-')})', AnsiTheme.MUTED)}"
                )
                self._print_line(f"    {self._ansi(doc_info.get('terms') or '-', AnsiTheme.DIM_TEXT)}")
            self._print_line(f"  {self._ansi('Escribe una pregunta concreta o usa /docs para revisar el corpus.', AnsiTheme.MUTED)}")
            self._rule(color=AnsiTheme.DIM_TEXT)
            self._print_line()
            return

        table = Table(title="[info]Contenidos[/]", box=box.ROUNDED, border_style="dim", show_header=True, header_style="info", pad_edge=False)
        table.add_column("Documento", style="text", overflow="fold")
        table.add_column("Págs.", style="muted", justify="right", no_wrap=True)
        table.add_column("Frag.", style="muted", justify="right", no_wrap=True)
        table.add_column("Términos frecuentes", style="dim", overflow="fold")
        for doc_info in docs_data:
            table.add_row(doc_info.get("name", "-"), str(doc_info.get("pages", "-")), str(doc_info.get("fragments", "-")), doc_info.get("terms") or "-")
        self.console.print()
        self.console.print(table)
        self.console.print("  [dim]Escribe una pregunta concreta o usa /docs para revisar el corpus.[/]")
        self.console.print()

    # misc
    def mode_change(self, mode: str, model: str = "") -> None:
        if self.backend != "rich":
            purpose = "consulta documental" if mode == "rag" else "conversación libre"
            color = self._mode_color(mode)
            self._rule(color=AnsiTheme.DIM_TEXT)
            self._print_line(f"  {self._ansi('modo:', AnsiTheme.DIM_TEXT)} {self._ansi(mode, color, AnsiTheme.BOLD)}")
            self._print_line(f"  {self._ansi('uso:', AnsiTheme.DIM_TEXT)} {self._ansi(purpose, AnsiTheme.MUTED)}")
            self._print_line(f"  {self._ansi('modelo:', AnsiTheme.DIM_TEXT)} {self._ansi(_short_model(model, 36), AnsiTheme.TEXT)}")
            self._rule(color=AnsiTheme.DIM_TEXT)
            return
        style = "mode.rag" if mode == "rag" else "mode.chat"
        purpose = "consulta documental" if mode == "rag" else "conversación libre"
        self.console.print(f"  [{style}]modo {mode}[/] [dim]{purpose} · {_short_model(model, 36)}[/]")

    def history_loaded(self, n: int) -> None:
        self.info(f"historial restaurado: {n} mensajes")

    def history_cleared(self) -> None:
        self.success("historial limpiado")

    def unknown_command(self, cmd: str) -> None:
        self.warning(f"Comando no reconocido: {cmd} (usa /ayuda)")

    def reindex_start(self) -> None:
        if self.backend != "rich":
            self._print_line()
            self._rule("Reindex", color=AnsiTheme.WARNING)
            self.warning("Reindexando documentos: se reconstruirá la base vectorial.")
            return
        self.console.print()
        self.warning("Reindexando documentos: se reconstruirá la base vectorial.")

    def reindex_complete(self, total: int) -> None:
        self.success(f"Reindexación completada: {total} fragmentos")
        self.warning("Reinicia el programa para usar la nueva base de datos")

    def farewell(self) -> None:
        if self.backend != "rich":
            self._print_line()
            self._rule(color=AnsiTheme.DIM_TEXT)
            self._print_line(f"  {self._ansi('Sesión finalizada.', AnsiTheme.DIM_TEXT)}")
            self._print_line()
            return
        self.console.print()
        self.console.print("  [dim]Sesión finalizada.[/]")
        self.console.print()

    def no_results(self) -> None:
        if self.backend != "rich":
            self._rule("Sin Resultados", color=AnsiTheme.WARNING)
            self._print_line(f"  {self._ansi('No se encontró información relevante en los documentos.', AnsiTheme.TEXT)}")
            self._print_line(f"  {self._ansi('- La información puede no estar indexada', AnsiTheme.MUTED)}")
            self._print_line(f"  {self._ansi('- La pregunta puede necesitar más detalle', AnsiTheme.MUTED)}")
            self._print_line(f"  {self._ansi('- El tema puede estar fuera del corpus', AnsiTheme.MUTED)}")
            self._print_line(f"  {self._ansi('Prueba con /temas o reformula la consulta.', AnsiTheme.DIM_TEXT)}")
            self._rule(color=AnsiTheme.DIM_TEXT)
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
        if self.backend != "rich":
            self._rule("Fuera de Ámbito", color=AnsiTheme.WARNING)
            self.warning(f"Pregunta fuera de ámbito (score {score:.4f} < {threshold})")
            self._print_line(f"     {self._ansi('Usa /temas para explorar el corpus.', AnsiTheme.DIM_TEXT)}")
            self._rule(color=AnsiTheme.DIM_TEXT)
            return
        self.warning(f"Pregunta fuera de ámbito (score {score:.4f} < {threshold})")
        self.console.print("     [dim]Usa /temas para explorar el corpus.[/]")

    def question_too_short(self) -> None:
        if self.backend != "rich":
            self._print_line(f"  {self._ansi('Pregunta demasiado corta. Formula una pregunta concreta o usa /chat.', AnsiTheme.DIM_TEXT)}")
            return
        self.console.print("  [dim]Pregunta demasiado corta. Formula una pregunta concreta o usa /chat.[/]")

    def no_pdfs(self, folder: str) -> None:
        self.warning(f"No existe la carpeta de PDFs o está vacía: {folder}")


ui = Display()
