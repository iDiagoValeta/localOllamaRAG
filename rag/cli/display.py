"""
MonkeyGrab CLI display backends.

Rich is the default renderer outside Windows. On Windows the interactive path
uses prompt_toolkit + ANSI output to keep the prompt stable while still
providing a colored CLI. Plain fallback is used when stdout is not a TTY (pipes,
non-interactive scripts).

All user-facing terminal output in the CLI goes through the ``ui`` singleton at
the bottom of this file, which ensures a single visual language across the
three backends.
"""

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  CONFIGURATION
#  +-- 1. Imports
#  +-- 2. Palette (single source of truth)
#  +-- 3. Module-level helpers
#
#  DATA CLASSES
#  +-- 4. QueryTimer
#  +-- 5. SessionStats
#
#  DISPLAY CLASS
#  +-- 6. __init__ + backend selection
#  +-- 7. Primitives (lines, rules, themed tables)
#  +-- 8. Branding (logo, init_panel, welcome, farewell)
#  +-- 9. Status messages (success/warning/error/info/debug)
#  +-- 10. Pipeline feedback (phases, query_summary)
#  +-- 11. Input (prompt, autocompletion, persistent history)
#  +-- 12. Response streaming
#  +-- 13. Stats / Docs / Topics
#  +-- 14. Edge-case messages (no_pdfs, out_of_scope, ...)
#
#  ENTRY
#  +-- 15. ui singleton
#
# ─────────────────────────────────────────────

from __future__ import annotations

# ─────────────────────────────────────────────
# SECTION 1: IMPORTS
# ─────────────────────────────────────────────

import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text
from rich.theme import Theme as RichTheme

from rag.cli.commands import COMMANDS, all_command_names

try:
    from colorama import just_fix_windows_console
except ImportError:  # pragma: no cover - dependency expected at runtime
    just_fix_windows_console = None

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.shortcuts import print_formatted_text
    from prompt_toolkit.shortcuts import prompt as ptk_prompt

    PTK_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency expected at runtime
    ANSI = None
    Completer = object  # type: ignore[assignment,misc]
    Completion = None  # type: ignore[assignment]
    FileHistory = None  # type: ignore[assignment]
    PromptSession = None  # type: ignore[assignment]
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


# ─────────────────────────────────────────────
# SECTION 2: PALETTE (single source of truth)
# ─────────────────────────────────────────────


@dataclass(frozen=True)
class _Color:
    """A color defined once, rendered in Rich (hex) and ANSI 256 consistently."""

    name: str
    hex: str
    ansi256: int
    bold: bool = False

    @property
    def rich_style(self) -> str:
        return f"{'bold ' if self.bold else ''}{self.hex}"

    @property
    def ansi(self) -> str:
        return f"\033[38;5;{self.ansi256}m"


class Palette:
    """Colors used across the CLI. One source, two consumers (Rich + ANSI)."""

    BRAND = _Color("brand", "#d7af5f", 179, bold=True)
    BRAND_DIM = _Color("brand.dim", "#af875f", 137)
    CHAT = _Color("mode.chat", "#87afd7", 110)
    RAG = _Color("mode.rag", "#5fafaf", 79)
    INFO = _Color("info", "#87afd7", 110)
    SUCCESS = _Color("success", "#87af5f", 71)
    WARNING = _Color("warning", "#d7af5f", 179)
    ERROR = _Color("error", "#d75f5f", 167)
    TEXT = _Color("text", "#dadada", 252)
    MUTED = _Color("muted", "#a8a8a8", 246)
    DIM = _Color("dim", "#6c6c6c", 240)
    OFF = _Color("off", "#606060", 242)

    @classmethod
    def all(cls) -> List[_Color]:
        return [c for c in cls.__dict__.values() if isinstance(c, _Color)]


def _build_rich_theme() -> RichTheme:
    return RichTheme({c.name: c.rich_style for c in Palette.all()})


MONKEYGRAB_THEME = _build_rich_theme()

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"


# ─────────────────────────────────────────────
# SECTION 3: MODULE-LEVEL HELPERS
# ─────────────────────────────────────────────


BackendName = Literal["rich", "prompt_toolkit", "plain"]


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


def _format_duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes}m{secs:0.0f}s"


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


# ─────────────────────────────────────────────
# SECTION 4: QueryTimer
# ─────────────────────────────────────────────


class QueryTimer:
    """Records wall-clock durations of the phases inside a RAG query.

    Usage::

        t = QueryTimer()
        t.mark("search")     # after hybrid search returns
        t.mark("rerank")     # after reranker
        t.mark("generate")   # after generation
        t.phase_durations()  # -> [("search", 0.8), ("rerank", 0.3), ...]
        t.total              # -> 2.4

    Each ``mark(name)`` closes the previous phase; the first ``mark`` opens the
    first phase relative to construction time.
    """

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._checkpoints: List[Tuple[str, float]] = []

    def mark(self, name: str) -> None:
        self._checkpoints.append((name, time.perf_counter() - self._start))

    def phase_durations(self) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        prev = 0.0
        for name, t in self._checkpoints:
            out.append((name, max(0.0, t - prev)))
            prev = t
        return out

    @property
    def total(self) -> float:
        return time.perf_counter() - self._start


# ─────────────────────────────────────────────
# SECTION 5: SessionStats
# ─────────────────────────────────────────────


@dataclass
class SessionStats:
    """Aggregate counters shown in the farewell summary."""

    start: float = field(default_factory=time.time)
    rag_queries: int = 0
    chat_queries: int = 0
    rag_time: float = 0.0
    chat_time: float = 0.0
    models_used: set = field(default_factory=set)

    def tick_rag(self, elapsed: float, model: Optional[str] = None) -> None:
        self.rag_queries += 1
        self.rag_time += elapsed
        if model:
            self.models_used.add(model)

    def tick_chat(self, elapsed: float, model: Optional[str] = None) -> None:
        self.chat_queries += 1
        self.chat_time += elapsed
        if model:
            self.models_used.add(model)

    @property
    def total_queries(self) -> int:
        return self.rag_queries + self.chat_queries

    @property
    def duration(self) -> float:
        return time.time() - self.start


# ─────────────────────────────────────────────
# SECTION 6: DISPLAY CLASS — INIT + BACKEND
# ─────────────────────────────────────────────


if PTK_AVAILABLE:

    class _SlashCompleter(Completer):  # type: ignore[misc]
        """Complete slash-commands only when the line starts with '/'.

        Avoids noisy popup suggestions during regular RAG/chat prompts.
        """

        def __init__(self, commands: List[str]) -> None:
            self._commands = sorted(set(commands))

        def get_completions(self, document, complete_event):  # type: ignore[override]
            text = document.text_before_cursor
            if not text.startswith("/"):
                return
            for cmd in self._commands:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))


class Display:
    """Centralized terminal interface manager for MonkeyGrab."""

    def __init__(self) -> None:
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
        self._ptk_session = self._build_ptk_session()

    def _build_ptk_session(self):
        """Create a PromptSession with persistent history + slash completer."""
        if not PTK_AVAILABLE or self.backend not in {"prompt_toolkit", "rich"}:
            return None
        try:
            history_path = os.environ.get(
                "MONKEYGRAB_HISTORY",
                os.path.join(os.path.expanduser("~"), ".monkeygrab_history"),
            )
            parent = os.path.dirname(history_path)
            if parent and not os.path.isdir(parent):
                os.makedirs(parent, exist_ok=True)
            return PromptSession(
                history=FileHistory(history_path),
                completer=_SlashCompleter(all_command_names()),
                complete_while_typing=False,
            )
        except Exception:
            return None

    # ─────────────────────────────────────────────
    # SECTION 7: PRIMITIVES
    # ─────────────────────────────────────────────

    def _ansi(self, text: str, *styles: str) -> str:
        return f"{''.join(styles)}{text}{ANSI_RESET}"

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

    def _rule(self, title: str = "", char: str = "─", color: Optional[_Color] = None) -> None:
        width = max(40, shutil.get_terminal_size(fallback=(88, 24)).columns)
        ansi = (color or Palette.DIM).ansi
        if not title:
            self._print_line(self._ansi(char * width, ansi))
            return
        label = f" {title} "
        side = max(2, (width - len(label)) // 2)
        rule = f"{char * side}{label}{char * max(2, width - len(label) - side)}"
        self._print_line(self._ansi(rule[:width], ansi))

    def _status_prefix(self, color: _Color, label: str, msg: str) -> None:
        self._print_line(
            f"  {self._ansi(label, color.ansi, ANSI_BOLD)} {self._ansi(msg, Palette.MUTED.ansi)}"
        )

    def _mode_color(self, mode: str) -> _Color:
        return Palette.RAG if mode == "rag" else Palette.CHAT

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

    def _themed_table(
        self,
        title: Optional[str] = None,
        *,
        header: bool = True,
        box_style=box.ROUNDED,
    ) -> Table:
        """Return a Table preconfigured with the MonkeyGrab visual language."""
        return Table(
            title=f"[info]{title}[/]" if title else None,
            box=box_style,
            border_style="dim",
            show_header=header,
            header_style="info",
            pad_edge=False,
        )

    # ─────────────────────────────────────────────
    # SECTION 8: BRANDING
    # ─────────────────────────────────────────────

    def logo(self) -> None:
        if self.backend == "rich":
            title = Text()
            title.append("Monkey", style="brand")
            title.append("Grab", style="brand.dim")
            title.append("  ·  ", style="dim")
            title.append("local PDF RAG", style="muted")
            self.console.rule(title, style="brand.dim")
            self.console.print(
                "  [dim]TFG · ETSINF · UPV · 2025-2026[/]",
                highlight=False,
            )
            return
        self._rule("MonkeyGrab · local PDF RAG", "═", Palette.BRAND)
        self._print_line(
            f"  {self._ansi('TFG · ETSINF · UPV · 2025-2026', Palette.DIM.ansi)}"
        )

    def init_panel(self, info: Dict[str, Any]) -> None:
        mode = info.get("mode", "chat")
        mode_style = "mode.rag" if mode == "rag" else "mode.chat"

        if self.backend != "rich":
            self._init_panel_ansi(info, mode)
            return

        corpus = Table.grid(padding=(0, 2))
        corpus.add_column(style="dim", no_wrap=True)
        corpus.add_column(style="text")
        corpus.add_row("PDFs", str(info.get("total_documentos", 0)))
        corpus.add_row("Fragmentos", str(info.get("total_fragmentos", 0)))
        corpus.add_row("Carpeta", str(info.get("docs_folder", "-")))
        corpus.add_row("Colección", str(info.get("collection_name", "-")))

        details = Table.grid(padding=(0, 2))
        details.add_column(style="dim", no_wrap=True)
        details.add_column(style="muted")
        details.add_row("Extractor", str(info.get("extractor", "-")))
        details.add_row("Búsqueda", str(info.get("busqueda", "-")))
        rr = "off"
        if info.get("reranker") == "on":
            rr = f"{info.get('reranker_model', '-')}, {info.get('reranker_device', '-')}"
        details.add_row("Reranker", rr)
        details.add_row(
            "Chunks",
            f"{info.get('chunk_size', '-')}c, overlap {info.get('chunk_overlap', '-')}c",
        )

        models = self._themed_table(header=True, box_style=box.SIMPLE_HEAVY)
        models.add_column("Rol", style="dim", width=12, no_wrap=True)
        models.add_column("Modelo", style="text", overflow="ellipsis")
        models.add_column("Tag", style="muted", width=12, overflow="ellipsis")
        models.add_column("Uso", style="muted", width=12, no_wrap=True)
        model_width = 44 if self.console.width >= 110 else 28
        model_rows = [
            ("RAG", info.get("modelo_rag", ""), "respuestas"),
            ("Chat", info.get("modelo_chat", ""), "general"),
            ("Embeddings", info.get("modelo_embedding", ""), "índice"),
            (
                "Contextual",
                info.get("modelo_contextual", ""),
                "indexación" if info.get("contextual") else "off",
            ),
            (
                "RECOMP",
                info.get("modelo_recomp", ""),
                "síntesis" if info.get("recomp") else "off",
            ),
            (
                "OCR",
                info.get("modelo_ocr", ""),
                "imágenes" if info.get("images") else "off",
            ),
        ]
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
        cells: List[Text] = []
        for name, enabled in flag_items:
            cell = Text()
            cell.append(name, style="dim")
            cell.append("=")
            cell.append_text(_state_label(enabled))
            cells.append(cell)
        for idx in range(0, len(cells), 3):
            flags.add_row(*cells[idx : idx + 3])

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
        self.console.print(
            Panel(flags, title="[info]Flags[/]", border_style="dim", box=box.ROUNDED)
        )
        self.console.print()

    def _init_panel_ansi(self, info: Dict[str, Any], mode: str) -> None:
        mode_color = self._mode_color(mode)
        self._print_line(
            f"{self._ansi('Modo inicial:', Palette.DIM.ansi)} "
            f"{self._ansi(mode, mode_color.ansi, ANSI_BOLD)}"
            f"{self._ansi('  |  usa /ayuda para comandos', Palette.DIM.ansi)}"
        )
        self._rule(color=Palette.DIM)
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
                f"  {self._ansi(f'{key}:', Palette.DIM.ansi)} "
                f"{self._ansi(str(value), Palette.TEXT.ansi)}"
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
        self._print_line(
            f"  {self._ansi('Flags:', Palette.DIM.ansi)} {self._ansi(flags, Palette.MUTED.ansi)}"
        )
        self._rule(color=Palette.DIM)
        self._print_line()

    def welcome(self) -> None:
        if self.backend != "rich":
            self._print_line()
            self._rule("Modos", color=Palette.INFO)
            self._print_line(
                f"  {self._ansi('CHAT', Palette.CHAT.ansi, ANSI_BOLD)} "
                f"{self._ansi('Conversación libre con historial.', Palette.MUTED.ansi)}"
            )
            self._print_line(
                f"  {self._ansi('RAG', Palette.RAG.ansi, ANSI_BOLD)}  "
                f"{self._ansi('Consulta los PDFs indexados y muestra fuentes.', Palette.MUTED.ansi)}"
            )
            self._print_line()
            self._rule("Comandos", color=Palette.BRAND)
            for cmd, desc in COMMANDS:
                self._print_line(
                    f"  {self._ansi(f'{cmd:<12}', Palette.BRAND.ansi, ANSI_BOLD)} "
                    f"{self._ansi(desc, Palette.TEXT.ansi)}"
                )
            self._print_line(
                f"  {self._ansi('Atajos:', Palette.DIM.ansi)} "
                f"{self._ansi('flechas ↑/↓ = historial · Tab sobre / = autocompletar comandos', Palette.MUTED.ansi)}"
            )
            self._rule(color=Palette.DIM)
            self._print_line()
            return

        self.console.print()
        modes = Table.grid(padding=(0, 2))
        modes.add_column(no_wrap=True)
        modes.add_column(style="muted")
        modes.add_row("[mode.chat]CHAT[/]", "Conversación libre con historial.")
        modes.add_row("[mode.rag]RAG[/]", "Consulta los PDFs indexados y muestra fuentes.")

        commands = self._themed_table(title="Comandos", header=False)
        commands.add_column("Comando", style="brand", width=12, no_wrap=True)
        commands.add_column("Descripción", style="muted")
        for cmd, desc in COMMANDS:
            commands.add_row(cmd, desc)

        shortcuts = Table.grid(padding=(0, 2))
        shortcuts.add_column(style="dim", no_wrap=True)
        shortcuts.add_column(style="muted")
        shortcuts.add_row("↑ / ↓", "Recuperar consultas anteriores del historial.")
        shortcuts.add_row("Tab", "Autocompletar sobre un comando que empiece por '/'.")
        shortcuts.add_row("Ctrl-C / Ctrl-D", "Salir guardando el historial.")

        self.console.print(
            Panel(modes, title="[info]Modos[/]", border_style="dim", box=box.ROUNDED)
        )
        self.console.print(commands)
        self.console.print(
            Panel(shortcuts, title="[info]Atajos[/]", border_style="dim", box=box.ROUNDED)
        )
        self.console.print()

    # ─────────────────────────────────────────────
    # SECTION 9: STATUS MESSAGES
    # ─────────────────────────────────────────────

    def success(self, msg: str) -> None:
        if self.backend != "rich":
            self._status_prefix(Palette.SUCCESS, "✓", msg)
            return
        self.console.print(f"  [success]✓[/] [muted]{msg}[/]")

    def warning(self, msg: str) -> None:
        if self.backend != "rich":
            self._status_prefix(Palette.WARNING, "!", msg)
            return
        self.console.print(f"  [warning]![/] [muted]{msg}[/]")

    def error(self, msg: str) -> None:
        if self.backend != "rich":
            self._status_prefix(Palette.ERROR, "✗", msg)
            return
        self.console.print(f"  [error]✗[/] [muted]{msg}[/]")

    def info(self, msg: str) -> None:
        if self.backend != "rich":
            self._status_prefix(Palette.INFO, "·", msg)
            return
        self.console.print(f"  [info]·[/] [dim]{msg}[/]")

    def debug(self, msg: str) -> None:
        if not self._debug_mode:
            return
        if self.backend != "rich":
            self._status_prefix(Palette.DIM, "debug", msg)
            return
        self.console.print(f"  [dim]⊡ {msg}[/]")

    def exception(self, title: str, exc: Exception) -> None:
        detail = str(exc).strip() or exc.__class__.__name__
        advice = "Comprueba que Ollama esté activo y que el modelo exista localmente."
        if self.backend != "rich":
            self._print_line()
            self._rule(title, color=Palette.ERROR)
            self._print_line(f"  {self._ansi(detail, Palette.TEXT.ansi)}")
            self._print_line(f"  {self._ansi(advice, Palette.DIM.ansi)}")
            self._rule(color=Palette.DIM)
            return
        panel = Panel(
            f"[muted]{detail}[/]\n\n[dim]{advice}[/]",
            title=f"[error]{title}[/]",
            border_style="error",
            box=box.ROUNDED,
            expand=False,
        )
        self.console.print(panel)

    def ollama_status(self, ok: bool, detail: str) -> None:
        """Render the Ollama health check result at startup."""
        if ok:
            self.success(detail)
            return
        if self.backend != "rich":
            self._print_line()
            self._rule("Ollama", color=Palette.WARNING)
            self._print_line(f"  {self._ansi(detail, Palette.TEXT.ansi)}")
            self._print_line(
                f"  {self._ansi('Arranca el servidor con `ollama serve` antes de consultar.', Palette.DIM.ansi)}"
            )
            self._rule(color=Palette.DIM)
            return
        panel = Panel(
            f"[muted]{detail}[/]\n\n"
            "[dim]Arranca el servidor con `ollama serve` antes de consultar.[/]",
            title="[warning]Ollama[/]",
            border_style="warning",
            box=box.ROUNDED,
            expand=False,
        )
        self.console.print(panel)

    # ─────────────────────────────────────────────
    # SECTION 10: PIPELINE FEEDBACK
    # ─────────────────────────────────────────────

    def pipeline_start(self, message: str = "Buscando en documentos...") -> Optional[Status]:
        self.pipeline_stop()
        if self.backend != "rich":
            self.info(message)
            return None
        self._status = self.console.status(
            f"[info]{message}[/]", spinner="dots", spinner_style="info"
        )
        self._status.start()
        return self._status

    def pipeline_phase(self, name: str, message: str = "") -> None:
        """Update the active spinner with a labelled phase.

        ``name`` is the short phase tag (e.g. "búsqueda", "rerank") and
        ``message`` is optional extra context.
        """
        label = name.strip().capitalize() if name else ""
        if message:
            line = f"▸ {label} · {message}" if label else f"▸ {message}"
        else:
            line = f"▸ {label}" if label else "▸"

        if self.backend != "rich":
            self.info(line)
            return
        if self._status:
            self._status.update(f"[info]{line}[/]")
        else:
            self.console.print(f"  [info]{line}[/]")

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

    def query_summary(
        self,
        timer: QueryTimer,
        *,
        n_fragmentos: int,
        best_score: Optional[float] = None,
    ) -> None:
        """Compact one-liner with query timings and retrieval metrics."""
        phases = timer.phase_durations()
        if self.backend != "rich":
            bits = [f"total {_format_duration(timer.total)}"]
            for name, dur in phases:
                if dur >= 0.05:
                    bits.append(f"{name} {_format_duration(dur)}")
            bits.append(f"{n_fragmentos} frag.")
            if best_score is not None:
                bits.append(f"score {best_score:.2f}")
            line = " · ".join(bits)
            self._print_line(
                f"  {self._ansi('·', Palette.INFO.ansi)} {self._ansi(line, Palette.DIM.ansi)}"
            )
            return

        text = Text()
        text.append("·  ", style="info")
        text.append("total ", style="dim")
        text.append(_format_duration(timer.total), style="muted")
        for name, dur in phases:
            if dur < 0.05:
                continue
            text.append("  ·  ", style="dim")
            text.append(f"{name} ", style="dim")
            text.append(_format_duration(dur), style="muted")
        text.append("  ·  ", style="dim")
        text.append(f"{n_fragmentos} frag.", style="muted")
        if best_score is not None:
            text.append("  ·  ", style="dim")
            text.append(f"score {best_score:.2f}", style="muted")
        self.console.print(text)

    # ─────────────────────────────────────────────
    # SECTION 11: INPUT
    # ─────────────────────────────────────────────

    def prompt(self, mode: str, model: str = "") -> str:
        model_short = _short_model(model, max_len=22, keep_tag=False)
        return f"monkeygrab {mode} {model_short} ▸ "

    def _ansi_prompt(self, mode: str, model: str = "") -> str:
        model_short = _short_model(model, max_len=22, keep_tag=False)
        mode_color = self._mode_color(mode)
        return (
            f"{Palette.BRAND.ansi}{ANSI_BOLD}monkeygrab{ANSI_RESET} "
            f"{mode_color.ansi}{ANSI_BOLD}{mode}{ANSI_RESET} "
            f"{Palette.DIM.ansi}{model_short}{ANSI_RESET} "
            f"{mode_color.ansi}{ANSI_BOLD}▸ {ANSI_RESET}"
        )

    def read_input(self, mode: str, model: str = "") -> str:
        self.pipeline_stop()
        if self.backend == "prompt_toolkit":
            return self._read_input_ptk(mode, model)
        if self.backend == "plain":
            return input(self.prompt(mode, model))
        # Rich: prefer PTK session if available (for history + completion),
        # otherwise fall back to Rich's own input which lacks those features.
        self.console.file.flush()
        if self._ptk_session is not None:
            return self._read_input_ptk(mode, model)
        mode_style = "mode.rag" if mode == "rag" else "mode.chat"
        model_short = _short_model(
            model, max_len=max(18, min(34, self.console.width - 28))
        )
        prompt_text = Text()
        prompt_text.append("monkeygrab", style="brand")
        prompt_text.append(" ")
        prompt_text.append(mode, style=mode_style)
        prompt_text.append(" ")
        prompt_text.append(model_short, style="dim")
        prompt_text.append(" ▸ ", style=mode_style)
        return self.console.input(prompt_text)

    def _read_input_ptk(self, mode: str, model: str) -> str:
        try:
            if self._ptk_session is not None:
                return self._ptk_session.prompt(ANSI(self._ansi_prompt(mode, model)))
            return ptk_prompt(ANSI(self._ansi_prompt(mode, model)))
        except Exception:
            self.backend = "plain"
            self.safe_tty = True
            return input(self.prompt(mode, model))

    # ─────────────────────────────────────────────
    # SECTION 12: RESPONSE STREAMING
    # ─────────────────────────────────────────────

    def begin_stream(self) -> None:
        self._stream_bold = False
        self._stream_line_start = True
        self._pending_stars = 0

    def _flush_pending_stars(self, next_char: str = "") -> bool:
        consumed_space = False
        while self._pending_stars >= 2:
            self._stream_bold = not self._stream_bold
            self._write_stream_text(
                ANSI_BOLD if self._stream_bold else ANSI_RESET + Palette.TEXT.ansi
            )
            self._pending_stars -= 2
        if self._pending_stars == 1:
            if self._stream_line_start and next_char == " ":
                self._write_stream_text(
                    f"{Palette.MUTED.ansi}• {Palette.TEXT.ansi}"
                )
                self._stream_line_start = False
                consumed_space = True
            else:
                self._write_stream_text("*")
            self._pending_stars = 0
        return consumed_space

    def response_header(self, mode: str, model: str = "") -> None:
        model_short = _short_model(model, max_len=36)
        label = "RAG" if mode == "rag" else "Chat"
        if self.backend != "rich":
            self._print_line()
            self._rule(f"{label} · {model_short}", color=self._mode_color(mode))
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
                self._write_stream_text(ANSI_RESET + Palette.TEXT.ansi)
                self._stream_bold = False
            self._print_line()
            return
        self.console.print()

    def render_response(self, text: str) -> None:
        if self.backend != "rich":
            body = self._plain_markdown((text or "(sin respuesta)").strip())
            for line in body.splitlines() or ["(sin respuesta)"]:
                self._print_line(f"  {self._ansi(line.rstrip(), Palette.TEXT.ansi)}")
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
            self._rule(f"Fuentes ({len(sources_map)})", color=Palette.BRAND)
            for doc, pages in sorted(sources_map.items()):
                self._print_line(
                    f"  {self._ansi(_short_model(doc, max_len=56, keep_tag=True), Palette.TEXT.ansi)} "
                    f"{self._ansi(f'(páginas: {_safe_pages(pages)})', Palette.MUTED.ansi)}"
                )
            return

        table = self._themed_table(header=False, box_style=None)
        table.add_column("Documento", style="muted", overflow="fold")
        table.add_column("Páginas", style="dim", no_wrap=True)
        for doc, pages in sorted(sources_map.items()):
            table.add_row(_short_model(doc, max_len=56, keep_tag=True), _safe_pages(pages))
        self.console.print(
            Panel(
                table,
                title=f"[dim]Fuentes ({len(sources_map)})[/]",
                border_style="dim",
                box=box.ROUNDED,
            )
        )

    def response_footer(self, sources: Optional[int] = None) -> None:
        if self.backend != "rich":
            if sources is not None:
                self._print_line(
                    f"  {self._ansi('·', Palette.DIM.ansi)} "
                    f"{self._ansi(f'{sources} fuentes citadas' if sources else 'sin fuentes', Palette.DIM.ansi)}"
                )
            self._rule(color=Palette.DIM)
            self._print_line()
            return
        if sources is not None:
            tag = f"{sources} fuentes citadas" if sources else "sin fuentes"
            self.console.print(f"  [dim]· {tag}[/]")
        self.console.rule(style="dim")
        self.console.print()

    # ─────────────────────────────────────────────
    # SECTION 13: STATS / DOCS / TOPICS
    # ─────────────────────────────────────────────

    def stats_table(
        self,
        total_fragments: int,
        docs: List[str],
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        info = info or {}
        rows = [
            ("PDFs", info.get("total_documentos", len(docs))),
            ("Documentos indexados", len(docs)),
            ("Fragmentos", total_fragments),
            ("Carpeta", info.get("docs_folder", "-")),
            ("Base vectorial", info.get("path_db", "-")),
            ("Colección", info.get("collection_name", "-")),
        ]

        if self.backend != "rich":
            self._print_line()
            self._rule("Estado", color=Palette.INFO)
            for key, value in rows:
                self._print_line(
                    f"  {self._ansi(f'{key}:', Palette.DIM.ansi)} "
                    f"{self._ansi(str(value), Palette.TEXT.ansi)}"
                )
            self._rule(color=Palette.DIM)
            self._print_line()
            return

        table = self._themed_table(title="Estado")
        table.add_column("Métrica", style="dim", no_wrap=True)
        table.add_column("Valor", style="text", overflow="fold")
        for key, value in rows:
            table.add_row(key, str(value))
        self.console.print()
        self.console.print(table)
        self.console.print()

    def docs_table(self, docs: List[Any]) -> None:
        if not docs:
            self.warning("No hay documentos indexados.")
            return
        if self.backend != "rich":
            self._print_line()
            self._rule("Documentos", color=Palette.BRAND)
            for idx, item in enumerate(docs, 1):
                if isinstance(item, dict):
                    descr = (
                        f"(págs: {item.get('pages', '-')}, "
                        f"frag: {item.get('fragments', '-')}, "
                        f"tipos: {item.get('formats', '-')})"
                    )
                    self._print_line(
                        f"  {self._ansi(f'{idx}.', Palette.DIM.ansi)} "
                        f"{self._ansi(item.get('name', '-'), Palette.TEXT.ansi)} "
                        f"{self._ansi(descr, Palette.MUTED.ansi)}"
                    )
                else:
                    self._print_line(
                        f"  {self._ansi(f'{idx}.', Palette.DIM.ansi)} "
                        f"{self._ansi(str(item), Palette.TEXT.ansi)}"
                    )
            self._rule(color=Palette.DIM)
            self._print_line()
            return

        table = self._themed_table(title="Documentos")
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
        tip = "Escribe una pregunta concreta o usa /docs para revisar el corpus."

        if self.backend != "rich":
            self._print_line()
            self._rule("Contenidos", color=Palette.INFO)
            for doc_info in docs_data:
                descr = (
                    f"(págs: {doc_info.get('pages', '-')}, "
                    f"frag: {doc_info.get('fragments', '-')})"
                )
                self._print_line(
                    f"  {self._ansi(doc_info.get('name', '-'), Palette.TEXT.ansi)} "
                    f"{self._ansi(descr, Palette.MUTED.ansi)}"
                )
                self._print_line(
                    f"    {self._ansi(doc_info.get('terms') or '-', Palette.DIM.ansi)}"
                )
            self._print_line(f"  {self._ansi(tip, Palette.MUTED.ansi)}")
            self._rule(color=Palette.DIM)
            self._print_line()
            return

        table = self._themed_table(title="Contenidos")
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
        self.console.print(f"  [dim]{tip}[/]")
        self.console.print()

    # ─────────────────────────────────────────────
    # SECTION 14: EDGE-CASE MESSAGES + MODE CHANGE
    # ─────────────────────────────────────────────

    def mode_change(self, mode: str, model: str = "") -> None:
        purpose = "consulta documental" if mode == "rag" else "conversación libre"
        if self.backend != "rich":
            color = self._mode_color(mode)
            self._rule(color=Palette.DIM)
            self._print_line(
                f"  {self._ansi('modo:', Palette.DIM.ansi)} "
                f"{self._ansi(mode, color.ansi, ANSI_BOLD)}"
            )
            self._print_line(
                f"  {self._ansi('uso:', Palette.DIM.ansi)} "
                f"{self._ansi(purpose, Palette.MUTED.ansi)}"
            )
            self._print_line(
                f"  {self._ansi('modelo:', Palette.DIM.ansi)} "
                f"{self._ansi(_short_model(model, 36), Palette.TEXT.ansi)}"
            )
            self._rule(color=Palette.DIM)
            return
        style = "mode.rag" if mode == "rag" else "mode.chat"
        self.console.print(
            f"  [{style}]modo {mode}[/] [dim]{purpose} · {_short_model(model, 36)}[/]"
        )

    def history_loaded(self, n: int) -> None:
        self.info(f"historial restaurado: {n} mensajes")

    def history_cleared(self) -> None:
        self.success("historial limpiado")

    def unknown_command(self, cmd: str) -> None:
        self.warning(f"Comando no reconocido: {cmd} (usa /ayuda)")

    def reindex_start(self) -> None:
        if self.backend != "rich":
            self._print_line()
            self._rule("Reindex", color=Palette.WARNING)
            self.warning("Reindexando documentos: se reconstruirá la base vectorial.")
            return
        self.console.print()
        self.warning("Reindexando documentos: se reconstruirá la base vectorial.")

    def reindex_complete(self, total: int) -> None:
        self.success(f"Reindexación completada: {total} fragmentos")
        self.warning("Reinicia el programa para usar la nueva base de datos")

    def farewell(self, stats: Optional[SessionStats] = None) -> None:
        if stats is not None and stats.total_queries > 0:
            self._session_summary(stats)
        if self.backend != "rich":
            self._print_line()
            self._rule(color=Palette.DIM)
            self._print_line(f"  {self._ansi('Sesión finalizada.', Palette.DIM.ansi)}")
            self._print_line()
            return
        self.console.print()
        self.console.print("  [dim]Sesión finalizada.[/]")
        self.console.print()

    def _session_summary(self, stats: SessionStats) -> None:
        models = ", ".join(sorted(_short_model(m, max_len=24) for m in stats.models_used)) or "-"
        rows = [
            ("Duración", _format_duration(stats.duration)),
            ("Consultas RAG", f"{stats.rag_queries} · {_format_duration(stats.rag_time)}"),
            ("Consultas CHAT", f"{stats.chat_queries} · {_format_duration(stats.chat_time)}"),
            ("Modelos usados", models),
        ]
        if self.backend != "rich":
            self._print_line()
            self._rule("Resumen de sesión", color=Palette.INFO)
            for key, value in rows:
                self._print_line(
                    f"  {self._ansi(f'{key}:', Palette.DIM.ansi)} "
                    f"{self._ansi(str(value), Palette.TEXT.ansi)}"
                )
            self._rule(color=Palette.DIM)
            return
        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim", no_wrap=True)
        table.add_column(style="text")
        for key, value in rows:
            table.add_row(key, str(value))
        self.console.print()
        self.console.print(
            Panel(
                table,
                title="[info]Resumen de sesión[/]",
                border_style="dim",
                box=box.ROUNDED,
                expand=False,
            )
        )

    def no_results(self) -> None:
        if self.backend != "rich":
            self._rule("Sin resultados", color=Palette.WARNING)
            self._print_line(
                f"  {self._ansi('No se encontró información relevante en los documentos.', Palette.TEXT.ansi)}"
            )
            self._print_line(
                f"  {self._ansi('- La información puede no estar indexada', Palette.MUTED.ansi)}"
            )
            self._print_line(
                f"  {self._ansi('- La pregunta puede necesitar más detalle', Palette.MUTED.ansi)}"
            )
            self._print_line(
                f"  {self._ansi('- El tema puede estar fuera del corpus', Palette.MUTED.ansi)}"
            )
            self._print_line(
                f"  {self._ansi('Prueba con /temas o reformula la consulta.', Palette.DIM.ansi)}"
            )
            self._rule(color=Palette.DIM)
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
            self._rule("Fuera de ámbito", color=Palette.WARNING)
            self.warning(f"Pregunta fuera de ámbito (score {score:.4f} < {threshold})")
            self._print_line(
                f"     {self._ansi('Usa /temas para explorar el corpus.', Palette.DIM.ansi)}"
            )
            self._rule(color=Palette.DIM)
            return
        self.warning(f"Pregunta fuera de ámbito (score {score:.4f} < {threshold})")
        self.console.print("     [dim]Usa /temas para explorar el corpus.[/]")

    def question_too_short(self) -> None:
        msg = "Pregunta demasiado corta. Formula una pregunta concreta o usa /chat."
        if self.backend != "rich":
            self._print_line(f"  {self._ansi(msg, Palette.DIM.ansi)}")
            return
        self.console.print(f"  [dim]{msg}[/]")

    def no_pdfs(self, folder: str) -> None:
        self.warning(f"No existe la carpeta de PDFs o está vacía: {folder}")


# ─────────────────────────────────────────────
# SECTION 15: ENTRY (singleton)
# ─────────────────────────────────────────────

ui = Display()
