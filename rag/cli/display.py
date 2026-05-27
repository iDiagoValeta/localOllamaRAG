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
#  +-- 14. Edge-case messages (no_pdfs, question_too_short, ...)
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

from rag.cli.commands import ALIASES, COMMANDS, primary_commands

try:
    from colorama import just_fix_windows_console
except ImportError:  # pragma: no cover - dependency expected at runtime
    just_fix_windows_console = None

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion, FuzzyWordCompleter
    from prompt_toolkit.formatted_text import ANSI, HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.shortcuts import print_formatted_text
    from prompt_toolkit.shortcuts import prompt as ptk_prompt

    PTK_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency expected at runtime
    ANSI = None
    Completer = object  # type: ignore[assignment,misc]
    Completion = None  # type: ignore[assignment]
    FuzzyWordCompleter = None  # type: ignore[assignment]
    HTML = None  # type: ignore[assignment]
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


def _state_label(enabled: Any, on_label: str = "on", off_label: str = "off") -> Text:
    text = Text()
    if _coerce_bool(enabled):
        text.append(on_label, style="success")
    else:
        text.append(off_label, style="off")
    return text


def _safe_pages(pages: Iterable[Any]) -> str:
    values = sorted({p for p in pages if isinstance(p, int)})
    if not values:
        return "-"
    shown = ", ".join(str(v + 1) for v in values[:12])
    if len(values) > 12:
        shown += f" +{len(values) - 12}"
    return f"pp. {shown}"


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
    # Rich output on every TTY (incl. Windows). The "rich" backend already
    # delegates input to the prompt_toolkit session, so history, the fuzzy
    # completer and the bottom toolbar are preserved. "prompt_toolkit" stays
    # available only as an explicit MONKEYGRAB_CLI_BACKEND escape hatch.
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

    class _FuzzySlashCompleter(Completer):  # type: ignore[misc]
        """Fuzzy-complete slash-commands only when the line starts with '/'.

        Delegates to FuzzyWordCompleter so partial/out-of-order characters work
        (e.g. '/sta' → '/stats', '/rei' → '/reindex') while still avoiding
        noisy popup suggestions during regular RAG/chat input.
        """

        def __init__(self, commands: List[str]) -> None:
            self._fuzzy = FuzzyWordCompleter(sorted(set(commands)))

        def get_completions(self, document, complete_event):  # type: ignore[override]
            if not document.text_before_cursor.startswith("/"):
                return
            yield from self._fuzzy.get_completions(document, complete_event)


class Display:
    """Centralized terminal interface manager for MonkeyGrab."""

    # Single visual language for the Rich backend (minimalist refined):
    # light rounded panels, tight padding, airy tables — no heavy boxes.
    PANEL_BOX = box.ROUNDED
    PANEL_PADDING = (0, 1)
    TABLE_BOX = box.MINIMAL
    # Below this width side-by-side panels stack vertically instead of truncating.
    NARROW_WIDTH = 90

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
        # State for the persistent bottom toolbar (prompt_toolkit only)
        self._toolbar_mode: str = "chat"
        self._toolbar_model: str = ""
        # Language: "es" (default), "en", or "ca", controlled by MONKEYGRAB_LANG env var
        _raw_lang = os.getenv("MONKEYGRAB_LANG", "es").strip().lower()
        self._lang: str = _raw_lang if _raw_lang in ("es", "en", "ca") else "es"
        self._ptk_session = self._build_ptk_session()
        self._verify_rich_backend()

    def _verify_rich_backend(self) -> None:
        """Probe the Rich console once; degrade to plain on failure.

        Worst case (an exotic console without VT support) falls back to the
        legacy ANSI path instead of raising mid-session.
        """
        if self.backend != "rich":
            return
        try:
            self.console.print("", end="")
        except Exception:
            self.backend = "plain"
            self.safe_tty = True

    def _term_width(self) -> int:
        """Current terminal width, preferring Rich's measured console width."""
        if self.backend == "rich":
            try:
                return self.console.width
            except Exception:
                pass
        return max(40, shutil.get_terminal_size(fallback=(88, 24)).columns)

    def _build_ptk_session(self):
        """Create a PromptSession with persistent history, fuzzy completer, and toolbar."""
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
                completer=_FuzzySlashCompleter([cmd for cmd, _ in primary_commands(self._lang)]),
                complete_while_typing=False,
                bottom_toolbar=self._get_toolbar,
            )
        except Exception:
            return None

    def _get_toolbar(self):
        """Dynamic bottom toolbar rendered on every prompt redraw."""
        if not PTK_AVAILABLE or HTML is None:
            return ""
        mode_color = "#5fafaf" if self._toolbar_mode == "rag" else "#87afd7"
        shortcuts = self._s("toolbar.shortcuts")
        return HTML(
            f' <b><style fg="{mode_color}">{self._toolbar_mode.upper()}</style></b>'
            f'  <style fg="#a8a8a8">{self._toolbar_model}</style>'
            f'  <style fg="#606060">{shortcuts}</style>'
        )

    def _s(self, key: str, **kwargs) -> str:
        """Return the localized string for *key* in the active language.

        Thin wrapper around ``rag.cli.strings.s`` that automatically injects
        the ``Display`` instance's current language so callers don't have to
        pass it explicitly.

        Keys not found in the active language fall back to Spanish, then to the
        key itself, so raw values (paths, numbers, model names) pass through
        unchanged.
        """
        from rag.cli.strings import s as _str
        return _str(key, lang=self._lang, **kwargs)

    def _state_word(self, enabled: Any) -> str:
        return self._s("state.on") if _coerce_bool(enabled) else self._s("state.off")

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
        width = self._term_width()
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

    def _panel(
        self,
        content: Any,
        title: Optional[str] = None,
        *,
        border: str = "dim",
        expand: bool = True,
    ) -> Panel:
        """Build a panel with the single MonkeyGrab visual language."""
        return Panel(
            content,
            title=title,
            border_style=border,
            box=self.PANEL_BOX,
            padding=self.PANEL_PADDING,
            expand=expand,
        )

    def _two_panels(self, left: Panel, right: Panel) -> Any:
        """Lay two panels side by side, stacking them on narrow terminals."""
        if self._term_width() < self.NARROW_WIDTH:
            return Group(left, right)
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=2)
        grid.add_row(left, right)
        return grid

    def _themed_table(
        self,
        title: Optional[str] = None,
        *,
        header: bool = True,
        box_style=TABLE_BOX,
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

    def _models_table(self, info: Dict[str, Any]) -> Table:
        """Build the reusable models table from runtime info."""
        models = self._themed_table(header=True)
        models.add_column(self._s("models.role"), style="dim", width=12, no_wrap=True)
        models.add_column(self._s("models.model"), style="text", overflow="ellipsis")
        models.add_column(self._s("models.tag"), style="muted", width=12, overflow="ellipsis")
        models.add_column(self._s("models.use"), style="muted", width=12, no_wrap=True)
        model_width = 44 if self._term_width() >= 110 else 28
        model_rows = [
            (self._s("role.rag"), info.get("modelo_rag", ""), self._s("model.use.rag")),
            (self._s("role.chat"), info.get("modelo_chat", ""), self._s("model.use.chat")),
            (self._s("role.embed"), info.get("modelo_embedding", ""), self._s("model.use.embed")),
            (
                self._s("role.contextual"),
                info.get("modelo_contextual", ""),
                self._s("model.use.contextual") if info.get("contextual") else "off",
            ),
            (
                self._s("role.recomp"),
                info.get("modelo_recomp", ""),
                self._s("model.use.recomp") if info.get("recomp") else "off",
            ),
            (
                self._s("role.ocr"),
                info.get("modelo_ocr", ""),
                self._s("model.use.ocr") if info.get("images") else "off",
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
        return models

    def _build_flags(self, info: Dict[str, Any]) -> Text:
        """Build a compact flags line with checkmarks."""
        flag_items = [
            ("hybrid", info.get("hybrid")),
            ("rerank", info.get("reranker") == "on"),
            ("contextual", info.get("contextual")),
            ("recomp", info.get("recomp")),
            ("images", info.get("images")),
            ("expand", info.get("expand")),
        ]
        text = Text()
        text.append(f"  {self._s('pipeline.flags')}: ", style="dim")
        for i, (name, enabled) in enumerate(flag_items):
            if i > 0:
                text.append("  ")
            if enabled:
                text.append("✓ ", style="success")
                text.append(name, style="muted")
            else:
                text.append("✗ ", style="off")
                text.append(name, style="dim")
        return text

    # ─────────────────────────────────────────────
    # SECTION 8: BRANDING
    # ─────────────────────────────────────────────

    def logo(self) -> None:
        """Minimal brand header — just the name, no decorative rule."""
        if self.backend == "rich":
            self.console.print()
            self.console.print("  [brand]MonkeyGrab[/]  [dim]local PDF RAG[/]")
            self.console.print()
            return
        self._print_line()
        self._print_line(
            f"  {self._ansi('MonkeyGrab', Palette.BRAND.ansi, ANSI_BOLD)}"
            f"  {self._ansi('local PDF RAG', Palette.DIM.ansi)}"
        )
        self._print_line()

    def init_panel(self, info: Dict[str, Any]) -> None:
        """Compact startup info — no box, just clean inline text."""
        mode = info.get("mode", "chat")
        n_pdfs = info.get("total_documentos", 0)
        n_frags = info.get("total_fragmentos", 0)
        mode_style = "mode.rag" if mode == "rag" else "mode.chat"
        mode_label = self._s("mode.rag.label") if mode == "rag" else self._s("mode.chat.label")

        if self.backend != "rich":
            self._init_panel_ansi(info, mode)
            return

        self.console.print(
            f"  [dim]{n_pdfs} PDFs  ·  {n_frags} fragments[/]"
            f"  [{mode_style}]{mode_label.lower()} mode[/]"
        )
        self.console.print(f"  [dim]{self._s('init.use_help')}[/]")
        self.console.print()

    def _typewriter_panel(self, panel: Panel, delay: float = 0.008) -> None:
        """Kept for API compatibility; renders immediately (no animation)."""
        self.console.print(panel)

    def _init_panel_ansi(self, info: Dict[str, Any], mode: str) -> None:
        """ANSI fallback: compact startup info."""
        mode_color = self._mode_color(mode)
        n_pdfs = info.get("total_documentos", 0)
        n_frags = info.get("total_fragmentos", 0)
        mode_label = self._s("mode.rag.label") if mode == "rag" else self._s("mode.chat.label")
        self._print_line(
            f"  {self._ansi(str(n_pdfs), Palette.TEXT.ansi, ANSI_BOLD)}"
            f" {self._ansi('PDFs', Palette.DIM.ansi)}"
            f"  {self._ansi('·', Palette.DIM.ansi)}"
            f"  {self._ansi(str(n_frags), Palette.TEXT.ansi, ANSI_BOLD)}"
            f" {self._ansi('fragments', Palette.DIM.ansi)}"
            f"  {self._ansi(mode_label.lower() + ' mode', mode_color.ansi, ANSI_BOLD)}"
        )
        self._print_line(f"  {self._ansi(self._s('init.use_help'), Palette.DIM.ansi)}")
        self._print_line()

    def welcome(self) -> None:
        if self.backend != "rich":
            self._print_line()
            for cmd, desc_key in primary_commands(self._lang):
                self._print_line(
                    f"  {self._ansi(f'{cmd:<10}', Palette.BRAND.ansi, ANSI_BOLD)}"
                    f" {self._ansi(self._s(desc_key), Palette.MUTED.ansi)}"
                )
            self._print_line()
            self._print_line(f"  {self._ansi(self._s('help.shortcuts.inline'), Palette.DIM.ansi)}")
            self._print_line()
            return

        self.console.print()
        for cmd, desc_key in primary_commands(self._lang):
            self.console.print(f"  [brand]{cmd:<10}[/] [muted]{self._s(desc_key)}[/]")
        self.console.print()
        self.console.print(f"  [dim]{self._s('help.shortcuts.inline')}[/]")
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
        advice = self._s("exception.advice")
        if self.backend != "rich":
            self._print_line()
            self._rule(title, color=Palette.ERROR)
            self._print_line(f"  {self._ansi(detail, Palette.TEXT.ansi)}")
            self._print_line(f"  {self._ansi(advice, Palette.DIM.ansi)}")
            self._rule(color=Palette.DIM)
            return
        self.console.print(
            self._panel(
                f"[muted]{detail}[/]\n\n[dim]{advice}[/]",
                title=f"[error]{title}[/]",
                border="error",
                expand=False,
            )
        )

    def ollama_status(self, ok: bool, detail: str) -> None:
        """Render the Ollama health check result at startup."""
        if ok:
            self.success(detail)
            return
        advice = self._s("ollama.advice")
        ollama_title = self._s("ollama.title")
        if self.backend != "rich":
            self._print_line()
            self._rule(ollama_title, color=Palette.WARNING)
            self._print_line(f"  {self._ansi(detail, Palette.TEXT.ansi)}")
            self._print_line(f"  {self._ansi(advice, Palette.DIM.ansi)}")
            self._rule(color=Palette.DIM)
            return
        self.console.print(
            self._panel(
                f"[muted]{detail}[/]\n\n[dim]{advice}[/]",
                title=f"[warning]{ollama_title}[/]",
                border="warning",
                expand=False,
            )
        )

    # ─────────────────────────────────────────────
    # SECTION 10: PIPELINE FEEDBACK
    # ─────────────────────────────────────────────

    def pipeline_start(self, message: Optional[str] = None) -> Optional[Status]:
        if message is None:
            message = self._s("pipeline.start")
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
        mode_label = "rag" if mode == "rag" else "chat"
        return f"{mode_label} > "

    def _ansi_prompt(self, mode: str, model: str = "") -> str:
        mode_color = self._mode_color(mode)
        mode_label = "rag" if mode == "rag" else "chat"
        return (
            f"{mode_color.ansi}{ANSI_BOLD}{mode_label}{ANSI_RESET}"
            f"{Palette.DIM.ansi} > {ANSI_RESET}"
        )

    def read_input(self, mode: str, model: str = "") -> str:
        self.pipeline_stop()
        self._toolbar_mode = mode
        self._toolbar_model = _short_model(model, max_len=22, keep_tag=False)
        if self.backend == "prompt_toolkit":
            return self._read_input_ptk(mode, model)
        if self.backend == "plain":
            return input(self.prompt(mode, model))
        self.console.file.flush()
        if self._ptk_session is not None:
            return self._read_input_ptk(mode, model)
        mode_style = "mode.rag" if mode == "rag" else "mode.chat"
        mode_label = "rag" if mode == "rag" else "chat"
        prompt_text = Text()
        prompt_text.append(mode_label, style=f"{mode_style} bold")
        prompt_text.append(" > ", style="dim")
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
        """Blank line before a response — no decorative rule."""
        if self.backend != "rich":
            self._print_line()
            return
        self.console.print()

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
        empty = self._s("response.empty")
        if self.backend != "rich":
            body = self._plain_markdown((text or empty).strip())
            for line in body.splitlines() or [empty]:
                self._print_line(f"  {self._ansi(line.rstrip(), Palette.TEXT.ansi)}")
            self._print_line()
            return
        self.console.print()
        self.console.print(Markdown(text), width=min(self._term_width() - 4, 100))
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

        n_src = len(sources_map)
        src_title = self._s("sources.title", n=n_src)
        if self.backend != "rich":
            self._print_line()
            self._rule(src_title, color=Palette.BRAND)
            for doc, pages in sorted(sources_map.items()):
                self._print_line(
                    f"  {self._ansi(_short_model(doc, max_len=56, keep_tag=True), Palette.TEXT.ansi)} "
                    f"{self._ansi(f'({_safe_pages(pages)})', Palette.MUTED.ansi)}"
                )
            return

        table = self._themed_table(header=False)
        table.add_column(self._s("sources.col.doc"), style="muted", overflow="fold")
        table.add_column(self._s("sources.col.pages"), style="dim", no_wrap=True)
        for doc, pages in sorted(sources_map.items()):
            table.add_row(_short_model(doc, max_len=56, keep_tag=True), _safe_pages(pages))
        self.console.rule(f"[dim]{src_title}[/]", style="dim")
        self.console.print(table)

    def response_footer(self, sources: Optional[int] = None) -> None:
        if self.backend != "rich":
            if sources is not None:
                src_text = (self._s("sources.footer.cited", n=sources)
                            if sources else self._s("sources.footer.none"))
                self._print_line(
                    f"  {self._ansi('·', Palette.DIM.ansi)} "
                    f"{self._ansi(src_text, Palette.DIM.ansi)}"
                )
            self._print_line()
            return
        if sources is not None:
            tag = (self._s("sources.footer.cited", n=sources)
                   if sources else self._s("sources.footer.none"))
            self.console.print(f"  [dim]· {tag}[/]")
        self.console.print()

    def response_footer_rag(
        self,
        fragments: List[Dict[str, Any]],
        timer: "QueryTimer",
    ) -> None:
        """Compact post-RAG footer: timing + source lines."""
        if not fragments:
            self.response_footer()
            return

        sources_map: Dict[str, Dict[str, Any]] = {}
        for frag in fragments:
            meta = frag.get("metadata", {}) or {}
            doc = meta.get("source", "?")
            page = meta.get("page", None)
            score = frag.get("score_reranker") or frag.get("score_final") or 0.0
            entry = sources_map.setdefault(doc, {"pages": set(), "score": 0.0})
            entry["pages"].add(page)
            entry["score"] = max(entry["score"], float(score))

        n_frags = len(fragments)
        total_dur = _format_duration(timer.total)

        if self.backend != "rich":
            self._print_line()
            timing = f"· {total_dur}  ·  {n_frags} frags"
            self._print_line(f"  {self._ansi(timing, Palette.DIM.ansi)}")
            for doc, data in sorted(sources_map.items()):
                pages_str = _safe_pages(data["pages"])
                short_doc = _short_model(doc, max_len=44, keep_tag=True)
                self._print_line(
                    f"  {self._ansi('↳', Palette.DIM.ansi)}"
                    f" {self._ansi(short_doc, Palette.MUTED.ansi)}"
                    f" {self._ansi(pages_str, Palette.DIM.ansi)}"
                )
            self._print_line()
            return

        self.console.print()
        self.console.print(f"  [dim]· {total_dur}  ·  {n_frags} frags[/]")
        for doc, data in sorted(sources_map.items()):
            pages_str = _safe_pages(data["pages"])
            short_doc = _short_model(doc, max_len=44, keep_tag=True)
            self.console.print(f"  [dim]↳[/] [muted]{short_doc}[/] [dim]{pages_str}[/]")
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
            (self._s("corpus.pdfs"), info.get("total_documentos", len(docs))),
            (self._s("corpus.docs_indexed"), len(docs)),
            (self._s("corpus.fragments"), total_fragments),
            (self._s("corpus.folder"), info.get("docs_folder", "-")),
            (self._s("corpus.vector_db"), info.get("path_db", "-")),
            (self._s("corpus.collection"), info.get("collection_name", "-")),
        ]

        if self.backend != "rich":
            self._print_line()
            self._rule(self._s("stats.title"), color=Palette.INFO)
            for key, value in rows:
                self._print_line(
                    f"  {self._ansi(f'{key}:', Palette.DIM.ansi)} "
                    f"{self._ansi(str(value), Palette.TEXT.ansi)}"
                )
            self._rule(color=Palette.DIM)
            self._print_line()
            return

        table = self._themed_table(title=self._s("stats.title"))
        table.add_column(self._s("stats.metric_col"), style="dim", no_wrap=True)
        table.add_column(self._s("stats.value_col"), style="text", overflow="fold")
        for key, value in rows:
            table.add_row(key, str(value))
        self.console.print()
        self.console.print(table)
        self.console.print()

    def stats_dashboard(
        self,
        total_fragments: int,
        docs: List[Any],
        info: Optional[Dict[str, Any]] = None,
        session: Optional[Any] = None,
    ) -> None:
        """Consolidated stats dashboard replacing the separate stats_table + init_panel combo.

        Shows corpus metrics, pipeline configuration, active models, flags,
        and session statistics in a single non-redundant view triggered by
        the ``/stats`` command.

        Args:
            total_fragments: Total chunk count in the vector DB.
            docs: List of indexed document names or summary dicts.
            info: Runtime metadata dict from ``_runtime_info``.
            session: Optional SessionStats for live session metrics.
        """
        info = info or {}
        n_docs = len(docs)
        if docs and isinstance(docs[0], dict):
            n_docs = sum(1 for d in docs if d.get("fragments", 0) > 0)

        if self.backend != "rich":
            rr_ansi = (
                f"on  ({info.get('reranker_device', '')})"
                if info.get("reranker") == "on" else "off"
            )
            self._print_line()
            self._print_line(
                f"  {self._ansi(str(info.get('total_documentos', n_docs)), Palette.TEXT.ansi, ANSI_BOLD)}"
                f" {self._ansi('PDFs', Palette.DIM.ansi)}"
                f"  {self._ansi('·', Palette.DIM.ansi)}"
                f"  {self._ansi(str(total_fragments), Palette.TEXT.ansi, ANSI_BOLD)}"
                f" {self._ansi('fragments', Palette.DIM.ansi)}"
                f"  {self._ansi('·', Palette.DIM.ansi)}"
                f"  {self._ansi(info.get('collection_name', '-'), Palette.MUTED.ansi)}"
            )
            self._print_line(
                f"  {self._ansi(info.get('docs_folder', '-'), Palette.DIM.ansi)}"
                f"  {self._ansi('·', Palette.DIM.ansi)}"
                f"  {self._ansi(info.get('path_db', '-'), Palette.DIM.ansi)}"
            )
            self._print_line()
            chunk_info = f"{info.get('chunk_size', '-')}c / {info.get('chunk_overlap', '-')}c"
            self._print_line(
                f"  {self._ansi(self._s(str(info.get('busqueda', '-'))), Palette.MUTED.ansi)}"
                f"  {self._ansi('·', Palette.DIM.ansi)}"
                f"  {self._ansi('reranker ' + rr_ansi, Palette.MUTED.ansi)}"
                f"  {self._ansi('·', Palette.DIM.ansi)}"
                f"  {self._ansi(chunk_info, Palette.DIM.ansi)}"
            )
            flag_parts = []
            for name, enabled in [
                ("hybrid", info.get("hybrid")),
                ("rerank", info.get("reranker") == "on"),
                ("contextual", info.get("contextual")),
                ("recomp", info.get("recomp")),
                ("images", info.get("images")),
                ("expand", info.get("expand")),
            ]:
                mark = self._ansi("✓", Palette.SUCCESS.ansi) if enabled else self._ansi("✗", Palette.DIM.ansi)
                flag_parts.append(f"{mark} {name}")
            self._print_line(f"  {'  '.join(flag_parts)}")
            if session is not None and session.total_queries > 0:
                self._session_summary(session)
            else:
                self._print_line()
            return

        self.console.print()

        # --- Corpus ---
        self.console.print(
            f"  [text]{info.get('total_documentos', n_docs)}[/] [dim]PDFs[/]"
            f"  [dim]·[/]  [text]{total_fragments}[/] [dim]fragments[/]"
            f"  [dim]·[/]  [dim]{info.get('collection_name', '-')}[/]"
        )
        self.console.print(
            f"  [dim]{info.get('docs_folder', '-')}  ·  {info.get('path_db', '-')}[/]"
        )
        self.console.print()

        # --- Pipeline ---
        rr = (f"reranker on  ({info.get('reranker_device', '')})"
              if info.get("reranker") == "on" else "reranker off")
        self.console.print(
            f"  [dim]{self._s(str(info.get('busqueda', '-')))}[/]"
            f"  [dim]·[/]  [dim]{rr}[/]"
            f"  [dim]·[/]  [dim]{info.get('chunk_size', '-')}c / {info.get('chunk_overlap', '-')}c overlap[/]"
        )
        self.console.print(self._build_flags(info))
        self.console.print()

        # --- Models ---
        self.console.print(self._models_table(info))

        # --- Session ---
        if session is not None and session.total_queries > 0:
            self._session_summary(session)
        else:
            self.console.print()

    def docs_table(self, docs: List[Any]) -> None:
        if not docs:
            self.warning(self._s("docs.none"))
            return
        if self.backend != "rich":
            self._print_line()
            self._rule(self._s("docs.title"), color=Palette.BRAND)
            for idx, item in enumerate(docs, 1):
                if isinstance(item, dict):
                    descr = self._s(
                        "docs.inline.meta",
                        pages=item.get("pages", "-"),
                        frags=item.get("fragments", "-"),
                        pct=item.get("pct_corpus", "-"),
                        types=item.get("formats", "-"),
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

        table = self._themed_table(title=self._s("docs.title"))
        table.add_column(self._s("docs.col.num"), style="dim", width=4, justify="right")
        table.add_column(self._s("docs.col.doc"), style="text", overflow="fold")
        table.add_column(self._s("docs.col.pages"), style="muted", justify="right", no_wrap=True)
        table.add_column(self._s("docs.col.frags"), style="muted", justify="right", no_wrap=True)
        table.add_column(self._s("docs.col.pct"), style="info", justify="right", no_wrap=True)
        table.add_column(self._s("docs.col.types"), style="dim", no_wrap=True)
        for idx, item in enumerate(docs, 1):
            if isinstance(item, dict):
                table.add_row(
                    str(idx),
                    item.get("name", "-"),
                    str(item.get("pages", "-")),
                    str(item.get("fragments", "-")),
                    item.get("pct_corpus", "-"),
                    item.get("formats", "-"),
                )
            else:
                table.add_row(str(idx), str(item), "-", "-", "-", "-")
        self.console.print()
        self.console.print(table)
        self.console.print()

    def topics_display(self, docs_data: List[Dict[str, Any]]) -> None:
        if not docs_data:
            self.warning(self._s("docs.none"))
            return
        tip = self._s("topics.tip")

        if self.backend != "rich":
            self._print_line()
            self._rule(self._s("topics.title"), color=Palette.INFO)
            for doc_info in docs_data:
                analizados = doc_info.get("analizados")
                descr = self._s(
                    "topics.inline.meta",
                    pages=doc_info.get("pages", "-"),
                    frags=doc_info.get("fragments", "-"),
                    chunks=analizados if analizados is not None else "-",
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

        table = self._themed_table(title=self._s("topics.title"))
        table.add_column(self._s("topics.col.doc"), style="text", overflow="fold")
        table.add_column(self._s("topics.col.pages"), style="muted", justify="right", no_wrap=True)
        table.add_column(self._s("topics.col.frags"), style="muted", justify="right", no_wrap=True)
        table.add_column(self._s("topics.col.chunks"), style="dim", justify="right", no_wrap=True)
        table.add_column(self._s("topics.col.terms"), style="dim", overflow="fold")
        for doc_info in docs_data:
            analizados = doc_info.get("analizados")
            chunks_str = str(analizados) if analizados is not None else "-"
            table.add_row(
                doc_info.get("name", "-"),
                str(doc_info.get("pages", "-")),
                str(doc_info.get("fragments", "-")),
                chunks_str,
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
        self._toolbar_mode = mode
        self._toolbar_model = _short_model(model, max_len=22, keep_tag=False)
        mode_label = self._s("mode.rag.label") if mode == "rag" else self._s("mode.chat.label")
        if self.backend != "rich":
            color = self._mode_color(mode)
            self._print_line(
                f"\n  {self._ansi('→', color.ansi, ANSI_BOLD)}"
                f" {self._ansi(mode_label.lower() + ' mode', color.ansi)}"
            )
            return
        style = "mode.rag" if mode == "rag" else "mode.chat"
        self.console.print(f"\n  [{style}]→ {mode_label.lower()} mode[/]")

    def history_loaded(self, n: int) -> None:
        self.info(self._s("history.loaded", n=n))

    def history_cleared(self) -> None:
        self.console.clear()
        self.success(self._s("history.cleared"))

    def unknown_command(self, cmd: str, suggestions: Optional[List[str]] = None) -> None:
        self.warning(self._s("unknown_cmd.msg", cmd=cmd))
        if suggestions:
            hint = "  ".join(suggestions)
            suggestion_text = self._s("unknown_cmd.suggestion", hint=hint)
            if self.backend != "rich":
                self._print_line(f"     {self._ansi(suggestion_text, Palette.DIM.ansi)}")
            else:
                self.console.print(f"     [dim]{suggestion_text}[/]")
        else:
            hint_text = self._s("unknown_cmd.hint")
            if self.backend != "rich":
                self._print_line(f"     {self._ansi(hint_text, Palette.DIM.ansi)}")
            else:
                self.console.print(f"     [dim]{hint_text}[/]")

    def reindex_start(self) -> None:
        reindex_title = self._s("reindex.title")
        reindex_msg = self._s("reindex.start")
        if self.backend != "rich":
            self._print_line()
            self._rule(reindex_title, color=Palette.WARNING)
            self.warning(reindex_msg)
            return
        self.console.print()
        self.warning(reindex_msg)

    def reindex_complete(self, total: int) -> None:
        self.success(self._s("reindex.complete", total=total))
        self.warning(self._s("reindex.restart"))

    def farewell(self, stats: Optional[SessionStats] = None) -> None:
        farewell_msg = self._s("farewell.msg")
        suffix = ""
        if stats is not None and stats.total_queries > 0:
            suffix = (
                f"  ·  {stats.total_queries} queries"
                f"  ·  {_format_duration(stats.duration)}"
            )
        if self.backend != "rich":
            self._print_line()
            self._print_line(f"  {self._ansi(farewell_msg + suffix, Palette.DIM.ansi)}")
            self._print_line()
            return
        self.console.print()
        self.console.print(f"  [dim]{farewell_msg}{suffix}[/]")
        self.console.print()

    def _session_summary(self, stats: SessionStats) -> None:
        """Compact session stats for /stats and farewell."""
        models = ", ".join(sorted(_short_model(m, max_len=24) for m in stats.models_used)) or "-"
        total = stats.total_queries
        dur = _format_duration(stats.duration)
        if self.backend != "rich":
            self._print_line()
            self._print_line(
                f"  {self._ansi(self._s('farewell.title') + ':', Palette.DIM.ansi)}"
                f" {self._ansi(f'{total} queries  ·  {dur}', Palette.TEXT.ansi)}"
            )
            if models != "-":
                self._print_line(f"  {self._ansi(models, Palette.MUTED.ansi)}")
            self._print_line()
            return
        self.console.print(
            f"\n  [dim]{self._s('farewell.title')}:[/] {total} queries  [dim]·[/]  {dur}"
        )
        if models != "-":
            self.console.print(f"  [muted]{models}[/]")
        self.console.print()

    def no_results(self) -> None:
        if self.backend != "rich":
            self._rule("—", color=Palette.WARNING)
            self._print_line(
                f"  {self._ansi(self._s('no_results.msg'), Palette.TEXT.ansi)}"
            )
            self._print_line(
                f"  {self._ansi(self._s('no_results.reason1'), Palette.MUTED.ansi)}"
            )
            self._print_line(
                f"  {self._ansi(self._s('no_results.reason2'), Palette.MUTED.ansi)}"
            )
            self._print_line(
                f"  {self._ansi(self._s('no_results.reason3'), Palette.MUTED.ansi)}"
            )
            self._print_line(
                f"  {self._ansi(self._s('no_results.tip'), Palette.DIM.ansi)}"
            )
            self._rule(color=Palette.DIM)
            return
        self.console.print(
            self._panel(
                f"[muted]{self._s('no_results.msg')}[/]\n\n"
                f"[dim]{self._s('no_results.reason1')}\n"
                f"{self._s('no_results.reason2')}\n"
                f"{self._s('no_results.reason3')}[/]\n\n"
                f"[dim]{self._s('no_results.tip')}[/]",
                border="warning",
                expand=False,
            )
        )

    def question_too_short(self) -> None:
        msg = self._s("question_too_short")
        if self.backend != "rich":
            self._print_line(f"  {self._ansi(msg, Palette.DIM.ansi)}")
            return
        self.console.print(f"  [dim]{msg}[/]")

    def no_pdfs(self, folder: str) -> None:
        self.warning(self._s("no_pdfs", folder=folder))


# ─────────────────────────────────────────────
# SECTION 15: ENTRY (singleton)
# ─────────────────────────────────────────────

ui = Display()
