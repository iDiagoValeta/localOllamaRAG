"""
MonkeyGrab CLI -- Theme and Visual Identity (legacy).

This ANSI theme is retained for reference only. The active CLI uses
``rag.cli.display.MONKEYGRAB_THEME``.

Dark professional color palette, terminal typography, iconography,
and ASCII logo carrying the MonkeyGrab brand identity.

Inspired by the CLIs of Claude Code and Gemini CLI:
muted colors, low contrast, subtle accents.
"""


# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  CONFIGURATION
#  +-- 1. Imports and cross-platform color support
#  +-- 2. THEME CLASS          256-color ANSI palette and icons
#  +-- 3. ASCII LOGO           MonkeyGrab brand banner
#  +-- 4. SYSTEM MESSAGES      mode labels and status strings
#
# ─────────────────────────────────────────────

import os

# ─────────────────────────────────────────────
# SECTION 1: IMPORTS AND CROSS-PLATFORM COLOR SUPPORT
# ─────────────────────────────────────────────
try:
    from colorama import init as _colorama_init
    _colorama_init()
except ImportError:
    pass

# ─────────────────────────────────────────────
# SECTION 2: THEME CLASS
# ─────────────────────────────────────────────

class Theme:
    """
    Visual design system for the MonkeyGrab CLI.

    Dark palette with subtle accents. All values are ANSI 256-color
    escape sequences for maximum terminal compatibility.
    """

    RESET       = "\033[0m"
    BOLD        = "\033[1m"
    DIM         = "\033[2m"
    ITALIC      = "\033[3m"
    UNDERLINE   = "\033[4m"

    TEXT         = "\033[38;5;253m"
    TEXT_MUTED   = "\033[38;5;245m"
    TEXT_DIM     = "\033[38;5;240m"

    BG_DARK      = "\033[38;5;236m"
    BG_MEDIUM    = "\033[38;5;238m"
    BORDER       = "\033[38;5;240m"

    BRAND        = "\033[38;5;172m"
    BRAND_LIGHT  = "\033[38;5;179m"
    BRAND_DIM    = "\033[38;5;130m"

    PURPLE       = "\033[38;5;141m"
    PURPLE_DIM   = "\033[38;5;97m"
    CYAN         = "\033[38;5;73m"
    CYAN_DIM     = "\033[38;5;66m"
    GREEN        = "\033[38;5;71m"
    GREEN_DIM    = "\033[38;5;65m"
    RED          = "\033[38;5;167m"
    RED_DIM      = "\033[38;5;131m"
    YELLOW       = "\033[38;5;178m"
    YELLOW_DIM   = "\033[38;5;136m"
    BLUE         = "\033[38;5;67m"
    BLUE_DIM     = "\033[38;5;60m"

    MONO_BRIGHT  = "\033[38;5;179m"
    MONO_BODY    = "\033[38;5;172m"
    MONO_SHADOW  = "\033[38;5;130m"
    MONO_DARK    = "\033[38;5;94m"
    MONO_ACCENT  = "\033[38;5;73m"

    BOX_H        = "─"
    BOX_H_BOLD   = "━"
    BOX_V        = "│"
    BOX_TL       = "╭"
    BOX_TR       = "╮"
    BOX_BL       = "╰"
    BOX_BR       = "╯"
    BOX_TEE_R    = "├"
    BOX_TEE_L    = "┤"
    BOX_DOUBLE_H = "═"

    ICON_OK      = "✓"
    ICON_FAIL    = "✗"
    ICON_WARN    = "!"
    ICON_INFO    = "·"
    ICON_ARROW   = "›"
    ICON_DOT     = "•"
    ICON_SEARCH  = "⊙"
    ICON_GEAR    = "⚙"
    ICON_DOC     = "◈"
    ICON_CHAT    = "◆"
    ICON_RAG     = "◇"
    ICON_SPARK   = "✦"

    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    @staticmethod
    def terminal_width() -> int:
        """Return the current terminal width in columns."""
        try:
            return os.get_terminal_size().columns
        except Exception:
            return 100

    @classmethod
    def hline(cls, char: str = None, width: int = None, color: str = None) -> str:
        """Generate a horizontal line spanning the terminal width.

        Args:
            char: Character used to draw the line (default: BOX_H).
            width: Override width in columns (default: terminal width).
            color: ANSI color escape to apply (default: BORDER).

        Returns:
            A colored string representing the horizontal line.
        """
        w = width or cls.terminal_width()
        c = char or cls.BOX_H
        col = color or cls.BORDER
        return f"{col}{c * w}{cls.RESET}"

    @classmethod
    def styled(cls, text: str, *styles: str) -> str:
        """Apply multiple ANSI style sequences to a text string.

        Args:
            text: The text to style.
            *styles: One or more ANSI escape sequences to prepend.

        Returns:
            The styled string with a trailing RESET.
        """
        prefix = "".join(styles)
        return f"{prefix}{text}{cls.RESET}"

# ─────────────────────────────────────────────
# SECTION 3: ASCII LOGO
# ─────────────────────────────────────────────

def get_logo(modelo_desc: str = "") -> str:
    """Generate the MonkeyGrab ASCII logo with a dark gradient.

    Args:
        modelo_desc: Base model description to display in the subtitle.

    Returns:
        A multi-line string containing the colored ASCII logo.
    """
    T = Theme

    logo = f"""
{T.MONO_DARK}                __------__{T.RESET}
{T.MONO_BODY}              /~{T.MONO_BRIGHT}          {T.MONO_BODY}~\\{T.RESET}
{T.MONO_BODY}             |  {T.MONO_BRIGHT}  //^\\\\//^\\{T.MONO_BODY}|{T.RESET}
{T.MONO_BODY}           /~~\\  ||{T.MONO_BRIGHT} o{T.MONO_BODY}| |{T.MONO_BRIGHT}o{T.MONO_BODY}|:~\\{T.RESET}
{T.MONO_BODY}          | |6   ||{T.MONO_DARK}___|_|_{T.MONO_BODY}||:|{T.RESET}
{T.MONO_BODY}           \\__.  /{T.MONO_BRIGHT}      o  {T.MONO_BODY}\\/{T.RESET}'
{T.MONO_SHADOW}            |   ({T.MONO_BRIGHT}       O   {T.MONO_SHADOW}){T.RESET}
{T.MONO_DARK}   /~~~~\\{T.MONO_DARK}    `\\  \\         /{T.RESET}
{T.MONO_DARK}  | |~~\\ |{T.MONO_BODY}     )  ~------~`\\{T.RESET}
{T.MONO_DARK} /' |  | |{T.MONO_BODY}   /     ____ /~~~)\\{T.RESET}
{T.MONO_DARK}(_/'   | | |{T.MONO_BODY}     /'    |    ( |{T.RESET}
{T.MONO_DARK}       | | |     \\    /   __)/ \\{T.RESET}
{T.MONO_DARK}       \\  \\ \\      \\/    /' \\   `\\{T.RESET}
{T.MONO_SHADOW}         \\  \\|\\        /   | |\\___|{T.RESET}
{T.MONO_SHADOW}           \\ |  \\____/     | |{T.RESET}
{T.MONO_SHADOW}           /^~>  \\{T.MONO_DARK}        _/ <{T.RESET}
{T.MONO_BODY}          |  |         \\       \\{T.RESET}
{T.MONO_BODY}          |  | \\        \\        \\{T.RESET}
{T.MONO_BODY}          -^-\\  \\{T.MONO_DARK}       |        ){T.RESET}
{T.MONO_DARK}               `\\_______/^\\______/{T.RESET}

{T.BOLD}{T.BRAND}              M O N K E Y G R A B{T.RESET}"""

    return logo


# ─────────────────────────────────────────────
# SECTION 4: SYSTEM MESSAGES
# ─────────────────────────────────────────────

MESSAGES = {
    "farewell": f"{Theme.TEXT_DIM}Hasta luego. Sesión finalizada.{Theme.RESET}",

    "no_results": (
        "No se encontró información sobre tu pregunta en los documentos.\n\n"
        f"  {Theme.TEXT_DIM}Posibles causas:{Theme.RESET}\n"
        f"  {Theme.BORDER}│{Theme.RESET} La información no está en los documentos indexados\n"
        f"  {Theme.BORDER}│{Theme.RESET} La pregunta podría formularse de otra manera\n"
        f"  {Theme.BORDER}│{Theme.RESET} El tema está fuera del alcance del corpus\n\n"
        f"  {Theme.TEXT_DIM}Sugerencia: reformula la pregunta o usa {Theme.BRAND}/temas{Theme.TEXT_DIM} para explorar.{Theme.RESET}"
    ),

    "out_of_scope": (
        "La pregunta parece estar fuera del ámbito de los documentos.\n\n"
        f"  {Theme.TEXT_DIM}Usa {Theme.BRAND}/temas{Theme.TEXT_DIM} para ver contenidos, "
        f"o {Theme.BRAND}/docs{Theme.TEXT_DIM} para listar documentos.{Theme.RESET}"
    ),

    "question_too_short": (
        f"  {Theme.TEXT_DIM}Pregunta demasiado corta para consulta documental.\n"
        f"  Formula una pregunta concreta o usa {Theme.BRAND}/chat{Theme.TEXT_DIM} para conversar.{Theme.RESET}"
    ),
}
