"""
MonkeyGrab CLI commands: single source of truth.

Both the dispatcher in ``app.py`` and the help / autocompletion in
``display.py`` read from this module, so the slash-command list cannot drift
between the prompt, the ``/ayuda`` screen and the autocompleter.
"""

from __future__ import annotations

from typing import List, Tuple


COMMANDS: List[Tuple[str, str]] = [
    ("/rag",     "Activar modo RAG"),
    ("/chat",    "Activar modo CHAT"),
    ("/limpiar", "Limpiar historial"),
    ("/stats",   "Estado, modelos y base vectorial"),
    ("/docs",    "Documentos indexados"),
    ("/temas",   "Resumen de contenidos"),
    ("/reindex", "Reconstruir el índice"),
    ("/ayuda",   "Mostrar esta ayuda"),
    ("/salir",   "Terminar la sesión"),
]

ALIASES: List[Tuple[str, str]] = [
    ("/clear", "/limpiar"),
    ("/help",  "/ayuda"),
    ("/exit",  "/salir"),
]


def all_command_names() -> List[str]:
    """All accepted slash tokens (primary + aliases), useful for autocomplete."""
    names = [cmd for cmd, _ in COMMANDS]
    names.extend(alias for alias, _ in ALIASES)
    return names
