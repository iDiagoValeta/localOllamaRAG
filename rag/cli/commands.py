"""
MonkeyGrab CLI commands: single source of truth.

Both the dispatcher in ``app.py`` and the help / autocompletion in
``display.py`` read from this module, so the slash-command list cannot drift
between the prompt, the ``/ayuda`` screen and the autocompleter.
"""

from __future__ import annotations

from typing import List, Tuple


COMMANDS: List[Tuple[str, str]] = [
    ("/rag",     "cmd.rag.desc"),
    ("/chat",    "cmd.chat.desc"),
    ("/limpiar", "cmd.limpiar.desc"),
    ("/stats",   "cmd.stats.desc"),
    ("/docs",    "cmd.docs.desc"),
    ("/temas",   "cmd.temas.desc"),
    ("/reindex", "cmd.reindex.desc"),
    ("/ayuda",   "cmd.ayuda.desc"),
    ("/salir",   "cmd.salir.desc"),
]

# 3-tuple: (alias, canonical_command, description_key_for_help)
ALIASES: List[Tuple[str, str, str]] = [
    ("/clear", "/limpiar", "alias.clear.desc"),
    ("/help",  "/ayuda",   "alias.help.desc"),
    ("/exit",  "/salir",   "alias.exit.desc"),
]


def all_command_names() -> List[str]:
    """All accepted slash tokens (primary + aliases), useful for autocomplete."""
    names = [cmd for cmd, _ in COMMANDS]
    names.extend(alias for alias, *_ in ALIASES)
    return names
