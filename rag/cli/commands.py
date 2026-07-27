"""MonkeyGrab CLI commands: single source of truth.

Both the dispatcher in ``app.py`` and the help / autocompletion in
``display.py`` read from this module, so the slash-command list cannot drift
between the prompt, the ``/ayuda`` screen and the autocompleter.

Primary command lists per language
-----------------------------------
- ES: /ayuda  /limpiar  /temas  /salir  (canonical names)
- EN: /help   /clear    /topics /exit
- CA: /ajuda  /netejar  /temes  /eixir

All tokens from all languages are always accepted by the dispatcher.
"""

from __future__ import annotations

import os
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
    # EN
    ("/clear",  "/limpiar", "alias.clear.desc"),
    ("/help",   "/ayuda",   "alias.help.desc"),
    ("/exit",   "/salir",   "alias.exit.desc"),
    ("/topics", "/temas",   "cmd.temas.desc"),
    # CA
    ("/netejar", "/limpiar", "alias.clear.desc"),
    ("/ajuda",   "/ayuda",   "alias.help.desc"),
    ("/eixir",   "/salir",   "alias.exit.desc"),
    ("/temes",   "/temas",   "cmd.temas.desc"),
]


_PRIMARY_EN: List[Tuple[str, str]] = [
    ("/rag",     "cmd.rag.desc"),
    ("/chat",    "cmd.chat.desc"),
    ("/clear",   "cmd.limpiar.desc"),
    ("/stats",   "cmd.stats.desc"),
    ("/docs",    "cmd.docs.desc"),
    ("/topics",  "cmd.temas.desc"),
    ("/reindex", "cmd.reindex.desc"),
    ("/help",    "cmd.ayuda.desc"),
    ("/exit",    "cmd.salir.desc"),
]

_PRIMARY_CA: List[Tuple[str, str]] = [
    ("/rag",     "cmd.rag.desc"),
    ("/chat",    "cmd.chat.desc"),
    ("/netejar", "cmd.limpiar.desc"),
    ("/stats",   "cmd.stats.desc"),
    ("/docs",    "cmd.docs.desc"),
    ("/temes",   "cmd.temas.desc"),
    ("/reindex", "cmd.reindex.desc"),
    ("/ajuda",   "cmd.ayuda.desc"),
    ("/eixir",   "cmd.salir.desc"),
]


def primary_commands(lang: str = None) -> List[Tuple[str, str]]:
    """Return the help-screen command list for *lang*.

    Each entry is (command_token, description_key).  The description keys are
    shared across languages; strings.py provides the localized text.
    """
    if lang is None:
        lang = os.getenv("MONKEYGRAB_LANG", "es").strip().lower()
    if lang == "en":
        return _PRIMARY_EN
    if lang == "ca":
        return _PRIMARY_CA
    return COMMANDS


def all_command_names() -> List[str]:
    """All accepted slash tokens (primary + aliases), useful for autocomplete."""
    names = [cmd for cmd, _ in COMMANDS]
    names.extend(alias for alias, *_ in ALIASES)
    return names
