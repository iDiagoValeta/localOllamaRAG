"""Auxiliary implementation module for rag.chat_pdfs.

This module keeps business logic split out of the public facade. Runtime
configuration stays owned by rag.chat_pdfs and is read lazily through ``cfg``
(a live reference to that module), so web/API toggles and test monkeypatches
are observed without any per-call synchronization.
"""

import json
import logging
import os
from typing import Dict, List

from rag.engine.atomic import escribir_json_atomico
from rag.engine.runtime import get_runtime

cfg = get_runtime()
# HISTORY PERSISTENCE


def cargar_historial() -> List[Dict[str, str]]:
    """Load CHAT history from disk.

    Returns:
        List of message dicts. Empty list if the file is missing or corrupt.
    """
    try:
        if os.path.exists(cfg.HISTORIAL_PATH):
            with open(cfg.HISTORIAL_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and "chat" in data:
                    return data["chat"]
                if isinstance(data, list):
                    return data
    except Exception as e:
        logging.warning(f"Error loading history: {e}")

    return []


def guardar_historial(historial: List[Dict[str, str]]) -> None:
    """Persist CHAT history to disk (last MAX_HISTORIAL_MENSAJES entries).

    The payload is serialized to a temporary file next to the target and swapped
    in with ``os.replace``, so a failure halfway through ``json.dump`` (full
    disk, killed process) leaves the previous history untouched instead of
    truncated. Errors are still only logged: a lost save must not abort the turn
    that triggered it.

    Args:
        historial: Full list of message dicts to save.
    """
    try:
        escribir_json_atomico(cfg.HISTORIAL_PATH, historial[-cfg.MAX_HISTORIAL_MENSAJES:])
    except Exception as e:
        logging.warning(f"Error saving history: {e}")

def limpiar_historial(historial: List[Dict[str, str]]) -> None:
    """Clear history in-place and persist the empty state.

    Args:
        historial: The history list to clear.
    """
    historial.clear()
    guardar_historial(historial)


