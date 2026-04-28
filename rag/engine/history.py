"""Auxiliary implementation module for rag.chat_pdfs.

This module keeps business logic split out of the public facade. Runtime
configuration remains owned by rag.chat_pdfs and is synchronized before each
function call so web/API toggles and test monkeypatches keep working.
"""

import base64
import contextlib
import io
import json
import logging
import os
import re
import requests
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import ollama
from pypdf import PdfReader

from rag.cli.display import ui
from rag.engine.runtime import sync_runtime_globals


def _sync_runtime_globals() -> None:
    sync_runtime_globals(globals())


_sync_runtime_globals()
# SECTION 5: HISTORY PERSISTENCE
# ─────────────────────────────────────────────


def cargar_historial() -> List[Dict[str, str]]:
    """Load CHAT history from disk.

    Returns:
        List of message dicts. Empty list if the file is missing or corrupt.
    """
    try:
        if os.path.exists(HISTORIAL_PATH):
            with open(HISTORIAL_PATH, 'r', encoding='utf-8') as f:
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

    Args:
        historial: Full list of message dicts to save.
    """
    try:
        historial_recortado = historial[-MAX_HISTORIAL_MENSAJES:]
        with open(HISTORIAL_PATH, 'w', encoding='utf-8') as f:
            json.dump(historial_recortado, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Error saving history: {e}")


def limpiar_historial(historial: List[Dict[str, str]]) -> None:
    """Clear history in-place and persist the empty state.

    Args:
        historial: The history list to clear.
    """
    historial.clear()
    guardar_historial(historial)


# ─────────────────────────────────────────────



def _with_runtime_sync(func):
    def wrapper(*args, **kwargs):
        _sync_runtime_globals()
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__module__ = func.__module__
    return wrapper


for _name, _obj in list(globals().items()):
    if callable(_obj) and getattr(_obj, "__module__", None) == __name__ and _name not in {
        "_sync_runtime_globals", "_with_runtime_sync"
    }:
        globals()[_name] = _with_runtime_sync(_obj)

_sync_runtime_globals()

