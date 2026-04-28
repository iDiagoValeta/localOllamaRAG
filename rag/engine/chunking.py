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
# SECTION 6: PREPROCESSING AND CHUNKING
# ─────────────────────────────────────────────


def extraer_header_markdown(texto: str) -> str:
    """Extract the last Markdown header (``# ...``) from the text.

    Args:
        texto: Raw text that may contain Markdown headers.

    Returns:
        The last header string found, or empty string if none.
    """
    headers = re.findall(r'^(#{1,4}\s+.+)$', texto, re.MULTILINE)
    return headers[-1].strip() if headers else ""


def dividir_en_chunks(
    texto: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, str]]:
    """Split text into chunks by Markdown sections with overlap.

    Uses a recursive separator strategy to break content at natural
    boundaries (paragraphs, sentences, commas, spaces). Each chunk
    preserves its nearest Markdown header for context.

    Args:
        texto: Source text to split.
        chunk_size: Maximum character length per chunk.
        overlap: Number of trailing characters from the previous chunk
            to prepend to the next one.

    Returns:
        List of dicts with ``"text"`` and ``"header"`` keys.
    """
    if not texto or not texto.strip():
        return []

    texto = re.sub(r'~~`?[^~]*`?~~', '', texto)
    texto = re.sub(r'(?<!\*)\*{1,2}(?!\*)', '', texto)
    texto = re.sub(r'`([^`\n]{1,3})`', r'\1', texto)
    texto = re.sub(r'\n{3,}', '\n\n', texto)

    header_pattern = re.compile(
        r'^(?:#{1,4}\s+.+|\*\*(?:[A-Z0-9].+?)\*\*\s*)$',
        re.MULTILINE
    )

    secciones = []
    last_end = 0
    current_header = ""

    for match in header_pattern.finditer(texto):
        contenido_previo = texto[last_end:match.start()].strip()
        if contenido_previo:
            secciones.append({"header": current_header, "content": contenido_previo})

        raw_header = match.group(0).strip()
        current_header = re.sub(r'^\*\*(.+?)\*\*$', r'\1', raw_header).strip()
        last_end = match.end()

    contenido_final = texto[last_end:].strip()
    if contenido_final:
        secciones.append({"header": current_header, "content": contenido_final})

    if not secciones:
        secciones = [{"header": "", "content": texto.strip()}]

    separadores = ["\n\n", "\n", ". ", ".\n", "! ", "? ", "; ", ", ", " "]

    def _split_recursivo(text: str, max_size: int, depth: int = 0) -> List[str]:
        """Recursively split text using hierarchical separators."""
        if len(text) <= max_size:
            return [text] if text.strip() else []

        for sep_idx, separador in enumerate(separadores):
            if separador not in text:
                continue

            partes = text.split(separador)
            resultado = []
            chunk_actual = ""

            for i, parte in enumerate(partes):
                parte_con_sep = parte + separador if i < len(partes) - 1 else parte

                if len(chunk_actual) + len(parte_con_sep) <= max_size:
                    chunk_actual += parte_con_sep
                else:
                    if chunk_actual.strip():
                        resultado.append(chunk_actual.strip())

                    if len(parte_con_sep) > max_size and depth < len(separadores) - 1:
                        resultado.extend(_split_recursivo(parte_con_sep, max_size, depth + 1))
                        chunk_actual = ""
                    else:
                        while len(parte_con_sep) > max_size:
                            resultado.append(parte_con_sep[:max_size].strip())
                            parte_con_sep = parte_con_sep[max_size:]
                        chunk_actual = parte_con_sep

            if chunk_actual.strip():
                resultado.append(chunk_actual.strip())

            if resultado:
                return resultado

        resultado = []
        for i in range(0, len(text), max_size):
            fragmento = text[i:i + max_size].strip()
            if fragmento:
                resultado.append(fragmento)
        return resultado

    fragmentos_raw = []
    for seccion in secciones:
        header = seccion["header"]
        content = seccion["content"]

        header_prefix = f"{header}\n" if header else ""
        espacio_contenido = chunk_size - len(header_prefix)

        if espacio_contenido < MIN_CHUNK_LENGTH:
            espacio_contenido = chunk_size
            header_prefix = ""

        partes = _split_recursivo(content, espacio_contenido)

        for parte in partes:
            texto_chunk = (header_prefix + parte).strip()
            if len(texto_chunk) >= MIN_CHUNK_LENGTH:
                fragmentos_raw.append({"text": texto_chunk, "header": header})

    if not fragmentos_raw:
        if len(texto.strip()) >= MIN_CHUNK_LENGTH:
            return [{"text": texto.strip()[:chunk_size], "header": ""}]
        return []

    chunks_finales = []
    for i, frag in enumerate(fragmentos_raw):
        texto_chunk = frag["text"]

        if i > 0 and overlap > 0:
            prev_text = fragmentos_raw[i - 1]["text"]
            overlap_text = prev_text[-overlap:]
            space_idx = overlap_text.find(' ')
            if space_idx > 0:
                overlap_text = overlap_text[space_idx + 1:]
            if overlap_text and overlap_text not in texto_chunk[:overlap + 50]:
                texto_chunk = overlap_text + " " + texto_chunk

        chunks_finales.append({
            "text": texto_chunk.strip(),
            "header": frag["header"]
        })

    return chunks_finales


def expandir_con_chunks_adyacentes(
    chunk_id: str,
    metadata: Dict[str, Any],
    n_vecinos: int = 1
) -> List[str]:
    """Build IDs for neighboring chunks (same-page and cross-page) for context expansion.

    Args:
        chunk_id: Identifier of the anchor chunk.
        metadata: Metadata dict with ``source``, ``page``, ``chunk``,
            and optionally ``total_chunks_in_page``.
        n_vecinos: How many neighbors to include on each side.

    Returns:
        List of neighboring chunk IDs.
    """
    archivo = metadata['source']
    pagina = metadata['page']
    chunk_num = metadata.get('chunk', 0)
    total_in_page = metadata.get('total_chunks_in_page', None)

    ids_adyacentes = []

    for i in range(1, n_vecinos + 1):
        if chunk_num - i >= 0:
            ids_adyacentes.append(f"{archivo}_pag{pagina}_chunk{chunk_num - i}")

    if chunk_num == 0 and pagina > 0:
        for last_c in range(3):
            ids_adyacentes.append(f"{archivo}_pag{pagina - 1}_chunk{last_c}")

    if total_in_page is not None:
        for i in range(1, n_vecinos + 1):
            if chunk_num + i < total_in_page:
                ids_adyacentes.append(f"{archivo}_pag{pagina}_chunk{chunk_num + i}")

        if chunk_num >= total_in_page - 1:
            for first_c in range(min(2, n_vecinos + 1)):
                ids_adyacentes.append(f"{archivo}_pag{pagina + 1}_chunk{first_c}")
    else:
        ids_adyacentes.append(f"{archivo}_pag{pagina + 1}_chunk0")

    return ids_adyacentes


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

