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
# SECTION 10: CONTEXT AND GENERATION
# ─────────────────────────────────────────────


def _es_continuacion_parrafo(linea_previa: str, linea_actual: str) -> bool:
    """Heuristic to detect whether the current line continues a paragraph.

    Avoids false paragraph breaks caused by PDF extraction double-spacing.

    Args:
        linea_previa: The preceding line (right-stripped).
        linea_actual: The current line (stripped).

    Returns:
        ``True`` if the current line likely continues the paragraph.
    """
    if not linea_previa or not linea_actual:
        return False

    prev_stripped = linea_previa.rstrip()
    curr_stripped = linea_actual.strip()

    if not prev_stripped or not curr_stripped:
        return False

    prev_end = prev_stripped[-1]

    if prev_end in '.?!':
        return False

    if re.match(r'^\d+[.\)]', curr_stripped):
        return False
    if curr_stripped.startswith('#'):
        return False
    if re.match(r'^[-*•]\s', curr_stripped):
        return False
    if curr_stripped.startswith('**'):
        return False

    if prev_end in ',;(':
        return True
    if curr_stripped[0].islower():
        return True

    if prev_end not in '.?!:':
        return True

    return False


def _reunir_parrafos(texto: str) -> str:
    """Re-join lines split by PDF extraction using look-ahead for continuations.

    Args:
        texto: Text with potentially broken paragraph lines.

    Returns:
        Text with properly joined paragraphs.
    """
    lines = texto.split('\n')
    result = []
    buffer = ""

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if not stripped:
            if buffer:
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1

                if j < len(lines) and _es_continuacion_parrafo(buffer, lines[j].strip()):
                    buffer += ' ' + lines[j].strip()
                    i = j + 1
                    continue
                else:
                    result.append(buffer)
                    buffer = ""
                    result.append("")
                    i = j if j < len(lines) else i + 1
                    continue
            else:
                result.append("")
                i += 1
                continue

        if not buffer:
            buffer = stripped
        elif _es_continuacion_parrafo(buffer, stripped):
            buffer += ' ' + stripped
        else:
            result.append(buffer)
            buffer = stripped

        i += 1

    if buffer:
        result.append(buffer)

    return '\n'.join(result)


def optimizar_texto_contexto(texto: str) -> str:
    """Remove PDF noise (box artifacts, footers, double spacing).

    Typical savings are 30-50% of characters.

    Args:
        texto: Raw text extracted from a PDF chunk.

    Returns:
        Cleaned text ready for LLM context.
    """
    texto = re.sub(r'#{1,6}\s*□', '', texto)
    texto = re.sub(r'^□\s*$', '', texto, flags=re.MULTILINE)

    texto = re.sub(r'^#{1,6}\s+', '', texto, flags=re.MULTILINE)

    texto = re.sub(
        r'^[A-ZÀ-Ú][a-zà-ú]+ [A-ZÀ-Ú][a-zà-ú]+,\s+[A-ZÀ-Ú].*?\d+\s*/\s*\d+\s+\w+.*$',
        '', texto, flags=re.MULTILINE
    )

    texto = re.sub(r'\*{0,2}Solución:?\*{0,2}\s*La\s+\d+\s*', '', texto)

    texto = re.sub(r'\*\*(.+?)\*\*', r'\1', texto)

    texto = re.sub(r'[ \t]{2,}', ' ', texto)

    texto = re.sub(r'[ \t]+$', '', texto, flags=re.MULTILINE)

    texto = _reunir_parrafos(texto)

    texto = re.sub(r'^\s+$', '', texto, flags=re.MULTILINE)

    texto = re.sub(r'(?m)^\s*\d{1,3}\s*$', '', texto)   # standalone PDF page numbers

    texto = re.sub(r'\n{3,}', '\n\n', texto)

    return texto.strip()


def _marcar_fragmento_incompleto(texto: str) -> str:
    """Append ``[incomplete fragment]`` if text lacks closing punctuation.

    Args:
        texto: Fragment text to inspect.

    Returns:
        Original text, possibly with an appended incompleteness marker.
    """
    stripped = texto.rstrip()
    if not stripped:
        return texto
    _CLOSING = frozenset('.?!:")]')
    if stripped[-1] not in _CLOSING and not stripped[-1].isdigit():
        return texto + '\n[excerpt ends mid-sentence]'
    return texto


def _texto_fuente_fragmento(doc: str) -> str:
    """Return chunk body without the contextual-retrieval summary prefix.

    Indexed chunks store ``<summary>\\n\\n<body>`` using the *literal* 6-char
    sequence ``\\n\\n`` (backslash-n-backslash-n) as separator, intentionally
    distinct from real paragraph breaks (``\n\n``).  Do NOT replace with real
    newlines: PDF body text naturally contains ``\n\n``, so a real-newline
    separator would produce false splits on the first paragraph break.

    Args:
        doc: Raw ``fragment['doc']`` string.

    Returns:
        Source body text, or full ``doc`` if no separator is present.
    """
    if "\\n\\n" in doc:
        _, cuerpo = doc.split("\\n\\n", 1)
        return cuerpo.strip()
    return doc.strip()


def _strip_ollama_think_blocks(text: str) -> str:
    """Remove ``...</think>`` wrappers often emitted by Qwen-style Ollama models."""
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


_RECOMP_FACTS_HEADER = "## Facts relevant to the question"


def _normalizar_salida_recomp(texto: str) -> str:
    """Ensure RECOMP briefing has the expected markdown header when possible."""
    t = texto.strip()
    if not t:
        return t
    if _RECOMP_FACTS_HEADER.lower() in t.lower():
        return t
    if re.search(r"^-\s+\S", t, flags=re.MULTILINE):
        return f"{_RECOMP_FACTS_HEADER}\n{t}"
    return t


def construir_contexto_para_modelo(fragmentos: List[Dict[str, Any]]) -> str:
    """Build the context string for the RAG model from retrieved fragments.

    Output format per fragment::

        --- [Fragment N] ---
        [Fragment Context]            <- only if Contextual Retrieval summary exists
        ...
        [Source Text]
        ...
        [incomplete fragment]         <- only if the chunk is truncated

    Fragments are separated by double newlines. PDF text optimization
    is applied when ``USAR_OPTIMIZACION_CONTEXTO`` is enabled.

    Args:
        fragmentos: List of retrieved chunk dicts with ``doc`` and ``metadata``.

    Returns:
        Formatted context string ready for the ``<context>`` tag.
    """
    fragmentos_ordenados = sorted(
        fragmentos,
        key=lambda f: (f['metadata']['source'], f['metadata']['page'], f['metadata'].get('chunk', 0))
    )

    textos_originales = [frag['doc'] for frag in fragmentos_ordenados]
    chars_original = sum(len(t) for t in textos_originales)

    fragmentos_formateados = []
    for i, frag in enumerate(fragmentos_ordenados, 1):
        texto = optimizar_texto_contexto(frag['doc']) if USAR_OPTIMIZACION_CONTEXTO else frag['doc']
        if not texto:
            continue

        if '\\n\\n' in texto:
            ctx_summary, raw_content = texto.split('\\n\\n', 1)
            raw_content = _marcar_fragmento_incompleto(raw_content.strip())
            texto = f"[Fragment Context]\n{ctx_summary.strip()}\n\n[Source Text]\n{raw_content}"
        else:
            texto = _marcar_fragmento_incompleto(texto.strip())

        fragmentos_formateados.append(f"--- [Fragment {i}] ---\n{texto}")

    resultado = "\n\n".join(fragmentos_formateados)

    chars_optimizado = len(resultado)
    if chars_original > 0 and LOGGING_METRICAS:
        ahorro = chars_original - chars_optimizado
        pct = (ahorro / chars_original) * 100 if ahorro > 0 else 0
        logging.info(
            f"Optimized context: {chars_original} -> {chars_optimizado} chars "
            f"({ahorro} saved, {pct:.1f}%)"
        )

    return resultado


def sintetizar_contexto_recomp(fragmentos: List[Dict[str, Any]], query_usuario: str = "") -> str:
    """Synthesize context using MODELO_RECOMP instead of raw chunks.

    Uses the original user question and a fixed markdown outline (``## Facts
    relevant to the question`` + bullets). Evidence is taken from chunk
    *body* only, omitting contextual-retrieval summaries, to avoid
    meta-descriptive prose in the briefing.

    Falls back to ``construir_contexto_para_modelo`` if synthesis is
    disabled, fails, or produces too little output.

    Args:
        fragmentos: Retrieved chunk dicts.
        query_usuario: Original user question (required for focused synthesis).

    Returns:
        Synthesized context string or raw formatted context on fallback.
    """
    if not USAR_RECOMP_SYNTHESIS or not fragmentos:
        return construir_contexto_para_modelo(fragmentos)

    textos_preparados = []
    for f in fragmentos:
        cuerpo = _texto_fuente_fragmento(f.get("doc", "") or "")
        content = cuerpo.replace("\n", " ").strip()
        content = re.sub(r'\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]', '', content)  # strip citation markers [38, 2, 9]
        if content:
            n = len(textos_preparados) + 1
            textos_preparados.append(f"Fragment {n}:\n{content}")

    contexto_raw = "\n\n".join(textos_preparados)
    if not contexto_raw.strip():
        return construir_contexto_para_modelo(fragmentos)

    q = (query_usuario or "").strip()
    bloque_pregunta = (
        f"## User question\n{q}\n"
        if q
        else "## User question\n(No question provided; extract the main technical facts from the excerpts.)\n"
    )

    system_prompt = (
        "You compress retrieved document excerpts into a brief briefing for a downstream "
        "answer model.\n"
        "GROUNDING:\n"
        "- Use ONLY information stated in the evidence excerpts. No outside knowledge.\n"
        "- Preserve technical terms, notation, formulas, and numbers exactly as written.\n"
        "ENUMERATION (critical):\n"
        "- If the question asks for a list or a count (e.g. 'three types', 'two ways'), "
        "search ALL fragments and enumerate EVERY item you find, even if items are spread "
        "across different fragments. Never say an item 'is not mentioned' if it appears "
        "anywhere in the excerpts.\n"
        "- When an excerpt ends with [excerpt ends mid-sentence], the list may continue in "
        "another fragment — collect items from ALL fragments before writing your bullets.\n"
        "STYLE:\n"
        "- Write ONLY facts that help answer the user question. Do NOT describe the documents, "
        "the paper, or the excerpts (forbidden openers: \"This paper\", \"The excerpt\", "
        "\"This section\", \"The document\", \"The text\", \"The fragment\").\n"
        "- Do NOT restate meta-summaries; every bullet must be substantive content from the excerpts.\n"
        "- Do NOT cite fragment numbers, sources, or page numbers.\n"
        "OUTPUT FORMAT (exactly this structure, markdown):\n"
        "## Facts relevant to the question\n"
        "- (first grounded fact)\n"
        "- (second grounded fact)\n"
        "Use one bullet per distinct fact; merge duplicates. If nothing in the excerpts bears "
        "on the question, output exactly one bullet: "
        "\"Insufficient evidence in the excerpts to answer the question.\"\n"
        "Language: same as the evidence excerpts (or the user question if excerpts mix languages)."
    )

    user_prompt = (
        f"{bloque_pregunta}"
        "## Evidence excerpts (verbatim from retrieval; may be partial)\n"
        f"{contexto_raw}\n\n"
        "Produce the briefing using the required OUTPUT FORMAT. No text before "
        "## Facts relevant to the question."
    )

    try:
        payload = {
            "model": MODELO_RECOMP,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 10000,
                "top_p": 0.9,
                "repeat_penalty": 1.15,
                "num_ctx": 16384,
            },
        }
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=60)
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "")

        sintesis = _strip_ollama_think_blocks(raw.strip())
        sintesis = _normalizar_salida_recomp(sintesis)

        if len(sintesis) < 20:
            logging.info(
                "RECOMP: falling back to raw chunks (synthesis too short after "
                "stripping think blocks; check %s / OLLAMA_RECOMP_MODEL)",
                MODELO_RECOMP,
            )
            return construir_contexto_para_modelo(fragmentos)

        if _RECOMP_FACTS_HEADER.lower() not in sintesis.lower():
            logging.info(
                "RECOMP: falling back to raw chunks (missing '%s' in model output)",
                _RECOMP_FACTS_HEADER,
            )
            return construir_contexto_para_modelo(fragmentos)

        return sintesis

    except Exception as e:
        logging.warning(f"Critical error in RECOMP synthesis ({MODELO_RECOMP}): {e}")
        return construir_contexto_para_modelo(fragmentos)



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

