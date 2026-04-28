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
OLLAMA_BASE_URL = "http://localhost:11434"


def _ollama_generate_stream(model: str, prompt: str, options: dict):
    """Stream tokens from the Ollama ``/api/generate`` endpoint.

    The system prompt is intentionally omitted here: it is baked directly
    into the model's Modelfile, so re-sending it via the API would be
    redundant and would override the Modelfile default unnecessarily.

    ``think=False`` avoids spending ``num_predict`` on a reasoning trace when
    ``MODELO_RAG`` (or any substitute) is a thinking model.

    Args:
        model: Ollama model name.
        prompt: User prompt.
        options: Generation options (temperature, top_p, etc.).

    Yields:
        Parsed JSON objects from each streamed line.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "think": False,
        "options": options,
    }
    with requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, stream=True) as resp:
        for line in resp.iter_lines():
            if line:
                yield json.loads(line)


def _preparar_mensaje_usuario_rag(pregunta: str, fragmentos: List[Dict[str, Any]]) -> str:
    """Build the user message sent to the generator: question + <context>."""
    if USAR_RECOMP_SYNTHESIS:
        contexto_str = sintetizar_contexto_recomp(fragmentos, query_usuario=pregunta)
    else:
        contexto_str = construir_contexto_para_modelo(fragmentos)
    return f"{pregunta}\n\n<context>{contexto_str}</context>"


def _generar_respuesta_stream(mensaje_usuario: str, on_token=None) -> str:
    """Stream the generator response, optionally forwarding tokens to a callback."""
    respuesta_completa = ""
    for chunk in _ollama_generate_stream(
        model=MODELO_RAG,
        prompt=mensaje_usuario,
        options={"temperature": 0.15, "top_p": 0.9, "repeat_penalty": 1.15, "num_ctx": 16384},
    ):
        content = chunk.get("response", "")
        if content:
            respuesta_completa += content
            if on_token is not None:
                on_token(content)
    return respuesta_completa


def generar_respuesta(
    pregunta: str,
    fragmentos: List[Dict[str, Any]],
    metricas: Optional[Dict[str, Any]] = None,
    on_token=None,
) -> str:
    """Generate a RAG response and optionally stream tokens through a callback.

    Builds context from fragments, streams the response via Ollama,
    optionally emits visible tokens through ``on_token``, and saves a
    debug dump.

    Args:
        pregunta: User question.
        fragmentos: Retrieved context fragments.
        metricas: Pipeline metrics for debug logging.
        on_token: Optional callable receiving each streamed token.

    Returns:
        Complete response text.
    """
    mensaje_usuario = _preparar_mensaje_usuario_rag(pregunta, fragmentos)
    respuesta_completa = _generar_respuesta_stream(mensaje_usuario, on_token=on_token)
    guardar_debug_rag(pregunta, mensaje_usuario, respuesta_completa, fragmentos, metricas=metricas)
    return respuesta_completa


def generar_respuesta_silenciosa(pregunta: str, fragmentos: List[Dict[str, Any]], metricas: Optional[Dict[str, Any]] = None) -> str:
    """Generate a RAG response silently (no terminal output).

    Same as ``generar_respuesta`` but without printing or debug dump.

    Args:
        pregunta: User question.
        fragmentos: Retrieved context fragments.
        metricas: Pipeline metrics (unused here, kept for API compat).

    Returns:
        Complete response text.
    """
    mensaje_usuario = _preparar_mensaje_usuario_rag(pregunta, fragmentos)
    return _generar_respuesta_stream(mensaje_usuario)


def evaluar_pregunta_rag(
    pregunta: str,
    collection: chromadb.Collection
) -> Tuple[str, List[str]]:
    """Run the full RAG pipeline silently for evaluation (RAGAS).

    Performs search, filtering, neighbor expansion, and generation
    without any terminal output.

    Args:
        pregunta: The evaluation question.
        collection: ChromaDB collection to search.

    Returns:
        Tuple of (response text, list of context strings).
    """
    import io
    import contextlib
    if len(pregunta.strip()) < MIN_LONGITUD_PREGUNTA_RAG:
        return ("", [])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        fragmentos_ranked, mejor_score, _ = realizar_busqueda_hibrida(pregunta, collection)
        if not fragmentos_ranked:
            return ("", [])
        # UMBRAL_RELEVANCIA is calibrated for reranker-scale scores, not RRF fusion scores.
        if (
            USAR_RERANKER
            and mejor_score < UMBRAL_RELEVANCIA
            and not EVAL_RAGBENCH_RERANKER_LOW_SCORE_FALLBACK
        ):
            return ("", [])
        if USAR_RERANKER:
            fragmentos_filtrados = [
                f for f in fragmentos_ranked
                if f.get('score_reranker', f.get('score_final', 0)) >= UMBRAL_SCORE_RERANKER
            ]
            if fragmentos_filtrados:
                fragmentos_ranked = fragmentos_filtrados
            elif not EVAL_RAGBENCH_RERANKER_LOW_SCORE_FALLBACK:
                return ("", [])
        fragmentos_finales = fragmentos_ranked[:TOP_K_FINAL]
        ids_usados = {f['id'] for f in fragmentos_finales}
        if EXPANDIR_CONTEXTO and fragmentos_finales and 'chunk' in fragmentos_finales[0]['metadata']:
            for frag in list(fragmentos_finales):  # snapshot: don't expand neighbors of neighbors
                ids_vecinos = expandir_con_chunks_adyacentes(frag['id'], frag['metadata'], n_vecinos=1)
                if ids_vecinos:
                    try:
                        vecinos = collection.get(ids=ids_vecinos, include=['documents', 'metadatas'])
                        for v_doc, v_meta in zip(vecinos['documents'], vecinos['metadatas']):
                            v_id = f"{v_meta['source']}_pag{v_meta['page']}_chunk{v_meta.get('chunk', 0)}"
                            if v_id not in ids_usados:
                                fragmentos_finales.append({
                                    'doc': v_doc, 'metadata': v_meta, 'distancia': float('inf'),
                                    'score_final': 0.0, 'id': v_id
                                })
                                ids_usados.add(v_id)
                    except Exception:
                        pass
        contexto_total = sum(len(f['doc']) for f in fragmentos_finales)
        if contexto_total > MAX_CONTEXTO_CHARS:
            fragmentos_truncados = []
            chars_acum = 0
            for f in fragmentos_finales:
                if chars_acum + len(f['doc']) > MAX_CONTEXTO_CHARS:
                    break
                fragmentos_truncados.append(f)
                chars_acum += len(f['doc'])
            fragmentos_finales = fragmentos_truncados
        respuesta = generar_respuesta_silenciosa(pregunta, fragmentos_finales)
        contexts = [f['doc'] for f in fragmentos_finales]
        return (respuesta, contexts)


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

