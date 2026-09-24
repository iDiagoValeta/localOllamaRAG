"""Auxiliary implementation module for rag.chat_pdfs.

This module keeps business logic split out of the public facade. Runtime
configuration stays owned by rag.chat_pdfs and is read lazily through ``cfg``
(a live reference to that module), so web/API toggles and test monkeypatches
are observed without any per-call synchronization.
"""

import logging
import re
import requests
from typing import Any, Dict, List

from monkeygrab.application.context_assembly import (
    _es_continuacion_parrafo as _es_continuacion_parrafo,
    _marcar_fragmento_incompleto as _marcar_fragmento_incompleto,
    _reunir_parrafos as _reunir_parrafos,
    _texto_fuente_fragmento as _texto_fuente_fragmento,
    RECOMP_FACTS_HEADER as _RECOMP_FACTS_HEADER,
    build_context_for_model,
    normalize_recomp_output as _normalizar_salida_recomp,
    optimize_context_text,
    strip_ollama_think_blocks as _strip_ollama_think_blocks,
)
from monkeygrab.domain.chunk_metadata import ChunkMetadata
from monkeygrab.domain.fragment import Fragment
from rag.engine.runtime import get_runtime

cfg = get_runtime()
# CONTEXT AND GENERATION


def optimizar_texto_contexto(texto: str) -> str:
    """Remove PDF noise (box artifacts, footers, double spacing).

    Typical savings are 30-50% of characters.

    Args:
        texto: Raw text extracted from a PDF chunk.

    Returns:
        Cleaned text ready for LLM context.
    """
    # Implementation lives in monkeygrab.application.context_assembly
    # (optimize_context_text) -- a literal port, no config dependency.
    return optimize_context_text(texto)


def construir_contexto_para_modelo(fragmentos: List[Dict[str, Any]]) -> str:
    """Build the context string for the RAG model from retrieved fragments.

    Output format per fragment::

        --- [Fragment N] ---
        [Fragment Context]            <- only if Contextual Retrieval summary exists
        ...
        [Source Text]
        ...
        [excerpt ends mid-sentence]   <- only if the chunk is truncated

    Fragments are separated by double newlines. PDF text optimization
    is applied when ``USAR_OPTIMIZACION_CONTEXTO`` is enabled.

    Args:
        fragmentos: List of retrieved chunk dicts with ``doc`` and ``metadata``.

    Returns:
        Formatted context string ready for the ``<context>`` tag.
    """
    # Implementation lives in monkeygrab.application.context_assembly
    # (build_context_for_model). Fragment/ChunkMetadata here only need to
    # carry source/page/chunk/doc -- the sorting key and formatting logic
    # never touch any other metadata field -- so the conversion is lossless
    # for what this function returns (a plain string).
    fragmentos_dominio = [
        Fragment(
            doc=f['doc'],
            metadata=ChunkMetadata(
                source=f['metadata']['source'],
                page=f['metadata']['page'],
                chunk=f['metadata'].get('chunk', 0),
            ),
        )
        for f in fragmentos
    ]
    resultado, metrics = build_context_for_model(fragmentos_dominio, cfg.USAR_OPTIMIZACION_CONTEXTO)

    chars_original = metrics['chars_original']
    chars_optimizado = metrics['chars_optimized']
    if chars_original > 0 and cfg.LOGGING_METRICAS:
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
    if not cfg.USAR_RECOMP_SYNTHESIS or not fragmentos:
        return cfg.construir_contexto_para_modelo(fragmentos)

    textos_preparados = []
    for f in fragmentos:
        cuerpo = _texto_fuente_fragmento(f.get("doc", "") or "")
        if cfg.USAR_OPTIMIZACION_CONTEXTO:
            cuerpo = optimizar_texto_contexto(cuerpo)
            cuerpo = _marcar_fragmento_incompleto(cuerpo)
        content = cuerpo.replace("\n", " ").strip()
        content = re.sub(r'\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]', '', content)  # strip citation markers [38, 2, 9]
        if content:
            n = len(textos_preparados) + 1
            textos_preparados.append(f"Fragment {n}:\n{content}")

    contexto_raw = "\n\n".join(textos_preparados)
    if not contexto_raw.strip():
        return cfg.construir_contexto_para_modelo(fragmentos)

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
            "model": cfg.MODELO_RECOMP,
            "keep_alive": cfg.OLLAMA_KEEP_ALIVE,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 1500,
                "top_p": 0.9,
                "repeat_penalty": 1.15,
                "num_ctx": cfg.OLLAMA_RECOMP_NUM_CTX,
            },
        }
        resp = requests.post(f"{cfg.OLLAMA_BASE_URL}/api/chat", json=payload, timeout=cfg.OLLAMA_REQUEST_TIMEOUT)
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "")

        sintesis = _strip_ollama_think_blocks(raw.strip())
        sintesis = _normalizar_salida_recomp(sintesis)

        if len(sintesis) < 20:
            logging.info(
                "RECOMP: falling back to raw chunks (synthesis too short after "
                "stripping think blocks; check %s / OLLAMA_RECOMP_MODEL)",
                cfg.MODELO_RECOMP,
            )
            return cfg.construir_contexto_para_modelo(fragmentos)

        if _RECOMP_FACTS_HEADER.lower() not in sintesis.lower():
            logging.info(
                "RECOMP: falling back to raw chunks (missing '%s' in model output)",
                _RECOMP_FACTS_HEADER,
            )
            return cfg.construir_contexto_para_modelo(fragmentos)

        return sintesis

    except Exception as e:
        logging.warning(f"Critical error in RECOMP synthesis ({cfg.MODELO_RECOMP}): {e}")
        return cfg.construir_contexto_para_modelo(fragmentos)


