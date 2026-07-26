"""context_assembly -- fragment text cleanup and final context-string formatting.

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  +-- SECTION 1: PARAGRAPH REJOINING   -- undo PDF-extraction line breaks
#  +-- SECTION 2: TEXT OPTIMIZATION     -- strip PDF noise
#  +-- SECTION 3: FRAGMENT FORMATTING   -- per-fragment marking/splitting
#  +-- SECTION 4: CONTEXT ASSEMBLY      -- build_context_for_model
#  +-- SECTION 5: RECOMP TEXT HELPERS   -- pure helpers used by Answer's RECOMP step
#
# ─────────────────────────────────────────────

Moved verbatim from ``rag/engine/context.py`` (all pure string transforms;
no I/O, no Ollama). ``build_context_for_model`` takes ``Sequence[Fragment]``
instead of the original's ``List[Dict]`` -- same fields, read via
``Fragment.doc``/``Fragment.metadata`` instead of dict keys.

``rag.engine.context.sintetizar_contexto_recomp`` itself (the function that
calls an LLM to synthesize a facts briefing) is not moved here -- prompting
an LLM is orchestration, owned by the ``Answer`` use case, which composes
these pure helpers around a ``ChatModel`` port call.
"""

import re
from typing import Any, Dict, List, Sequence, Tuple

from monkeygrab.domain.fragment import Fragment


# ─────────────────────────────────────────────
# SECTION 1: PARAGRAPH REJOINING
# ─────────────────────────────────────────────


def _es_continuacion_parrafo(linea_previa: str, linea_actual: str) -> bool:
    """Heuristic to detect whether the current line continues a paragraph.

    Avoids false paragraph breaks caused by PDF extraction double-spacing.
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
    """Re-join lines split by PDF extraction using look-ahead for continuations."""
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


# ─────────────────────────────────────────────
# SECTION 2: TEXT OPTIMIZATION
# ─────────────────────────────────────────────


def optimize_context_text(texto: str) -> str:
    """Remove PDF noise (box artifacts, footers, double spacing).

    Moved from ``rag.engine.context.optimizar_texto_contexto``. Typical
    savings are 30-50% of characters.

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

    # SURPRISE (see test_optimizar_does_not_strip_standalone_page_number_
    # followed_by_more_text): this only strips a page number that is the
    # LAST line of the text. A page number sandwiched between two paragraphs
    # gets glued onto the next paragraph by _reunir_parrafos above before
    # this regex ever sees it standalone, and survives. Reproduced as-is.
    texto = re.sub(r'(?m)^\s*\d{1,3}\s*$', '', texto)   # standalone PDF page numbers

    texto = re.sub(r'\n{3,}', '\n\n', texto)

    return texto.strip()


# ─────────────────────────────────────────────
# SECTION 3: FRAGMENT FORMATTING
# ─────────────────────────────────────────────


def _marcar_fragmento_incompleto(texto: str) -> str:
    """Append ``[excerpt ends mid-sentence]`` if text lacks closing punctuation."""
    stripped = texto.rstrip()
    if not stripped:
        return texto
    _CLOSING = frozenset('.?!:")]')
    if stripped[-1] not in _CLOSING and not stripped[-1].isdigit():
        return texto + '\n[excerpt ends mid-sentence]'
    return texto


def _texto_fuente_fragmento(doc: str) -> str:
    """Return chunk body without the contextual-retrieval summary prefix.

    Indexed chunks store ``<summary>\\n\\n<body>`` using the *literal* 4-char
    sequence ``\\n\\n`` (backslash-n-backslash-n) as separator, intentionally
    distinct from real paragraph breaks (``\n\n``). Do NOT replace with real
    newlines: PDF body text naturally contains ``\n\n``, so a real-newline
    separator would produce false splits on the first paragraph break.
    """
    if "\\n\\n" in doc:
        _, cuerpo = doc.split("\\n\\n", 1)
        return cuerpo.strip()
    return doc.strip()


# ─────────────────────────────────────────────
# SECTION 4: CONTEXT ASSEMBLY
# ─────────────────────────────────────────────


def build_context_for_model(
    fragmentos: Sequence[Fragment],
    optimize_text: bool,
) -> Tuple[str, Dict[str, Any]]:
    """Build the context string for the RAG model from retrieved fragments.

    Moved from ``rag.engine.context.construir_contexto_para_modelo``. Output
    format per fragment::

        --- [Fragment N] ---
        [Fragment Context]            <- only if Contextual Retrieval summary exists
        ...
        [Source Text]
        ...
        [excerpt ends mid-sentence]   <- only if the chunk is truncated

    Fragments are separated by double newlines.

    Unlike the original, this does not log a savings percentage as a side
    effect (``LOGGING_METRICAS``-gated ``logging.info`` call) -- pure
    functions in this layer return their metrics instead of emitting them;
    see the module docstring of ``monkeygrab.application`` for the project-
    wide decision on where metrics/logging live.

    Args:
        fragmentos: Retrieved fragments.
        optimize_text: Whether to run ``optimize_context_text`` per fragment
            (``USAR_OPTIMIZACION_CONTEXTO``).

    Returns:
        Tuple of (formatted context string ready for the ``<context>`` tag,
        metrics dict with ``chars_original``/``chars_optimized``).
    """
    fragmentos_ordenados = sorted(
        fragmentos,
        key=lambda f: (f.metadata.source, f.metadata.page, f.metadata.chunk),
    )

    textos_originales = [frag.doc for frag in fragmentos_ordenados]
    chars_original = sum(len(t) for t in textos_originales)

    fragmentos_formateados = []
    for i, frag in enumerate(fragmentos_ordenados, 1):
        texto = optimize_context_text(frag.doc) if optimize_text else frag.doc
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

    metrics = {"chars_original": chars_original, "chars_optimized": len(resultado)}
    return resultado, metrics


# ─────────────────────────────────────────────
# SECTION 5: RECOMP TEXT HELPERS
# ─────────────────────────────────────────────


def strip_ollama_think_blocks(text: str) -> str:
    """Remove ``<think>...</think>`` wrappers often emitted by Qwen-style Ollama models.

    Moved from ``rag.engine.context._strip_ollama_think_blocks``.
    """
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


RECOMP_FACTS_HEADER = "## Facts relevant to the question"


def normalize_recomp_output(texto: str) -> str:
    """Ensure a RECOMP briefing has the expected markdown header when possible.

    Moved from ``rag.engine.context._normalizar_salida_recomp``.
    """
    t = texto.strip()
    if not t:
        return t
    if RECOMP_FACTS_HEADER.lower() in t.lower():
        return t
    if re.search(r"^-\s+\S", t, flags=re.MULTILINE):
        return f"{RECOMP_FACTS_HEADER}\n{t}"
    return t
