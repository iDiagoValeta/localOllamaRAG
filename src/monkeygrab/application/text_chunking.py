"""text_chunking -- hierarchical markdown chunking and neighbor-id expansion.

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  +-- TextChunk               -- chunking output before position metadata exists
#  +-- split_markdown_into_chunks  -- moved from rag.engine.chunking.dividir_en_chunks
#  +-- adjacent_chunk_ids          -- moved from rag.engine.chunking.expandir_con_chunks_adyacentes
#
# ─────────────────────────────────────────────

Both functions are a literal port of ``rag/engine/chunking.py``, with every
``cfg.*`` read replaced by an explicit parameter -- the structural fix for
the stale-default-argument bug documented in
``tests/characterization/test_stale_default_config_bug.py`` (a function that
wants config now takes it as a required parameter, so there is no default to
go stale against). Equivalence with the original is asserted field-for-field
in ``tests/unit/application/test_text_chunking_equivalence.py``.

Every surprising behavior documented by
``tests/characterization/test_chunking.py`` is reproduced on purpose, not
fixed here: the raw-truncation fallback when every recursively-split piece
of every section falls below ``min_chunk_length``, the unreachable
bold-pseudo-header branch, and overlap being taken from the previous
*chunk's rendered text* (header included) rather than its body. Fixing any
of these would break the equivalence this migration exists to prove; see the
inline comments at each site for the specific characterization test that
pins it.
"""

import re
from dataclasses import dataclass
from typing import List

from monkeygrab.domain.chunk_metadata import ChunkMetadata


# ─────────────────────────────────────────────
# TEXT CHUNK
# ─────────────────────────────────────────────


@dataclass(frozen=True)
class TextChunk:
    """One chunk of split text, before positional metadata (page, index) exists.

    Mirrors the ``{"text": ..., "header": ...}`` dicts
    ``rag.engine.chunking.dividir_en_chunks`` returns. Not a ``monkeygrab.
    domain.Chunk``: that entity requires a ``ChunkMetadata`` (source, page,
    chunk index) that only ``IndexCorpus`` can assign once it knows this
    piece's position within the document.

    Attributes:
        text: The chunk's text, header already prepended when a header applies.
        header: The nearest Markdown header above this chunk (``""`` if none).
    """

    text: str
    header: str


# ─────────────────────────────────────────────
# MARKDOWN CHUNKING
# ─────────────────────────────────────────────


def split_markdown_into_chunks(
    texto: str,
    chunk_size: int,
    overlap: int,
    min_chunk_length: int,
) -> List[TextChunk]:
    """Split text into chunks by Markdown sections with overlap.

    Uses a recursive separator strategy to break content at natural
    boundaries (paragraphs, sentences, commas, spaces). Each chunk preserves
    its nearest Markdown header for context.

    Literal port of ``rag.engine.chunking.dividir_en_chunks``; every
    ``cfg.MIN_CHUNK_LENGTH`` read becomes the ``min_chunk_length`` parameter.

    Args:
        texto: Source text to split.
        chunk_size: Maximum character length per chunk.
        overlap: Number of trailing characters from the previous chunk to
            prepend to the next one.
        min_chunk_length: Minimum character length for a chunk to be kept.

    Returns:
        List of ``TextChunk``.
    """
    if not texto or not texto.strip():
        return []

    texto = re.sub(r'~~`?[^~]*`?~~', '', texto)
    # Strips single/double "*" runs -- this is what makes the bold-pseudo-header
    # branch of header_pattern below unreachable in practice (the "**" markers
    # are gone before header detection ever runs); see
    # test_bold_pseudo_header_is_not_recognized_as_a_header.
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

        if espacio_contenido < min_chunk_length:
            espacio_contenido = chunk_size
            header_prefix = ""

        partes = _split_recursivo(content, espacio_contenido)

        for parte in partes:
            texto_chunk = (header_prefix + parte).strip()
            if len(texto_chunk) >= min_chunk_length:
                fragmentos_raw.append({"text": texto_chunk, "header": header})

    if not fragmentos_raw:
        # SURPRISE (see test_pieces_below_min_chunk_length_are_dropped_...):
        # if every section's pieces landed below min_chunk_length, fall back
        # to ONE raw-truncated chunk with header="" -- markdown noise and
        # header syntax survive uncleaned. Reproduced as-is.
        if len(texto.strip()) >= min_chunk_length:
            return [TextChunk(text=texto.strip()[:chunk_size], header="")]
        return []

    chunks_finales = []
    for i, frag in enumerate(fragmentos_raw):
        texto_chunk = frag["text"]

        if i > 0 and overlap > 0:
            # Overlap is taken from the previous chunk's *rendered* text
            # (header prefix included, per test_overlap_prepends_trailing_
            # words_of_previous_chunk), not its body alone.
            prev_text = fragmentos_raw[i - 1]["text"]
            overlap_text = prev_text[-overlap:]
            space_idx = overlap_text.find(' ')
            if space_idx > 0:
                overlap_text = overlap_text[space_idx + 1:]
            if overlap_text and overlap_text not in texto_chunk[:overlap + 50]:
                texto_chunk = overlap_text + " " + texto_chunk

        chunks_finales.append(TextChunk(text=texto_chunk.strip(), header=frag["header"]))

    return chunks_finales


# ─────────────────────────────────────────────
# NEIGHBOR EXPANSION
# ─────────────────────────────────────────────


def adjacent_chunk_ids(metadata: ChunkMetadata, n_neighbors: int = 1) -> List[str]:
    """Build IDs for neighboring chunks (same-page and cross-page) for context expansion.

    Literal port of ``rag.engine.chunking.expandir_con_chunks_adyacentes``.
    The original also takes a ``chunk_id`` parameter, but its body never
    reads it (every id it builds is derived from ``metadata`` alone) -- it is
    dropped here rather than reproduced as unused.

    Args:
        metadata: Position/format of the anchor chunk.
        n_neighbors: How many neighbors to include on each side.

    Returns:
        List of neighboring chunk IDs.
    """
    archivo = metadata.source
    pagina = metadata.page
    chunk_num = metadata.chunk
    total_in_page = metadata.total_chunks_in_page

    ids_adyacentes: List[str] = []

    for i in range(1, n_neighbors + 1):
        if chunk_num - i >= 0:
            ids_adyacentes.append(f"{archivo}_pag{pagina}_chunk{chunk_num - i}")

    if chunk_num == 0 and pagina > 0:
        for last_c in range(3):
            ids_adyacentes.append(f"{archivo}_pag{pagina - 1}_chunk{last_c}")

    if total_in_page is not None:
        for i in range(1, n_neighbors + 1):
            if chunk_num + i < total_in_page:
                ids_adyacentes.append(f"{archivo}_pag{pagina}_chunk{chunk_num + i}")

        if chunk_num >= total_in_page - 1:
            for first_c in range(min(2, n_neighbors + 1)):
                ids_adyacentes.append(f"{archivo}_pag{pagina + 1}_chunk{first_c}")
    else:
        ids_adyacentes.append(f"{archivo}_pag{pagina + 1}_chunk0")

    return ids_adyacentes
