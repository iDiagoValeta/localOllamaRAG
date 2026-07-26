"""Characterization tests for rag/engine/chunking.py -- pre-migration snapshot.

These tests document what ``dividir_en_chunks`` and
``expandir_con_chunks_adyacentes`` DO today, not what they should do. They
exist so the hexagonal-architecture migration (see
docs/design/2026-07-26-monkeygrab-v2.md, phase F1) can move this logic
without silently changing chunk boundaries, headers or neighbor-expansion
IDs. Do not edit these assertions during the migration: if one fails, the
migrated code changed behavior.

Chunking is a pure function of its arguments (no I/O, no Ollama), so every
test here calls ``dividir_en_chunks`` with explicit ``chunk_size``/``overlap``
to stay independent of the ambient CHUNK_SIZE/CHUNK_OVERLAP configuration
(the stale-default-argument bug covered separately in
test_stale_default_config_bug.py).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag.chat_pdfs as rag


# ─────────────────────────────────────────────
# dividir_en_chunks -- separator hierarchy and overlap
# ─────────────────────────────────────────────


def test_splits_on_paragraph_boundary_when_content_exceeds_chunk_size():
    """Content that fits per-paragraph but not as a whole is split on ``\\n\\n``.

    Each paragraph individually stays above MIN_CHUNK_LENGTH (150 by default),
    so none of the recursive-splitting fallbacks kick in and the split
    boundary should land exactly between paragraphs.
    """
    p1 = ("Paragraph one covers the general background and motivation for this "
          "technical report in a reasonably long sentence structure that spans "
          "well beyond one hundred fifty characters total length here.")
    p2 = ("Paragraph two extends the discussion with additional context about the "
          "evaluation setup used across all reported experiments and also runs "
          "past the same minimum length threshold comfortably now.")
    p3 = ("Paragraph three closes the section by summarizing the key takeaways "
          "relevant to the reader before moving onward, again comfortably longer "
          "than the minimum chunk length threshold required here.")
    texto = f"{p1}\n\n{p2}\n\n{p3}"

    chunks = rag.dividir_en_chunks(texto, chunk_size=300, overlap=0)

    assert [c["text"] for c in chunks] == [p1, p2, p3]
    assert all(c["header"] == "" for c in chunks)


def test_overlap_prepends_trailing_words_of_previous_chunk():
    """Overlap takes the last N chars of the previous *full* chunk (header
    included), trims to the next whole-word boundary, and prepends it.

    This means overlap text can duplicate part of a header that was baked
    into the previous chunk -- captured here as current behavior, not a
    recommendation.
    """
    p1 = ("Paragraph one covers the general background and motivation for this "
          "technical report in a reasonably long sentence structure that spans "
          "well beyond one hundred fifty characters total length here.")
    p2 = ("Paragraph two extends the discussion with additional context about the "
          "evaluation setup used across all reported experiments and also runs "
          "past the same minimum length threshold comfortably now.")
    p3 = ("Paragraph three closes the section by summarizing the key takeaways "
          "relevant to the reader before moving onward, again comfortably longer "
          "than the minimum chunk length threshold required here.")
    texto = f"{p1}\n\n{p2}\n\n{p3}"

    chunks = rag.dividir_en_chunks(texto, chunk_size=300, overlap=50)

    assert chunks[0]["text"] == p1  # first chunk has no predecessor: no overlap
    assert chunks[1]["text"] == "one hundred fifty characters total length here. " + p2
    assert chunks[2]["text"] == "same minimum length threshold comfortably now. " + p3


def test_markdown_header_is_captured_and_prefixed_to_its_chunks():
    """A ``#``/``##`` heading becomes the chunk's ``header`` field AND is
    physically prepended to the chunk text (as ``"{header}\\n{content}"``).
    """
    texto = (
        "# Introduction\n\n"
        "This is the first paragraph of the introduction section, long enough "
        "to comfortably clear the minimum chunk length even after the header "
        "prefix is added back on top of it here.\n\n"
        "## Methods\n\n"
        "The methods section describes the experimental setup in enough detail "
        "to comfortably clear the minimum chunk length threshold even with its "
        "own header prefix counted against it."
    )

    chunks = rag.dividir_en_chunks(texto, chunk_size=500, overlap=0)

    assert [c["header"] for c in chunks] == ["# Introduction", "## Methods"]
    assert chunks[0]["text"].startswith("# Introduction\n")
    assert chunks[1]["text"].startswith("## Methods\n")


def test_bold_pseudo_header_is_not_recognized_as_a_header():
    """SURPRISE: ``**Bold**`` lines are documented as an alternate header
    syntax (see the ``header_pattern`` regex), but the markdown-noise cleanup
    that runs first (``re.sub(r'(?<!\\*)\\*{1,2}(?!\\*)', '', texto)``) strips
    ``**`` from the text BEFORE header detection ever sees it. The bold-header
    branch of ``header_pattern`` can therefore never match in practice --
    a bold line is just body text with no captured header.
    """
    texto = (
        "**Resumen**\n"
        "Contenido del resumen con suficiente longitud para superar el minimo "
        "de caracteres requerido por la configuracion actual del sistema de "
        "chunking en este momento concreto."
    )

    chunks = rag.dividir_en_chunks(texto, chunk_size=400, overlap=0)

    assert len(chunks) == 1
    assert chunks[0]["header"] == ""
    # The "**" markers are gone but the word "Resumen" survives as plain text.
    assert chunks[0]["text"].startswith("Resumen\n")


def test_pieces_below_min_chunk_length_are_dropped_and_whole_section_falls_back_to_raw_truncation():
    """SURPRISE: if every recursively-split piece of a section ends up below
    MIN_CHUNK_LENGTH once the header prefix is added, ``fragmentos_raw`` stays
    empty for that section. When this happens for *every* section, the
    function falls back to returning ONE unprocessed chunk: the raw input
    text (markdown noise, header syntax and all) truncated to ``chunk_size``,
    with ``header=""``.

    This is a real trap: a small ``chunk_size`` relative to MIN_CHUNK_LENGTH
    can silently discard proper section/header structure for content that
    would otherwise have chunked cleanly at a larger size.
    """
    texto = (
        "## Methods\n\n"
        "The methods section describes the experimental setup in detail, "
        "including data collection procedures, preprocessing steps, and the "
        "statistical tests applied to validate the results obtained during "
        "the study."
    )

    # chunk_size=200 leaves ~189 chars for content after the header prefix;
    # the recursive split (on ", ") yields two ~90-130 char pieces, both
    # under MIN_CHUNK_LENGTH (150) once you add the header back.
    chunks = rag.dividir_en_chunks(texto, chunk_size=200, overlap=0)

    assert len(chunks) == 1
    assert chunks[0]["header"] == ""
    # Raw fallback: literal "## Methods" markdown syntax survives, unlike
    # the normal path where it would be split off into the header field.
    assert chunks[0]["text"] == texto.strip()[:200]


def test_empty_and_whitespace_only_input_produce_no_chunks():
    assert rag.dividir_en_chunks("", chunk_size=300, overlap=50) == []
    assert rag.dividir_en_chunks("   \n  ", chunk_size=300, overlap=50) == []


# ─────────────────────────────────────────────
# expandir_con_chunks_adyacentes -- neighbor ID construction
# ─────────────────────────────────────────────


def test_expand_neighbors_mid_page_returns_prev_and_next_chunk_ids():
    metadata = {"source": "paperA.pdf", "page": 2, "chunk": 3, "total_chunks_in_page": 6}

    ids = rag.expandir_con_chunks_adyacentes("paperA.pdf_pag2_chunk3", metadata, n_vecinos=1)

    assert ids == ["paperA.pdf_pag2_chunk2", "paperA.pdf_pag2_chunk4"]


def test_expand_neighbors_first_chunk_of_page_pulls_last_three_of_previous_page():
    """When ``chunk == 0`` and this isn't page 0, the function assumes (does
    not verify) the previous page had at least 3 chunks and always requests
    chunk indices 0, 1, 2 of the previous page -- plus the normal next-chunk
    lookup for the current page since ``total_chunks_in_page`` is known here.
    """
    metadata = {"source": "paperA.pdf", "page": 2, "chunk": 0, "total_chunks_in_page": 6}

    ids = rag.expandir_con_chunks_adyacentes("paperA.pdf_pag2_chunk0", metadata, n_vecinos=1)

    assert ids == [
        "paperA.pdf_pag1_chunk0",
        "paperA.pdf_pag1_chunk1",
        "paperA.pdf_pag1_chunk2",
        "paperA.pdf_pag2_chunk1",
    ]


def test_expand_neighbors_last_chunk_of_page_pulls_first_two_of_next_page():
    metadata = {"source": "paperA.pdf", "page": 2, "chunk": 5, "total_chunks_in_page": 6}

    ids = rag.expandir_con_chunks_adyacentes("paperA.pdf_pag2_chunk5", metadata, n_vecinos=1)

    assert ids == [
        "paperA.pdf_pag2_chunk4",
        "paperA.pdf_pag3_chunk0",
        "paperA.pdf_pag3_chunk1",
    ]


def test_expand_neighbors_without_total_chunks_in_page_only_guesses_next_page_chunk0():
    """Without ``total_chunks_in_page`` (e.g. legacy metadata), the function
    can't tell if this is the last chunk of the page, so it always appends a
    guess at the next page's first chunk in addition to the previous chunk.
    """
    metadata = {"source": "paperA.pdf", "page": 1, "chunk": 2}

    ids = rag.expandir_con_chunks_adyacentes("paperA.pdf_pag1_chunk2", metadata, n_vecinos=1)

    assert ids == ["paperA.pdf_pag1_chunk1", "paperA.pdf_pag2_chunk0"]


def test_expand_neighbors_page_zero_chunk_zero_has_no_previous_page_fallback():
    """Page 0 is never treated as having a "previous page" even at chunk 0."""
    metadata = {"source": "paperA.pdf", "page": 0, "chunk": 0, "total_chunks_in_page": 4}

    ids = rag.expandir_con_chunks_adyacentes("paperA.pdf_pag0_chunk0", metadata, n_vecinos=1)

    assert ids == ["paperA.pdf_pag0_chunk1"]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
