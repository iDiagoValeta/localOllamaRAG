"""Characterization tests for rag/engine/context.py -- pre-migration snapshot.

Covers ``construir_contexto_para_modelo`` (final fragment formatting sent to
the generator) and ``optimizar_texto_contexto`` (PDF noise cleanup). Pure
string transforms; no I/O, no Ollama.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag.chat_pdfs as rag

# Indexed chunks store "<summary><LITERAL \n\n><body>" where the separator is
# the literal 4-character sequence backslash-n-backslash-n, deliberately
# distinct from a real paragraph break. Building it explicitly here (rather
# than typing "\\n\\n" inline) keeps the test's intent obvious.
CONTEXTUAL_SEP = "\\n\\n"


# ─────────────────────────────────────────────
# construir_contexto_para_modelo
# ─────────────────────────────────────────────


def test_context_is_sorted_by_source_page_chunk_and_renumbered_from_one():
    """Fragments are re-sorted by (source, page, chunk) regardless of the
    order retrieval ranked them in -- the RAG generator sees document order,
    not relevance order."""
    fragmentos = [
        {"doc": "Body from b.pdf page 1.", "metadata": {"source": "b.pdf", "page": 1, "chunk": 0}},
        {"doc": "Second chunk of a.pdf page 0.", "metadata": {"source": "a.pdf", "page": 0, "chunk": 1}},
        {"doc": "First chunk of a.pdf page 0.", "metadata": {"source": "a.pdf", "page": 0, "chunk": 0}},
    ]

    resultado = rag.construir_contexto_para_modelo(fragmentos)

    assert resultado == (
        "--- [Fragment 1] ---\n"
        "First chunk of a.pdf page 0.\n\n"
        "--- [Fragment 2] ---\n"
        "Second chunk of a.pdf page 0.\n\n"
        "--- [Fragment 3] ---\n"
        "Body from b.pdf page 1."
    )


def test_context_splits_contextual_retrieval_summary_from_source_body():
    """A fragment whose ``doc`` contains the literal contextual-retrieval
    separator is rendered as two labeled sections instead of one block."""
    fragmentos = [{
        "doc": f"Summary about the topic{CONTEXTUAL_SEP}Actual body text without terminal punctuation",
        "metadata": {"source": "a.pdf", "page": 0, "chunk": 0},
    }]

    resultado = rag.construir_contexto_para_modelo(fragmentos)

    assert resultado == (
        "--- [Fragment 1] ---\n"
        "[Fragment Context]\n"
        "Summary about the topic\n\n"
        "[Source Text]\n"
        "Actual body text without terminal punctuation\n"
        "[excerpt ends mid-sentence]"
    )


def test_context_marks_fragments_that_do_not_end_in_closing_punctuation():
    """Missing terminal punctuation (or a closing bracket/quote/digit) gets an
    explicit ``[excerpt ends mid-sentence]`` marker appended; a fragment that
    already ends cleanly does not."""
    fragmentos = [
        {"doc": "Ends cleanly.", "metadata": {"source": "a.pdf", "page": 0, "chunk": 0}},
        {"doc": "Cut off mid", "metadata": {"source": "a.pdf", "page": 0, "chunk": 1}},
    ]

    resultado = rag.construir_contexto_para_modelo(fragmentos)

    assert "Ends cleanly.\n\n" in resultado
    assert resultado.endswith("Cut off mid\n[excerpt ends mid-sentence]")


# ─────────────────────────────────────────────
# optimizar_texto_contexto
# ─────────────────────────────────────────────


def test_optimizar_collapses_double_spaces_and_strips_bold_markers():
    assert rag.optimizar_texto_contexto("This   is  a   test.") == "This is a test."
    assert rag.optimizar_texto_contexto("This is **bold** text.") == "This is bold text."


def test_optimizar_strips_box_artifact_headings():
    texto = "### □\nBody text after box heading."
    assert rag.optimizar_texto_contexto(texto) == "Body text after box heading."


def test_optimizar_strips_trailing_standalone_page_number():
    """A page-number line at the very END of the text (nothing follows it) is
    correctly stripped by the standalone-page-number regex."""
    texto = "Body text ends here.\n\n42"
    assert rag.optimizar_texto_contexto(texto) == "Body text ends here."


def test_optimizar_does_not_strip_standalone_page_number_followed_by_more_text():
    """SURPRISE: the standalone-page-number regex only fires on what's left
    AFTER paragraph rejoining (``_reunir_parrafos``) runs. Paragraph rejoining
    treats a lone numeric line as a "continuation" of whatever follows it
    (nothing in ``_es_continuacion_parrafo`` special-cases pure-digit lines),
    so a page number sandwiched between two paragraphs gets glued onto the
    NEXT paragraph before the page-number regex ever sees it standalone --
    and survives in the output. Page numbers only get removed when they are
    the last line of the text (see the sibling test above).
    """
    texto = "First paragraph ends here.\n\n42\n\nSecond paragraph starts here."

    resultado = rag.optimizar_texto_contexto(texto)

    assert resultado == "First paragraph ends here.\n\n42 Second paragraph starts here."


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
