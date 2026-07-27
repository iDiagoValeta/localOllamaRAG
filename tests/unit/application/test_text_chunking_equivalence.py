"""Equivalence tests: monkeygrab.application.text_chunking vs rag.engine.chunking.

Runs the SAME inputs through the moved implementation and the untouched
original side by side and asserts identical output. This is the proof that
``split_markdown_into_chunks``/``adjacent_chunk_ids`` are a faithful move of
``rag.dividir_en_chunks``/``rag.expandir_con_chunks_adyacentes``, not a
reimplementation -- fixtures are the same ones
``tests/characterization/test_chunking.py`` pins, including every documented
surprise (bold-pseudo-header non-match, the below-min-length raw-truncation
fallback, overlap taken from the previous chunk's rendered text).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag.chat_pdfs as rag  # noqa: E402

from monkeygrab.application.text_chunking import (  # noqa: E402
    adjacent_chunk_ids,
    split_markdown_into_chunks,
)
from monkeygrab.domain.chunk_metadata import ChunkMetadata  # noqa: E402

MIN_CHUNK_LENGTH = 150  # matches rag.chat_pdfs's default (RAG_MIN_CHUNK_LENGTH unset)


def _both(texto, chunk_size, overlap, min_chunk_length=MIN_CHUNK_LENGTH):
    """Run both implementations and return (ours_as_dicts, theirs)."""
    ours = split_markdown_into_chunks(texto, chunk_size, overlap, min_chunk_length)
    ours_as_dicts = [{"text": c.text, "header": c.header} for c in ours]
    theirs = rag.dividir_en_chunks(texto, chunk_size=chunk_size, overlap=overlap)
    return ours_as_dicts, theirs


def test_paragraph_split_matches_original():
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

    ours, theirs = _both(texto, chunk_size=300, overlap=0)

    assert ours == theirs
    assert [c["text"] for c in ours] == [p1, p2, p3]  # pinned, not just cross-checked


def test_overlap_matches_original():
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

    ours, theirs = _both(texto, chunk_size=300, overlap=50)

    assert ours == theirs


def test_markdown_header_matches_original():
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

    ours, theirs = _both(texto, chunk_size=500, overlap=0)

    assert ours == theirs


def test_bold_pseudo_header_non_match_matches_original():
    texto = (
        "**Resumen**\n"
        "Contenido del resumen con suficiente longitud para superar el minimo "
        "de caracteres requerido por la configuracion actual del sistema de "
        "chunking en este momento concreto."
    )

    ours, theirs = _both(texto, chunk_size=400, overlap=0)

    assert ours == theirs


def test_below_min_length_raw_truncation_fallback_matches_original():
    texto = (
        "## Methods\n\n"
        "The methods section describes the experimental setup in detail, "
        "including data collection procedures, preprocessing steps, and the "
        "statistical tests applied to validate the results obtained during "
        "the study."
    )

    ours, theirs = _both(texto, chunk_size=200, overlap=0)

    assert ours == theirs


def test_empty_and_whitespace_only_input_matches_original():
    ours_empty, theirs_empty = _both("", chunk_size=300, overlap=50)
    ours_ws, theirs_ws = _both("   \n  ", chunk_size=300, overlap=50)

    assert ours_empty == theirs_empty == []
    assert ours_ws == theirs_ws == []


def test_min_chunk_length_parameter_matches_original_cfg_override(monkeypatch):
    """cfg.MIN_CHUNK_LENGTH is read live inside dividir_en_chunks's body (not a
    stale default, per test_stale_default_config_bug.py's contrast case) --
    so mutating it and passing the same value as our explicit parameter must
    keep both implementations in lockstep."""
    texto = "Body text of a single short paragraph that is not empty at all."  # 65 chars

    monkeypatch.setattr(rag, "MIN_CHUNK_LENGTH", 10)
    ours, theirs = _both(texto, chunk_size=2000, overlap=0, min_chunk_length=10)
    assert ours == theirs != []

    monkeypatch.setattr(rag, "MIN_CHUNK_LENGTH", 1000)
    ours, theirs = _both(texto, chunk_size=2000, overlap=0, min_chunk_length=1000)
    assert ours == theirs == []


def _metadata(**kwargs):
    kwargs.setdefault("chunk", 0)
    return ChunkMetadata(**kwargs)


def test_mid_page_neighbor_ids_match_original():
    meta_dict = {"source": "paperA.pdf", "page": 2, "chunk": 3, "total_chunks_in_page": 6}
    meta = _metadata(source="paperA.pdf", page=2, chunk=3, total_chunks_in_page=6)

    ours = adjacent_chunk_ids(meta, n_neighbors=1)
    theirs = rag.expandir_con_chunks_adyacentes("paperA.pdf_pag2_chunk3", meta_dict, n_vecinos=1)

    assert ours == theirs == ["paperA.pdf_pag2_chunk2", "paperA.pdf_pag2_chunk4"]


def test_first_chunk_of_page_neighbor_ids_match_original():
    meta_dict = {"source": "paperA.pdf", "page": 2, "chunk": 0, "total_chunks_in_page": 6}
    meta = _metadata(source="paperA.pdf", page=2, chunk=0, total_chunks_in_page=6)

    ours = adjacent_chunk_ids(meta, n_neighbors=1)
    theirs = rag.expandir_con_chunks_adyacentes("paperA.pdf_pag2_chunk0", meta_dict, n_vecinos=1)

    assert ours == theirs


def test_last_chunk_of_page_neighbor_ids_match_original():
    meta_dict = {"source": "paperA.pdf", "page": 2, "chunk": 5, "total_chunks_in_page": 6}
    meta = _metadata(source="paperA.pdf", page=2, chunk=5, total_chunks_in_page=6)

    ours = adjacent_chunk_ids(meta, n_neighbors=1)
    theirs = rag.expandir_con_chunks_adyacentes("paperA.pdf_pag2_chunk5", meta_dict, n_vecinos=1)

    assert ours == theirs


def test_missing_total_chunks_in_page_neighbor_ids_match_original():
    meta_dict = {"source": "paperA.pdf", "page": 1, "chunk": 2}
    meta = _metadata(source="paperA.pdf", page=1, chunk=2, total_chunks_in_page=None)

    ours = adjacent_chunk_ids(meta, n_neighbors=1)
    theirs = rag.expandir_con_chunks_adyacentes("paperA.pdf_pag1_chunk2", meta_dict, n_vecinos=1)

    assert ours == theirs


def test_page_zero_chunk_zero_neighbor_ids_match_original():
    meta_dict = {"source": "paperA.pdf", "page": 0, "chunk": 0, "total_chunks_in_page": 4}
    meta = _metadata(source="paperA.pdf", page=0, chunk=0, total_chunks_in_page=4)

    ours = adjacent_chunk_ids(meta, n_neighbors=1)
    theirs = rag.expandir_con_chunks_adyacentes("paperA.pdf_pag0_chunk0", meta_dict, n_vecinos=1)

    assert ours == theirs


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
