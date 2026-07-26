"""Equivalence tests: monkeygrab.application.context_assembly vs rag.engine.context.

Same fixtures as ``tests/characterization/test_context_building.py``, run
through both the moved ``build_context_for_model``/``optimize_context_text``
and the untouched originals. ``build_context_for_model`` takes
``Sequence[Fragment]`` instead of the original's ``List[Dict]`` -- fixtures
are translated once via ``_fragment`` and fed to both sides from the same
source data.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag.chat_pdfs as rag  # noqa: E402

from monkeygrab.application.context_assembly import (  # noqa: E402
    build_context_for_model,
    optimize_context_text,
)
from monkeygrab.domain.chunk_metadata import ChunkMetadata  # noqa: E402
from monkeygrab.domain.fragment import Fragment  # noqa: E402

CONTEXTUAL_SEP = "\\n\\n"


def _fragment(doc, source, page, chunk=0):
    return Fragment(doc=doc, metadata=ChunkMetadata(source=source, page=page, chunk=chunk))


def _as_dict(frag):
    return {"doc": frag.doc, "metadata": {"source": frag.metadata.source, "page": frag.metadata.page, "chunk": frag.metadata.chunk}}


# ─────────────────────────────────────────────
# build_context_for_model vs construir_contexto_para_modelo
# ─────────────────────────────────────────────


def test_sorted_and_renumbered_output_matches_original(monkeypatch):
    monkeypatch.setattr(rag, "USAR_OPTIMIZACION_CONTEXTO", False)
    fragments = [
        _fragment("Body from b.pdf page 1.", "b.pdf", 1, 0),
        _fragment("Second chunk of a.pdf page 0.", "a.pdf", 0, 1),
        _fragment("First chunk of a.pdf page 0.", "a.pdf", 0, 0),
    ]

    ours, _metrics = build_context_for_model(fragments, optimize_text=False)
    theirs = rag.construir_contexto_para_modelo([_as_dict(f) for f in fragments])

    assert ours == theirs


def test_contextual_summary_split_matches_original(monkeypatch):
    monkeypatch.setattr(rag, "USAR_OPTIMIZACION_CONTEXTO", False)
    fragments = [_fragment(
        f"Summary about the topic{CONTEXTUAL_SEP}Actual body text without terminal punctuation",
        "a.pdf", 0, 0,
    )]

    ours, _metrics = build_context_for_model(fragments, optimize_text=False)
    theirs = rag.construir_contexto_para_modelo([_as_dict(f) for f in fragments])

    assert ours == theirs


def test_incomplete_marker_matches_original(monkeypatch):
    monkeypatch.setattr(rag, "USAR_OPTIMIZACION_CONTEXTO", False)
    fragments = [
        _fragment("Ends cleanly.", "a.pdf", 0, 0),
        _fragment("Cut off mid", "a.pdf", 0, 1),
    ]

    ours, _metrics = build_context_for_model(fragments, optimize_text=False)
    theirs = rag.construir_contexto_para_modelo([_as_dict(f) for f in fragments])

    assert ours == theirs


def test_with_optimization_enabled_matches_original(monkeypatch):
    """Same fixture as above, but with USAR_OPTIMIZACION_CONTEXTO True (the
    default) so optimize_context_text is exercised inside the assembly path too."""
    monkeypatch.setattr(rag, "USAR_OPTIMIZACION_CONTEXTO", True)
    fragments = [
        _fragment("This   is  a   test.\n\n42", "a.pdf", 0, 0),
        _fragment(f"Doc summary{CONTEXTUAL_SEP}### □\nBody after box heading, no closing punctuation", "a.pdf", 1, 0),
    ]

    ours, _metrics = build_context_for_model(fragments, optimize_text=True)
    theirs = rag.construir_contexto_para_modelo([_as_dict(f) for f in fragments])

    assert ours == theirs


# ─────────────────────────────────────────────
# optimize_context_text vs optimizar_texto_contexto
# ─────────────────────────────────────────────


def test_collapse_and_bold_strip_matches_original():
    assert optimize_context_text("This   is  a   test.") == rag.optimizar_texto_contexto("This   is  a   test.")
    assert optimize_context_text("This is **bold** text.") == rag.optimizar_texto_contexto("This is **bold** text.")


def test_box_artifact_heading_strip_matches_original():
    texto = "### □\nBody text after box heading."
    assert optimize_context_text(texto) == rag.optimizar_texto_contexto(texto)


def test_trailing_page_number_strip_matches_original():
    texto = "Body text ends here.\n\n42"
    assert optimize_context_text(texto) == rag.optimizar_texto_contexto(texto)


def test_sandwiched_page_number_survives_to_match_original():
    """The documented SURPRISE (glued to the following paragraph, not
    stripped) must reproduce identically."""
    texto = "First paragraph ends here.\n\n42\n\nSecond paragraph starts here."
    assert optimize_context_text(texto) == rag.optimizar_texto_contexto(texto)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
