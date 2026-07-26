"""Equivalence tests: monkeygrab.application.answer's char budget vs
rag.engine.generation._limitar_fragmentos_por_chars.

Same fixtures as
``tests/characterization/test_generation_budget.py``'s char-budget tests,
run through both the moved ``_limit_by_char_budget`` (part of the ``Answer``
use case, per the task's use-case boundary -- Answer starts with "limitar
por presupuesto de caracteres") and the untouched original.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag.chat_pdfs  # noqa: E402,F401 -- must import before rag.engine.generation (circular import guard)
from rag.engine.generation import _limitar_fragmentos_por_chars  # noqa: E402

from monkeygrab.application.answer import _limit_by_char_budget  # noqa: E402
from monkeygrab.domain.chunk_metadata import ChunkMetadata  # noqa: E402
from monkeygrab.domain.fragment import Fragment  # noqa: E402


def _fragment(id_, doc):
    source, rest = id_.split("_pag", 1)
    page_str, chunk_str = rest.split("_chunk", 1)
    return Fragment(doc=doc, metadata=ChunkMetadata(source=source, page=int(page_str), chunk=int(chunk_str)))


def _as_dict(frag):
    return {"id": frag.id, "doc": frag.doc}


def test_no_truncation_when_total_fits_matches_original(monkeypatch):
    monkeypatch.setattr(rag.chat_pdfs, "MAX_CONTEXTO_CHARS", 100)
    fragments = [_fragment("a_pag0_chunk0", "x" * 50), _fragment("b_pag0_chunk0", "y" * 50)]

    ours_kept, ours_discarded = _limit_by_char_budget(fragments, max_context_chars=100)
    theirs_kept, theirs_discarded = _limitar_fragmentos_por_chars([_as_dict(f) for f in fragments])

    assert [f.id for f in ours_kept] == [d["id"] for d in theirs_kept] == ["a_pag0_chunk0", "b_pag0_chunk0"]
    assert ours_discarded == theirs_discarded == 0


def test_inclusive_boundary_then_drop_matches_original(monkeypatch):
    monkeypatch.setattr(rag.chat_pdfs, "MAX_CONTEXTO_CHARS", 100)
    fragments = [
        _fragment("a_pag0_chunk0", "x" * 60),
        _fragment("b_pag0_chunk0", "y" * 40),  # 60 + 40 == 100: exactly at budget, kept
        _fragment("c_pag0_chunk0", "z" * 1),   # would push to 101: dropped
    ]

    ours_kept, ours_discarded = _limit_by_char_budget(fragments, max_context_chars=100)
    theirs_kept, theirs_discarded = _limitar_fragmentos_por_chars([_as_dict(f) for f in fragments])

    assert [f.id for f in ours_kept] == [d["id"] for d in theirs_kept] == ["a_pag0_chunk0", "b_pag0_chunk0"]
    assert ours_discarded == theirs_discarded == 1


def test_drops_from_first_overflow_onward_matches_original(monkeypatch):
    monkeypatch.setattr(rag.chat_pdfs, "MAX_CONTEXTO_CHARS", 50)
    fragments = [
        _fragment("a_pag0_chunk0", "x" * 60),  # overflows immediately: 60 > 50
        _fragment("b_pag0_chunk0", "y" * 10),  # would fit alone, but never considered
    ]

    ours_kept, ours_discarded = _limit_by_char_budget(fragments, max_context_chars=50)
    theirs_kept, theirs_discarded = _limitar_fragmentos_por_chars([_as_dict(f) for f in fragments])

    assert ours_kept == theirs_kept == []
    assert ours_discarded == theirs_discarded == 2


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
