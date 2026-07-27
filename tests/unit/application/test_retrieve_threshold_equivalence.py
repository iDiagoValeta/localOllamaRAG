"""Equivalence tests: monkeygrab.application.retrieve's threshold filter vs
rag.engine.generation._filtrar_por_umbral_reranker.

Same fixtures as
``tests/characterization/test_generation_budget.py``'s threshold-filter
tests, run through both the moved ``_filter_by_reranker_threshold`` (now
part of the ``Retrieve`` use case, per the task's use-case boundary --
"filtrar por umbral" is Retrieve's last step, not Answer's) and the
untouched original. Fixtures are translated once via ``_fragment`` into
``Fragment``s and fed to both sides from the same source data.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Bound to an explicit alias, not plain `import rag.chat_pdfs`: that form binds
# the name `rag` to the root package, which carries none of the config
# attributes this test patches, so any later reference reads the wrong object.
import rag.chat_pdfs as chat_pdfs  # noqa: E402 -- must precede rag.engine.generation (circular import guard)
from rag.engine.generation import _filtrar_por_umbral_reranker  # noqa: E402

from monkeygrab.application.retrieve import _filter_by_reranker_threshold  # noqa: E402
from monkeygrab.domain.chunk_metadata import ChunkMetadata  # noqa: E402
from monkeygrab.domain.fragment import Fragment  # noqa: E402


def _fragment(id_, score_reranker=None, score_final=0.0):
    source, rest = id_.split("_pag", 1)
    page_str, chunk_str = rest.split("_chunk", 1)
    return Fragment(
        doc="x",
        metadata=ChunkMetadata(source=source, page=int(page_str), chunk=int(chunk_str)),
        score_reranker=score_reranker,
        score_final=score_final,
    )


def _as_dict(frag):
    # score_reranker must be OMITTED (not set to None) when the fragment was
    # never reranked: the original's `fragmento.get("score_reranker",
    # fragmento.get("score_final", 0))` only falls back to score_final when
    # the key is ABSENT -- dict.get returns None, not the fallback, for a
    # key that is present with value None. Fragment.score_reranker's default
    # of None must translate to "key absent" here to match that semantics.
    d = {"id": frag.id, "score_final": frag.score_final}
    if frag.score_reranker is not None:
        d["score_reranker"] = frag.score_reranker
    return d


def test_inclusive_boundary_matches_original():
    fragments = [
        _fragment("above_pag0_chunk0", score_reranker=0.9),
        _fragment("at_threshold_pag0_chunk0", score_reranker=0.65),
        _fragment("just_below_pag0_chunk0", score_reranker=0.649999),
        _fragment("far_below_pag0_chunk0", score_reranker=0.1),
    ]

    ours = _filter_by_reranker_threshold(fragments, usar_reranker=True, threshold=0.65)
    theirs = _call_original(fragments, usar_reranker=True, threshold=0.65)

    assert [f.id for f in ours] == [d["id"] for d in theirs]


def _call_original(fragments, usar_reranker, threshold):
    """rag.engine.generation._filtrar_por_umbral_reranker reads cfg.USAR_RERANKER /
    cfg.UMBRAL_SCORE_RERANKER live from rag.chat_pdfs -- patch them for the
    duration of one call so this stays a plain equivalence check without
    importing rag.chat_pdfs's monkeypatch fixtures into every test."""

    original_usar, original_umbral = (
        chat_pdfs.USAR_RERANKER,
        chat_pdfs.UMBRAL_SCORE_RERANKER,
    )
    chat_pdfs.USAR_RERANKER, chat_pdfs.UMBRAL_SCORE_RERANKER = usar_reranker, threshold
    try:
        return _filtrar_por_umbral_reranker([_as_dict(f) for f in fragments])
    finally:
        chat_pdfs.USAR_RERANKER, chat_pdfs.UMBRAL_SCORE_RERANKER = (
            original_usar,
            original_umbral,
        )


def test_fallback_to_score_final_matches_original():
    fragments = [
        _fragment("no_reranker_score_ok_pag0_chunk0", score_reranker=None, score_final=0.7),
        _fragment("no_reranker_score_low_pag0_chunk0", score_reranker=None, score_final=0.2),
    ]

    ours = _filter_by_reranker_threshold(fragments, usar_reranker=True, threshold=0.65)
    theirs = _call_original(fragments, usar_reranker=True, threshold=0.65)

    assert [f.id for f in ours] == [d["id"] for d in theirs] == ["no_reranker_score_ok_pag0_chunk0"]


def test_disabled_flag_passthrough_matches_original():
    fragments = [
        _fragment("a_pag0_chunk0", score_reranker=0.9),
        _fragment("b_pag0_chunk0", score_reranker=None, score_final=0.0),
        _fragment("c_pag0_chunk0", score_reranker=None, score_final=0.0),
    ]

    ours = _filter_by_reranker_threshold(fragments, usar_reranker=False, threshold=0.65)
    theirs = _call_original(fragments, usar_reranker=False, threshold=0.65)

    assert [f.id for f in ours] == [d["id"] for d in theirs] == ["a_pag0_chunk0", "b_pag0_chunk0", "c_pag0_chunk0"]


def test_all_below_returns_empty_matches_original():
    fragments = [
        _fragment("a_pag0_chunk0", score_reranker=0.1),
        _fragment("b_pag0_chunk0", score_reranker=0.2),
    ]

    ours = _filter_by_reranker_threshold(fragments, usar_reranker=True, threshold=0.65)
    theirs = _call_original(fragments, usar_reranker=True, threshold=0.65)

    assert ours == theirs == []


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
