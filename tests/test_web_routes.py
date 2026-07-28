"""Tests for the web layer's contract with the pipeline.

Covers the routes the frontend depends on, and the shape of the fragment dicts
the source panel is handed.
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monkeygrab.domain.chunk_metadata import ChunkMetadata
from monkeygrab.domain.fragment import Fragment
from rag.engine.wiring import fragment_to_dict
from rag.web.app import _format_sources, app


def test_upload_route_is_preserved_for_api_compatibility():
    routes = {str(rule) for rule in app.url_map.iter_rules()}

    assert "/api/upload" in routes
    assert "/api/reindex" in routes


def _fragment(page, score_reranker=None, score_final=0.0):
    return Fragment(
        doc="text",
        metadata=ChunkMetadata(source="paper.pdf", page=page, chunk=0),
        score_reranker=score_reranker,
        score_final=score_final,
    )


def test_sources_panel_handles_fragments_that_were_never_reranked():
    """With reranking off no fragment carries a reranker score, and the panel
    falls back to score_final to pick each document's best page.

    That fallback is `frag.get("score_reranker", frag["score_final"])`, which a
    present-but-None key silently defeats: the panel then compares None against
    None and raises. Two fragments of one document are the minimum needed to
    reach that comparison.
    """
    fragments = [
        fragment_to_dict(_fragment(page=0, score_final=0.1)),
        fragment_to_dict(_fragment(page=1, score_final=0.2)),
    ]

    sources = _format_sources(fragments)

    assert sources == [{"document": "paper.pdf", "pages": [1, 2], "best_page": 2}]


def test_sources_panel_prefers_the_reranker_score_when_there_is_one():
    fragments = [
        fragment_to_dict(_fragment(page=0, score_reranker=0.9, score_final=0.1)),
        fragment_to_dict(_fragment(page=1, score_reranker=0.2, score_final=0.99)),
    ]

    sources = _format_sources(fragments)

    assert sources == [{"document": "paper.pdf", "pages": [1, 2], "best_page": 1}]
