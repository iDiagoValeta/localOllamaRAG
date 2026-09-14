"""format_sources groups ranked fragments by document with 1-based pages."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rag.engine.sources import format_sources


def test_groups_by_document_and_keeps_the_best_scored_page():
    fragments = [
        {"metadata": {"source": "b.pdf", "page": 4}, "score_reranker": 0.9},
        {"metadata": {"source": "a.md", "page": 0}, "score_reranker": 0.8},
        {"metadata": {"source": "b.pdf", "page": 1}, "score_reranker": 0.7},
    ]
    assert format_sources(fragments) == [
        {"document": "a.md", "pages": [1], "best_page": 1},
        {"document": "b.pdf", "pages": [2, 5], "best_page": 5},
    ]


def test_falls_back_to_score_final_when_not_reranked():
    fragments = [
        {"metadata": {"source": "x.pdf", "page": 2}, "score_final": 0.2},
        {"metadata": {"source": "x.pdf", "page": 7}, "score_final": 0.5},
    ]
    assert format_sources(fragments)[0]["best_page"] == 8
