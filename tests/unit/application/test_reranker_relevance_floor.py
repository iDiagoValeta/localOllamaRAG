"""The opt-in relevance floor judges a question by its best reranked fragment."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from monkeygrab.application.retrieve import _filter_by_reranker_threshold
from monkeygrab.config.app_config import AppConfig
from monkeygrab.domain.chunk_metadata import ChunkMetadata
from monkeygrab.domain.fragment import Fragment


def _fragment(source, score):
    return Fragment(doc="", metadata=ChunkMetadata(source=source, page=0, chunk=0), score_reranker=score)


def _sources(fragments):
    return [f.metadata.source for f in fragments]


def test_a_question_whose_best_fragment_reaches_the_floor_keeps_every_fragment():
    # Measured on a lab-notes corpus: the relevant table chunk scored 0.097 and
    # a multi-part question's second document 0.028, both below 0.65.
    fragments = [_fragment("notes.txt", 0.311), _fragment("protocol.md", 0.028), _fragment("park.pdf", 0.003)]
    kept = _filter_by_reranker_threshold(fragments, True, 0.65, floor=0.05)
    assert _sources(kept) == ["notes.txt", "protocol.md", "park.pdf"]


def test_an_off_topic_question_below_the_floor_keeps_nothing():
    fragments = [_fragment("park.pdf", 0.035), _fragment("park.pdf", 0.029)]
    assert _filter_by_reranker_threshold(fragments, True, 0.65, floor=0.05) == []


def test_named_sources_still_pass_below_the_floor():
    fragments = [_fragment("park.pdf", 0.01), _fragment("notes.txt", 0.02)]
    kept = _filter_by_reranker_threshold(fragments, True, 0.65, named_sources={"notes.txt"}, floor=0.05)
    assert _sources(kept) == ["notes.txt"]


def test_without_a_floor_the_per_fragment_threshold_is_unchanged():
    fragments = [_fragment("a.pdf", 0.9), _fragment("b.pdf", 0.3)]
    assert _sources(_filter_by_reranker_threshold(fragments, True, 0.65)) == ["a.pdf"]


def test_the_floor_is_read_from_the_environment_and_off_by_default(monkeypatch):
    monkeypatch.delenv("RAG_SUELO_SCORE_RERANKER", raising=False)
    assert AppConfig.from_env().reranking.relevance_floor == 0.0
    monkeypatch.setenv("RAG_SUELO_SCORE_RERANKER", "0.05")
    assert AppConfig.from_env().reranking.relevance_floor == 0.05
