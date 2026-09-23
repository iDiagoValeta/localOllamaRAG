"""Observability for two best-effort silent drops (no behavior change).

Covers: MinerU `_content_list_to_pages` blocks without `page_idx` are still
skipped, but counted and logged; jina-clip `_pump_stderr` still never raises,
but logs the reason at debug.
"""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monkeygrab.adapters.embedding import jina_clip_embedder as embedder_module  # noqa: E402
from monkeygrab.adapters.embedding.jina_clip_embedder import JinaClipEmbedder  # noqa: E402
from monkeygrab.adapters.extraction.mineru_extractor import _content_list_to_pages  # noqa: E402


def test_blocks_without_page_idx_are_skipped_but_counted_and_logged(caplog):
    blocks = [
        {"type": "text", "text": "hello", "page_idx": 0},
        {"type": "text", "text": "orphan without page"},
        {"type": "text", "text": "world", "page_idx": 0},
    ]

    with caplog.at_level(logging.WARNING, logger="monkeygrab.adapters.extraction.mineru_extractor"):
        pages = _content_list_to_pages(blocks)

    assert len(pages) == 1
    assert pages[0].page == 0
    assert pages[0].text == "hello\n\nworld"
    assert "orphan without page" not in pages[0].text
    assert any(
        "1" in record.message and "page_idx" in record.message for record in caplog.records
    )


def test_no_warning_when_every_block_has_page_idx(caplog):
    blocks = [{"type": "text", "text": "hello", "page_idx": 0}]

    with caplog.at_level(logging.WARNING, logger="monkeygrab.adapters.extraction.mineru_extractor"):
        pages = _content_list_to_pages(blocks)

    assert len(pages) == 1
    assert not [r for r in caplog.records if "page_idx" in r.message]


class _BrokenStderr:
    def __iter__(self):
        raise RuntimeError("pipe broken mid-read")


def test_pump_stderr_never_raises_and_logs_reason_at_debug(caplog):
    embedder = JinaClipEmbedder("fake-python", worker_script="fake-worker.py")
    tail: list = []
    process = SimpleNamespace(stderr=_BrokenStderr())

    with caplog.at_level(logging.DEBUG, logger=embedder_module.__name__):
        embedder._pump_stderr(process, tail)  # must not raise

    assert tail == []
    assert any(
        record.levelno == logging.DEBUG and "pipe broken" in record.message
        for record in caplog.records
    )
