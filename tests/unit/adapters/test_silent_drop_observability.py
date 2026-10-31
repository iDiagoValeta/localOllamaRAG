"""Observability for a best-effort silent drop (no behavior change).

Covers: MinerU `_content_list_to_pages` blocks without `page_idx` are still
skipped, but counted and logged.
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monkeygrab.adapters.extraction.mineru_extractor import _content_list_to_pages


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
