"""Tests for classifying a text chunk as a table.

A preserved table indexed as ordinary prose is invisible to a query that needs a
table, which is why table retrieval measured 0/5 on the flattening path even
though the numbers were present in the text. These tests pin the classification
that makes the difference.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monkeygrab.application.index_corpus import _text_chunk_format  # noqa: E402


def test_html_table_is_marked_as_a_table():
    chunk = (
        "Table 1: Maximum path lengths per layer type.\n"
        "<table><tr><td>Layer Type</td><td>Complexity</td></tr>"
        "<tr><td>Self-Attention</td><td>O(n^2 · d)</td></tr></table>"
    )
    assert _text_chunk_format(chunk) == "table"


def test_prose_is_not_marked_as_a_table():
    chunk = (
        "The encoder is composed of a stack of N = 6 identical layers, each with "
        "two sub-layers."
    )
    assert _text_chunk_format(chunk) == "markdown"


def test_detection_survives_tag_attributes_and_casing():
    """Extractors differ in how they emit the tag; the content is the same."""
    for variant in (
        '<table border="1">',
        "<TABLE>",
        "<Table class='data'>",
    ):
        assert _text_chunk_format(f"caption\n{variant}<tr><td>x</td></tr></table>") == "table"


def test_prose_merely_mentioning_the_word_table_is_not_a_table():
    """The word appears constantly in papers; only real markup counts.

    Otherwise every sentence referring to "Table 3" would be indexed as a table
    and the distinction would carry no information.
    """
    chunk = (
        "As shown in Table 3, the big model outperforms all previously published "
        "models. The table also reports training cost."
    )
    assert _text_chunk_format(chunk) == "markdown"
