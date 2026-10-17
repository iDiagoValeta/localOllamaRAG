"""Citation-marker stripping for RECOMP input (issue #311).

``strip_citation_markers`` must remove only brackets that read as citations,
not numeric notation that happens to be bracketed. The distinction matters
because the same text is also what the generator sees when RECOMP is off.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from monkeygrab.application.context_assembly import strip_citation_markers  # noqa: E402


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("as shown in [12].", "as shown in."),
        ("prior work [3, 7, 9] shows", "prior work shows"),
        ("see [38, 2, 9].", "see."),
        ("results end here [7]", "results end here"),
        ("evidence [1], more", "evidence, more"),
    ],
)
def test_citation_markers_are_removed(texto, esperado):
    assert strip_citation_markers(texto) == esperado


@pytest.mark.parametrize(
    "texto",
    [
        "use x[0] first",
        "use x[1] first",
        "scaled to [0, 1]",
        "scores in [0.5, 1.5]",
        "range [3-7]",
        "pair [1, 2]",
        "[12] starts the text",
    ],
)
def test_numeric_notation_is_preserved(texto):
    assert strip_citation_markers(texto) == texto
