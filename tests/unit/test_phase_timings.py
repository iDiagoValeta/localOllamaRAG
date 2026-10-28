"""What the phase-timing probe counts as embedding time.

The probe exists to split a query's wall clock by phase (issue #12). Query
embedding reaches the embedder through ``embed_query`` whenever the adapter
offers it, and image retrieval through ``embed_image``; timing only ``embed``
leaves the embedding phase short by exactly the calls the product makes
(#369). The wrapper is exercised with a stand-in embedder, so this test loads
no model and touches no GPU -- the probe's own module only imports stdlib.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diagnostics.phase_timings import _Timer, _timed


class _Embedder:
    """Stand-in exposing the three embedding entry points the probe wraps."""

    def embed(self, text, *, keep_alive=None):
        return [0.0]

    def embed_query(self, text):
        return [0.0]

    def embed_image(self, image_path, *, caption=""):
        return [0.0]


def test_embedding_phase_counts_text_query_and_image_calls():
    timer = _Timer()
    embedder = _timed(_Embedder(), timer, "embed")

    embedder.embed("a document")
    embedder.embed_query("a query")
    embedder.embed_image("figure.pdf")

    _, calls = timer.snapshot()
    assert calls["embed"] == 3
    assert calls["vector_search"] == 0
