"""Binds index_fingerprint's _EMBEDDING_ID to the model the adapter loads.

Nothing in either monkeygrab.application or monkeygrab.adapters can make this
comparison itself: the dependency rule (see test_architecture_boundaries.py)
forbids application from importing adapters, so index_fingerprint cannot read
jina_clip_worker's constants directly. A plain test module has no such
restriction and can import both, which is what turns "remember to bump both
strings together" into something a green suite actually checks.
"""

from monkeygrab.adapters.embedding.jina_clip_worker import _MODEL_NAME, _TRUNCATE_DIM
from monkeygrab.application.index_fingerprint import _EMBEDDING_ID


def test_embedding_id_names_the_model_and_dimension_the_worker_actually_loads():
    # jina_clip_worker.py is the process that calls
    # SentenceTransformer(_MODEL_NAME, ...) and encodes with
    # truncate_dim=_TRUNCATE_DIM. Bumping either without a matching bump to
    # _EMBEDDING_ID would silently leave the recipe fingerprint claiming an
    # embedding no index was actually built with.
    assert _EMBEDDING_ID == f"{_MODEL_NAME}@{_TRUNCATE_DIM}"
