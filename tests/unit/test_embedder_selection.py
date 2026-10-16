"""Unit tests for selecting the embedder (gemma2 default, jina opt-in).

The default configuration uses embeddinggemma-2 -- index path suffix ``_eg2``,
embedding id ``google/embeddinggemma-2@512``, ``LlamaServerEmbedder`` -- and
jina-clip-v2 must still change all three places that name an embedder. The
historical ``_EMBEDDING_ID`` constant stays the jina-clip-v2 string so existing
indexes remain valid. ``Retrieve`` must also prefer an embedder's
``embed_query`` hook when it has one, since embeddinggemma-2 is asymmetric.
"""

import pytest

from monkeygrab.application.index_fingerprint import (
    _EMBEDDING_ID,
    compute_index_fingerprint,
    index_recipe,
)
from monkeygrab.application.retrieve import Retrieve
from monkeygrab.composition import build_embedder
from monkeygrab.config.app_config import AppConfig
from monkeygrab.config.paths import derive_db_paths


def _jina_config():
    return AppConfig().with_overrides(**{"models.embedder": "jina"})


# derive_db_paths

def test_derive_db_paths_uses_eg2_slug_by_default(tmp_path):
    path_db, collection_name = derive_db_paths(str(tmp_path / "en"), str(tmp_path))

    assert path_db.endswith("en_eg2")
    assert collection_name == "docs_en"


def test_derive_db_paths_uses_jina_clip_slug_for_jina(tmp_path):
    path_db, collection_name = derive_db_paths(
        str(tmp_path / "en"), str(tmp_path), embedder="jina"
    )

    assert path_db.endswith("en_jina_clip")
    assert collection_name == "docs_en"


def test_derive_db_paths_rejects_an_unknown_embedder(tmp_path):
    with pytest.raises(ValueError, match="Valid"):
        derive_db_paths(str(tmp_path / "en"), str(tmp_path), embedder="other")


# index fingerprint

def test_default_embedding_id_is_embeddinggemma2():
    assert _EMBEDDING_ID == "jinaai/jina-clip-v2@512"
    assert index_recipe(AppConfig())["embedding"] == "google/embeddinggemma-2@512"


def test_jina_embedding_id_keeps_the_historical_constant():
    recipe = index_recipe(_jina_config())

    assert recipe["embedding"] == _EMBEDDING_ID


def test_jina_changes_the_fingerprint():
    assert compute_index_fingerprint(_jina_config()) != compute_index_fingerprint(AppConfig())


# composition

def test_build_embedder_returns_llama_server_embedder_by_default():
    from monkeygrab.adapters.embedding.llama_server_embedder import LlamaServerEmbedder

    embedder = build_embedder(AppConfig())

    assert isinstance(embedder, LlamaServerEmbedder)


# Retrieve query embedding

class _VectorStoreDouble:
    def query(self, embedding, n_results):
        return []


class _RecordingEmbedder:
    """Embedder exposing only ``embed``; the plain path every real jina double uses."""

    def __init__(self):
        self.calls = []

    def embed(self, text, *, keep_alive=None):
        self.calls.append(("embed", text))
        return [1.0, 0.0]


class _QueryEmbedder:
    """Embedder that also exposes ``embed_query``, like embeddinggemma-2."""

    def __init__(self):
        self.calls = []

    def embed(self, text, *, keep_alive=None):
        self.calls.append(("embed", text))
        return [1.0, 0.0]

    def embed_query(self, text):
        self.calls.append(("embed_query", text))
        return [0.0, 1.0]


def _no_optional_stages_config():
    return AppConfig().with_overrides(
        **{
            "flags.usar_llm_query_decomposition": False,
            "flags.usar_busqueda_hibrida": False,
            "flags.usar_reranker": False,
        }
    )


def test_retrieve_prefers_embed_query_when_the_embedder_has_one():
    embedder = _QueryEmbedder()
    retrieve = Retrieve(embedder, _VectorStoreDouble(), _no_optional_stages_config())

    retrieve.run("gravity in the early universe")

    assert embedder.calls
    assert all(method == "embed_query" for method, _ in embedder.calls)


def test_retrieve_falls_back_to_embed_without_embed_query():
    embedder = _RecordingEmbedder()
    retrieve = Retrieve(embedder, _VectorStoreDouble(), _no_optional_stages_config())

    retrieve.run("gravity in the early universe")

    assert embedder.calls
    assert all(method == "embed" for method, _ in embedder.calls)
