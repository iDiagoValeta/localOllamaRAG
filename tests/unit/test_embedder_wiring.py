"""Unit tests for how the embedder is wired.

embeddinggemma-2 names the index path suffix ``_eg2``, the embedding id
``google/embeddinggemma-2@<width>`` and the ``LlamaServerEmbedder`` the stack
builds. ``Retrieve`` must also prefer an embedder's ``embed_query`` hook when it
has one, since embeddinggemma-2 is asymmetric.
"""

from monkeygrab.application.index_fingerprint import compute_index_fingerprint, index_recipe
from monkeygrab.application.retrieve import Retrieve
from monkeygrab.composition import build_embedder
from monkeygrab.config.app_config import AppConfig
from monkeygrab.config.models import EmbeddingRuntimeConfig
from monkeygrab.config.paths import derive_db_paths


# derive_db_paths

def test_derive_db_paths_uses_the_eg2_slug(tmp_path):
    path_db, collection_name = derive_db_paths(str(tmp_path / "en"), str(tmp_path))

    assert path_db.endswith("en_eg2")
    assert collection_name == "docs_en"


# index fingerprint

def test_embedding_id_names_embeddinggemma2_and_its_width():
    assert index_recipe(AppConfig())["embedding"] == "google/embeddinggemma-2@512"


def test_a_different_width_changes_the_fingerprint():
    narrower = AppConfig().with_overrides(
        **{"models.embedding": EmbeddingRuntimeConfig(dimensions=256)}
    )

    assert index_recipe(narrower)["embedding"] == "google/embeddinggemma-2@256"
    assert compute_index_fingerprint(narrower) != compute_index_fingerprint(AppConfig())


# composition

def test_build_embedder_returns_llama_server_embedder():
    from monkeygrab.adapters.embedding.llama_server_embedder import LlamaServerEmbedder

    embedder = build_embedder(AppConfig())

    assert isinstance(embedder, LlamaServerEmbedder)


# Retrieve query embedding

class _VectorStoreDouble:
    def query(self, embedding, n_results):
        return []


class _RecordingEmbedder:
    """Embedder exposing only ``embed``; the plain path every test double uses."""

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
