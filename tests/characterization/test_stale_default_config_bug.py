"""Regression tests for the historical stale-default configuration bug.

Several engine functions used to capture pipeline configuration in Python
default argument values evaluated at module-import time. Runtime changes to
``rag.chat_pdfs`` then affected parameters read in function bodies but not
parameters bound in signatures. ``dividir_en_chunks`` now resolves omitted
chunking values from the live runtime module; the tests below keep that
contract explicit and document the older stale-default locations that were
already migrated.
"""

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag.chat_pdfs as rag


def test_hot_changing_chunk_size_and_overlap_affect_omitted_arguments(monkeypatch):
    """Omitted chunking parameters are resolved from live runtime config.

    A hot change to both values must produce the same result as passing those
    values explicitly. This catches the old import-time ``cfg.CHUNK_SIZE`` and
    ``cfg.CHUNK_OVERLAP`` defaults without loading a second implementation.
    """
    monkeypatch.setattr(rag, "CHUNK_SIZE", 80)
    monkeypatch.setattr(rag, "CHUNK_OVERLAP", 10)
    monkeypatch.setattr(rag, "MIN_CHUNK_LENGTH", 10)

    texto = "word " * 100
    from_runtime = rag.dividir_en_chunks(texto)
    explicit = rag.dividir_en_chunks(texto, chunk_size=80, overlap=10)

    assert from_runtime == explicit
    assert len(from_runtime) > 1


def test_keyword_result_limit_now_follows_a_config_change():
    """Formerly the second stale-default location: ``busqueda_lexica_bm25``'s
    ``top_n`` default was bound to ``cfg.N_RESULTADOS_KEYWORD`` at import time.

    Lexical search now runs through ``Bm25LexicalIndex``, which takes the
    limit as a call argument supplied per run from ``AppConfig``. The limit a
    caller passes is the limit that applies.
    """
    from monkeygrab.adapters.lexical.bm25_index import Bm25LexicalIndex
    from monkeygrab.config.retrieval import RetrievalConfig
    from monkeygrab.domain.chunk_metadata import ChunkMetadata
    from monkeygrab.domain.fragment import Fragment

    class _FakeStore:
        def __init__(self, docs):
            self.fragments = [
                Fragment(doc=d, metadata=ChunkMetadata(source=f"doc{i}.pdf", page=0, chunk=i))
                for i, d in enumerate(docs)
            ]

        def count(self):
            return len(self.fragments)

        def get_page(self, limit, offset):
            return self.fragments

    # One distractor keeps BM25's idf for "alpha" positive, so the ten
    # matching documents all score above zero and the limit is what truncates.
    docs = [f"alpha token{i}" for i in range(10)] + ["unrelated distractor"]
    index = Bm25LexicalIndex(_FakeStore(docs), RetrievalConfig())

    assert len(index.search("alpha", top_n=10)) == 10
    assert len(index.search("alpha", top_n=3)) == 3


def test_final_top_k_is_no_longer_frozen_into_a_signature_default():
    """Formerly the third stale-default location: ``rerank_resultados``'s
    ``top_k`` default was bound to ``cfg.TOP_K_FINAL`` at import time.

    ``Reranker.rerank`` takes ``top_k`` as a required argument, so there is no
    default left to go stale. Checked at the signature level, which also keeps
    this suite from loading a CrossEncoder.
    """
    from monkeygrab.adapters.reranking.cross_encoder_reranker import CrossEncoderReranker

    top_k = inspect.signature(CrossEncoderReranker.rerank).parameters["top_k"]

    assert top_k.default is inspect.Parameter.empty


def test_contrast_min_chunk_length_is_read_live_inside_the_function_body(monkeypatch):
    """Contrast case: ``MIN_CHUNK_LENGTH`` is read as ``cfg.MIN_CHUNK_LENGTH``
    INSIDE ``dividir_en_chunks``'s body (not bound as a default argument), so
    a hot change to it IS honored immediately -- proving the defect above is
    specifically about default-argument binding, not about ``cfg`` being
    unobservable in general.
    """
    texto = "Body text of a single short paragraph that is not empty at all."  # 65 chars

    monkeypatch.setattr(rag, "MIN_CHUNK_LENGTH", 10)
    assert rag.dividir_en_chunks(texto, chunk_size=2000, overlap=0) != []

    monkeypatch.setattr(rag, "MIN_CHUNK_LENGTH", 1000)
    assert rag.dividir_en_chunks(texto, chunk_size=2000, overlap=0) == []


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
