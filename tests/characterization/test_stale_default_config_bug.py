"""Characterization test for THE LATENT BUG this suite exists to protect
against fixing accidentally: several engine functions capture pipeline
configuration in Python DEFAULT ARGUMENT VALUES, which are evaluated once
at module-import time (e.g. ``def dividir_en_chunks(texto,
chunk_size=cfg.CHUNK_SIZE, ...)`` in rag/engine/chunking.py). Reads inside
the function BODY (``cfg.MIN_CHUNK_LENGTH`` etc.) are dynamic and do observe
runtime config changes; the bound DEFAULTS never do.

Concretely: ``rag.chat_pdfs.set_pipeline_flags`` and the web control panel's
model/config endpoints mutate ``rag.chat_pdfs`` globals in-place, assuming
every engine function reads them live. That assumption is only true for
config read inside a function body, not for config baked into a default
argument. So a hot config change silently applies to some parameters and
not others, with no error or warning anywhere.

Design doc docs/design/2026-07-26-monkeygrab-v2.md (section on "Configuracion
en caliente funciona a medias") calls this out explicitly and section 4
("AppConfig inmutable inyectada") describes the fix: an immutable config
object passed explicitly into use cases, which makes "stale default" a
non-representable state.

THIS IS THE ONE TEST IN tests/characterization/ THAT IS EXPECTED TO CHANGE.
Once the migration eliminates the stale-default pattern (AppConfig injected
per-call, no more ``arg=cfg.X`` defaults), update this test to assert the
FIXED behavior: that a config change is honored by a call made with no
explicit override, instead of asserting today's broken passthrough.
"""

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag.chat_pdfs as rag


class _FakeCollection:
    """Minimal Chroma collection double for busqueda_lexica_bm25's full scan."""

    def __init__(self, docs):
        self.docs = docs
        self.metas = [{"source": f"doc{i}.pdf", "page": 0, "chunk": i} for i in range(len(docs))]
        self.ids = [f"doc{i}.pdf_pag0_chunk{i}" for i in range(len(docs))]

    def count(self):
        return len(self.docs)

    def get(self, limit=None, offset=0, include=None, **kwargs):
        end = None if limit is None else offset + limit
        return {
            "documents": self.docs[offset:end],
            "metadatas": self.metas[offset:end],
            "ids": self.ids[offset:end],
        }


def test_hot_changing_chunk_size_does_not_affect_calls_without_an_explicit_argument(monkeypatch):
    """``dividir_en_chunks``'s ``chunk_size`` default was bound to
    ``cfg.CHUNK_SIZE`` at import time (2000). Mutating ``rag.CHUNK_SIZE`` at
    runtime -- exactly what ``set_pipeline_flags``-style hot config changes
    do for OTHER parameters -- has NO effect on a call that relies on the
    default. The text below (500 chars) stays a single, unsplit chunk even
    though the "new" configured chunk_size (50) should have forced a split.
    """
    monkeypatch.setattr(rag, "CHUNK_SIZE", 50)

    texto = "word " * 100  # 500 chars, well above the "new" 50-char budget
    chunks = rag.dividir_en_chunks(texto)  # no explicit chunk_size: uses the stale default

    assert len(chunks) == 1
    assert len(chunks[0]["text"]) > 50  # the stale default (2000), not the live 50, was applied


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


def test_image_extraction_limit_now_follows_a_config_change():
    """Formerly the fourth stale-default location: ``extraer_imagenes_pdf``'s
    ``max_por_pagina`` default was bound to ``cfg.MAX_IMAGENES_POR_PAGINA`` at
    import time, so a config change never reached it.

    Image extraction now runs through ``PymupdfImageExtractor``, built per
    indexing run from an ``AppConfig``. Overriding the limit and rebuilding the
    extractor yields the new limit -- the fixed behavior, asserted here in
    place of the defect the other cases still document.
    """
    from monkeygrab.adapters.extraction.pymupdf_image_extractor import PymupdfImageExtractor
    from monkeygrab.config.app_config import AppConfig

    config = AppConfig.from_env()
    raised = config.with_overrides(**{"chunking.max_images_per_page": 999})

    assert PymupdfImageExtractor(config.chunking)._max_per_page != 999
    assert PymupdfImageExtractor(raised.chunking)._max_per_page == 999


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
