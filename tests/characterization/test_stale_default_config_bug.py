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


def test_hot_changing_keyword_result_limit_does_not_affect_calls_without_an_explicit_argument(monkeypatch):
    """Same defect, second location: ``busqueda_lexica_bm25``'s ``top_n``
    default was bound to ``cfg.N_RESULTADOS_KEYWORD`` (40) at import time.
    """
    monkeypatch.setattr(rag, "N_RESULTADOS_KEYWORD", 3)

    docs = [f"alpha token{i}" for i in range(10)]
    results, _metrics = rag.busqueda_lexica_bm25("alpha", _FakeCollection(docs))

    # All 10 positive BM25 matches come back; the "new" limit of 3 was never applied.
    assert len(results) == 10


def test_hot_changing_final_top_k_leaves_rerank_resultados_default_stale(monkeypatch):
    """Third location, checked at the signature level (invoking
    ``rerank_resultados`` for real would load a CrossEncoder/torch, which
    this suite deliberately avoids -- see module docstring on
    determinism/no-GPU). The bound default is enough to prove the same
    defect exists here independently of chunking and BM25."""
    default_before = inspect.signature(rag.rerank_resultados).parameters["top_k"].default
    monkeypatch.setattr(rag, "TOP_K_FINAL", 999)

    default_after = inspect.signature(rag.rerank_resultados).parameters["top_k"].default

    assert default_after == default_before  # unchanged despite the "live" TOP_K_FINAL mutation
    assert default_after != 999


def test_hot_changing_max_images_per_page_leaves_extraer_imagenes_pdf_default_stale(monkeypatch):
    """Fourth location: ``extraer_imagenes_pdf``'s ``max_por_pagina`` default
    was bound to ``cfg.MAX_IMAGENES_POR_PAGINA`` at import time. Checked at
    signature level to avoid a real PyMuPDF file-open call."""
    default_before = inspect.signature(rag.extraer_imagenes_pdf).parameters["max_por_pagina"].default
    monkeypatch.setattr(rag, "MAX_IMAGENES_POR_PAGINA", 999)

    default_after = inspect.signature(rag.extraer_imagenes_pdf).parameters["max_por_pagina"].default

    assert default_after == default_before
    assert default_after != 999


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
