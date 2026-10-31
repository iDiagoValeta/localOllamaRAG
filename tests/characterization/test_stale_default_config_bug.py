"""Regression tests for the historical stale-default configuration bug.

Several engine functions used to capture pipeline configuration in Python
default argument values evaluated at module-import time. Runtime changes to
``rag.chat_pdfs`` then affected parameters read in function bodies but not
parameters bound in signatures. Every one of those locations has since moved
to the core, where config is a required argument supplied per run from
``AppConfig``; the tests below keep that contract explicit, one per former
location. The chunking pair used to go through ``rag.dividir_en_chunks``, a
wrapper removed as dead code (issue #537); they now check the chunker the
product calls.
"""

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monkeygrab.application.text_chunking import split_markdown_into_chunks


def test_chunk_size_and_overlap_have_no_default_to_go_stale():
    """Formerly import-time ``cfg.CHUNK_SIZE``/``cfg.CHUNK_OVERLAP`` defaults.

    ``IndexCorpus`` passes both from ``AppConfig`` on every run, and the
    chunker requires them, so there is no default left to freeze. Checked at
    the signature level, like the reranker below.
    """
    params = inspect.signature(split_markdown_into_chunks).parameters

    assert params["chunk_size"].default is inspect.Parameter.empty
    assert params["overlap"].default is inspect.Parameter.empty


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


def test_contrast_min_chunk_length_takes_effect_on_the_next_call():
    """Contrast case: the value a caller passes is the value that applies,
    call by call -- proving the defect above was about default-argument
    binding, not about a changed setting being unobservable in general.
    """
    texto = "Body text of a single short paragraph that is not empty at all."  # 65 chars

    assert split_markdown_into_chunks(texto, 2000, 0, 10) != []
    assert split_markdown_into_chunks(texto, 2000, 0, 1000) == []


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
