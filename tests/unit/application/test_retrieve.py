"""Unit tests for monkeygrab.application.retrieve.Retrieve.

Covers the orchestration -- which query variants get embedded, which optional
stages run, and how reranking interacts with the relevance threshold -- using
hand-written port fakes, so nothing here touches Ollama, FAISS or a GPU.

The pieces Retrieve delegates to have their own files: RRF fusion in
test_rrf_fusion.py, the threshold filter in test_retrieve_threshold_equivalence.py,
and keyword extraction in test_keywords.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from monkeygrab.application.retrieve import Retrieve, _parse_subqueries  # noqa: E402
from monkeygrab.config.app_config import AppConfig  # noqa: E402
from monkeygrab.domain.chunk_metadata import ChunkMetadata  # noqa: E402
from monkeygrab.domain.fragment import Fragment  # noqa: E402


def test_parse_subqueries_strips_numbering_drops_short_lines_and_caps_at_three():
    """Models answer the decomposition prompt with numbered or bulleted lines
    despite being told not to, and sometimes pad with a stray fragment. The
    parser normalizes all of that: markers stripped, lines of 20 characters or
    fewer discarded as noise, and no more than three queries kept."""
    raw_response = (
        "1. How does the attention mechanism compute query-key similarity scores?\n"
        "- What role does multi-head attention play in the transformer encoder?\n"
        "short\n"  # <=20 chars: dropped
        "3) Why does the paper use scaled dot-product instead of additive attention?\n"
        "This is a fourth line that would be truncated by the top-3 cap anyway.\n"
    )

    parsed = _parse_subqueries(raw_response)

    assert parsed == [
        "How does the attention mechanism compute query-key similarity scores?",
        "What role does multi-head attention play in the transformer encoder?",
        "Why does the paper use scaled dot-product instead of additive attention?",
    ]


def test_parse_subqueries_returns_nothing_for_an_empty_response():
    assert _parse_subqueries("") == []
    assert _parse_subqueries("   \n  \n") == []


def _fragment(id_, score_final=0.0):
    source, rest = id_.split("_pag", 1)
    page_str, chunk_str = rest.split("_chunk", 1)
    return Fragment(
        doc=f"doc-{id_}",
        metadata=ChunkMetadata(source=source, page=int(page_str), chunk=int(chunk_str)),
        score_final=score_final,
    )


class FakeEmbedder:
    def __init__(self):
        self.calls = []
        self.keep_alive_calls = []

    def embed(self, text, *, keep_alive=None):
        self.calls.append(text)
        self.keep_alive_calls.append(keep_alive)
        return [0.0]


class FakeVectorStore:
    def __init__(self, hits_by_call=None, default_hits=None):
        self._hits_by_call = list(hits_by_call or [])
        self._default_hits = default_hits or []
        self.query_calls = 0

    def query(self, embedding, n_results):
        self.query_calls += 1
        if self._hits_by_call:
            return self._hits_by_call.pop(0)
        return self._default_hits


class FakeLexicalIndex:
    def __init__(self, hits=None):
        self._hits = hits or []
        self.search_calls = 0

    def search(self, query, top_n):
        self.search_calls += 1
        return self._hits


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, fragments, top_k):
        self.calls.append((query, len(fragments), top_k))
        # Simple deterministic re-score: reverse input order, mark score_reranker.
        import dataclasses
        return [dataclasses.replace(f, score_reranker=1.0 - i * 0.1) for i, f in enumerate(reversed(fragments[:top_k]))]


class FakeQueryDecomposer:
    def __init__(self, subqueries=None, raises=False):
        self._subqueries = subqueries or []
        self._raises = raises
        self.calls = []

    def generate(self, prompt, *, system=None, images=()):
        self.calls.append(prompt)
        if self._raises:
            raise RuntimeError("decomposer unavailable")
        return "\n".join(self._subqueries)


def _config(**overrides):
    cfg = AppConfig()
    if overrides:
        cfg = cfg.with_overrides(**overrides)
    return cfg


def test_semantic_only_search_when_hybrid_and_reranker_disabled():
    hits = [_fragment("a.pdf_pag0_chunk0"), _fragment("b.pdf_pag0_chunk0")]
    embedder = FakeEmbedder()
    store = FakeVectorStore(default_hits=hits)
    config = _config(**{"flags.usar_busqueda_hibrida": False, "flags.usar_reranker": False})

    result = Retrieve(embedder, store, config).run("short query")

    assert [f.id for f in result.fragments] == ["a.pdf_pag0_chunk0", "b.pdf_pag0_chunk0"]
    assert result.metrics["reranked"] is False
    # The question plus the keyword-derived fallback variant; no decomposition,
    # since the question is below the length threshold.
    assert result.metrics["query_variants"] == ["short query", "query short"]
    assert store.query_calls == 2


def test_query_decomposition_only_triggers_above_60_chars_and_adds_variants():
    short_question = "Short question."
    long_question = "A" * 61  # len > 60

    embedder = FakeEmbedder()
    store = FakeVectorStore(default_hits=[])
    decomposer = FakeQueryDecomposer(subqueries=[
        "First reasonably long sub-query about the topic at hand here.",
        "Second reasonably long sub-query about a different aspect entirely.",
    ])
    config = _config(**{"flags.usar_busqueda_hibrida": False, "flags.usar_reranker": False})

    result = Retrieve(embedder, store, config, query_decomposer=decomposer).run(short_question)
    assert decomposer.calls == []  # too short: decomposition never invoked
    # Original question plus the keyword fallback that stands in for the
    # sub-queries decomposition did not produce.
    assert result.metrics["query_variants"] == ["Short question.", "short question"]
    assert store.query_calls == 2

    embedder2 = FakeEmbedder()
    store2 = FakeVectorStore(default_hits=[])
    # A run of identical characters yields no usable keyword, so this run
    # isolates decomposition: original + 2 sub-queries, no fallback variant.
    result2 = Retrieve(embedder2, store2, config, query_decomposer=decomposer).run(long_question)
    assert len(decomposer.calls) == 1
    assert result2.metrics["keywords"] == []
    assert store2.query_calls == 3


def test_decomposer_failure_falls_back_to_original_question_only():
    long_question = "B" * 61
    embedder = FakeEmbedder()
    store = FakeVectorStore(default_hits=[])
    decomposer = FakeQueryDecomposer(raises=True)
    config = _config(**{"flags.usar_busqueda_hibrida": False, "flags.usar_reranker": False})

    result = Retrieve(embedder, store, config, query_decomposer=decomposer).run(long_question)

    assert result.metrics["sub_queries"] == []
    assert store.query_calls == 1


def test_missing_optional_ports_disable_their_stage_regardless_of_flags():
    """flags default to True for hybrid/reranker/decomposition, but with no
    lexical_index/reranker/query_decomposer wired in, those stages must be
    silently absent -- there is nothing to invoke."""
    hits = [_fragment("a.pdf_pag0_chunk0")]
    embedder = FakeEmbedder()
    store = FakeVectorStore(default_hits=hits)
    config = _config()  # all flags at their True defaults

    result = Retrieve(embedder, store, config).run("A question long enough to trigger decomposition normally right here.")

    assert result.metrics["reranked"] is False
    assert result.metrics["keyword_candidates"] == 0
    assert result.metrics["sub_queries"] == []


def test_reranker_runs_and_then_threshold_filters_its_output():
    hits = [_fragment(f"doc{i}.pdf_pag0_chunk0") for i in range(3)]
    embedder = FakeEmbedder()
    store = FakeVectorStore(default_hits=hits)
    reranker = FakeReranker()
    config = _config(**{
        "flags.usar_busqueda_hibrida": False,
        "reranking.score_threshold": 0.85,
    })

    result = Retrieve(embedder, store, config, reranker=reranker).run("short query")

    assert reranker.calls  # reranker was invoked
    # FakeReranker scores: 1.0, 0.9, 0.8 -- the threshold is inclusive (>=),
    # so 1.0 and 0.9 clear 0.85 and only 0.8 is filtered out.
    assert [round(f.score_reranker, 2) for f in result.fragments] == [1.0, 0.9]


def test_top_k_final_truncates_even_when_reranker_is_off():
    hits = [_fragment(f"doc{i}.pdf_pag0_chunk0") for i in range(5)]
    embedder = FakeEmbedder()
    store = FakeVectorStore(default_hits=hits)
    config = _config(**{
        "flags.usar_busqueda_hibrida": False,
        "flags.usar_reranker": False,
        "retrieval.top_k_final": 2,
    })

    result = Retrieve(embedder, store, config).run("short query")

    assert len(result.fragments) == 2


def test_single_query_variant_unloads_the_embedding_model_immediately():
    """With only the original question, that one call IS the last variant --
    it must ask to unload the embedding model so the RAG generator that runs
    next fits in VRAM.

    The question is all stopwords, so no keyword fallback variant is built
    and no decomposition runs at this length: exactly one embedding call.
    """
    embedder = FakeEmbedder()
    store = FakeVectorStore(default_hits=[])
    config = _config(**{"flags.usar_busqueda_hibrida": False, "flags.usar_reranker": False})

    result = Retrieve(embedder, store, config).run("the is of")

    assert result.metrics["query_variants"] == ["the is of"]
    assert embedder.keep_alive_calls == [0]


def test_only_the_last_of_several_query_variants_unloads_the_embedding_model():
    long_question = "C" * 61
    embedder = FakeEmbedder()
    store = FakeVectorStore(default_hits=[])
    decomposer = FakeQueryDecomposer(subqueries=[
        "First reasonably long sub-query about the topic at hand here.",
        "Second reasonably long sub-query about a different aspect entirely.",
    ])
    config = _config(**{"flags.usar_busqueda_hibrida": False, "flags.usar_reranker": False})

    Retrieve(embedder, store, config, query_decomposer=decomposer).run(long_question)

    # 3 variants (original + 2 sub-queries): only the last one unloads.
    assert embedder.keep_alive_calls == [None, None, 0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
