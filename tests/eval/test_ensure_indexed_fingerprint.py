"""Pins the write-after-verify invariant in run_eval.ensure_indexed.

ensure_indexed writes the index fingerprint only after confirming every
required paper actually ended up in the store -- an indexing attempt that
silently drops a paper must raise EvalSetupError instead. Nothing enforced
that order: hoisting the write above the raise passed every other test.

Needs the full engine importable, like ensure_indexed itself does (it
unconditionally imports OllamaChatModel/Bm25LexicalIndex/CrossEncoderReranker
before any of its own branching runs, regardless of which flags this test
forces off). The monkeygrab.adapters import below is what lets
tests/conftest.py's heuristic skip this file in the dependency-free fast
gate, the same way it already skips tests/unit/adapters/*.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

import run_eval  # noqa: E402  (tests/eval sibling module; sets up sys.path for monkeygrab)
from monkeygrab.adapters.lexical.bm25_index import Bm25LexicalIndex  # noqa: E402,F401
from monkeygrab.config.app_config import AppConfig  # noqa: E402


class _FakeStore:
    """Vector store double that never actually gains the required paper."""

    def __init__(self):
        self.fingerprint_written = False

    def count(self):
        return 0

    def read_fingerprint(self):
        return None

    def clear(self):
        pass

    def get_page(self, *_a, **_kw):
        # Empty on every call, however many indexing passes run -- simulates
        # an indexing attempt that completes without actually adding the
        # required paper.
        return []

    def write_fingerprint(self, value):
        self.fingerprint_written = True


class _FakeIndexCorpus:
    """Stands in for the real IndexCorpus: a no-op run() is enough to reach
    the post-index check without needing a real extractor/embedder."""

    def __init__(self, *_a, **_kw):
        pass

    def run(self, *_a, **_kw):
        pass


class _FakeStack:
    def __init__(self, store):
        self.extractor = None
        self.embedder = None
        self.vector_store = store


class _FakeRag:
    @staticmethod
    def set_docs_folder_runtime(*_a, **_kw):
        pass


def test_aborted_indexing_never_writes_a_fingerprint(monkeypatch, tmp_path):
    store = _FakeStore()
    fake_stack = _FakeStack(store)
    fake_config = AppConfig().with_overrides(
        **{
            "flags.usar_embeddings_imagen": False,
            "flags.usar_contextual_retrieval": False,
        }
    )

    monkeypatch.setattr(run_eval, "_eval_app_config", lambda rag, carpeta, **_kw: fake_config)
    monkeypatch.setattr("monkeygrab.composition.build_stack", lambda config: fake_stack)
    monkeypatch.setattr("monkeygrab.application.index_corpus.IndexCorpus", _FakeIndexCorpus)

    with pytest.raises(run_eval.EvalSetupError):
        run_eval.ensure_indexed(_FakeRag(), tmp_path, ["missing.pdf"], "dev set")

    assert store.fingerprint_written is False


class _PresentStore:
    """Cache-hit store: fingerprint matches and the required paper is present."""

    def __init__(self, source="paper.pdf"):
        self.source = source
        self.fingerprint_written = None

    def count(self):
        return 1

    def read_fingerprint(self):
        return "fp"

    def write_fingerprint(self, value):
        self.fingerprint_written = value

    def get_page(self, *_a, **_kw):
        return [SimpleNamespace(metadata=SimpleNamespace(source=self.source))]


class _CapturingRetrieve:
    calls = []

    def __init__(self, *_a, **kwargs):
        type(self).calls.append(kwargs)


def _stub_cache_hit_collaborators(monkeypatch, store, config):
    """Get ensure_indexed to the Retrieve constructor without a real index."""
    monkeypatch.setattr(run_eval, "_eval_app_config", lambda rag, carpeta, **_kw: config)
    monkeypatch.setattr("monkeygrab.composition.build_stack", lambda _c: _FakeStack(store))
    monkeypatch.setattr(
        "monkeygrab.application.index_fingerprint.compute_index_fingerprint",
        lambda _c: "fp",
    )
    monkeypatch.setattr(
        "monkeygrab.adapters.lexical.bm25_index.Bm25LexicalIndex",
        lambda *_a, **_kw: object(),
    )
    monkeypatch.setattr("monkeygrab.application.answer.Answer", lambda *_a, **_kw: object())
    monkeypatch.setattr("monkeygrab.application.retrieve.Retrieve", _CapturingRetrieve)
    monkeypatch.setattr("rag.engine.wiring.query_decomposer", lambda _c: "decomposer-sentinel")
    monkeypatch.setattr("rag.engine.wiring.rag_chat_model", lambda _c: object())
    _CapturingRetrieve.calls = []


def test_ensure_indexed_wires_query_decomposer_when_the_flag_is_on(monkeypatch, tmp_path):
    """Issue #64: the product wires query_decomposer whenever the flag is
    on; the gate used to pass None unconditionally. A test that only
    inspects config_overrides would still pass that old bug."""
    store = _PresentStore()
    config = AppConfig().with_overrides(
        **{
            "flags.usar_embeddings_imagen": False,
            "flags.usar_contextual_retrieval": False,
            "flags.usar_reranker": False,
            "flags.usar_llm_query_decomposition": True,
        }
    )
    _stub_cache_hit_collaborators(monkeypatch, store, config)

    run_eval.ensure_indexed(_FakeRag(), tmp_path, ["paper.pdf"], "dev set")

    assert _CapturingRetrieve.calls[0]["query_decomposer"] == "decomposer-sentinel"


def test_ensure_indexed_leaves_query_decomposer_off_when_the_flag_is_off(monkeypatch, tmp_path):
    store = _PresentStore()
    config = AppConfig().with_overrides(
        **{
            "flags.usar_embeddings_imagen": False,
            "flags.usar_contextual_retrieval": False,
            "flags.usar_reranker": False,
            "flags.usar_llm_query_decomposition": False,
        }
    )
    decomposer_calls = []

    def _should_not_run(_config):
        decomposer_calls.append(True)
        return "decomposer-sentinel"

    _stub_cache_hit_collaborators(monkeypatch, store, config)
    monkeypatch.setattr("rag.engine.wiring.query_decomposer", _should_not_run)

    run_eval.ensure_indexed(_FakeRag(), tmp_path, ["paper.pdf"], "dev set")

    assert _CapturingRetrieve.calls[0]["query_decomposer"] is None
    assert decomposer_calls == []
