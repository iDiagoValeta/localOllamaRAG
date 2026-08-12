"""Regression tests for rag/engine/wiring.py's release_gpu_models (#25).

Nothing in rag/ freed the jina-clip worker or the Cross-Encoder weights
before Ollama loaded the generator, so both stayed resident for the life of
the process. On an 8 GB card that is enough to keep Ollama from loading at
all -- it does not fail fast when memory is short, it blocks until the
request times out. tests/eval/run_eval.py's _release_gpu_models already
proved the fix (embedder.close() + reranker.release() before generation);
these tests cover the equivalent hook wired into this module's own caches.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag.chat_pdfs  # noqa: F401 -- import before rag.engine.wiring, see its module docstring
from rag.engine import wiring


class _RecordingReleasable:
    """Stand-in for JinaClipEmbedder/CrossEncoderReranker: records release calls."""

    def __init__(self, method_name):
        self.calls = 0
        self._method_name = method_name
        setattr(self, method_name, self._record)

    def _record(self):
        self.calls += 1


def _fake_embedder():
    return _RecordingReleasable("close")


def _fake_reranker():
    return _RecordingReleasable("release")


def setup_function(_func):
    wiring._embedder_cache.update(embedder=None)
    wiring._reranker_cache.update(reranker=None)


def teardown_function(_func):
    wiring._embedder_cache.update(embedder=None)
    wiring._reranker_cache.update(reranker=None)


def test_release_gpu_models_closes_cached_embedder_and_clears_the_slot():
    embedder = _fake_embedder()
    wiring._embedder_cache.update(embedder=embedder)

    wiring.release_gpu_models()

    assert embedder.calls == 1
    assert wiring._embedder_cache["embedder"] is None


def test_release_gpu_models_releases_cached_reranker_and_clears_the_slot():
    reranker = _fake_reranker()
    wiring._reranker_cache.update(reranker=reranker)

    wiring.release_gpu_models()

    assert reranker.calls == 1
    assert wiring._reranker_cache["reranker"] is None


def test_release_gpu_models_releases_both_caches_in_one_call():
    embedder = _fake_embedder()
    reranker = _fake_reranker()
    wiring._embedder_cache.update(embedder=embedder)
    wiring._reranker_cache.update(reranker=reranker)

    wiring.release_gpu_models()

    assert embedder.calls == 1
    assert reranker.calls == 1
    assert wiring._embedder_cache["embedder"] is None
    assert wiring._reranker_cache["reranker"] is None


def test_release_gpu_models_is_a_no_op_when_nothing_is_cached():
    # Neither cache populated -- must not raise (e.g. on a /chat-only session
    # that never triggered a RAG retrieval).
    wiring.release_gpu_models()

    assert wiring._embedder_cache["embedder"] is None
    assert wiring._reranker_cache["reranker"] is None


def test_release_gpu_models_does_not_double_release_on_a_second_call():
    """A second call must find the slot already cleared, not re-release the
    same instance -- CrossEncoderReranker.release()/JinaClipEmbedder.close()
    already tolerate that themselves, but release_gpu_models should not even
    attempt it once the cache has been swapped to None."""
    embedder = _fake_embedder()
    reranker = _fake_reranker()
    wiring._embedder_cache.update(embedder=embedder)
    wiring._reranker_cache.update(reranker=reranker)

    wiring.release_gpu_models()
    wiring.release_gpu_models()

    assert embedder.calls == 1
    assert reranker.calls == 1


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
