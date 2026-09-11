"""Tests for wiring.release_reranker() (issue #239).

CrossEncoderReranker.release() existed to give back its CUDA weights, but a
repo-wide grep found zero callers: wiring.reranker() cached one instance as a
singleton with no reset path, so once loaded its weights stayed resident for
the rest of the process's life. release_reranker() is the exposed release
point, mirroring release_embedder -- but unlike the embedder (whose worker
process is gone for good and must be rebuilt), CrossEncoderReranker reloads
its own weights lazily on the next rerank() call, so the fix drops the
cached instance's weights without discarding the singleton itself.

Doubles CrossEncoderReranker entirely: what is under test is the wiring
cache's own behaviour, not the sentence-transformers adapter (that adapter's
own release() is covered by
tests/unit/adapters/test_cross_encoder_reranker.py).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag.chat_pdfs  # noqa: E402,F401
from monkeygrab.config.app_config import AppConfig  # noqa: E402
from rag.engine import wiring  # noqa: E402


class _FakeReranker:
    """Stands in for CrossEncoderReranker: only the release contract matters."""

    def __init__(self):
        self.released = 0

    def release(self):
        self.released += 1


def _reset_cache():
    with wiring._reranker_cache_lock:
        wiring._reranker_cache["reranker"] = None


def _patch_class(monkeypatch):
    built = []

    def _build():
        instance = _FakeReranker()
        built.append(instance)
        return instance

    monkeypatch.setattr(wiring, "CrossEncoderReranker", _build)
    return built


def test_release_reranker_calls_release_on_the_cached_instance(monkeypatch):
    _reset_cache()
    built = _patch_class(monkeypatch)
    config = AppConfig()

    instance = wiring.reranker(config)
    wiring.release_reranker()

    assert built == [instance]
    assert instance.released == 1


def test_release_reranker_keeps_the_singleton_so_the_next_call_reuses_it(monkeypatch):
    """Unlike release_embedder (which evicts its slot because the embedder's
    worker process is gone for good), the reranker reloads its weights
    lazily on the next rerank() -- so wiring.reranker() must keep returning
    the SAME object after a release, not build a second one."""
    _reset_cache()
    built = _patch_class(monkeypatch)
    config = AppConfig()

    first = wiring.reranker(config)
    wiring.release_reranker()
    second = wiring.reranker(config)

    assert first is second
    assert len(built) == 1


def test_release_reranker_is_a_no_op_before_anything_was_ever_built():
    _reset_cache()
    wiring.release_reranker()  # must not raise


def test_release_reranker_reaches_the_cached_instance_even_called_twice(monkeypatch):
    _reset_cache()
    built = _patch_class(monkeypatch)
    config = AppConfig()
    wiring.reranker(config)

    wiring.release_reranker()
    wiring.release_reranker()

    assert built[0].released == 2
