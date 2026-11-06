"""Unit tests for monkeygrab.adapters.reranking.cross_encoder_reranker.CrossEncoderReranker.

Stubs sentence_transformers.CrossEncoder entirely -- no GPU, no model
download -- so these run in milliseconds.
"""

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from monkeygrab.adapters.reranking import cross_encoder_reranker as module
from monkeygrab.adapters.reranking.cross_encoder_reranker import CrossEncoderReranker
from monkeygrab.domain.chunk_metadata import ChunkMetadata
from monkeygrab.domain.fragment import Fragment


def _fragment(text, chunk=0):
    return Fragment(doc=text, metadata=ChunkMetadata(source="doc.pdf", page=0, chunk=chunk))


class FakeCrossEncoder:
    """Records construction args; .rank() returns a canned ranking."""

    instances = []

    def __init__(self, model_name_or_path, device, model_kwargs, **extra):
        self.model_name = model_name_or_path
        self.device = device
        self.model_kwargs = model_kwargs
        self.extra = extra
        self.rank_calls = []
        FakeCrossEncoder.instances.append(self)

    def rank(self, query, documents, top_k, return_documents, batch_size=32):
        self.rank_calls.append({"query": query, "documents": documents, "top_k": top_k, "batch_size": batch_size})
        # Return highest corpus_id first with a made-up descending score.
        return [
            {"corpus_id": i, "score": 1.0 - i * 0.1}
            for i in range(min(top_k, len(documents)))
        ]


@pytest.fixture(autouse=True)
def _reset_fake_instances():
    FakeCrossEncoder.instances = []
    yield
    FakeCrossEncoder.instances = []


def _patch_crossencoder(monkeypatch, factory=None):
    monkeypatch.setattr(module, "CrossEncoder", factory or FakeCrossEncoder)


def _force_cpu(monkeypatch):
    # Deterministic across dev machines with/without a GPU.
    monkeypatch.setenv("RERANKER_DEVICE", "cpu")


def test_batch_size_and_max_length_come_from_the_environment(monkeypatch):
    _patch_crossencoder(monkeypatch)
    _force_cpu(monkeypatch)
    monkeypatch.setenv("RERANKER_BATCH_SIZE", "8")
    monkeypatch.setenv("RERANKER_MAX_LENGTH", "1024")
    CrossEncoderReranker().rerank("q", [_fragment("a")], top_k=1)
    model = FakeCrossEncoder.instances[0]
    assert model.extra == {"max_length": 1024}
    assert model.rank_calls[0]["batch_size"] == 8


def test_without_tuning_env_the_reranker_keeps_its_defaults(monkeypatch):
    _patch_crossencoder(monkeypatch)
    _force_cpu(monkeypatch)
    monkeypatch.delenv("RERANKER_BATCH_SIZE", raising=False)
    monkeypatch.setenv("RERANKER_MAX_LENGTH", "nope")
    CrossEncoderReranker().rerank("q", [_fragment("a")], top_k=1)
    model = FakeCrossEncoder.instances[0]
    assert model.extra == {}
    assert model.rank_calls[0]["batch_size"] == 32


def test_reranker_uses_fixed_bge_model(monkeypatch):
    _patch_crossencoder(monkeypatch)
    _force_cpu(monkeypatch)

    CrossEncoderReranker().rerank("query", [_fragment("a")], top_k=1)

    assert FakeCrossEncoder.instances[0].model_name == "BAAI/bge-reranker-v2-m3"


def test_model_is_not_loaded_until_the_first_rerank_call(monkeypatch):
    _patch_crossencoder(monkeypatch)
    _force_cpu(monkeypatch)

    CrossEncoderReranker()  # construction alone must not load anything

    assert FakeCrossEncoder.instances == []


def test_rerank_with_no_fragments_returns_empty_list_without_loading_the_model(monkeypatch):
    _patch_crossencoder(monkeypatch)
    _force_cpu(monkeypatch)

    result = CrossEncoderReranker().rerank("query", [], top_k=5)

    assert result == []
    assert FakeCrossEncoder.instances == []


def test_rerank_sets_score_reranker_and_score_final_from_the_model(monkeypatch):
    _patch_crossencoder(monkeypatch)
    _force_cpu(monkeypatch)

    fragments = [_fragment("a", 0), _fragment("b", 1)]
    result = CrossEncoderReranker().rerank("query", fragments, top_k=2)

    assert [f.metadata.chunk for f in result] == [0, 1]
    assert result[0].score_reranker == pytest.approx(1.0)
    assert result[0].score_final == pytest.approx(1.0)
    assert result[1].score_reranker == pytest.approx(0.9)


def test_explicit_device_override_is_used_verbatim(monkeypatch):
    _patch_crossencoder(monkeypatch)

    CrossEncoderReranker(device="cuda").rerank("q", [_fragment("a")], top_k=1)

    assert FakeCrossEncoder.instances[0].device == "cuda"


def test_forced_device_load_failure_hard_fails_without_a_cpu_fallback(monkeypatch):
    """RERANKER_DEVICE='cpu' pins the device -- a load failure there must
    raise immediately, never silently trying another device."""
    monkeypatch.setenv("RERANKER_DEVICE", "cpu")

    def always_raises(model_name_or_path, device, model_kwargs):
        raise OSError("model weights not found")

    _patch_crossencoder(monkeypatch, factory=always_raises)

    with pytest.raises(RuntimeError, match="failed to load on 'cpu'"):
        CrossEncoderReranker().rerank("q", [_fragment("a")], top_k=1)


def test_auto_detected_cuda_failure_does_not_fall_back_to_cpu(monkeypatch):
    """An auto-selected CUDA device is still hard-fail, never changed silently."""
    monkeypatch.delenv("RERANKER_DEVICE", raising=False)

    fake_torch = type(sys)("torch")
    fake_torch.cuda = type(sys)("cuda")
    fake_torch.cuda.is_available = lambda: True
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    attempts = []

    def always_raises(model_name_or_path, device, model_kwargs):
        attempts.append(device)
        raise RuntimeError("CUDA out of memory")

    _patch_crossencoder(monkeypatch, factory=always_raises)

    with pytest.raises(RuntimeError, match="failed to load on 'cuda'"):
        CrossEncoderReranker().rerank("q", [_fragment("a")], top_k=1)

    assert attempts == ["cuda"]


def test_rerank_hard_fails_when_scoring_itself_raises(monkeypatch):
    """The model loads fine but .rank() blows up mid-inference -- must raise,
    not silently return the untouched input."""
    _force_cpu(monkeypatch)

    class LoadsButFailsToScore(FakeCrossEncoder):
        def rank(self, query, documents, top_k, return_documents):
            raise RuntimeError("inference error")

    _patch_crossencoder(monkeypatch, factory=LoadsButFailsToScore)

    with pytest.raises(RuntimeError, match="reranking failed"):
        CrossEncoderReranker().rerank("q", [_fragment("a")], top_k=1)


def test_release_drops_the_model_so_the_next_rerank_reloads_it(monkeypatch):
    """release() existed with no caller anywhere in the codebase (#239) and
    no test either -- this pins what it must do: forget the loaded model, so
    a later rerank() builds a fresh CrossEncoder instead of reusing the freed
    one."""
    _patch_crossencoder(monkeypatch)
    _force_cpu(monkeypatch)

    reranker = CrossEncoderReranker()
    reranker.rerank("q", [_fragment("a")], top_k=1)
    assert len(FakeCrossEncoder.instances) == 1

    reranker.release()
    reranker.rerank("q", [_fragment("a")], top_k=1)

    assert len(FakeCrossEncoder.instances) == 2


def test_release_before_any_rerank_call_does_not_load_anything(monkeypatch):
    _patch_crossencoder(monkeypatch)
    _force_cpu(monkeypatch)

    CrossEncoderReranker().release()  # must not raise, must not load the model

    assert FakeCrossEncoder.instances == []


def test_concurrent_first_reranks_load_the_model_only_once(monkeypatch):
    """Two threads reaching an unloaded reranker together share one load (#368).

    A second load would allocate a second full copy of the weights, which is
    the VRAM an 8GB card does not have to spare. The slow factory widens the
    window between the null check and the assignment, so an unguarded
    ``_ensure_model`` deterministically loads twice.
    """
    _force_cpu(monkeypatch)
    load_calls = []
    barrier = threading.Barrier(2)

    def slow_counting_factory(model_name_or_path, device, model_kwargs):
        load_calls.append(device)
        time.sleep(0.1)  # keep the loader busy while the other thread arrives
        return FakeCrossEncoder(model_name_or_path, device, model_kwargs)

    _patch_crossencoder(monkeypatch, factory=slow_counting_factory)

    reranker = CrossEncoderReranker()
    errors = []

    def worker():
        try:
            barrier.wait(timeout=5)
            reranker.rerank("q", [_fragment("a")], top_k=1)
        except BaseException as exc:  # re-surfaced below, never swallowed
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert not any(thread.is_alive() for thread in threads)
    assert load_calls == ["cpu"]
    assert len(FakeCrossEncoder.instances) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
