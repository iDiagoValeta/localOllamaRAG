"""The gate's preflight must load the reranker, not just reach Ollama and the embedder (issue #610).

A driver upgraded without a reboot left Ollama answering while PyTorch could
not initialise NVML, so every retrieval case of a 25-minute run recorded the
same infrastructure error. The preflight scores one pair on the device the run
will use and always releases the weights afterwards.

No torch and no model: a fake reranker stands in, so this file also runs in
the dependency-free fast gate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

import run_eval


class _FakeReranker:
    def __init__(self, error=None):
        self.error = error
        self.reranked = []
        self.released = False

    def rerank(self, query, fragments, top_k):
        self.reranked.append((query, len(fragments), top_k))
        if self.error:
            raise self.error
        return list(fragments)

    def release(self):
        self.released = True


def test_a_reranker_that_scores_passes_and_is_released():
    reranker = _FakeReranker()

    run_eval.preflight_reranker(reranker)

    assert reranker.reranked == [("preflight", 1, 1)]
    assert reranker.released


def test_a_load_failure_stops_the_run_with_the_cause_and_a_way_out():
    nvml = RuntimeError(
        "Cross-Encoder 'BAAI/bge-reranker-v2-m3' failed to load on 'cuda': "
        "NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED"
    )
    reranker = _FakeReranker(error=nvml)

    with pytest.raises(run_eval.EvalSetupError) as excinfo:
        run_eval.preflight_reranker(reranker)

    message = str(excinfo.value)
    assert "nvmlInit_v2_" in message
    assert "reboot" in message
    assert "USAR_RERANKER=false" in message
    assert excinfo.value.__cause__ is nvml
    assert reranker.released
