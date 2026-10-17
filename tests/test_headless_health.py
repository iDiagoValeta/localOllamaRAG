"""The headless health probe says exactly why the full stack cannot run."""
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from monkeygrab.config import AppConfig
from rag.headless import health
from rag.headless.health import _reset_probe_cache, cached_probe, probe


def _cuda_ok(python):
    return {"available": True, "device": "RTX 4060", "torch": "2.6.0"}


def _embedder_ok(config):
    return {"ok": True, "url": f"{config.models.embedding.base_url}/health"}


def _embedder_down(config):
    return {
        "ok": False,
        "url": f"{config.models.embedding.base_url}/health",
        "error": "Connection refused",
    }


def test_ok_when_isolated_env_and_cuda_are_present(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_RAG_BACKEND", "openai")
    report = probe(
        AppConfig.from_env(),
        isolated_python=lambda: "/repo/.venv-mineru/bin/python",
        cuda_probe=_cuda_ok,
        git_commit=lambda: "abc123",
        embedder_probe=_embedder_ok,
    )
    assert report.ok is True and report.reason is None
    body = report.to_dict()
    assert body["commit"] == "abc123"
    assert body["cuda"] == {"available": True, "device": "RTX 4060", "torch": "2.6.0"}
    assert body["isolated_env"] == {"present": True, "python": "/repo/.venv-mineru/bin/python"}
    assert body["roles"]["rag"] == {"backend": "openai", "model": AppConfig.from_env().models.rag}
    assert set(body["roles"]) == {"rag", "chat", "contextual", "recomp"}


def test_missing_isolated_env_is_the_reason():
    def missing():
        raise FileNotFoundError("The multimodal pipeline needs the isolated MinerU environment")

    report = probe(AppConfig.from_env(), isolated_python=missing, cuda_probe=_cuda_ok, git_commit=lambda: None)
    assert report.ok is False
    assert "isolated MinerU environment" in report.reason
    assert report.to_dict()["isolated_env"] == {"present": False, "python": None}


def test_cuda_unavailable_is_the_reason():
    report = probe(
        AppConfig.from_env(),
        isolated_python=lambda: "/p",
        cuda_probe=lambda python: {"available": False, "device": None, "torch": "2.6.0"},
        git_commit=lambda: None,
    )
    assert report.ok is False
    assert "CUDA" in report.reason
    assert report.commit is None


def test_cuda_probe_failure_is_reported_not_swallowed():
    def broken(python):
        raise RuntimeError("torch failed to import")

    report = probe(AppConfig.from_env(), isolated_python=lambda: "/p", cuda_probe=broken, git_commit=lambda: None)
    assert report.ok is False
    assert "torch failed to import" in report.reason


def test_an_unreachable_embedder_is_the_reason():
    report = probe(
        AppConfig.from_env(),
        isolated_python=lambda: "/p",
        cuda_probe=_cuda_ok,
        git_commit=lambda: "abc",
        embedder_probe=_embedder_down,
    )
    assert report.ok is False
    assert "embedder" in report.reason and "Connection refused" in report.reason
    assert report.embedder["ok"] is False


class _FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_the_embedder_probe_uses_the_appconfig_url_with_a_bounded_timeout(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_EMBEDDER_URL", "http://embedder.example:9000/")
    calls = []

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return _FakeResponse()

    monkeypatch.setattr(health.urllib.request, "urlopen", fake_urlopen)
    result = health._embedder_probe(AppConfig.from_env())
    assert result == {"ok": True, "url": "http://embedder.example:9000/health"}
    assert calls[0][0] == "http://embedder.example:9000/health"
    assert calls[0][1] <= 5


def test_cached_probe_rechecks_the_embedder_even_when_the_static_checks_are_cached():
    _reset_probe_cache()
    cuda_calls = []
    embedder_answers = [
        {"ok": True, "url": "u"},
        {"ok": False, "url": "u", "error": "Connection refused"},
    ]
    kwargs = {
        "isolated_python": lambda: "/p",
        "cuda_probe": _counting_cuda(cuda_calls),
        "git_commit": lambda: "abc",
        "embedder_probe": lambda config: embedder_answers.pop(0),
    }
    first = cached_probe(_config_with_rag("cache-embedder-model"), **kwargs)
    second = cached_probe(_config_with_rag("cache-embedder-model"), **kwargs)
    assert first.ok is True and first.embedder["ok"] is True
    assert second.ok is False and "embedder" in second.reason
    assert cuda_calls == ["/p"]


def test_cached_probe_lets_a_recovered_embedder_clear_the_reason():
    _reset_probe_cache()
    cuda_calls = []
    embedder_answers = [
        {"ok": False, "url": "u", "error": "Connection refused"},
        {"ok": True, "url": "u"},
    ]
    kwargs = {
        "isolated_python": lambda: "/p",
        "cuda_probe": _counting_cuda(cuda_calls),
        "git_commit": lambda: "abc",
        "embedder_probe": lambda config: embedder_answers.pop(0),
    }
    first = cached_probe(_config_with_rag("cache-embedder-recovery-model"), **kwargs)
    second = cached_probe(_config_with_rag("cache-embedder-recovery-model"), **kwargs)
    assert first.ok is False and "embedder" in first.reason
    assert second.ok is True and second.reason is None
    assert cuda_calls == ["/p"]


def _counting_cuda(calls, result=None):
    def cuda_probe(python):
        calls.append(python)
        return result or {"available": True, "device": "RTX 4060", "torch": "2.6.0"}

    return cuda_probe


def _config_with_rag(model):
    return AppConfig.from_env().with_overrides(**{"models.rag": model})


def test_cached_probe_probes_once_within_the_ttl():
    _reset_probe_cache()
    calls = []
    kwargs = {"isolated_python": lambda: "/p", "cuda_probe": _counting_cuda(calls),
              "git_commit": lambda: "abc", "embedder_probe": _embedder_ok}
    first = cached_probe(_config_with_rag("cache-hit-model"), **kwargs)
    second = cached_probe(_config_with_rag("cache-hit-model"), **kwargs)
    assert first.ok is True and second.ok is True
    assert calls == ["/p"]


def test_cached_probe_reprobes_once_the_ttl_expires():
    _reset_probe_cache()
    calls = []
    kwargs = {"isolated_python": lambda: "/p", "cuda_probe": _counting_cuda(calls),
              "git_commit": lambda: "abc", "ttl": 0, "embedder_probe": _embedder_ok}
    cached_probe(_config_with_rag("cache-ttl-model"), **kwargs)
    cached_probe(_config_with_rag("cache-ttl-model"), **kwargs)
    assert calls == ["/p", "/p"]


def test_cached_probe_reprobes_when_the_roles_change():
    _reset_probe_cache()
    calls = []
    kwargs = {"isolated_python": lambda: "/p", "cuda_probe": _counting_cuda(calls),
              "git_commit": lambda: "abc", "embedder_probe": _embedder_ok}
    cached_probe(_config_with_rag("cache-roles-a"), **kwargs)
    cached_probe(_config_with_rag("cache-roles-b"), **kwargs)
    assert calls == ["/p", "/p"]


def test_concurrent_stale_probes_share_one_computation():
    """A stale cache with a probe already running waits for it instead of probing again (issue #347)."""
    _reset_probe_cache()
    cuda_calls = []
    embedder_calls = []
    started = threading.Event()
    release = threading.Event()

    def cuda_probe(python):
        cuda_calls.append(python)
        started.set()
        release.wait(timeout=10)
        return {"available": True, "device": "RTX 4060", "torch": "2.6.0"}

    def embedder_probe(config):
        embedder_calls.append(config.models.rag)
        return _embedder_ok(config)

    kwargs = {"isolated_python": lambda: "/p", "cuda_probe": cuda_probe,
              "git_commit": lambda: "abc", "embedder_probe": embedder_probe}
    config = _config_with_rag("cache-single-flight-model")
    reports = []
    first = threading.Thread(target=lambda: reports.append(cached_probe(config, **kwargs)))
    first.start()
    assert started.wait(timeout=2), "the first probe never started"
    second_started = threading.Event()
    second = threading.Thread(
        target=lambda: (second_started.set(), reports.append(cached_probe(config, **kwargs)))
    )
    second.start()
    try:
        assert second_started.wait(timeout=2), "the second request never started"
        # Give it time to reach the probe (and, without coalescing, to enter a
        # second CUDA check) while the first one is still held.
        time.sleep(0.5)
        assert cuda_calls == ["/p"]
    finally:
        release.set()
        first.join(timeout=10)
        second.join(timeout=10)
    assert [report.ok for report in reports] == [True, True]
    assert cuda_calls == ["/p"]
    # The embedder is never cached: both requests still check it.
    assert len(embedder_calls) == 2


def test_cached_probe_caches_a_failure_report_too():
    _reset_probe_cache()
    calls = []
    config = _config_with_rag("cache-failure-model")
    kwargs = {"isolated_python": lambda: "/p",
              "cuda_probe": _counting_cuda(
                  calls, {"available": False, "device": None, "torch": "2.6.0"}),
              "git_commit": lambda: None, "embedder_probe": _embedder_ok}
    first = cached_probe(config, **kwargs)
    second = cached_probe(config, **kwargs)
    assert first.ok is False and second.ok is False
    assert calls == ["/p"]
