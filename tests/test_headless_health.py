"""The headless health probe says exactly why the full stack cannot run."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from monkeygrab.config import AppConfig
from rag.headless.health import probe


def _cuda_ok(python):
    return {"available": True, "device": "RTX 4060", "torch": "2.6.0"}


def test_ok_when_isolated_env_and_cuda_are_present(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_RAG_BACKEND", "openai")
    report = probe(
        AppConfig.from_env(),
        isolated_python=lambda: "/repo/.venv-mineru/bin/python",
        cuda_probe=_cuda_ok,
        git_commit=lambda: "abc123",
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
