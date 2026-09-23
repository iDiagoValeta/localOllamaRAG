"""MinerU CLI timeout hard-fails as RuntimeError.

Regression test: ``_run_mineru`` passes ``timeout`` to ``subprocess.run`` but
only caught ``OSError``; ``subprocess.TimeoutExpired`` (a ``SubprocessError``,
not an ``OSError``) escaped unwrapped, contradicting the ``Raises`` contract.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from monkeygrab.adapters.extraction import mineru_extractor as module  # noqa: E402
from monkeygrab.adapters.extraction.mineru_extractor import MineruExtractor  # noqa: E402


def _make_pdf(tmp_path: Path, name="paper.pdf") -> Path:
    """Create a fake PDF file for the extractor to accept as input."""
    pdf = tmp_path / name
    pdf.write_bytes(b"%PDF-1.4 fake pdf bytes")
    return pdf


def _make_bin(tmp_path: Path, name="mineru.exe") -> Path:
    """Create a fake executable file so binary resolution succeeds."""
    bin_path = tmp_path / name
    bin_path.write_text("fake binary", encoding="utf-8")
    return bin_path


def test_mineru_timeout_expired_raises_runtime_error(monkeypatch, tmp_path):
    """TimeoutExpired from subprocess.run must surface as RuntimeError.

    Args:
        monkeypatch: Pytest fixture to stub ``subprocess.run``.
        tmp_path: Pytest fixture providing an isolated temp directory.
    """
    pdf = _make_pdf(tmp_path)
    bin_path = _make_bin(tmp_path)

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    extractor = MineruExtractor(
        mineru_bin=str(bin_path), cache_dir=str(tmp_path / "cache"), timeout_seconds=123,
    )

    with pytest.raises(RuntimeError, match="timed out after 123s") as excinfo:
        extractor.extract(str(pdf))

    message = str(excinfo.value)
    assert ".venv-mineru" in message
    assert pdf.name in message
