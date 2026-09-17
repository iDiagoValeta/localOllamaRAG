"""Can the full multimodal stack run here, and on what?

Detection only. Every check names what is missing instead of inferring; the
service's ``/health`` returns this verbatim so the caller (Daimon) can show the
reason rather than a generic "unavailable".
"""
import dataclasses
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from monkeygrab.composition import _isolated_python
from monkeygrab.config.app_config import AppConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROLES = ("rag", "chat", "contextual", "recomp")

# How long a /health answer stays valid. The CUDA probe imports torch in the
# isolated interpreter (seconds per call, 300 s worst case), so an uncached
# probe turns Daimon's polling into worker exhaustion (issue #263). Roles and
# commit cannot change mid-process, and the key below still re-probes the
# moment the roles do.
_PROBE_CACHE_TTL_SECONDS = 60.0
_probe_cache_lock = threading.Lock()
_probe_cache: Dict[str, Any] = {"key": None, "report": None, "at": 0.0}


@dataclasses.dataclass(frozen=True)
class HealthReport:
    """Outcome of ``probe``; ``ok`` is False exactly when ``reason`` is set."""

    ok: bool
    reason: Optional[str]
    commit: Optional[str]
    cuda: Dict[str, Any]
    isolated_env: Dict[str, Any]
    roles: Dict[str, Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _cuda_probe(python: str) -> Dict[str, Any]:
    """Ask the ISOLATED interpreter whether CUDA is visible: that is where jina-clip runs.

    Same probe as ``tools/setup_environments.py``; duplicated here because
    ``tools/`` is repo tooling, not product, and the service cannot import it.
    """
    script = (
        "import torch, json; "
        "print(json.dumps({'available': torch.cuda.is_available(), "
        "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
        "'torch': torch.__version__}))"
    )
    result = subprocess.run([python, "-c", script], capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"torch failed to import in the isolated env: {result.stderr.strip()[:200]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _git_commit() -> Optional[str]:
    """HEAD of this checkout, or None when git is unavailable (a diagnostic, never a failure)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _roles(config: AppConfig) -> Dict[str, Dict[str, str]]:
    return {
        role: {"backend": getattr(config.models, f"{role}_backend"), "model": getattr(config.models, role)}
        for role in _ROLES
    }


def probe(
    config: AppConfig,
    *,
    isolated_python: Callable[[], str] = _isolated_python,
    cuda_probe: Callable[[str], Dict[str, Any]] = _cuda_probe,
    git_commit: Callable[[], Optional[str]] = _git_commit,
) -> HealthReport:
    """Check the isolated interpreter and CUDA; report roles and commit.

    Args:
        config: Current config, for the per-role backend and model names.
        isolated_python: Returns the isolated interpreter path or raises
            ``FileNotFoundError``. Injected for tests.
        cuda_probe: Runs the CUDA check in that interpreter. Injected for tests.
        git_commit: Returns HEAD or None. Injected for tests.

    Returns:
        A ``HealthReport``; ``ok`` only when both checks passed.
    """
    commit = git_commit()
    roles = _roles(config)
    try:
        python = isolated_python()
    except FileNotFoundError as exc:
        return HealthReport(False, str(exc), commit, {}, {"present": False, "python": None}, roles)
    isolated_env = {"present": True, "python": python}
    try:
        cuda = cuda_probe(python)
    except Exception as exc:
        return HealthReport(False, str(exc), commit, {}, isolated_env, roles)
    if not cuda.get("available"):
        return HealthReport(
            False, f"CUDA is not available to the isolated interpreter (torch {cuda.get('torch')})",
            commit, cuda, isolated_env, roles,
        )
    return HealthReport(True, None, commit, cuda, isolated_env, roles)


def cached_probe(
    config: AppConfig,
    *,
    ttl: float = _PROBE_CACHE_TTL_SECONDS,
    isolated_python: Callable[[], str] = _isolated_python,
    cuda_probe: Callable[[str], Dict[str, Any]] = _cuda_probe,
    git_commit: Callable[[], Optional[str]] = _git_commit,
) -> HealthReport:
    """``probe`` with a TTL cache, for the ``/health`` route (issue #263).

    A liveness endpoint must be cheap; the raw probe is not. The cached entry
    is keyed by the role/backend/model mapping, so a model switch re-probes
    immediately instead of serving a stale report for the rest of the TTL.

    Args:
        config: Current config, for the per-role backend and model names.
        ttl: Cache lifetime in seconds.
        isolated_python: Returns the isolated interpreter path or raises
            ``FileNotFoundError``. Injected for tests.
        cuda_probe: Runs the CUDA check in that interpreter. Injected for tests.
        git_commit: Returns HEAD or None. Injected for tests.

    Returns:
        The cached ``HealthReport`` when fresh, otherwise a fresh ``probe``.
    """
    key = json.dumps(_roles(config), sort_keys=True)
    with _probe_cache_lock:
        if _probe_cache["key"] == key and time.monotonic() - _probe_cache["at"] < ttl:
            return _probe_cache["report"]
    report = probe(
        config, isolated_python=isolated_python, cuda_probe=cuda_probe, git_commit=git_commit
    )
    with _probe_cache_lock:
        _probe_cache.update(key=key, report=report, at=time.monotonic())
    return report


def _reset_probe_cache() -> None:
    """Drop the cached ``/health`` report. Tests only."""
    with _probe_cache_lock:
        _probe_cache.update(key=None, report=None, at=0.0)
