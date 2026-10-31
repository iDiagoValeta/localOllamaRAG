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
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from monkeygrab.composition import isolated_python as _isolated_python_default
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
# Serializes the computation itself. Without it, every poll that arrived after
# the TTL expired started its own torch/CUDA subprocess, so a burst (or a slow
# probe) multiplied the very cost the cache exists to avoid (issue #347). Kept
# apart from the cache lock so readers of a fresh entry never queue behind a
# probe in flight.
_probe_refresh_lock = threading.Lock()
_probe_cache: Dict[str, Any] = {"key": None, "report": None, "at": 0.0}

# A dead llama-server must not hold Daimon's poll for long: 3 s is generous on
# loopback and keeps the whole probe within "a few seconds" even when the
# embedder never answers (issue #319).
_EMBEDDER_TIMEOUT_SECONDS = 3.0


@dataclasses.dataclass(frozen=True)
class HealthReport:
    """Outcome of ``probe``; ``ok`` is False exactly when ``reason`` is set."""

    ok: bool
    reason: Optional[str]
    commit: Optional[str]
    cuda: Dict[str, Any]
    isolated_env: Dict[str, Any]
    roles: Dict[str, Dict[str, str]]
    # Defaulted so reports built before the embedder check existed (and the
    # positionally constructed ones in tests) stay valid.
    embedder: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _cuda_probe(python: str) -> Dict[str, Any]:
    """Ask the ISOLATED interpreter whether CUDA is visible: that is where MinerU runs.

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


def _embedder_probe(config: AppConfig) -> Dict[str, Any]:
    """Ask llama-server's own ``/health`` whether the embedder is up (issue #319).

    Every index and retrieval run embeds through
    ``models.embedding.base_url``, so a stopped llama-server left ``/health``
    at ``ok`` until the first ``/rag`` returned 502. The URL is read from the
    same ``AppConfig`` the embedding adapter uses rather than from the
    environment again, so the check can never name a different server.

    Args:
        config: Current config; ``models.embedding.base_url`` is the endpoint.

    Returns:
        ``{"ok": True, "url": ...}`` on HTTP 200, otherwise ``ok`` False plus
        the ``error`` that explains it.
    """
    url = f"{config.models.embedding.base_url.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(url, timeout=_EMBEDDER_TIMEOUT_SECONDS) as response:  # nosec B310
            status = response.status
    except urllib.error.HTTPError as exc:
        return {"ok": False, "url": url, "error": f"HTTP {exc.code}"}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"ok": False, "url": url, "error": str(exc)}
    if status != 200:
        return {"ok": False, "url": url, "error": f"HTTP {status}"}
    return {"ok": True, "url": url}


def _embedder_reason(embedder: Dict[str, Any]) -> str:
    """One line saying which endpoint failed and why (see ``_embedder_probe``)."""
    return (
        f"embedder (llama-server) is not reachable at {embedder.get('url')}: "
        f"{embedder.get('error')}"
    )


def _with_embedder(report: "HealthReport", embedder: Dict[str, Any]) -> "HealthReport":
    """Fold a fresh embedder check into a cached report.

    A cached failure keeps its own ``reason``: the embedder is probed last in
    ``probe``, so an earlier failure is the one that explains why the stack
    cannot run here. Only a report that was otherwise ``ok`` can be flipped by
    an embedder that stopped answering.
    """
    if not report.ok or embedder.get("ok"):
        return dataclasses.replace(report, embedder=embedder)
    return dataclasses.replace(report, ok=False, reason=_embedder_reason(embedder), embedder=embedder)


def _roles(config: AppConfig) -> Dict[str, Dict[str, str]]:
    return {
        role: {"backend": getattr(config.models, f"{role}_backend"), "model": getattr(config.models, role)}
        for role in _ROLES
    }


def probe(
    config: AppConfig,
    *,
    isolated_python: Callable[[], str] = _isolated_python_default,
    cuda_probe: Callable[[str], Dict[str, Any]] = _cuda_probe,
    git_commit: Callable[[], Optional[str]] = _git_commit,
    embedder_probe: Callable[[AppConfig], Dict[str, Any]] = _embedder_probe,
) -> HealthReport:
    """Check the isolated interpreter, CUDA and the embedder; report roles and commit.

    Args:
        config: Current config, for the per-role backend and model names and
            for the embedder endpoint.
        isolated_python: Returns the isolated interpreter path or raises
            ``FileNotFoundError``. Injected for tests.
        cuda_probe: Runs the CUDA check in that interpreter. Injected for tests.
        git_commit: Returns HEAD or None. Injected for tests.
        embedder_probe: Checks the embedder endpoint. Injected for tests.

    Returns:
        A ``HealthReport``; ``ok`` only when every check passed.
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
    embedder = embedder_probe(config)
    if not embedder.get("ok"):
        return HealthReport(
            False, _embedder_reason(embedder), commit, cuda, isolated_env, roles, embedder,
        )
    return HealthReport(True, None, commit, cuda, isolated_env, roles, embedder)


def _fresh_report(key: str, ttl: float) -> Optional["HealthReport"]:
    """The cached report if ``key`` matches and it is within ``ttl``.

    The caller must hold ``_probe_cache_lock``.
    """
    if _probe_cache["key"] == key and time.monotonic() - _probe_cache["at"] < ttl:
        return _probe_cache["report"]
    return None


def cached_probe(
    config: AppConfig,
    *,
    ttl: float = _PROBE_CACHE_TTL_SECONDS,
    isolated_python: Callable[[], str] = _isolated_python_default,
    cuda_probe: Callable[[str], Dict[str, Any]] = _cuda_probe,
    git_commit: Callable[[], Optional[str]] = _git_commit,
    embedder_probe: Callable[[AppConfig], Dict[str, Any]] = _embedder_probe,
) -> HealthReport:
    """``probe`` with the expensive checks cached, for the ``/health`` route (issue #263).

    A liveness endpoint must be cheap; importing torch in the isolated
    interpreter is not. The cached entry is keyed by the role/backend/model
    mapping, so a model switch re-probes immediately instead of serving a
    stale report for the rest of the TTL. The embedder check is *not* cached:
    it is one HTTP GET on loopback, and a server that stops or starts has to
    show up on the next poll, not up to a TTL later (issue #319).

    Args:
        config: Current config, for the per-role backend and model names and
            for the embedder endpoint.
        ttl: Cache lifetime in seconds.
        isolated_python: Returns the isolated interpreter path or raises
            ``FileNotFoundError``. Injected for tests.
        cuda_probe: Runs the CUDA check in that interpreter. Injected for tests.
        git_commit: Returns HEAD or None. Injected for tests.
        embedder_probe: Checks the embedder endpoint. Injected for tests.

    Returns:
        The cached static checks plus a fresh embedder check.
    """
    key = json.dumps(_roles(config), sort_keys=True)
    with _probe_cache_lock:
        cached = _fresh_report(key, ttl)
    if cached is not None:
        return _with_embedder(cached, embedder_probe(config))
    with _probe_refresh_lock:
        # A probe that was already in flight may have refreshed the entry while
        # this request waited for that lock; reuse it instead of probing again.
        with _probe_cache_lock:
            cached = _fresh_report(key, ttl)
        if cached is not None:
            return _with_embedder(cached, embedder_probe(config))
        report = probe(
            config,
            isolated_python=isolated_python,
            cuda_probe=cuda_probe,
            git_commit=git_commit,
            embedder_probe=embedder_probe,
        )
        cached = report
        if not report.ok and report.embedder and not report.embedder.get("ok"):
            # The static checks passed and only the embedder was down (it is
            # probed last, so any earlier failure leaves ``embedder`` empty).
            # Cache that static success: otherwise the embedder's failure would
            # be replayed from the cache and a server that came back would stay
            # a 503 for the rest of the TTL, contradicting the fresh check below.
            cached = dataclasses.replace(report, ok=True, reason=None, embedder={})
        with _probe_cache_lock:
            _probe_cache.update(key=key, report=cached, at=time.monotonic())
    return report


def _reset_probe_cache() -> None:
    """Drop the cached ``/health`` report. Tests only."""
    with _probe_cache_lock:
        _probe_cache.update(key=None, report=None, at=0.0)
