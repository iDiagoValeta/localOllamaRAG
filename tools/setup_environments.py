#!/usr/bin/env python3
"""setup_environments.py -- build the two interpreters MonkeyGrab needs, and check them.

Issue #120. ``README.md`` said "there is no setup script for this yet", and the
consequence was that the isolated environment's contents lived only in prose:
MinerU runs out of its own interpreter (``.venv-mineru``) with a CUDA build of
torch, and a reader had to turn that into pip invocations, in the right order,
from the right index.

The order is the part that is not obvious and fails late. The CUDA torch build
comes first, from the PyTorch index, so the CPU wheel is never chosen, and
``mineru[core]`` is installed afterwards because it resolves a large dependency
tree of its own.

Usage:
    python tools/setup_environments.py            # build both, then check
    python tools/setup_environments.py --check    # check only, install nothing

Teardown is one line and always has been::

    rm -rf .venv .venv-mineru

What this does NOT do: pull Ollama models. It reports which ones the
configured roles ask for and which are missing, because that is gigabytes over
someone else's connection. It also never writes ``requirements*.txt`` -- those
are pinned and a protected zone (``AGENTS.md`` rule 5); it reads them.

Platform: Linux and macOS layouts (``bin/``) and Windows (``Scripts/``) are
both handled, but only the POSIX path is exercised on this machine.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

# The script runs from a checkout with no installed package, and the embedder
# endpoint lives in the product's AppConfig (AGENTS.md section 3). Resolving it
# here keeps this check naming the same server the pipeline will call.
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from monkeygrab.config import AppConfig

PRODUCT_VENV = REPO_ROOT / ".venv"
ISOLATED_VENV = REPO_ROOT / ".venv-mineru"

# The CUDA build of torch MinerU needs, and the index it comes from. Not read
# from a requirements file on purpose: there is no requirements file for this
# environment.
TORCH_PIN = "torch==2.6.0"
TORCHVISION_PIN = "torchvision==0.21.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"

# Pinned because every gate row in docs/model-history.md was measured on 3.4.5,
# the version is part of the index fingerprint (#426), and 4.x dropped the `-p`
# option the adapter passes (#573): an unpinned install breaks every indexing run.
MINERU_PIN = "mineru[core]==3.4.5"

OLLAMA_ROLE_VARS = {
    "rag": "OLLAMA_RAG_MODEL",
    "chat": "OLLAMA_CHAT_MODEL",
    "contextual": "OLLAMA_CONTEXTUAL_MODEL",
    "recomp": "OLLAMA_RECOMP_MODEL",
}
DEFAULT_OLLAMA_MODEL = "gemma4:e4b"

OK = "  ok   "
FAIL = " FAIL  "
WARN = " warn  "

# Same short bound the desktop app and the headless service use: a stopped
# server must not turn a setup check into a hang.
_EMBEDDER_HEALTH_TIMEOUT_SECONDS = 3.0


def _venv_python(venv: Path) -> Path:
    """The interpreter inside ``venv``, whichever layout this platform uses."""
    windows = venv / "Scripts" / "python.exe"
    return windows if windows.exists() else venv / "bin" / "python"


def _run(command: Sequence[str], *, label: str) -> None:
    """Run ``command``, streaming its output, and abort the script if it fails.

    Hard-fail rather than "continue and see": a half-built environment is the
    state this script exists to prevent someone from ending up in by hand.
    """
    print(f"\n$ {' '.join(str(part) for part in command)}")
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")


def build_product_venv() -> None:
    """``.venv`` -- the product's own interpreter, from the pinned requirements."""
    if not _venv_python(PRODUCT_VENV).exists():
        _run([sys.executable, "-m", "venv", str(PRODUCT_VENV)], label="creating .venv")
    python = str(_venv_python(PRODUCT_VENV))
    _run([python, "-m", "pip", "install", "--upgrade", "pip"], label="upgrading pip")
    _run(
        [
            python, "-m", "pip", "install",
            "-r", str(REPO_ROOT / "rag" / "requirements.txt"),
            "-r", str(REPO_ROOT / "rag" / "web" / "requirements.txt"),
            "pytest", "ruff",
        ],
        label="installing product dependencies",
    )


def build_isolated_venv() -> None:
    """``.venv-mineru`` -- MinerU's own interpreter.

    torch first (from the CUDA index, so the CPU wheel is never chosen), then
    ``mineru[core]`` (it resolves a large tree of its own).
    """
    if not _venv_python(ISOLATED_VENV).exists():
        _run([sys.executable, "-m", "venv", str(ISOLATED_VENV)], label="creating .venv-mineru")
    python = str(_venv_python(ISOLATED_VENV))
    _run([python, "-m", "pip", "install", "--upgrade", "pip"], label="upgrading pip")
    _run(
        [python, "-m", "pip", "install", TORCH_PIN, TORCHVISION_PIN, "--index-url", TORCH_INDEX],
        label="installing CUDA torch",
    )
    _run([python, "-m", "pip", "install", MINERU_PIN], label="installing MinerU")


def _check_interpreter(venv: Path, label: str) -> Tuple[bool, str]:
    python = _venv_python(venv)
    if not python.exists():
        return False, f"{label}: no interpreter at {python}"
    result = subprocess.run(
        [str(python), "--version"], capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        return False, f"{label}: {python} does not run"
    return True, f"{label}: {result.stdout.strip() or result.stderr.strip()}"


def _check_cuda() -> Tuple[bool, str]:
    """CUDA must be visible to the ISOLATED interpreter -- that is where MinerU runs.

    MinerU is the only component in ``.venv-mineru`` and the one that uses the
    GPU, so this reports the isolated interpreter's own view of the card.
    """
    python = _venv_python(ISOLATED_VENV)
    if not python.exists():
        return False, "CUDA: cannot check, .venv-mineru has no interpreter"
    probe = (
        "import torch, json; "
        "print(json.dumps({'available': torch.cuda.is_available(), "
        "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
        "'torch': torch.__version__}))"
    )
    result = subprocess.run([str(python), "-c", probe], capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        return False, f"CUDA: torch failed to import in .venv-mineru: {result.stderr.strip()[:200]}"
    info = json.loads(result.stdout.strip().splitlines()[-1])
    if not info["available"]:
        return False, f"CUDA: not available to .venv-mineru (torch {info['torch']})"
    return True, f"CUDA: {info['device']} (torch {info['torch']})"


def _check_mineru_binary() -> Tuple[bool, str]:
    for candidate in (ISOLATED_VENV / "bin" / "mineru", ISOLATED_VENV / "Scripts" / "mineru.exe"):
        if candidate.exists():
            result = subprocess.run(
                [str(candidate), "--version"], capture_output=True, text=True, timeout=120
            )
            # A binary that exists but prints nothing to either stream is the
            # case that used to raise IndexError out of a diagnostic whose one
            # job is to survive a broken install and describe it.
            lines = (result.stdout or result.stderr or "").strip().splitlines()
            version = lines[-1] if lines else "present, but --version printed nothing"
            if result.returncode != 0:
                # A file that exists but cannot run (a missing CUDA library, a
                # half-built venv) is not an installed MinerU; reporting it as
                # present is what issue #385 fixed.
                return False, f"MinerU: {candidate} --version exited {result.returncode} ({version})"
            return True, f"MinerU: {version}"
    return False, f"MinerU: no binary under {ISOLATED_VENV}"


def _settings_path() -> Path:
    """Where the app keeps its persisted choices.

    Mirrors ``DATA_DIR`` in ``rag/chat_pdfs.py``: ``MONKEYGRAB_DATA_DIR`` when
    set, the ``rag/`` package dir otherwise. Duplicated rather than imported
    because this script and its test must run on the standard library alone --
    ``rag.engine.settings`` reaches ``rag/chat_pdfs.py`` and pulls torch in.
    ``TestSettingsLocationDrift`` fails if the product's rule moves.
    """
    data_dir = os.getenv("MONKEYGRAB_DATA_DIR") or REPO_ROOT / "rag"
    return Path(data_dir).resolve() / "settings.json"


def _saved_roles() -> dict:
    """The model roles saved by the web UI, or ``{}`` if unreadable.

    Best-effort like ``rag/engine/settings.py``: a missing or corrupt file
    leaves the defaults standing rather than failing a check whose job is to
    report state.
    """
    try:
        # utf-8-sig for the same reason the product reads it that way: the file
        # may have been hand-edited with a tool that writes a BOM.
        with open(_settings_path(), "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    roles = data.get("roles") if isinstance(data, dict) else None
    if not isinstance(roles, dict):
        return {}
    return {
        role: model
        for role, model in roles.items()
        if isinstance(model, str) and model.strip()
    }


def _configured_models() -> List[str]:
    """The Ollama models the four roles resolve to.

    Same precedence the product runs under -- environment > ``settings.json`` >
    module default (``AGENTS.md`` §3). Reading only the first and last was
    issue #215: it recommended pulling gigabytes of a model this machine would
    never load, and stayed green when a role saved in the UI named a tag that
    was not pulled.
    """
    saved = _saved_roles()
    return sorted(
        {
            os.getenv(var) or saved.get(role) or DEFAULT_OLLAMA_MODEL
            for role, var in OLLAMA_ROLE_VARS.items()
        }
    )


def _check_ollama() -> Tuple[bool, str]:
    base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=5) as response:
            installed = {
                model["name"] for model in json.loads(response.read().decode("utf-8"))["models"]
            }
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        return False, f"Ollama: not reachable at {base_url} ({exc})"

    wanted = _configured_models()
    missing = [name for name in wanted if name not in installed]
    if missing:
        pulls = "; ".join(f"ollama pull {name}" for name in missing)
        return False, f"Ollama: reachable, missing model(s) for the configured roles -- {pulls}"
    return True, f"Ollama: reachable at {base_url}, all {len(wanted)} configured model(s) present"


# The command the root README documents. Without --mmproj the server cannot
# embed images and answers them with HTTP 500 (issue #565).
_EMBEDDER_COMMAND = (
    "llama-server -m embeddinggemma-2-Q8_0.gguf --mmproj mmproj-embeddinggemma-2-Q8_0.gguf "
    "--embedding --port {port} -b 2048 -ub 2048"
)


def _embedder_failure(health_url: str, reason: str) -> str:
    """One line naming the failing endpoint and how to start llama-server."""
    port = urllib.parse.urlsplit(health_url).port or 8091
    return (
        f"Embedder: llama-server is not reachable at {health_url} ({reason}). "
        f"Start it with `{_EMBEDDER_COMMAND.format(port=port)}`."
    )


def _embedder_vision_problem(base_url: str) -> str:
    """Explain why the embedder cannot embed images, or return ``""`` when it can.

    A server started without ``--mmproj`` answers ``/health`` and text
    embeddings normally, then fails every figure with HTTP 500 at indexing
    time; ``/props`` reports it up front as ``modalities.vision`` false.
    """
    props_url = f"{base_url}/props"
    try:
        with urllib.request.urlopen(  # nosec B310 -- local endpoint from app config
            props_url, timeout=_EMBEDDER_HEALTH_TIMEOUT_SECONDS
        ) as response:
            props = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return _embedder_failure(props_url, str(exc))
    modalities = props.get("modalities") if isinstance(props, dict) else None
    if isinstance(modalities, dict) and modalities.get("vision") is True:
        return ""
    port = urllib.parse.urlsplit(base_url).port or 8091
    return (
        f"Embedder: llama-server at {base_url} reports no vision support in /props, so "
        "image embeddings (USAR_EMBEDDINGS_IMAGEN) fail with HTTP 500. Restart it with "
        f"--mmproj: `{_EMBEDDER_COMMAND.format(port=port)}`."
    )


def _check_embedder() -> Tuple[bool, str]:
    """Ask llama-server's ``/health`` whether the embedder is up (issue #385).

    The embedder is an external prerequisite, like Ollama: this script does not
    build it. It still has to be checked, because every indexing and retrieval
    run embeds through it, and "all components present" was a claim made
    without asking. The URL comes from ``AppConfig`` so this can never name a
    different server from the one the pipeline calls. When image embeddings are
    on, ``/props`` must also report vision support (issue #565).
    """
    config = AppConfig.from_env()
    base_url = config.models.embedding.base_url.rstrip("/")
    health_url = f"{base_url}/health"
    try:
        with urllib.request.urlopen(  # nosec B310 -- local endpoint from app config
            health_url, timeout=_EMBEDDER_HEALTH_TIMEOUT_SECONDS
        ) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        return False, _embedder_failure(health_url, f"HTTP {exc.code}")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return False, _embedder_failure(health_url, str(exc))
    if status != 200:
        return False, _embedder_failure(health_url, f"HTTP {status}")
    if config.flags.usar_embeddings_imagen:
        problem = _embedder_vision_problem(base_url)
        if problem:
            return False, problem
    return True, f"Embedder: llama-server reachable at {health_url}"


def run_checks() -> int:
    """Every precondition, each reported separately. Returns a process exit code."""
    # Labelled explicitly rather than read off __name__: half the entries are
    # lambdas, and a label guessed from a function object is the kind of
    # pointer that names something real and wrong -- a raising CUDA probe
    # reported as "interpreter check" sends the reader to rebuild a venv that
    # is fine.
    #
    # The last field marks a component this script does not build: a missing
    # embedder or Ollama model needs a command run by hand, not a rebuild, so
    # it warns instead of failing.
    checks = [
        (".venv", lambda: _check_interpreter(PRODUCT_VENV, ".venv"), False),
        (".venv-mineru", lambda: _check_interpreter(ISOLATED_VENV, ".venv-mineru"), False),
        ("CUDA", _check_cuda, False),
        ("MinerU", _check_mineru_binary, False),
        ("Embedder", _check_embedder, True),
        ("Ollama", _check_ollama, True),
    ]
    print("\nChecking:")
    failures = 0
    warnings = 0
    for label, check, warns_only in checks:
        try:
            passed, message = check()
        except subprocess.TimeoutExpired:
            passed, message = False, f"{label}: timed out"
        except Exception as exc:  # a diagnostic must outlive what it diagnoses
            # Every check shells out to something that may be half-installed.
            # One raising must report that check as failed, not abort the run
            # and leave the remaining components unreported -- the state this
            # script exists to describe is exactly the broken one.
            passed, message = False, f"{label}: raised {type(exc).__name__}: {exc}"
        if passed:
            marker = OK
        elif warns_only:
            marker = WARN
            warnings += 1
        else:
            marker = FAIL
            failures += 1
        print(f"[{marker}] {message}")

    if failures:
        print(f"\n{failures} component(s) missing. Run without --check to build them.")
        return 1
    if warnings:
        print(
            f"\n{warnings} warning(s) above: everything this script builds is present, "
            "but something it does not build needs attention. Teardown: rm -rf .venv .venv-mineru"
        )
        return 0
    print("\nAll components present. Teardown: rm -rf .venv .venv-mineru")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify without installing anything. Answers 'is this machine set up?'.",
    )
    args = parser.parse_args(argv)

    if not args.check:
        build_product_venv()
        build_isolated_venv()
    return run_checks()


if __name__ == "__main__":
    raise SystemExit(main())
