#!/usr/bin/env python3
"""setup_environments.py -- build the two interpreters MonkeyGrab needs, and check them.

Issue #120. ``README.md`` said "there is no setup script for this yet", and the
consequence was that the isolated environment's contents lived only in a
docstring: ``src/monkeygrab/adapters/embedding/jina_clip_worker.py`` names
torch 2.6.0+cu124, transformers 4.57.6 and sentence-transformers 5.6.1, and a
reader had to turn that into pip invocations, in the right order, from the
right index.

The order is the part that is not obvious and fails late. ``mineru[core]``
resolves its own dependency tree, so installing it AFTER the pins upgrades
transformers past 4.x, and the failure surfaces much later as jina-clip's
``Tensor.item() cannot be called on meta tensors`` at index time -- the exact
error ``rag/requirements.txt`` cites as the reason the isolated venv exists.

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
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCT_VENV = REPO_ROOT / ".venv"
ISOLATED_VENV = REPO_ROOT / ".venv-mineru"

# The pins jina_clip_worker.py's docstring names, and the index torch's CUDA
# build comes from. Not read from a requirements file on purpose: there is no
# requirements file for this environment, and adding one would put a second
# source of truth beside that docstring.
TORCH_PIN = "torch==2.6.0"
TORCHVISION_PIN = "torchvision==0.21.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
TRANSFORMERS_PIN = "transformers==4.57.6"
SENTENCE_TRANSFORMERS_PIN = "sentence-transformers==5.6.1"
# jina-clip-v2's remote code imports these directly; they are not pulled in by
# anything above.
JINA_RUNTIME_DEPS = ("einops", "timm", "pillow")

WORKER_SCRIPT = REPO_ROOT / "src" / "monkeygrab" / "adapters" / "embedding" / "jina_clip_worker.py"

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
    """``.venv-mineru`` -- MinerU and jina-clip, in the order that survives.

    torch first (from the CUDA index, so the CPU wheel is never chosen),
    ``mineru[core]`` second (it resolves a large tree of its own), and the
    transformers/sentence-transformers pins LAST, so MinerU's resolution
    cannot leave transformers on 5.x -- which loads fine and then fails at
    index time inside jina-clip's remote code.
    """
    if not _venv_python(ISOLATED_VENV).exists():
        _run([sys.executable, "-m", "venv", str(ISOLATED_VENV)], label="creating .venv-mineru")
    python = str(_venv_python(ISOLATED_VENV))
    _run([python, "-m", "pip", "install", "--upgrade", "pip"], label="upgrading pip")
    _run(
        [python, "-m", "pip", "install", TORCH_PIN, TORCHVISION_PIN, "--index-url", TORCH_INDEX],
        label="installing CUDA torch",
    )
    _run([python, "-m", "pip", "install", "mineru[core]"], label="installing MinerU")
    _run(
        [python, "-m", "pip", "install", TRANSFORMERS_PIN, SENTENCE_TRANSFORMERS_PIN,
         *JINA_RUNTIME_DEPS],
        label="pinning the jina-clip runtime",
    )


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
    """CUDA must be visible to the ISOLATED interpreter -- that is where it is needed.

    jina_clip_worker.py hard-fails without it (~100 s/document on CPU makes
    indexing a real corpus infeasible), so a CPU-only isolated venv is a setup
    failure, not a slower setup.
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
            return True, f"MinerU: {version}"
    return False, f"MinerU: no binary under {ISOLATED_VENV}"


def _check_jina_worker() -> Tuple[bool, str]:
    """Actually start the worker. Loading is what fails, not importing.

    The first run downloads the model (~30 s afterwards, several minutes the
    first time), which is why this has a generous timeout and says so.
    """
    python = _venv_python(ISOLATED_VENV)
    # Name which of the two is absent. "one of these is missing" sends the
    # reader to check both, and the wrong one first half the time.
    if not python.exists():
        return False, f"jina-clip: cannot check, no interpreter at {python}"
    if not WORKER_SCRIPT.exists():
        return False, f"jina-clip: cannot check, worker script missing at {WORKER_SCRIPT}"
    print("       (starting the jina-clip worker; the first run downloads the model)")
    result = subprocess.run(
        [str(python), str(WORKER_SCRIPT)],
        input=json.dumps({"id": 1, "op": "text", "text": "setup probe"}) + "\n",
        capture_output=True,
        text=True,
        timeout=1800,
    )
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        message = json.loads(line)
        if message.get("event") == "ready":
            return True, f"jina-clip: worker ready, dim {message['dim']}"
        if message.get("event") == "fatal":
            error = message["error"]
            # "Installed but the card is busy" is not "not installed", and
            # rebuilding the venv fixes nothing. A gate run or a resident
            # Ollama model holding the 8 GB is the ordinary cause.
            if "out of memory" in error.lower():
                return False, (
                    "jina-clip: installed, but the GPU has no free memory right now "
                    "(another run or a resident Ollama model is holding it) -- "
                    "BUSY, not missing. Free the card and re-check."
                )
            return False, f"jina-clip: {error[:200]}"
    return False, f"jina-clip: worker produced no ready event (exit {result.returncode})"


def _configured_models() -> List[str]:
    """The Ollama models the four roles resolve to, environment first."""
    return sorted({os.getenv(var, DEFAULT_OLLAMA_MODEL) for var in OLLAMA_ROLE_VARS.values()})


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


def run_checks() -> int:
    """Every precondition, each reported separately. Returns a process exit code."""
    # Labelled explicitly rather than read off __name__: half the entries are
    # lambdas, and a label guessed from a function object is the kind of
    # pointer that names something real and wrong -- a raising CUDA probe
    # reported as "interpreter check" sends the reader to rebuild a venv that
    # is fine.
    checks = [
        (".venv", lambda: _check_interpreter(PRODUCT_VENV, ".venv")),
        (".venv-mineru", lambda: _check_interpreter(ISOLATED_VENV, ".venv-mineru")),
        ("CUDA", _check_cuda),
        ("MinerU", _check_mineru_binary),
        ("jina-clip", _check_jina_worker),
        ("Ollama", _check_ollama),
    ]
    print("\nChecking:")
    failures = 0
    warnings = 0
    for label, check in checks:
        try:
            passed, message = check()
        except subprocess.TimeoutExpired:
            passed, message = False, f"{label}: timed out"
        except Exception as exc:  # noqa: BLE001 -- a diagnostic must outlive what it diagnoses
            # Every check shells out to something that may be half-installed.
            # One raising must report that check as failed, not abort the run
            # and leave the remaining components unreported -- the state this
            # script exists to describe is exactly the broken one.
            passed, message = False, f"{label}: raised {type(exc).__name__}: {exc}"
        # Two states are not setup failures and must not read as one:
        # Ollama is the component this script does not build (a missing model
        # is a pull away), and a busy GPU means installed-and-occupied, which
        # rebuilding does not fix.
        not_a_missing_component = "Ollama:" in message or "BUSY, not missing" in message
        if passed:
            marker = OK
        elif not_a_missing_component:
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
