"""Documentation drift check: ``.env.example`` against the environment
variables the code actually reads.

README.md claims ``.env.example`` "documents every supported variable with
its default" (issue #32). That claim went false twice in one day, caught
only by a human audit each time, in opposite directions:

- ``MINERU_BIN`` / ``MINERU_CACHE_DIR`` were read by the extraction adapter
  and undocumented (issue #41, fixed in #60).
- ``OLLAMA_HOST`` was read by every Ollama probe the web control panel
  makes and stayed undocumented even after the PR meant to fix that class
  of bug (caught in review of #60).
- ``RAG_MAX_IMAGES_PER_PAGE`` / ``RAG_MIN_IMAGE_SIZE_PX`` /
  ``RAG_CAPTION_MARGIN_PX`` were documented while no code read any of them
  (issue #63, fixed in #67).

Periodic hand-auditing does not hold against drift that runs both ways, so
this module checks both directions mechanically:

1. Every variable the code reads must appear in ``.env.example``.
2. Every variable ``.env.example`` documents must be read somewhere.

Detection only, like ``test_docs_drift.py`` next to this file: a failure
here means a human reads the diff and decides whether the code or the doc
is wrong, then fixes that one, per ``docs/README.md``'s "the code is the
source of truth" rule. This file does not rewrite ``.env.example`` or infer
a default value for anything.

## Detection: four ways this codebase reads an environment variable

A plain grep for ``getenv`` misses three of the four mechanisms below,
which is exactly how the ``OLLAMA_HOST`` drift survived one full PR meant
to fix it:

1. ``os.getenv(name)`` / ``os.environ.get(name)`` / ``os.environ[name]``
   (a *read* subscript, not an assignment) / ``os.environ.pop(name, ...)``.
2. ``monkeygrab.config.env``'s typed readers -- ``read_env_str``,
   ``read_env_int``, ``read_env_float``, ``read_env_choice`` -- plus
   ``rag/chat_pdfs.py``'s own pre-``monkeygrab`` equivalents,
   ``_leer_env_int`` / ``_leer_env_float``. All six take the variable name
   as their first positional argument, so a literal string there is exactly
   as much a "read" as ``os.getenv("NAME")``.
3. ``env.setdefault(name, ...)`` on a dict assigned from
   ``os.environ.copy()`` -- how ``MINERU_MODEL_SOURCE`` reaches the MinerU
   subprocess (``src/monkeygrab/adapters/extraction/mineru_extractor.py``):
   the code only supplies a default when the inherited environment doesn't
   already have one, so a value set in ``.env`` silently wins. A literal
   grep for ``getenv`` does not see this mechanism at all.
4. The frontend: ``process.env.NAME`` / ``import.meta.env.NAME`` under
   ``rag/web/frontend/**/*.{ts,tsx}``.

Python detection walks the AST (``ast.parse`` + ``ast.walk``) rather than
grepping source text, for the same reason ``test_architecture_boundaries.py``
parses imports instead of pattern-matching them: a reformatted or
differently-spaced call site is invisible to a regex but not to the parser.
The frontend has no stdlib parser available and pulling in a JS/TS toolchain
would break this file's "stdlib + pytest only" requirement (see below), so
that half is a narrow regex over the two access forms actually used here.

## Scope boundary

In scope: every ``.py`` file in the repository (skipping VCS/build/vendor
directories that hold no first-party code -- see ``_EXCLUDED_DIR_NAMES``)
and every ``.ts``/``.tsx`` file under ``rag/web/frontend/`` (skipping
``node_modules``/``dist``). This deliberately includes ``tests/`` and
``tools/diagnostics/`` rather than narrowing the scan to "product code" and
hoping nothing env-reading lives outside it -- both already contain a real
variable, handled below by name rather than by carving out their directory.

Out of scope, each named explicitly in ``_CODE_READS_NOT_DOCUMENTED`` below
with its own reason -- never by skipping a directory or file wholesale:

- ``RUN_OLLAMA_INTEGRATION`` -- gates one opt-in integration test
  (``tests/test_nothink.py``); not a product configuration knob.
- ``OLLAMA_GEMMA4_TEST_MODEL`` -- overrides the model used by a manual
  diagnostics script (``tools/diagnostics/ollama_aux_nothink.py``); never
  read by the shipped product.
- ``DISABLE_HMR`` -- a Vite dev-server-only flag
  (``rag/web/frontend/vite.config.ts``); meaningless outside ``pnpm run dev``.
- ``LOCALAPPDATA``, ``XDG_DATA_HOME`` -- platform-provided variables
  ``rag/web/desktop.py`` reads as an OS install-location fallback; the OS
  sets these, not the user via ``.env``, and MonkeyGrab defines no default
  for either.

Nothing today is "documented but legitimately never read" -- that is what
``dead`` in ``test_every_documented_env_var_is_read_by_code`` catches, and
as of this writing it is empty. If a genuine case shows up later (a flag
staged ahead of the code that reads it, say), give it the same treatment:
a named entry with a reason, not a silently narrowed scan.

Dependencies: stdlib only (``ast``, ``re``, ``pathlib``) plus ``pytest`` --
this file runs in the fast CI gate's ``architecture`` job, which installs
nothing else (see ``tests/conftest.py``).
"""

import ast
import re
from pathlib import Path
from typing import Optional, Set

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = ROOT / ".env.example"
FRONTEND_ROOT = ROOT / "rag" / "web" / "frontend"

# Directories that hold no first-party code: skipping them keeps a full dev
# checkout (node_modules/ after `pnpm install`, a built dist/, caches) from
# ever being asked "does this variable belong in .env.example".
_EXCLUDED_DIR_NAMES = {
    ".git",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".venv",
    ".venv-mineru",
    ".pytest_cache",
    ".ruff_cache",
}

# See the module docstring's "Scope boundary" section for why each of these
# is read by the code but does not belong in .env.example.
_CODE_READS_NOT_DOCUMENTED = {
    "RUN_OLLAMA_INTEGRATION",
    "OLLAMA_GEMMA4_TEST_MODEL",
    "DISABLE_HMR",
    "LOCALAPPDATA",
    "XDG_DATA_HOME",
    "PYTORCH_CUDA_ALLOC_CONF",
}

# monkeygrab.config.env's typed readers, plus rag/chat_pdfs.py's own
# pre-monkeygrab equivalents: every one of these takes the env var name as
# its first positional argument, exactly like os.getenv(name).
_LITERAL_ARG_READER_NAMES = {
    "getenv",
    "read_env_str",
    "read_env_int",
    "read_env_float",
    "read_env_choice",
    "_leer_env_int",
    "_leer_env_float",
}

# Dict methods checked only when the receiver is os.environ itself (or a
# dict copied from it, see _environ_copy_targets) -- these same method names
# are common on plain dicts throughout this codebase (buckets.setdefault,
# kwargs.setdefault, ...) and would be false positives otherwise.
_ENVIRON_METHODS_WITH_NAME_ARG = {"get", "pop", "setdefault"}


def _is_excluded(path: Path) -> bool:
    return any(part in _EXCLUDED_DIR_NAMES for part in path.parts)


def _literal_str(node: Optional[ast.expr]) -> Optional[str]:
    """The string value of ``node`` if it is a literal, else None.

    A non-literal argument (a variable, an f-string, ...) means the variable
    name can't be determined statically -- that call site is skipped rather
    than guessed at. Every call site this codebase actually has passes a
    literal here; a dynamic name would be a new pattern this check does not
    yet cover, not a bug in the check itself.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_os_environ(node: ast.expr) -> bool:
    """True for the attribute access ``os.environ`` specifically."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _call_leaf_name(func: ast.expr) -> Optional[str]:
    """The bare or attribute name a call targets: ``getenv`` for both
    ``getenv(...)`` and ``os.getenv(...)``, ``read_env_int`` for both a bare
    call and ``env.read_env_int(...)``."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _environ_copy_targets(tree: ast.Module) -> Set[str]:
    """Names assigned directly from ``os.environ.copy()`` in this module.

    This is the mechanism behind ``MINERU_MODEL_SOURCE``: a subprocess is
    handed a copy of the environment, and ``.setdefault`` on that copy only
    fills in a value the user hasn't already supplied through the real
    environment -- so a name later passed to ``.setdefault`` on one of these
    targets counts as a read exactly like ``os.environ.setdefault`` would.
    """
    targets: Set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "copy"
            and _is_os_environ(node.value.func.value)
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    targets.add(target.id)
    return targets


def _python_env_reads(py_file: Path) -> Set[str]:
    """Every environment variable name ``py_file`` reads, by any of the
    mechanisms described in the module docstring's "Detection" section."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    environ_copies = _environ_copy_targets(tree)
    names: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            leaf = _call_leaf_name(node.func)
            if leaf in _LITERAL_ARG_READER_NAMES:
                if node.args:
                    value = _literal_str(node.args[0])
                    if value:
                        names.add(value)
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _ENVIRON_METHODS_WITH_NAME_ARG
                and node.args
            ):
                base = node.func.value
                reads_environ = _is_os_environ(base) or (
                    isinstance(base, ast.Name) and base.id in environ_copies
                )
                if reads_environ:
                    value = _literal_str(node.args[0])
                    if value:
                        names.add(value)
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            if _is_os_environ(node.value):
                value = _literal_str(node.slice)
                if value:
                    names.add(value)

    return names


_FRONTEND_ENV_PATTERN = re.compile(r"(?:process\.env|import\.meta\.env)\.([A-Za-z_][A-Za-z0-9_]*)")


def _frontend_env_reads(ts_file: Path) -> Set[str]:
    return set(_FRONTEND_ENV_PATTERN.findall(ts_file.read_text(encoding="utf-8")))


def _all_code_reads() -> Set[str]:
    names: Set[str] = set()
    for py_file in ROOT.rglob("*.py"):
        if _is_excluded(py_file):
            continue
        names.update(_python_env_reads(py_file))
    if FRONTEND_ROOT.is_dir():
        for pattern in ("*.ts", "*.tsx"):
            for ts_file in FRONTEND_ROOT.rglob(pattern):
                if _is_excluded(ts_file):
                    continue
                names.update(_frontend_env_reads(ts_file))
    return names


_DOCUMENTED_PATTERN = re.compile(r"^#([A-Za-z_][A-Za-z0-9_]*)=", re.MULTILINE)


def _documented_names() -> Set[str]:
    """Variable names from .env.example's commented ``#NAME=value`` lines.

    Every real entry is commented out (the file documents defaults, it does
    not set them), and the section-divider comments never have an ``=``
    right after an identifier, so this one pattern separates the two
    without needing to special-case the banner lines.
    """
    if not ENV_EXAMPLE.is_file():
        pytest.fail(f"{ENV_EXAMPLE} is missing -- nothing to check drift against.")
    return set(_DOCUMENTED_PATTERN.findall(ENV_EXAMPLE.read_text(encoding="utf-8")))


# Floors guard the guard: if either parser regressed to returning an empty
# (or near-empty) set, these turn that into a loud failure instead of two
# vacuously-passing tests. Set well below the current counts (46 documented,
# 45 distinct names read as of this writing) so ordinary additions or
# removals of a handful of variables never trip them.
_MIN_DOCUMENTED = 20
_MIN_CODE_READ = 20


def test_every_env_var_read_by_code_is_documented():
    """Direction 1: a variable the code reads must appear in .env.example,
    unless it's named in _CODE_READS_NOT_DOCUMENTED (see the module
    docstring's "Scope boundary" for why each entry there is excluded)."""
    code_read = _all_code_reads()
    assert len(code_read) >= _MIN_CODE_READ, (
        f"only {len(code_read)} env var read(s) detected across the repo -- "
        "the AST/regex scan likely regressed; see this file's docstring."
    )
    documented = _documented_names()

    missing = sorted(code_read - documented - _CODE_READS_NOT_DOCUMENTED)
    assert not missing, (
        "These variables are read by the code but missing from .env.example "
        f"(code -> doc drift): {missing}. Either add them to .env.example "
        "with their default, or -- if they're deliberately internal -- add "
        "them to _CODE_READS_NOT_DOCUMENTED in this file with a reason."
    )


def test_every_documented_env_var_is_read_by_code():
    """Direction 2: a variable .env.example documents must be read
    somewhere, or it is dead documentation -- exactly what happened with the
    three image-extraction knobs removed in #67."""
    documented = _documented_names()
    assert len(documented) >= _MIN_DOCUMENTED, (
        f"only {len(documented)} documented var(s) parsed from .env.example "
        "-- the regex likely regressed; see this file's docstring."
    )
    code_read = _all_code_reads()

    dead = sorted(documented - code_read)
    assert not dead, (
        "These variables are documented in .env.example but nothing in the "
        f"code reads them (doc -> code drift): {dead}. Either wire them up, "
        "or remove them from .env.example if they're no longer supported."
    )
