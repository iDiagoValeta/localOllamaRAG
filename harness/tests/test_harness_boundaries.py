"""harness/tests/test_harness_boundaries.py -- issue #31 spec section 1.

Modelled on tests/unit/test_architecture_boundaries.py: parses each harness/
production module's AST (not a source-text grep, which a reformatted or
aliased import could dodge) rather than trusting the prohibitions below by
convention.

Prohibitions enforced, and what each check does and does not cover:

1. **No production module imports `grade`** -- neither `import grade` nor
   `from <anything> import grade`. Scoped to harness/ EXCLUDING harness/tests/,
   since tests legitimately import `run_eval`/`grade` to verify these very
   prohibitions and to mirror-check constants (see
   harness/tests/test_evaluator.py's drift check). This is a direct-import
   check: harness/evaluator.py imports `run_eval` (which itself imports
   `grade` for its own scoring), and that is allowed -- the harness may read
   and run the evaluation, never redefine how it scores (spec section 0).
   What this does NOT cover: reaching `grade`'s functions off an
   already-imported `run_eval` module object (e.g.
   `run_eval.grade.grade_answer(...)`) without ever importing `grade`
   itself. No production harness code does this; if one ever did, it still
   could not WRITE grade.py (check 2), which is the substantive prohibition.
2. **No production module writes** ``tests/eval/grade.py``,
   ``gold_cases.jsonl`` or ``baseline_min_pass_rate.txt``. Detected by
   walking every AST ``Call`` node, flagging "write-like" calls
   (``.write_text``/``.write_bytes``/``.write``/``.writelines``, or
   ``open()``/``Path.open()`` with a mode containing ``w``/``a``/``x``), then
   checking whether any string literal in that call's subtree (target path,
   arguments, f-string pieces) contains one of the three protected
   filenames. Catches a literal path written directly or built from an
   f-string/join with a literal filename segment -- the realistic accidental
   case. Does NOT catch a fully dynamic path assembled from opaque variables
   with no literal filename anywhere in the call -- undetectable by static
   analysis without false positives, and not how a harness module would
   plausibly reach these paths by accident.
3. **No module under harness/ (including harness/tests/) imports
   `subprocess` or the third-party `git` package (GitPython).** A full ban
   on both names -- not scoped to "a git invocation" specifically, because
   that is simpler to make non-flaky, and it is what makes
   ledger.read_git_commit's "no subprocess" docstring claim checked rather
   than asserted. Does NOT catch `os.system`/`os.popen`/`ctypes`-based
   shelling out; none of this harness's modules have any reason to use
   those, and banning the entire `os` module (needed for path handling)
   would be impractical.

Item 5 of the spec's prohibition list ("commit, merge, push, or deploy
anything") is covered entirely by check 3: with no subprocess and no
GitPython import anywhere under harness/, there is no code path left that
could invoke git at all, let alone a mutating one. Item 4 ("write any
product configuration") is not covered by this file: harness/ never writes
outside its own ``ledger/`` directory (see harness/ledger.py, the only
module here that writes anything), which this file does not need a separate
check to see -- there is no `settings.json`/`.env`/`rag/chat_pdfs.py` string
anywhere in the package to accidentally target.
"""

import ast
import sys
from pathlib import Path
from typing import List, Set

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

HARNESS_DIR = ROOT / "harness"

PROTECTED_FILENAMES = ("grade.py", "gold_cases.jsonl", "baseline_min_pass_rate.txt")

_WRITE_METHOD_NAMES = {"write_text", "write_bytes", "write", "writelines"}


def _production_python_files() -> List[Path]:
    """Every .py file directly under harness/, excluding harness/tests/."""
    return sorted(
        p for p in HARNESS_DIR.rglob("*.py")
        if "tests" not in p.relative_to(HARNESS_DIR).parts
    )


def _all_python_files() -> List[Path]:
    return sorted(HARNESS_DIR.rglob("*.py"))


def _imports_name(tree: ast.AST, banned_name: str) -> bool:
    """True if ``tree`` imports ``banned_name`` as a (sub)module, or imports
    a name called ``banned_name`` from anywhere."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == banned_name:
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and banned_name in node.module.split("."):
                return True
            for alias in node.names:
                if alias.name == banned_name:
                    return True
    return False


def _string_constants(node: ast.AST) -> Set[str]:
    return {sub.value for sub in ast.walk(node) if isinstance(sub, ast.Constant) and isinstance(sub.value, str)}


def _mode_contains_write(call: ast.Call) -> bool:
    """For an ``open(...)``-shaped call: True if a mode argument suggests writing."""
    mode_value = None
    if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant) and isinstance(call.args[1].value, str):
        mode_value = call.args[1].value
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            mode_value = kw.value.value
    if mode_value is None:
        return False  # open()'s default mode is "r" -- not a write
    return any(flag in mode_value for flag in ("w", "a", "x"))


def _writes_a_protected_path(tree: ast.AST) -> Set[str]:
    """Protected filenames found inside any write-like ``Call``'s subtree."""
    hits: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_write_call = (
            (isinstance(func, ast.Attribute) and func.attr in _WRITE_METHOD_NAMES)
            or (isinstance(func, ast.Attribute) and func.attr == "open" and _mode_contains_write(node))
            or (isinstance(func, ast.Name) and func.id == "open" and _mode_contains_write(node))
        )
        if not is_write_call:
            continue
        literals = _string_constants(node)
        for filename in PROTECTED_FILENAMES:
            if any(filename in literal for literal in literals):
                hits.add(filename)
    return hits


def test_harness_directory_is_not_accidentally_empty():
    """Guards the guard: an empty/misnamed directory would make every check below vacuous."""
    assert len(_production_python_files()) >= 5


@pytest.mark.parametrize("path", _production_python_files(), ids=lambda p: str(p.relative_to(HARNESS_DIR)))
def test_no_harness_module_imports_grade(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert not _imports_name(tree, "grade"), f"{path} imports grade -- forbidden (issue #31 spec section 1)"


@pytest.mark.parametrize("path", _production_python_files(), ids=lambda p: str(p.relative_to(HARNESS_DIR)))
def test_no_harness_module_writes_a_protected_path(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = _writes_a_protected_path(tree)
    assert not hits, f"{path} appears to write protected path(s) {hits} -- forbidden (issue #31 spec section 1)"


@pytest.mark.parametrize("path", _all_python_files(), ids=lambda p: str(p.relative_to(HARNESS_DIR)))
def test_no_harness_module_imports_subprocess_or_gitpython(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert not _imports_name(tree, "subprocess"), f"{path} imports subprocess -- forbidden (issue #31 spec section 1)"
    assert not _imports_name(tree, "git"), f"{path} imports git (GitPython) -- forbidden (issue #31 spec section 1)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
