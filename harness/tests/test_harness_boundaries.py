"""harness/tests/test_harness_boundaries.py -- issue #31 spec section 1.

Modelled on tests/unit/test_architecture_boundaries.py: parses each harness/
production module's AST (not a source-text grep, which a reformatted or
aliased import could dodge) rather than trusting the prohibitions below by
convention.

Prohibitions enforced, and what each check does and does not cover:

1. **No production module imports `grade` or `importlib`.** Neither
   `import grade` nor `from <anything> import grade`, in any dotted form
   (`import tests.eval.grade` is caught too -- an earlier version of this
   check only compared the top-level segment of a plain `Import` node and
   missed it, found in a #65 PR review). `importlib` is banned outright:
   its only use inside this package would be `importlib.import_module("grade")`,
   a dynamic import no static "did you import grade" check can see by
   construction. Scoped to harness/ EXCLUDING harness/tests/, since tests
   legitimately import `run_eval`/`grade` to verify these very prohibitions
   and to mirror-check constants (see harness/tests/test_evaluator.py's
   drift check). What this does NOT cover: reaching `grade`'s functions off
   an already-imported `run_eval` module object (e.g.
   `run_eval.grade.grade_answer(...)`) without ever importing `grade` or
   `importlib` itself. No production harness code does this; if one ever
   did, it still could not WRITE grade.py (check 2), which is the
   substantive prohibition.
2. **No production module references the literal `grade.py` or
   `baseline_min_pass_rate.txt` filename ANYWHERE**, not only inside a
   write-like call. No production module needs either string at all (unlike
   `gold_cases.jsonl`, which `evaluator.py` legitimately reads) -- the
   simplest ban that closes every write path a #65 PR review demonstrated
   in one adversarial module (`shutil.copyfile`, `os.replace` via a variable
   defined elsewhere, `write_text` on a path built from a folded `"grade" +
   ".py"` concatenation) without needing to enumerate every write-shaped API
   individually. String folding handles simple `"a" + "b"` concatenation
   (the exact obfuscation demonstrated); it does NOT handle f-string
   placeholders, `str.join`/`%`/`.format()`, or encoded/programmatically
   built strings (base64, `chr()`-built strings, reading a filename from
   another file at runtime) -- closing every possible obfuscation is not
   this test's job, and attempting to would risk false positives for little
   real protection.
3. **No production module writes `gold_cases.jsonl` specifically.**
   Unlike the two filenames above, this one must be allowed to appear as a
   literal (``evaluator.GOLD_FILE``), so it is checked only inside
   write-like calls: `.write_text`/`.write_bytes`/`.write`/`.writelines`,
   `open()`/`Path.open()` with a mode containing `w`/`a`/`x`, and
   `os.remove`/`os.unlink`/`os.replace`/`os.rename` /
   `shutil.copyfile`/`copy`/`copy2`/`move` (added after the same PR review
   demonstrated `os.remove`/`os.replace` bypassing the original, narrower
   write-method list). Does NOT catch a fully dynamic path assembled from
   opaque variables with no literal filename anywhere in the call --
   undetectable by static analysis without false positives.
4. **No module under harness/ (including harness/tests/) imports
   `subprocess` or the third-party `git` package (GitPython).** A full ban
   on both names -- not scoped to "a git invocation" specifically, because
   that is simpler to make non-flaky, and it is what makes
   ledger.read_git_commit's "no subprocess" docstring claim checked rather
   than asserted. Does NOT catch `os.system`/`os.popen`/`ctypes`-based
   shelling out; none of this harness's modules have any reason to use
   those, and banning the entire `os` module (needed for path handling)
   would be impractical.

Item 5 of the spec's prohibition list ("commit, merge, push, or deploy
anything") is covered entirely by check 4: with no subprocess and no
GitPython import anywhere under harness/, there is no code path left that
could invoke git at all, let alone a mutating one. Item 4 ("write any
product configuration") is not covered by this file: harness/ never writes
outside its own ``ledger/`` directory (see harness/ledger.py, the only
module here that writes anything), which this file does not need a separate
check to see -- there is no `settings.json`/`.env`/`rag/chat_pdfs.py` string
anywhere in the package to accidentally target.

``test_adversarial_bypass_module_is_caught_by_every_check`` is a regression
test for the exact six-bypass module a #65 PR review used to show the
original version of this file passed vacuously: it parses that module's
source directly (never written under harness/) and asserts every check
above flags at least one of its six attacks, so "these checks pass" means
something more than "today's real files happen to be clean."
"""

import ast
import sys
from pathlib import Path
from typing import List, Optional, Set

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

HARNESS_DIR = ROOT / "harness"

# Banned as a literal ANYWHERE in production code (check 2) -- no production
# module has a legitimate reason to reference either filename at all.
_LITERAL_BANNED_FILENAMES = ("grade.py", "baseline_min_pass_rate.txt")

# Allowed as a literal (evaluator.py reads it), banned only inside a
# write-like call (check 3).
_GOLD_CASES_FILENAME = "gold_cases.jsonl"

_WRITE_METHOD_NAMES = {"write_text", "write_bytes", "write", "writelines"}
_OS_WRITE_FUNCS = {"remove", "unlink", "replace", "rename"}
_SHUTIL_WRITE_FUNCS = {"copyfile", "copy", "copy2", "move"}


def _production_python_files() -> List[Path]:
    """Every .py file directly under harness/, excluding harness/tests/."""
    return sorted(
        p for p in HARNESS_DIR.rglob("*.py")
        if "tests" not in p.relative_to(HARNESS_DIR).parts
    )


def _all_python_files() -> List[Path]:
    return sorted(HARNESS_DIR.rglob("*.py"))


def _imports_name(tree: ast.AST, banned_name: str) -> bool:
    """True if ``tree`` imports ``banned_name`` as a (sub)module at any depth
    of a dotted path, or imports a name called ``banned_name`` from anywhere.

    Checks every segment of a plain ``import a.b.c`` (not just ``a``) -- a
    #65 PR review found ``import tests.eval.grade`` invisible to a version
    of this check that only looked at the first segment.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if banned_name in alias.name.split("."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and banned_name in node.module.split("."):
                return True
            for alias in node.names:
                if alias.name == banned_name:
                    return True
    return False


def _docstring_constant_ids(tree: ast.AST) -> Set[int]:
    """``id()`` of every ``Constant`` node that is a module/class/function docstring.

    A docstring is an inert string in that specific position -- nothing in
    this package (or plausibly any package) reads ``__doc__`` back out to
    build a filesystem path, so excluding it from the literal scan cannot
    hide a real reference. Needed because this test's own module docstring,
    and several production modules' docstrings, legitimately explain these
    prohibitions in prose (mentioning "grade.py"/"baseline_min_pass_rate.txt"
    as words, not as code references) -- without this exclusion the literal
    scan flags its own documentation.
    """
    ids: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def _fold_string_literals(tree: ast.AST, exclude_ids: Optional[Set[int]] = None) -> Set[str]:
    """Every (non-excluded) string literal under ``tree``, plus simple
    ``"a" + "b"`` concatenations folded together.

    Handles the exact obfuscation a #65 PR review used: splitting a
    protected filename across a ``+`` so no single ``Constant`` node equals
    or contains it (``"grade" + ".py"``). Does not attempt to fold
    f-strings, ``str.join``/``%``/``.format()``, or anything programmatically
    built -- see the module docstring for why that line is drawn here.

    Args:
        tree: Subtree to scan.
        exclude_ids: ``id()``s of ``Constant`` nodes to skip (docstrings --
            see ``_docstring_constant_ids``). Only meaningful for a
            whole-file scan; a write-call's own argument subtree never
            contains a docstring node, so callers scoped to one call omit
            this.
    """
    exclude_ids = exclude_ids or set()
    literals: Set[str] = set()

    def _folded(node: ast.AST):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = _folded(node.left), _folded(node.right)
            if left is not None and right is not None:
                return left + right
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in exclude_ids:
                continue
            literals.add(node.value)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            folded = _folded(node)
            if folded is not None:
                literals.add(folded)
    return literals


def _references_banned_literal_anywhere(tree: ast.AST) -> Set[str]:
    literals = _fold_string_literals(tree, exclude_ids=_docstring_constant_ids(tree))
    return {name for name in _LITERAL_BANNED_FILENAMES if any(name in lit for lit in literals)}


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


def _is_write_like_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute):
        if func.attr in _WRITE_METHOD_NAMES:
            return True
        if func.attr == "open" and _mode_contains_write(node):
            return True
        if isinstance(func.value, ast.Name):
            if func.value.id == "os" and func.attr in _OS_WRITE_FUNCS:
                return True
            if func.value.id == "shutil" and func.attr in _SHUTIL_WRITE_FUNCS:
                return True
        return False
    if isinstance(func, ast.Name):
        if func.id == "open" and _mode_contains_write(node):
            return True
        if func.id in _OS_WRITE_FUNCS:  # covers `from os import remove` then `remove(...)`
            return True
    return False


def _writes_gold_cases(tree: ast.AST) -> bool:
    """True if any write-like ``Call`` anywhere in ``tree`` references ``gold_cases.jsonl``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_write_like_call(node):
            if any(_GOLD_CASES_FILENAME in lit for lit in _fold_string_literals(node)):
                return True
    return False


def test_harness_directory_is_not_accidentally_empty():
    """Guards the guard: an empty/misnamed directory would make every check below vacuous."""
    assert len(_production_python_files()) >= 5


@pytest.mark.parametrize("path", _production_python_files(), ids=lambda p: str(p.relative_to(HARNESS_DIR)))
def test_no_harness_module_imports_grade_or_importlib(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert not _imports_name(tree, "grade"), f"{path} imports grade -- forbidden (issue #31 spec section 1)"
    assert not _imports_name(tree, "importlib"), f"{path} imports importlib -- forbidden (issue #31 spec section 1)"


@pytest.mark.parametrize("path", _production_python_files(), ids=lambda p: str(p.relative_to(HARNESS_DIR)))
def test_no_harness_module_references_grade_py_or_baseline_literal(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = _references_banned_literal_anywhere(tree)
    assert not hits, f"{path} references protected literal(s) {hits} -- forbidden (issue #31 spec section 1)"


@pytest.mark.parametrize("path", _production_python_files(), ids=lambda p: str(p.relative_to(HARNESS_DIR)))
def test_no_harness_module_writes_gold_cases(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert not _writes_gold_cases(tree), f"{path} appears to write gold_cases.jsonl -- forbidden (issue #31 spec section 1)"


@pytest.mark.parametrize("path", _all_python_files(), ids=lambda p: str(p.relative_to(HARNESS_DIR)))
def test_no_harness_module_imports_subprocess_or_gitpython(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert not _imports_name(tree, "subprocess"), f"{path} imports subprocess -- forbidden (issue #31 spec section 1)"
    assert not _imports_name(tree, "git"), f"{path} imports git (GitPython) -- forbidden (issue #31 spec section 1)"


# REGRESSION TEST -- proves the checks above actually catch something.

_ADVERSARIAL_MODULE_SOURCE = '''
import importlib
import tests.eval.grade
import shutil
import os
from pathlib import Path

EVAL = Path("tests/eval")
baseline_path = EVAL / "baseline_min_pass_rate.txt"


def attack():
    importlib.import_module("grade")
    shutil.copyfile("src.py", EVAL / "grade.py")
    os.remove(EVAL / "gold_cases.jsonl")
    os.replace("tmp", baseline_path)
    (EVAL / ("grade" + ".py")).write_text("evil")
'''


def test_adversarial_bypass_module_is_caught_by_every_check():
    """The exact six-bypass module a #65 PR review used to show the original
    boundary test passed vacuously. Every one of its attacks must be caught
    by at least one check, proving the fix rather than trusting it."""
    tree = ast.parse(_ADVERSARIAL_MODULE_SOURCE, filename="<adversarial>")

    assert _imports_name(tree, "grade"), "dotted `import tests.eval.grade` should be caught"
    assert _imports_name(tree, "importlib"), "`import importlib` should be caught"

    literal_hits = _references_banned_literal_anywhere(tree)
    assert "grade.py" in literal_hits, "grade.py (direct literal and folded '+') should be caught"
    assert "baseline_min_pass_rate.txt" in literal_hits, "baseline literal (via a variable) should be caught"

    assert _writes_gold_cases(tree), "os.remove(..., gold_cases.jsonl) should be caught"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
