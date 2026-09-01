"""No statement in this repository sits after a `return`, `raise`, `break` or
`continue` in the same block.

Issue #111. ``harness/loop.py::_build_entry`` carried two consecutive
``return ledger_mod.LedgerEntry(...)`` statements for months. PR #94 (issue
#92, recovery mode) added the new construction above the one it replaced and
did not delete the old one, so 26 lines could never execute.

No behaviour was wrong -- the first return is the live, correct one. What
made it worth a check rather than a shrug is that **the two copies had
diverged, and in the direction that matters**: the dead one did not pass
``regression_baseline_iteration``, the field whose entire purpose is making
it auditable which baseline an entry was paired against. A reader saw two
plausible constructions of the same object; any edit removing or reordering
the first would have silently dropped recovery mode's provenance. Redundant
is noise. Redundant *and* subtly wrong is a trap.

Nothing in the repo would have caught it. ``ruff.toml`` enables no
unreachable-code rule, and the failure is invisible to a reader who does not
happen to scroll past the first ``return`` of a 30-line constructor call --
which is exactly the shape a merge resolution leaves behind.

## Scope

Every ``.py`` file in the repository except the virtualenvs: the defect is a
merge-resolution artifact, and merges happen in ``src/``, ``rag/``, ``tests/``
and ``harness/`` alike. It was zero everywhere else when this test was
written (2026-09-01), so the check costs no triage -- it starts green and
stays that way or something real happened.

Standard library only, so this runs in the fast gate's ``architecture`` job.
``test_the_check_catches_a_module_that_has_unreachable_code`` parses a
deliberately broken source string, so "this passes" means the detector works,
not merely that today's files happen to be clean -- the same guard-the-guard
pattern as ``harness/tests/test_harness_boundaries.py``'s adversarial module.
"""

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Directories that are installed, generated or vendored -- not this repo's code.
_SKIP_DIRS = frozenset(
    {".venv", ".venv-mineru", "venv", "node_modules", "__pycache__", ".git", "build", "dist"}
)

# Statements after which nothing in the same block can run. `sys.exit()` and
# `os._exit()` are deliberately absent: they are ordinary calls as far as the
# parser is concerned, and treating them as terminal would need a name
# resolution this check has no business doing.
_TERMINAL = (ast.Return, ast.Raise, ast.Continue, ast.Break)

# Every attribute of an AST node that holds a list of statements. A dead
# statement can hide in an `else:` or a `finally:` as easily as in a body.
_BLOCK_FIELDS = ("body", "orelse", "finalbody")


def _unreachable_statements(tree: ast.AST):
    """``(terminal_line, dead_line, dead_node_type)`` for each dead statement.

    One report per block: the first statement after the terminal is enough to
    locate the problem, and listing the remaining twenty-five would bury it.
    """
    found = []
    for node in ast.walk(tree):
        for field in _BLOCK_FIELDS:
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for index, statement in enumerate(block[:-1]):
                if isinstance(statement, _TERMINAL):
                    dead = block[index + 1]
                    found.append((statement.lineno, dead.lineno, type(dead).__name__))
                    break
    return found


def _repository_python_files():
    return sorted(
        path
        for path in _REPO_ROOT.rglob("*.py")
        if not _SKIP_DIRS.intersection(path.relative_to(_REPO_ROOT).parts)
    )


def test_the_file_list_is_not_accidentally_empty():
    """Guards the guard: a bad skip list would make the check below vacuous."""
    files = _repository_python_files()
    assert len(files) > 100, f"only {len(files)} python files found; the skip list is too broad"


@pytest.mark.parametrize(
    "path",
    _repository_python_files(),
    ids=lambda p: str(p.relative_to(_REPO_ROOT)),
)
def test_module_has_no_unreachable_statements(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    dead = _unreachable_statements(tree)
    assert not dead, (
        f"{path.relative_to(_REPO_ROOT)} has unreachable code: "
        + "; ".join(
            f"line {dead_line} ({kind}) cannot run, the block returns/raises at line {terminal}"
            for terminal, dead_line, kind in dead
        )
        + ". A merge resolution usually leaves this behind -- check whether the "
        "live copy and the dead one have drifted apart before deleting either."
    )


_MODULE_WITH_DEAD_CODE = '''
def build(value):
    if value is None:
        value = 0
    return {"live": value, "provenance": "kept"}
    return {"live": value}


def loop(items):
    for item in items:
        continue
        print(item)
'''


def test_the_check_catches_a_module_that_has_unreachable_code():
    """The #111 shape, plus a dead statement after `continue` in a loop body."""
    dead = _unreachable_statements(ast.parse(_MODULE_WITH_DEAD_CODE))
    assert len(dead) == 2, dead
    assert [kind for _, _, kind in dead] == ["Return", "Expr"]


def test_the_check_does_not_flag_a_terminal_that_ends_its_block():
    """An `if`/`else` where both arms return is the normal shape, not a defect."""
    source = '''
def classify(value):
    if value > 0:
        return "positive"
    else:
        return "not positive"
'''
    assert _unreachable_statements(ast.parse(source)) == []
