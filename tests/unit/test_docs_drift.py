"""Documentation drift check: enumerations in .claude/CLAUDE.md against the
code that actually defines them.

docs/README.md states the rule -- "if a document and the code disagree, the
code is right" -- but nothing enforced it (GitHub issue #32), and drift kept
being caught by human review instead of CI. This module closes that gap for
the subset of claims that are mechanically checkable: documented
*enumerations* with a single, stable source of truth in the code.

Detection only. This test does not generate or rewrite prose -- when a check
fails, the fix is a human reading the diff and deciding whether the code or
the doc is wrong, per docs/README.md's rule, and writing that sentence
themselves.

Two checks:

1. ``.claude/CLAUDE.md`` rule 7's "Pipeline entry points" and "Runtime
   switches" bullets against the module-level names ``rag/chat_pdfs.py``
   actually binds. Rule 7 calls the file's re-export block "the authoritative
   list" and warns that renaming a symbol there breaks the web app, the CLI
   and the tests at once.
2. ``.claude/CLAUDE.md`` section 9's CLI slash-command line against
   ``rag/cli/commands.py``, which its own docstring calls the "single source
   of truth for slash-commands". Checked both ways: a documented command
   that no longer exists, and a registered command nobody documented.

Both source files are parsed with ``ast`` rather than imported: importing
``rag.chat_pdfs`` or ``rag.cli.commands`` (the latter transitively, through
``rag/cli/__init__.py``) pulls in third-party packages (``dotenv``, ``rich``,
...) that the fast CI gate's ``architecture`` job -- which runs this file --
does not install.
"""

import ast
import re
from pathlib import Path
from typing import Set

import pytest

ROOT = Path(__file__).resolve().parents[2]

CLAUDE_MD = ROOT / ".claude" / "CLAUDE.md"
CHAT_PDFS = ROOT / "rag" / "chat_pdfs.py"
CLI_COMMANDS = ROOT / "rag" / "cli" / "commands.py"


def _assign_target_names(target: ast.expr) -> Set[str]:
    """Names bound by one assignment target, recursing into tuple/list unpacking."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: Set[str] = set()
        for elt in target.elts:
            names.update(_assign_target_names(elt))
        return names
    return set()


def _collect_names_from_body(body: list, names: Set[str]) -> None:
    """Walk one statement list and add every name it binds, recursing into
    try/except so a guarded import isn't invisible to the walk.

    rag/chat_pdfs.py already wraps one import in ``try/except ImportError``
    -- the ``dotenv`` import, so an absent optional dependency doesn't crash
    the module. A re-export using the same guard is a real possibility, not
    a hypothetical -- and a walk of ``tree.body`` alone would drop it, making
    this check report a symbol as unbound when it is simply guarded.
    """
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_assign_target_names(target))
        elif isinstance(node, ast.AnnAssign):
            names.update(_assign_target_names(node.target))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Try):
            _collect_names_from_body(node.body, names)
            for handler in node.handlers:
                _collect_names_from_body(handler.body, names)
            _collect_names_from_body(node.orelse, names)
            _collect_names_from_body(node.finalbody, names)


def _module_level_names(py_file: Path) -> Set[str]:
    """Every name reachable as ``module.<name>``: defs, classes, plain and
    annotated assignments (including tuple unpacking), and imported symbols
    -- including ones bound inside a top-level try/except.

    Imported symbols count because rag/chat_pdfs.py's public API is a
    re-export block (``from rag.engine.generation import generar_respuesta``,
    etc.) -- a name bound by import is exactly as reachable from a caller
    doing ``import rag.chat_pdfs as rag_engine`` as one defined locally.
    """
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    names: Set[str] = set()
    _collect_names_from_body(tree.body, names)
    return names


# Symbols documented today: 7 in "Pipeline entry points", 7 in "Runtime
# switches" (count the backtick-quoted names in .claude/CLAUDE.md rule 7).
# The floor is set below either total, so trimming a symbol or two from a
# bullet doesn't trip it -- but if the capture regex below ever narrows (for
# example because a reflowed bullet's continuation lines fall outside it),
# the parsed count collapses toward zero, and this floor turns that into a
# loud assertion failure instead of a quietly-passing, under-checked test.
_RULE7_MIN_SYMBOLS_PER_BULLET = 5


def _rule7_documented_symbols() -> Set[str]:
    """Backtick-quoted names from rule 7's "Pipeline entry points" and
    "Runtime switches" bullets.

    Deliberately narrow: only these two bullets are matched, by their fixed
    bold lead-in text. Each bullet is captured from right after its label to
    the next bullet marker or a blank line, not just the rest of the first
    physical line -- a bullet that reflows across multiple lines (this one is
    232 characters in its current single-line form) is still read in full. A
    change to unrelated prose elsewhere in rule 7 cannot silently widen or
    narrow what gets checked.
    """
    text = CLAUDE_MD.read_text(encoding="utf-8")
    symbols: Set[str] = set()
    for label in ("Pipeline entry points:", "Runtime switches (web control panel):"):
        match = re.search(
            rf"\*\*{re.escape(label)}\*\*(.*?)(?=\n   - |\n\s*\n|\Z)",
            text,
            re.DOTALL,
        )
        if not match:
            pytest.fail(
                f"CLAUDE.md rule 7 no longer has a '{label}' bullet in the "
                "expected '**label:** ...' form -- update this check's parsing."
            )
        found = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", match.group(1)))
        assert len(found) >= _RULE7_MIN_SYMBOLS_PER_BULLET, (
            f"CLAUDE.md rule 7's '{label}' bullet now parses to only "
            f"{len(found)} symbol(s) ({sorted(found)}) -- either the bullet "
            "legitimately shrank (lower this floor to match) or the capture "
            "regex silently narrowed (fix this check's parsing)."
        )
        symbols.update(found)
    return symbols


def _documented_slash_commands() -> Set[str]:
    """Backtick-quoted slash tokens from section 9's CLI command summary.

    Captures the whole block after the heading, up to the next blank line or
    heading, rather than just the first physical line -- that line is 152
    characters today, so a reflow across two lines is a realistic edit, and
    a single-line capture would report the wrapped commands as "registered
    but undocumented" while they sit one line below where the test just read.
    """
    text = CLAUDE_MD.read_text(encoding="utf-8")
    match = re.search(
        r"### CLI slash commands\s*\n\n(.*?)(?=\n\s*\n|\n#|\Z)",
        text,
        re.DOTALL,
    )
    if not match:
        pytest.fail(
            "CLAUDE.md section 9 no longer has a 'CLI slash commands' line "
            "in the expected place -- update this check's parsing."
        )
    # [\w-]+, not [A-Za-z]+: a command token may contain digits, hyphens or
    # underscores, none of which the stricter class would match.
    return set(re.findall(r"`(/[\w-]+)`", match.group(1)))


def _cli_registered_commands() -> Set[str]:
    """Every slash token rag/cli/commands.py declares: COMMANDS (primary
    tokens) plus ALIASES (aliases across the three UI languages).

    commands.py's own docstring calls it the "single source of truth for
    slash-commands", and rag/cli/app.py validates its dispatcher against
    exactly these two lists when the CLI starts
    (``_validate_commands_registry``). So checking commands.py here is
    equivalent to checking what the CLI actually registers, without needing
    to parse app.py's dispatch dict directly.
    """
    tree = ast.parse(CLI_COMMANDS.read_text(encoding="utf-8"), filename=str(CLI_COMMANDS))
    declared: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in ("COMMANDS", "ALIASES"):
                try:
                    declared[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pytest.fail(
                        f"rag/cli/commands.py's {target.id} is no longer a "
                        "literal ast.literal_eval can parse -- update this "
                        "check's parsing."
                    )

    missing = {"COMMANDS", "ALIASES"} - declared.keys()
    if missing:
        pytest.fail(
            f"rag/cli/commands.py no longer defines {sorted(missing)} at "
            "module level -- update this check's parsing."
        )

    # token, *_ (not token, _): tolerant of extra fields per entry, matching
    # ALIASES below -- a plain 2-tuple unpack raises on any entry that later
    # grows a third field, which is exactly the kind of change this file
    # should survive without needing an edit.
    tokens = {token for token, *_ in declared["COMMANDS"]}
    tokens.update(alias for alias, *_ in declared["ALIASES"])
    return tokens


def test_rule7_pipeline_entry_points_and_runtime_switches_are_exported():
    """Every symbol rule 7 promises under those two bullets must actually be
    bound at rag/chat_pdfs.py's module level -- otherwise the doc promises a
    public API the code no longer has."""
    documented = _rule7_documented_symbols()
    exported = _module_level_names(CHAT_PDFS)
    missing = sorted(documented - exported)
    assert not missing, (
        "CLAUDE.md rule 7 lists these symbols as part of rag/chat_pdfs.py's "
        f"public API, but the module no longer binds them: {missing}. Fix "
        "whichever is wrong -- the code, if it dropped/renamed something "
        "callers rely on, or the doc, per docs/README.md's rule that the "
        "code is right."
    )


def test_cli_slash_commands_match_documentation():
    """CLAUDE.md's slash-command list and rag/cli/commands.py must name the
    same set of tokens, in both directions."""
    documented = _documented_slash_commands()
    registered = _cli_registered_commands()
    missing_from_code = sorted(documented - registered)
    missing_from_docs = sorted(registered - documented)
    assert not missing_from_code and not missing_from_docs, (
        "CLAUDE.md section 9's CLI slash-command list and "
        "rag/cli/commands.py disagree. "
        f"Documented but no longer registered: {missing_from_code}. "
        f"Registered but undocumented: {missing_from_docs}."
    )
