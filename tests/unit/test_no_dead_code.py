"""No product symbol or module in this repository is reachable only from tests.

Issue #526. Ruff's ``F`` rules catch unused imports and unused locals; they say
nothing about a function, a class, a method or a whole module that only the
test suite still imports. That gap is where product code goes to die: it keeps
passing its own tests, keeps aging, and stops describing anything the shipped
pipeline does. ``harness/slow_tier.py``, ``harness.evaluator.blind_set_case_ids``
and ``QuizQuestion.correct_option`` were three instances of it found by hand.

The rule enforced here is the one a reader would apply: a product symbol is
dead when no consumer names it as code. A reference is a ``NAME`` token from
``tokenize`` outside any ``def``/``class`` line that declares that same name,
in any consumer file. Comments, docstrings, f-strings and every other string
are discussion of the symbol, not a use of it, and neither is a second
``def`` that happens to share the name. ``test_*.py`` and ``conftest.py`` are
not consumers because a test that imports a symbol proves only that the test
imports it.

## What counts as product, and who consumes it

``*.py`` under ``src/``, ``rag/``, ``harness/``, ``tools/`` and ``packaging/``,
minus any file under a ``tests/`` directory and any ``test_*.py`` /
``conftest.py``. Consumers are every ``*.py`` in the repository that is not one
of those test files, wherever it lives: product modules calling each other,
``tools/`` scripts, ``tests/eval``'s gate (a real consumer of the product, by
design) and any helper someone adds beside the tests. Virtualenvs,
``node_modules``, ``.git`` and build output are not code.

## Blind spots

The check counts code names; it does not resolve them. These stay unreported by
construction, and shrinking the list is a separate change:

- Module constants, class attributes, ``Enum`` members and dataclass fields:
  only ``def``/``class`` declarations are candidates, so a dead constant or a
  dead field is invisible.
- Nested functions, nested classes and the methods of a nested class: only
  ``tree.body`` and the methods directly inside a top-level class are walked.
- Classes exempted by a registrar decorator. Any attribute decorator counts
  (see "Exemptions"), so ``@dataclasses.dataclass(...)`` exempts the class and
  a whole dead dataclass would go unreported; its methods are still checked.
- Code under a module-level ``if``/``try`` is never declared as a symbol.
- A name reached only through a string, e.g. ``getattr(x, "is_unusable")``:
  the string is not a reference, so the attribute must be in ``_WHITELIST``
  with the reason it is looked up dynamically.
- A symbol mentioned only in prose (README, ``docs/``, a comment) is dead
  here even if a person would call that documentation a consumer. Entry points
  are the exception and are detected separately below.

## Exemptions, each with its reason

- Dunder methods (``__init__``, ``__str__``, ...) are the Python data model
  calling them, not a reader.
- A function decorated by a registrar (``@app.route(...)``,
  ``@bp.errorhandler(...)``, ``@app.before_request``) is referenced by a
  framework through the decorator, so no name lookup will ever show it.
  Attribute decorators count whether or not they are called: Flask registers
  both shapes. A bare ``@property`` is not attribute access and not a
  registrar, and ``QuizQuestion.correct_option`` was exactly a dead
  ``@property``.
- The names re-exported by ``rag/chat_pdfs.py`` are its public API (AGENTS.md
  section 1 rule 7); the facade exists so callers can reach them from one
  place, and a grep for the name in the product will not find them because
  that is the point of a facade.
- ``_WHITELIST`` below is deliberately tiny and every entry says why. A symbol
  whitelisted "just in case" is how the check stops meaning anything, so the
  staleness test fails when an entry stops corresponding to a real finding.

Modules get the same treatment: a product module no consumer imports is dead
unless it is an entry point (``__init__.py``, ``__main__.py``, a
``if __name__ == "__main__"`` block, or a path named by a workflow, a
PyInstaller spec or the README).

## Standard library only

This runs in the fast gate's ``architecture`` job, which installs pytest and
nothing else (``tests/conftest.py``). ``ast``, ``io`` and ``tokenize`` are
enough. The detector is exercised against synthetic sources by the ``guard the
guard`` tests at the bottom, so a green run means the check works, not merely
that today's tree happens to be clean.
"""

import ast
import io
import os
import sys
import tokenize
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pytest

# Before 3.12 tokenize returns a whole f-string as one STRING token, so names
# used inside {...} would read as unreferenced and live code would fail here.
# CI runs 3.13; an older local interpreter skips rather than reports noise.
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="needs PEP 701 f-string tokens (Python 3.12+)"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Directories that are installed, generated or vendored -- not this repo's code.
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        ".venv-mineru",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
    }
)

_PRODUCT_ROOTS = ("src", "rag", "harness", "tools", "packaging")

_DEFINITION_KEYWORDS = frozenset({"def", "class"})

# ``path:symbol`` -> why the symbol must stay despite having no caller.
_WHITELIST: Dict[str, str] = {
    "rag/headless/health.py:_reset_probe_cache": (
        "Test hook: clears the module-level probe cache between health probes; "
        "the runtime only ever reads it through cached_probe."
    ),
    "rag/web/desktop.py:readline": (
        "Part of the file-like protocol _NullStream stands in for when a "
        "windowed build has no std streams; the runtime calls it, not our code."
    ),
    "rag/web/desktop.py:isatty": (
        "Same protocol as readline above: logging and the display backend call "
        "isatty() on sys.stdout, which _NullStream replaces when there is no "
        "console, so nothing in the repository names it."
    ),
    "src/monkeygrab/adapters/embedding/llama_server_embedder.py:is_unusable": (
        "Optional capability read by name, not by import: wiring.py asks "
        'getattr(instance, "is_unusable", False) so embedders without an '
        "opinion on liveness need only satisfy the port structurally."
    ),
    "rag/chat_pdfs.py:_restaurar_roles_y_flags_por_defecto": (
        "Test hook: tests/conftest.py restores the facade's import-time model "
        "roles and flags after a test module imported rag.web.app (issue #231). "
        "It lives in the facade because the globals it resets live there."
    ),
}


def _is_test_file(path: Path) -> bool:
    return path.name == "conftest.py" or path.name.startswith("test_")


def _python_files() -> List[Path]:
    """Every first-party ``.py`` file, pruning skipped trees instead of walking them.

    ``rglob`` would descend into ``.venv`` and ``node_modules`` (tens of
    thousands of files) only to reject each one; pruning at the directory
    level keeps this check cheap enough for the fast gate.
    """
    files = []
    for directory, subdirectories, filenames in os.walk(_REPO_ROOT):
        subdirectories[:] = [name for name in subdirectories if name not in _SKIP_DIRS]
        files.extend(Path(directory) / name for name in filenames if name.endswith(".py"))
    return sorted(files)


def _product_files() -> List[Path]:
    """Product modules: first-party code an interface or tool ships."""
    return [
        path
        for path in _python_files()
        if path.relative_to(_REPO_ROOT).parts[0] in _PRODUCT_ROOTS
        and "tests" not in path.relative_to(_REPO_ROOT).parts
        and not _is_test_file(path)
    ]


def _consumer_files() -> List[Path]:
    """Every non-test module, including the evaluation gate's own scripts."""
    return [path for path in _python_files() if not _is_test_file(path)]


def _module_name(path: Path) -> str:
    """``src/monkeygrab/domain/chunk.py`` -> ``monkeygrab.domain.chunk``."""
    parts = list(path.relative_to(_REPO_ROOT).with_suffix("").parts)
    if parts[0] == "src":
        parts = parts[1:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _package_of(path: Path) -> str:
    """The package a relative import resolves against."""
    name = _module_name(path)
    return name if path.name == "__init__.py" else name.rpartition(".")[0]


def _import_targets(path: Path, tree: ast.AST) -> Set[str]:
    """Every dotted module a consumer imports, relative imports resolved."""
    package = _package_of(path)
    targets: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                drop = node.level - 1
                parts = parts[:-drop] if drop else parts
                module = ".".join([*parts, node.module] if node.module else parts)
            else:
                module = node.module or ""
            targets.add(module)
            for alias in node.names:
                if alias.name != "*":
                    targets.add(f"{module}.{alias.name}" if module else alias.name)
    return {target for target in targets if target}


def _used_names(sources: Iterable[str], names: Optional[Set[str]] = None) -> Set[str]:
    """Every name used as code in ``sources``.

    Only NAME tokens count. A name inside a comment, a docstring or a string
    is text about the symbol, not a use of it, and treating it as a reference
    is how a dead symbol hides behind the note that mentions it. ``tokenize``
    is what draws that line: on 3.12+ an f-string arrives as parts, and its
    literal chunks are FSTRING_MIDDLE, never NAME, so they are text as well.

    Lines declaring a ``def``/``class`` are excluded for the name they declare.
    A definition is where a symbol is born, not where it is used; without this,
    two dead methods sharing a name (``is_empty`` on three dataclasses) each
    count as the other's caller.

    ``names`` narrows the scan to the symbols under test: the repository has
    millions of words and a few thousand symbols, so tracking every name
    would cost far more than the question needs.
    """
    used: Set[str] = set()
    for text in sources:
        tokens = [
            token
            for token in tokenize.generate_tokens(io.StringIO(text).readline)
            if token.type == tokenize.NAME
        ]
        declared_on: Dict[str, Set[int]] = {}
        for keyword, name_token in zip(tokens, tokens[1:]):
            if keyword.string in _DEFINITION_KEYWORDS:
                declared_on.setdefault(name_token.string, set()).add(name_token.start[0])
        for token in tokens:
            name = token.string
            if names is not None and name not in names:
                continue
            if token.start[0] in declared_on.get(name, ()):
                continue
            used.add(name)
    return used


def _is_registrar(decorator: ast.expr) -> bool:
    """``@app.route(...)`` registers its function; ``@property`` does not.

    ``@app.before_request`` is not a call and registers just as much, so any
    attribute decorator counts. The distinction that matters is attribute
    access versus a bare name: ``@property`` is the latter.
    """
    return isinstance(decorator, ast.Attribute) or (
        isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
    )


def _declared_symbols(tree: ast.AST) -> List[Tuple[str, int, Sequence[ast.expr]]]:
    """Top-level functions/classes and the methods of those classes."""
    declared = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            declared.append((node.name, node.lineno, node.decorator_list))
        elif isinstance(node, ast.ClassDef):
            declared.append((node.name, node.lineno, node.decorator_list))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    declared.append((child.name, child.lineno, child.decorator_list))
    return declared


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _unused_symbols(
    tree: ast.AST,
    used_names: Set[str],
    reexports: Set[str],
) -> List[Tuple[str, int]]:
    """Symbols whose name is never used as code outside its own declaration."""
    unused = []
    for name, line, decorators in _declared_symbols(tree):
        if _is_dunder(name) or name in reexports or any(_is_registrar(d) for d in decorators):
            continue
        if name in used_names:
            continue
        unused.append((name, line))
    return unused


def _chat_pdfs_reexports() -> Set[str]:
    """Names ``rag/chat_pdfs.py`` imports and never uses itself: its facade.

    Importing a name and not referencing it anywhere in the module is exactly
    what a re-export looks like. A name the module also calls is an ordinary
    internal import and stays subject to the check.
    """
    path = _REPO_ROOT / "rag" / "chat_pdfs.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(alias.asname or alias.name for alias in node.names)
    used = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    return imported - used


def _entry_point_mentions() -> Tuple[str, str]:
    """Names that make a module an entry point: (README+workflows, specs).

    Specs are separate because they reference the frozen entry point by
    basename inside ``os.path.join(...)``, never by path or dotted name.
    """
    general = []
    readme = _REPO_ROOT / "README.md"
    if readme.exists():
        general.append(readme.read_text(encoding="utf-8"))
    for path in _REPO_ROOT.glob(".github/workflows/*.yml"):
        general.append(path.read_text(encoding="utf-8"))
    specs = [path.read_text(encoding="utf-8") for path in _REPO_ROOT.glob("packaging/*.spec")]
    return "\n".join(general), "\n".join(specs)


def _is_entry_point(path: Path, source: str, mentions: Tuple[str, str]) -> bool:
    if path.name in {"__init__.py", "__main__.py"}:
        return True
    if "__name__" in source and "__main__" in source:
        return True
    general, specs = mentions
    relative = path.relative_to(_REPO_ROOT).as_posix()
    return _module_name(path) in general or relative in general or path.name in specs


def _product_analysis():
    """Scan the consumer sources once and return the raw findings and the trees."""
    sources = []
    import_targets: Set[str] = set()
    for path in _consumer_files():
        text = path.read_text(encoding="utf-8")
        sources.append(text)
        import_targets |= _import_targets(path, ast.parse(text, filename=str(path)))

    product_trees = {}
    declared_names: Set[str] = set()
    for path in _product_files():
        label = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        product_trees[label] = tree
        declared_names.update(name for name, _, _ in _declared_symbols(tree))

    used_names = _used_names(sources, names=declared_names)
    reexports = _chat_pdfs_reexports()

    dead_symbols = []
    for label, tree in product_trees.items():
        for name, line in _unused_symbols(tree, used_names, reexports):
            dead_symbols.append((label, name, line))

    mentions = _entry_point_mentions()
    dead_modules = []
    for path in _product_files():
        label = path.relative_to(_REPO_ROOT).as_posix()
        if _module_name(path) in import_targets:
            continue
        if _is_entry_point(path, path.read_text(encoding="utf-8"), mentions):
            continue
        dead_modules.append(label)
    return dead_symbols, dead_modules, product_trees


@pytest.fixture(scope="module")
def analysis():
    return _product_analysis()


def test_the_file_lists_are_not_accidentally_empty():
    """Guards the guard: a bad skip list would make everything below vacuous."""
    assert len(_product_files()) > 50, f"only {len(_product_files())} product files found"
    assert len(_consumer_files()) > len(_product_files())


def test_no_product_symbol_is_referenced_only_by_its_own_definition(analysis):
    dead_symbols, _, _ = analysis
    unwhitelisted = [
        (label, name, line)
        for label, name, line in dead_symbols
        if f"{label}:{name}" not in _WHITELIST
    ]
    assert not unwhitelisted, (
        "Product symbols that nothing outside their own def/class line references:\n"
        + "\n".join(
            f"  {label}:{name} (line {line})" for label, name, line in unwhitelisted
        )
        + "\nDelete it, or add \"path:symbol\" to _WHITELIST at the top of "
        "tests/unit/test_no_dead_code.py with the reason it must stay."
    )


def test_no_product_module_is_imported_only_by_tests(analysis):
    _, dead_modules, _ = analysis
    assert not dead_modules, (
        "Product modules no consumer imports (tests do not count as consumers):\n"
        + "\n".join(f"  {label}" for label in dead_modules)
        + "\nDelete it, or make it an entry point (an `if __name__ == \"__main__\"` "
        "block, a workflow, a PyInstaller spec or the README)."
    )


def test_the_whitelist_has_no_stale_entries(analysis):
    dead_symbols, _, product_trees = analysis
    still_dead = {f"{label}:{name}" for label, name, _ in dead_symbols}
    stale = []
    for key in _WHITELIST:
        label, _, name = key.partition(":")
        declared = _declared_symbols(product_trees[label]) if label in product_trees else []
        if key not in still_dead or name not in {symbol for symbol, _, _ in declared}:
            stale.append(key)
    assert not stale, (
        "Whitelist entries that no longer correspond to a dead symbol:\n"
        + "\n".join(f"  {key}" for key in stale)
        + "\nRemove them: a whitelist that outlives its symbol hides the next one."
    )


# Synthetic sources, so a green run proves the detector works, not the tree.

_SYNTHETIC_PRODUCT = '''
def live():
    return 1


def dead():
    return 2


class Thing:
    def used(self):
        return live()

    def forgotten(self):
        return 3

    @app.route("/x")
    def handler(self):
        return 4

    @app.before_request
    def hook(self):
        return 5

    def __str__(self):
        return "thing"
'''

_SYNTHETIC_CONSUMER = "from mod import live\n\nThing().used()\n"


def test_the_check_catches_a_function_and_a_method_nothing_references():
    """Only ``dead`` and ``forgotten``; the rest have a caller or are exempt."""
    used = _used_names([_SYNTHETIC_PRODUCT, _SYNTHETIC_CONSUMER])
    found = _unused_symbols(
        ast.parse(_SYNTHETIC_PRODUCT),
        used,
        reexports={"live"},
    )
    assert [name for name, _ in found] == ["dead", "forgotten"], found


def test_the_check_catches_a_module_nothing_imports():
    """``from harness import ledger`` imports ``harness.ledger``, not the package."""
    tree = ast.parse("from harness import ledger\nimport rag.engine.history\n")
    targets = _import_targets(_REPO_ROOT / "tools" / "script.py", tree)
    assert "harness.ledger" in targets
    assert "rag.engine.history" in targets
    assert "tools.script" not in targets


def test_a_plain_property_is_not_a_registrar():
    """``@property`` looks like a decorator but is not framework registration."""
    source = '''
class Quiz:
    @property
    def correct_option(self):
        return self.options[self.correct_index]
'''
    used = _used_names([source, "Quiz()\n"])
    found = _unused_symbols(ast.parse(source), used, reexports=set())
    assert [name for name, _ in found] == ["correct_option"], found


def test_homonymous_dead_properties_and_a_merely_mentioned_name_are_reported():
    """The shapes the old token scan missed: two ``def`` lines rescuing each
    other, and a name that only appears in a comment and a string."""
    product = '''
class Summary:
    @property
    def is_empty(self):
        return not self.sections


class Outline:
    @property
    def is_empty(self):
        return not self.nodes


def forgotten():
    return 1
'''
    consumer = (
        '"""forgotten is described here, which is not a use."""\n'
        "# is_empty used to matter too.\n"
        "Summary()\n"
        "Outline()\n"
        'NOTE = "forgotten"\n'
    )
    used = _used_names([product, consumer])
    found = _unused_symbols(ast.parse(product), used, reexports=set())
    assert [name for name, _ in found] == ["is_empty", "is_empty", "forgotten"], found
