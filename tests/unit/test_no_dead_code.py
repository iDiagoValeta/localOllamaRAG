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
- None for ``rag/chat_pdfs.py``'s re-exports. The facade's own import line
  is not a use: a name only the facade re-exports, with no consumer reaching
  it through ``rag_engine.<name>`` or an import, is dead like any other. This
  used to be a blanket exemption, and it hid two whole legacy modules and
  four functions (issue #537).
- ``_WHITELIST`` below is deliberately tiny and every entry says why. A symbol
  whitelisted "just in case" is how the check stops meaning anything, so the
  staleness test fails when an entry stops corresponding to a real finding.

## Flask routes

The registrar exemption above stops a handler from being reported as a
*symbol*, and with it stops anyone noticing that no client asks for the route
(issue #557: ``GET /api/stats``, ``/api/docs`` and ``/api/topics`` survived
that way). Routes get their own check: every ``/api/...`` route of
``rag/web/app.py`` must be requested by a string in the frontend sources
(``rag/web/frontend/src``) or in the headless service (``rag/headless``).
Tests are not consumers, as everywhere else in this file. Paths outside
``/api/`` (the page, the bundle, the logos) are requested by the browser
itself, not by a string in the frontend, so they are not checked. The match
is by path only, not by HTTP method: ``GET`` and ``POST`` on one path are kept
alive by either caller. The ``api`` object in ``App.tsx`` is where those strings
live, so a member of it that no ``api.<name>`` call reaches is reported too:
its URL would otherwise vouch for a route nobody requests.

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
import re
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


def _unused_symbols(tree: ast.AST, used_names: Set[str]) -> List[Tuple[str, int]]:
    """Symbols whose name is never used as code outside its own declaration."""
    unused = []
    for name, line, decorators in _declared_symbols(tree):
        if _is_dunder(name) or any(_is_registrar(d) for d in decorators):
            continue
        if name in used_names:
            continue
        unused.append((name, line))
    return unused


_FACADE = _REPO_ROOT / "rag" / "chat_pdfs.py"


def _facade_reexports(source: str) -> Set[str]:
    """Names the facade imports and never uses itself.

    Importing a name and not referencing it anywhere in the module is exactly
    what a re-export looks like. A name the module also calls is an ordinary
    internal import, and its use there counts.
    """
    tree = ast.parse(source)
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


_WEB_APP = _REPO_ROOT / "rag" / "web" / "app.py"
_FRONTEND_SRC = _REPO_ROOT / "rag" / "web" / "frontend" / "src"
_HEADLESS = _REPO_ROOT / "rag" / "headless"


_ROUTE_DECORATORS = frozenset({"route", "get", "post", "put", "delete", "patch"})


def _api_routes(source: str) -> List[str]:
    """Every ``/api/...`` path registered with ``@<app>.route("...")`` or the
    method shortcuts ``@<app>.get/post/put/delete/patch("...")``."""
    routes = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr in _ROUTE_DECORATORS
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
                and isinstance(decorator.args[0].value, str)
                and decorator.args[0].value.startswith("/api/")
            ):
                routes.append(decorator.args[0].value)
    return routes


def _frontend_code(source: str) -> str:
    """A TypeScript source with comments removed and ``${API_BASE}`` resolved.

    The frontend writes ``fetch(`${API_BASE}/init`)``, so the path a route is
    matched against only exists once the constant is substituted. Comments are
    dropped for the reason ``tokenize`` drops them above: ``// /api/init
    answers ok: false`` discusses a route, it does not request it.
    """
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    source = re.sub(r"(?m)(^|\s)//.*$", r"\1", source)
    base = re.search(r"const\s+API_BASE\s*=\s*['\"]([^'\"]*)['\"]", source)
    return source.replace("${API_BASE}", base.group(1)) if base else source


def _python_strings(source: str) -> str:
    """The string literals of a Python source, docstrings excluded."""
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    return "\n".join(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    )


def _is_frontend_test(path: Path) -> bool:
    """``*.test.ts(x)``, ``*.spec.ts(x)`` and anything under ``__tests__``."""
    infixes = path.name.split(".")[1:-1]
    # Relative to src/: a checkout under a directory called __tests__ must not
    # turn every frontend file into a test.
    relative = path.relative_to(_FRONTEND_SRC).parts
    return "__tests__" in relative or "test" in infixes or "spec" in infixes


def _frontend_files() -> List[Path]:
    """Frontend sources that count as consumers: tests are not, as in Python."""
    return [
        path
        for path in sorted(_FRONTEND_SRC.rglob("*"))
        if path.suffix in {".ts", ".tsx"} and not _is_frontend_test(path)
    ]


def _all_frontend_code() -> str:
    return "\n".join(
        _frontend_code(path.read_text(encoding="utf-8")) for path in _frontend_files()
    )


def _api_members(code: str) -> List[str]:
    """Member names of the frontend's ``api`` object; raises when it cannot parse any."""
    body = re.search(r"(?ms)^(?:export )?const api(?:: [\w<>\[\], ]+)? = \{\n(.*?)^\};", code)
    if not body:
        # Silence here would turn the whole check into a pass; a renamed or
        # reshaped object has to be noticed and this pattern updated.
        raise AssertionError("no `const api = {...}` object found in the frontend sources")
    members = re.findall(r"(?m)^  (\w+):", body.group(1))
    if not members:
        raise AssertionError(
            "the `api` object was found but no member matched (expected two-space "
            "indented `name:` keys); update the pattern instead of passing on nothing"
        )
    return members


def _uncalled_api_members(code: str) -> List[str]:
    """Members of the frontend's ``const api = {...}`` that no ``api.<name>`` reaches.

    A declared member carries a URL string, so left alone it keeps a route
    "requested" long after the UI stopped calling it (issue #557). knip does
    not flag unused object members, hence this check.
    """
    members = _api_members(code)
    return [name for name in members if not re.search(rf"\bapi\.{name}\b", code)]


def _route_consumer_text() -> str:
    """The code the web routes can be requested from: frontend plus headless."""
    parts = [_all_frontend_code()]
    parts += [
        _python_strings(path.read_text(encoding="utf-8"))
        for path in sorted(_HEADLESS.glob("*.py"))
    ]
    return "\n".join(parts)


def _unrequested_routes(routes: Iterable[str], consumer_text: str) -> List[str]:
    """Routes whose path no string in ``consumer_text`` requests.

    A parameterised route (``/api/docs/<path:filename>``) is requested by any
    string that starts with its static part, because the frontend appends the
    value (``/api/docs/${encodeURIComponent(name)}``). A fixed route is
    requested by its exact path: ``/api/docs`` must not be rescued by the
    ``/api/docs/...`` URL of its parameterised sibling, so the character after
    it cannot continue the path.
    """
    unrequested = []
    for route in routes:
        static, parameterised, _ = route.partition("<")
        tail = "" if parameterised else r"(?![\w/-])"
        if not re.search(re.escape(static) + tail, consumer_text):
            unrequested.append(route)
    return unrequested


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
    facade_source = ""
    import_targets: Set[str] = set()
    for path in _consumer_files():
        text = path.read_text(encoding="utf-8")
        if path == _FACADE:
            facade_source = text
        else:
            sources.append(text)
        import_targets |= _import_targets(path, ast.parse(text, filename=str(path)))

    product_trees = {}
    declared_names: Set[str] = set()
    for path in _product_files():
        label = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        product_trees[label] = tree
        declared_names.update(name for name, _, _ in _declared_symbols(tree))

    # The facade's re-export lines name every symbol they re-export, so they
    # are left out: only a caller beyond the facade keeps such a name alive.
    used_names = _used_names(sources, names=declared_names) | (
        _used_names([facade_source], names=declared_names) - _facade_reexports(facade_source)
    )

    dead_symbols = []
    for label, tree in product_trees.items():
        for name, line in _unused_symbols(tree, used_names):
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


def test_the_route_consumers_are_not_accidentally_empty():
    """Guards the guard: no routes or no consumer text would make the next test vacuous."""
    assert len(_api_routes(_WEB_APP.read_text(encoding="utf-8"))) > 10
    assert "/api/init" in _route_consumer_text()


def test_every_api_member_of_the_frontend_is_called():
    code = _all_frontend_code()
    uncalled = _uncalled_api_members(code)
    assert "init" in _api_members(code)  # the parse is alive, not vacuously empty
    assert not uncalled, (
        "Members of the `api` object in App.tsx that no `api.<name>` call reaches:\n"
        + "\n".join(f"  api.{name}" for name in uncalled)
        + "\nDelete the member: its URL would keep a dead route looking requested."
    )


def test_no_web_route_is_requested_by_nothing():
    unrequested = _unrequested_routes(
        _api_routes(_WEB_APP.read_text(encoding="utf-8")), _route_consumer_text()
    )
    assert not unrequested, (
        "Routes of rag/web/app.py that neither the frontend nor the headless service "
        "request (tests do not count as consumers):\n"
        + "\n".join(f"  {route}" for route in unrequested)
        + "\nDelete the route, or make the caller request it."
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
    found = _unused_symbols(ast.parse(_SYNTHETIC_PRODUCT), used)
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
    found = _unused_symbols(ast.parse(source), used)
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
    found = _unused_symbols(ast.parse(product), used)
    assert [name for name, _ in found] == ["is_empty", "is_empty", "forgotten"], found


def test_a_name_only_the_facade_reexports_is_reported():
    """The re-export line is not a use; ``rag_engine.<name>`` elsewhere is."""
    facade = "from rag.engine.generation import evaluar, generar\n"
    assert _facade_reexports(facade) == {"evaluar", "generar"}
    product = "def evaluar():\n    return 1\n\n\ndef generar():\n    return 2\n"
    consumer = "rag_engine.generar()\n"
    used = _used_names([product, consumer]) | (
        _used_names([facade]) - _facade_reexports(facade)
    )
    found = _unused_symbols(ast.parse(product), used)
    assert [name for name, _ in found] == ["evaluar"], found


def test_a_route_nothing_requests_is_reported():
    """Only ``/api/gone``: ``/api/used`` is built through ``API_BASE``, the
    parameterised route by its static prefix, and the page is not an API route."""
    app = (
        '@app.route("/")\n'
        "def index(): ...\n"
        '@app.route("/api/used", methods=["GET"])\n'
        "def used(): ...\n"
        '@app.route("/api/gone", methods=["GET"])\n'
        "def gone(): ...\n"
        '@app.route("/api/docs/<path:name>", methods=["DELETE"])\n'
        "def delete(name): ...\n"
    )
    frontend = _frontend_code(
        "const API_BASE = '/api';\n"
        "// /api/gone is only discussed here\n"
        "fetch(`${API_BASE}/used`);\n"
        "fetch(`${API_BASE}/docs/${encodeURIComponent(name)}`);\n"
    )
    assert _api_routes(app) == ["/api/used", "/api/gone", "/api/docs/<path:name>"]
    assert _unrequested_routes(_api_routes(app), frontend) == ["/api/gone"]


def test_an_api_member_nothing_calls_is_reported():
    """``stats`` is declared and never called; ``init`` is called, and a comment
    mentioning ``api.stats`` does not count as a call."""
    code = _frontend_code(
        "const API_BASE = '/api';\n"
        "const api = {\n"
        "  init: () =>\n"
        "    fetch(`${API_BASE}/init`),\n"
        "\n"
        "  stats: () =>\n"
        "    fetch(`${API_BASE}/stats`),\n"
        "};\n"
        "// api.stats used to be shown here\n"
        "await api.init();\n"
    )
    assert _uncalled_api_members(code) == ["stats"]


@pytest.mark.parametrize(
    "declaration",
    ["const api = {", "export const api = {", "const api: Api = {"],
)
def test_the_api_object_is_found_in_each_declaration_form(declaration):
    code = f"{declaration}\n  stats: () => 1,\n  init: () => 2,\n}};\napi.init();\n"
    assert _uncalled_api_members(code) == ["stats"]


def test_a_missing_api_object_fails_instead_of_passing():
    """A renamed object must not make the member check vacuously green."""
    with pytest.raises(AssertionError):
        _uncalled_api_members("const client = {\n  init: () => 1,\n};\n")


def test_an_api_object_whose_members_do_not_parse_fails_instead_of_passing():
    """Four-space indentation, quoted keys and method shorthand all parse to
    nothing; that has to be an error, not an empty list of findings."""
    for body in ("    stats: () => 1,\n", '  "stats": () => 1,\n', "  stats() { return 1; },\n"):
        with pytest.raises(AssertionError):
            _uncalled_api_members(f"const api = {{\n{body}}};\n")


def test_a_parent_directory_named_tests_does_not_empty_the_consumers(tmp_path, monkeypatch):
    src = tmp_path / "__tests__" / "checkout" / "src"
    src.mkdir(parents=True)
    (src / "App.tsx").write_text("fetch('/api/x');\n")
    monkeypatch.setattr(sys.modules[__name__], "_FRONTEND_SRC", src)
    assert _frontend_files() == [src / "App.tsx"]


def test_frontend_test_files_are_not_consumers():
    """A route or member reached only from a test is dead, as in the Python gate."""
    root = _FRONTEND_SRC
    for name in (
        "a.test.ts",
        "a.test.tsx",
        "a.spec.ts",
        "a.spec.tsx",
        "__tests__/a.ts",
        "lib/__tests__/deep/a.tsx",
    ):
        assert _is_frontend_test(root / name), name
    for name in ("App.tsx", "lib/types.ts", "contest.ts", "test.ts", "latest.tsx"):
        assert not _is_frontend_test(root / name), name


def test_a_route_reached_only_by_a_frontend_test_is_reported(tmp_path, monkeypatch):
    """End to end over a throwaway src/: the test's URL does not vouch for the
    route and its ``api.<name>`` call does not vouch for the member."""
    (tmp_path / "__tests__").mkdir()
    (tmp_path / "App.tsx").write_text(
        "const API_BASE = '/api';\nconst api = {\n  stats: () =>\n"
        "    fetch(`${API_BASE}/stats`),\n};\n"
    )
    (tmp_path / "App.test.tsx").write_text("api.stats();\n")
    (tmp_path / "__tests__" / "b.ts").write_text("fetch('/api/other');\n")
    monkeypatch.setattr(sys.modules[__name__], "_FRONTEND_SRC", tmp_path)
    code = _all_frontend_code()
    assert _uncalled_api_members(code) == ["stats"]
    assert _unrequested_routes(["/api/other"], code) == ["/api/other"]
    assert "api.stats();" not in code


def test_method_shortcut_decorators_register_routes_too():
    """``@app.get("/api/x")`` is a route like ``@app.route``; a bare ``@cache``
    or a non-/api path is not one."""
    app = (
        '@app.get("/api/a")\n'
        "def a(): ...\n"
        '@bp.post("/api/b")\n'
        "def b(): ...\n"
        '@app.delete("/page")\n'
        "def c(): ...\n"
        "@cache\n"
        "def d(): ...\n"
    )
    assert _api_routes(app) == ["/api/a", "/api/b"]
    assert _unrequested_routes(_api_routes(app), "fetch('/api/a')") == ["/api/b"]


def test_a_fixed_route_is_not_kept_alive_by_its_parameterised_sibling():
    """``/api/docs`` and ``/api/docs/<name>`` are different URLs; a client of the
    second says nothing about the first, and a headless string counts like the
    frontend's while its docstring does not."""
    frontend = "fetch('/api/docs/' + name);"
    assert _unrequested_routes(["/api/docs"], frontend) == ["/api/docs"]
    assert _unrequested_routes(["/api/docs"], "fetch('/api/docs?x=1');") == []
    headless = _python_strings('"""Mentions /api/old."""\nURL = "/api/new"\n')
    assert _unrequested_routes(["/api/old", "/api/new"], headless) == ["/api/old"]
