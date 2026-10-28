"""Structure tests for ``rag/``: layering, private imports, the composition root
and import cycles.

Issue #527. ``tests/unit/test_architecture_boundaries.py`` protects the four
pure layers of ``src/monkeygrab``; outside them, nothing stopped the engine from
importing the web layer, one package from reaching into another's private
names, or a new import cycle from appearing. This module parses every product
module's imports, so the check fails on the source as written rather than on a
list someone remembered to update. It reads import statements and calls by
name only: attribute access on an imported module (``mod._x``), ``getattr``,
``importlib`` and star imports are outside what it sees.

1. **Layers.** ``rag/engine`` may not import ``rag.web`` or ``rag.headless``;
   ``rag.web`` and ``rag.headless`` may not import each other. Imports inside
   functions count: a deferred import of the web app is still the engine
   depending on the web app.
2. **Private names.** A module may not import a name starting with ``_`` from
   another top-level package (``monkeygrab``, ``rag``, ``harness``, ``tools``).
   Within the same top level it is allowed (``rag.chat_pdfs`` re-exports
   ``rag.engine``'s legacy private names on purpose), so ``rag.web`` reaching
   into ``rag.engine._x`` passes this rule.
3. **Composition root.** Only ``src/monkeygrab/adapters/``,
   ``src/monkeygrab/composition.py`` and ``rag/engine/wiring.py`` instantiate
   adapter classes. Importing a function or a constant from an adapter stays
   allowed (``rag/chat_pdfs.py`` imports ``resolve_reranker_device``), because
   only construction is the composition root's job.
4. **Cycles.** The module-level import graph is split into strongly connected
   components with Tarjan's algorithm. A cycle in ``KNOWN_CYCLES`` must still
   exist, and any other one fails: the list only shrinks. Imports inside
   functions do not count, which is the known way to break a cycle.

Standard library only (``ast``, ``os``, ``pathlib``): this runs in the fast
gate's ``architecture`` job, which installs nothing but pytest. The helpers are
exercised against synthetic sources at the bottom, so a green run means the
checks work, not merely that today's tree happens to be clean.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Mapping, Set, Tuple

import pytest

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

# Directories this repository ships. Tests are not product: a test that imports
# a private name is exercising it, not depending on it.
_PRODUCT_ROOTS = ("src", "rag", "harness", "tools")

# First-level packages a private import may cross. ``src`` maps to
# ``monkeygrab``; the other three are their own top level.
_PACKAGES = frozenset({"monkeygrab", "rag", "harness", "tools"})

_ADAPTER_PREFIX = "monkeygrab.adapters"

# The composition root: the adapters build each other, and these two files turn
# an AppConfig into the adapters the use cases receive.
_COMPOSITION_SITES = frozenset({"monkeygrab.composition", "rag.engine.wiring"})

# Importing downward is the normal direction; these pairs must never point up.
_FORBIDDEN_LAYER_IMPORTS = {
    "rag.engine": ("rag.web", "rag.headless"),
    "rag.web": ("rag.headless",),
    "rag.headless": ("rag.web",),
}

# Baseline, not a licence: every entry is a cycle that exists today and must
# keep existing. ``config/__init__`` re-exports ``AppConfig`` while
# ``app_config`` imports its sibling ``env`` through the package
# (``from monkeygrab.config import env``), so importing either one imports the
# other. Harmless as written -- ``env`` is imported first -- and shrinking the
# list is a separate change.
KNOWN_CYCLES = (
    frozenset({"monkeygrab.config", "monkeygrab.config.app_config"}),
)


def _python_files() -> List[Path]:
    """Every first-party ``.py`` file, pruning skipped trees instead of walking them."""
    files = []
    for directory, subdirectories, filenames in os.walk(_REPO_ROOT):
        subdirectories[:] = [name for name in subdirectories if name not in _SKIP_DIRS]
        files.extend(Path(directory) / name for name in filenames if name.endswith(".py"))
    return sorted(files)


def _is_test_file(path: Path) -> bool:
    return path.name == "conftest.py" or path.name.startswith("test_")


def _module_name(path: Path) -> str:
    """``src/monkeygrab/domain/chunk.py`` -> ``monkeygrab.domain.chunk``."""
    parts = list(path.relative_to(_REPO_ROOT).with_suffix("").parts)
    if parts[0] == "src":
        parts = parts[1:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _product_sources() -> Dict[str, str]:
    """Product modules by dotted name, keyed the way an import names them."""
    sources = {}
    for path in _python_files():
        relative = path.relative_to(_REPO_ROOT)
        if relative.parts[0] not in _PRODUCT_ROOTS:
            continue
        if "tests" in relative.parts or _is_test_file(path):
            continue
        sources[_module_name(path)] = path.read_text(encoding="utf-8")
    return sources


def _is_package(module: str) -> bool:
    """True when ``module`` names a package (an ``__init__.py``) on disk."""
    relative = Path(*module.split("."))
    return any((base / relative / "__init__.py").is_file() for base in (_REPO_ROOT, _REPO_ROOT / "src"))


def _base_module(node: ast.AST, module: str) -> str:
    """The module an ``ImportFrom`` names, relative imports resolved.

    A package's ``__init__`` is keyed by the package name, and its relative
    imports resolve against the package itself, not its parent: ``from ..web``
    in ``rag/engine/__init__.py`` means ``rag.web``.
    """
    if not isinstance(node, ast.ImportFrom):
        return ""
    if not node.level:
        return node.module or ""
    package = module if _is_package(module) else module.rpartition(".")[0]
    parts = package.split(".") if package else []
    drop = node.level - 1
    if drop:
        parts = parts[:-drop]
    return ".".join([*parts, node.module] if node.module else parts)


def _resolve_import(node: ast.AST, module: str) -> Set[str]:
    """Every dotted target an import names, including the imported names.

    ``from rag.engine import wiring`` yields both ``rag.engine`` and
    ``rag.engine.wiring``: the first is the package boundary an ``__init__``
    import crosses, the second is the module an import of a submodule reaches.
    """
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}
    if not isinstance(node, ast.ImportFrom):
        return set()
    base = _base_module(node, module)
    targets = {base} if base else set()
    for alias in node.names:
        if alias.name != "*":
            targets.add(f"{base}.{alias.name}" if base else alias.name)
    return targets


def _belongs_to(module: str, layer: str) -> bool:
    return module == layer or module.startswith(layer + ".")


def _layer_violations(sources: Mapping[str, str]) -> List[Tuple[str, str, int]]:
    """``(importer, imported target, line)`` for every upward import."""
    violations = []
    for module, source in sources.items():
        tree = ast.parse(source, filename=module)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for target in _resolve_import(node, module):
                for layer, forbidden in _FORBIDDEN_LAYER_IMPORTS.items():
                    if not _belongs_to(module, layer):
                        continue
                    if any(_belongs_to(target, other) for other in forbidden):
                        violations.append((module, target, node.lineno))
    return violations


def _private_import_violations(sources: Mapping[str, str]) -> List[Tuple[str, str, int]]:
    """``(importer, imported name, line)`` for private names crossing a top level."""
    violations = []
    for module, source in sources.items():
        importer_top = module.split(".")[0]
        tree = ast.parse(source, filename=module)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = _base_module(node, module)
                source_top = base.split(".")[0] if base else ""
                for alias in node.names:
                    if (
                        alias.name.startswith("_")
                        and source_top in _PACKAGES
                        and source_top != importer_top
                    ):
                        violations.append((module, f"{base}.{alias.name}", node.lineno))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if (
                        parts[0] in _PACKAGES
                        and parts[0] != importer_top
                        and any(part.startswith("_") for part in parts[1:])
                    ):
                        violations.append((module, alias.name, node.lineno))
    return violations


def _call_name(node: ast.AST) -> str:
    """The dotted text of a pure ``Name``/``Attribute`` chain, else ``""``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else ""
    return ""


def _adapter_bindings(tree: ast.AST, module: str) -> Dict[str, str]:
    """Map each local name bound by an import to the dotted path it refers to."""
    bindings = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = _base_module(node, module)
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                bindings[local] = f"{base}.{alias.name}" if base else alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    bindings[alias.asname] = alias.name
                else:
                    root = alias.name.split(".")[0]
                    bindings[root] = root
    return bindings


def _adapter_call_violations(sources: Mapping[str, str]) -> List[Tuple[str, str, int]]:
    """``(importer, adapter class, line)`` for construction outside the root."""
    violations = []
    for module, source in sources.items():
        if _belongs_to(module, _ADAPTER_PREFIX) or module in _COMPOSITION_SITES:
            continue
        tree = ast.parse(source, filename=module)
        bindings = _adapter_bindings(tree, module)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            if not name:
                continue
            root, _, rest = name.partition(".")
            resolved = bindings.get(root, root)
            dotted = f"{resolved}.{rest}" if rest else resolved
            if not _belongs_to(dotted, _ADAPTER_PREFIX):
                continue
            if dotted.rsplit(".", 1)[-1][:1].isupper():
                violations.append((module, dotted, node.lineno))
    return violations


class _ModuleLevelImports(ast.NodeVisitor):
    """Collect imports that run when the module is imported.

    A function body runs later, if ever -- placing an import there is exactly
    how a cycle is broken today, so the cycle graph must not see it.
    """

    def __init__(self):
        self.nodes: List[ast.AST] = []

    def visit_Import(self, node):
        self.nodes.append(node)

    def visit_ImportFrom(self, node):
        self.nodes.append(node)

    def visit_FunctionDef(self, node):
        pass

    visit_AsyncFunctionDef = visit_FunctionDef


def _import_graph(sources: Mapping[str, str]) -> Dict[str, Set[str]]:
    """Dotted module -> modules it imports at import time."""
    known = set(sources)
    graph: Dict[str, Set[str]] = {}
    for module, source in sources.items():
        visitor = _ModuleLevelImports()
        visitor.visit(ast.parse(source, filename=module))
        edges: Set[str] = set()
        for node in visitor.nodes:
            for target in _resolve_import(node, module):
                # ``from pkg import name`` may name a submodule or a symbol.
                # Trim back to the longest known module prefix.
                parts = target.split(".")
                while parts:
                    candidate = ".".join(parts)
                    if candidate in known:
                        if candidate != module:
                            edges.add(candidate)
                        break
                    parts.pop()
        graph[module] = edges
    return graph


def _strongly_connected_components(graph: Mapping[str, Set[str]]) -> List[Set[str]]:
    """Tarjan's algorithm, iterative in spirit but recursive in size (the tree is small)."""
    index: Dict[str, int] = {}
    low: Dict[str, int] = {}
    stack: List[str] = []
    on_stack: Set[str] = set()
    components: List[Set[str]] = []
    counter = [0]

    def connect(node):
        index[node] = low[node] = counter[0]
        counter[0] += 1
        stack.append(node)
        on_stack.add(node)
        for successor in graph[node]:
            if successor not in index:
                connect(successor)
                low[node] = min(low[node], low[successor])
            elif successor in on_stack:
                low[node] = min(low[node], index[successor])
        if low[node] == index[node]:
            component: Set[str] = set()
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.add(member)
                if member == node:
                    break
            components.append(component)

    for node in graph:
        if node not in index:
            connect(node)
    return components


def _import_cycles(sources: Mapping[str, str]) -> Set[frozenset]:
    """Cycles of two or more modules; a module importing itself is not one."""
    return {
        frozenset(component)
        for component in _strongly_connected_components(_import_graph(sources))
        if len(component) > 1
    }


def _format(violations: List[Tuple[str, str, int]]) -> str:
    return "\n".join(
        f"  {module} -> {target} (line {line})" for module, target, line in violations
    )


@pytest.fixture(scope="module")
def product_sources():
    return _product_sources()


def test_the_product_sources_are_not_accidentally_empty(product_sources):
    """Guards the guard: a bad skip list or root list would make everything vacuous."""
    assert len(product_sources) > 50, f"only {len(product_sources)} product modules found"
    for layer in _FORBIDDEN_LAYER_IMPORTS:
        assert any(_belongs_to(module, layer) for module in product_sources), layer


def test_no_layer_imports_a_layer_above_it(product_sources):
    violations = _layer_violations(product_sources)
    assert not violations, (
        "rag/ imports across a layer boundary it must not cross "
        "(rag.engine -> rag.web|rag.headless, rag/web <-> rag/headless):\n"
        + _format(violations)
    )


def test_no_private_name_crosses_a_top_level_package(product_sources):
    violations = _private_import_violations(product_sources)
    assert not violations, (
        "Private names imported across top-level packages; make the name public "
        "if another package legitimately needs it:\n" + _format(violations)
    )


def test_only_the_composition_root_instantiates_adapters(product_sources):
    violations = _adapter_call_violations(product_sources)
    assert not violations, (
        "Adapter classes instantiated outside the composition root "
        "(src/monkeygrab/adapters/, src/monkeygrab/composition.py, "
        "rag/engine/wiring.py); add a builder there instead:\n" + _format(violations)
    )


def test_no_new_import_cycles(product_sources):
    unexpected = sorted(
        (cycle for cycle in _import_cycles(product_sources) if cycle not in KNOWN_CYCLES),
        key=lambda cycle: sorted(cycle),
    )
    assert not unexpected, (
        "New module-level import cycles (move one side inside a function, the "
        "usual fix):\n"
        + "\n".join(f"  {', '.join(sorted(cycle))}" for cycle in unexpected)
    )


def test_each_known_cycle_is_still_a_cycle(product_sources):
    cycles = _import_cycles(product_sources)
    stale = sorted(
        (known for known in KNOWN_CYCLES if known not in cycles),
        key=lambda cycle: sorted(cycle),
    )
    assert not stale, (
        "KNOWN_CYCLES lists cycles that no longer exist; remove them from "
        "KNOWN_CYCLES so the baseline only shrinks:\n"
        + "\n".join(f"  {', '.join(sorted(cycle))}" for cycle in stale)
    )


# Synthetic sources, so a green run proves the checks work, not the tree.


def test_the_check_catches_a_layering_import_even_inside_a_function():
    sources = {
        "rag.engine.debug": "def dump():\n    import rag.web.app\n",
        "rag.web.app": "import rag.engine.debug\n",
    }
    assert _layer_violations(sources) == [("rag.engine.debug", "rag.web.app", 2)]


def test_a_relative_import_in_a_package_init_resolves_against_the_package():
    """``from ..web`` in ``rag/engine/__init__.py`` is ``rag.web``, not ``web``."""
    sources = {"rag.engine": "from ..web import app\n"}
    assert set(_layer_violations(sources)) == {("rag.engine", "rag.web", 1), ("rag.engine", "rag.web.app", 1)}


def test_the_check_catches_a_private_import_only_across_top_level_packages():
    sources = {
        "rag.engine.indexing": "from monkeygrab.composition import _isolated_python\n",
        "monkeygrab.other": "from monkeygrab.composition import _isolated_python\n",
    }
    assert _private_import_violations(sources) == [
        ("rag.engine.indexing", "monkeygrab.composition._isolated_python", 1)
    ]


def test_the_check_catches_an_adapter_class_built_outside_the_composition_root():
    sources = {
        "rag.engine.indexing": (
            "from monkeygrab.adapters.extraction.mineru_extractor import MineruExtractor\n"
            "extractor = MineruExtractor()\n"
        ),
        "rag.chat_pdfs": (
            "from monkeygrab.adapters.reranking.cross_encoder_reranker import "
            "resolve_reranker_device\n"
            "device = resolve_reranker_device(0)\n"
        ),
    }
    assert _adapter_call_violations(sources) == [
        (
            "rag.engine.indexing",
            "monkeygrab.adapters.extraction.mineru_extractor.MineruExtractor",
            2,
        )
    ]


def test_the_cycle_detector_finds_a_module_level_cycle():
    sources = {"a": "import b\n", "b": "from a import value\n"}
    assert _import_cycles(sources) == {frozenset({"a", "b"})}


def test_the_cycle_detector_ignores_an_import_inside_a_function():
    sources = {"a": "def load():\n    import b\n", "b": "from a import value\n"}
    assert _import_cycles(sources) == set()
