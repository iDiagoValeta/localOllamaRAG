"""Architecture boundary test: domain/, ports/, config/ and application/ stay
layered correctly -- infrastructure-free, and each importing only the layers
below it.

docs/design/2026-07-26-monkeygrab-v2.md, section 4: "Cuatro capas con
dependencias en un solo sentido: interfaces -> aplicacion -> dominio, y
adaptadores implementando puertos que la aplicacion declara. El dominio no
importa nada de infraestructura." This test enforces two independent things,
both by AST-parsing each module's import statements (not a source-text grep,
which a reformatted or aliased import could dodge):

1. **No infrastructure at all** (``test_package_imports_nothing_outside_the_standard_library``):
   every top-level import target in domain/ports/config/application must be
   part of the Python standard library or the ``monkeygrab`` package itself.
   A whitelist of "what's allowed" (stdlib) is used rather than a denylist of
   "known infra libraries" (chromadb, ollama, fitz, torch, ...) so a *new*
   infrastructure dependency added later is caught automatically instead of
   requiring this test to be updated in lockstep. This also transitively
   blocks importing ``rag`` (the legacy service-locator package): it is
   neither stdlib nor ``monkeygrab``.

2. **Per-layer boundaries within monkeygrab itself**
   (``test_package_imports_only_allowed_monkeygrab_layers``): (1) alone is
   NOT enough, because it only records "monkeygrab" as the top-level import
   target for every ``monkeygrab.*`` import, regardless of which submodule --
   ``from monkeygrab.adapters.vectorstore.chroma_store import ChromaVectorStore``
   written inside ``domain/`` would PASS check (1) while dragging chromadb in
   transitively. This second check keeps the FULL dotted import path and
   verifies the targeted monkeygrab *subpackage* against a per-layer
   whitelist: domain -> only domain; ports -> domain/ports; config -> only
   config; application -> domain/ports/config/application. None of the four
   may import adapters (adapters implement ports the application layer
   declares -- the dependency never points the other way) or application may
   not import "rag" either, already covered by check (1).
"""

import ast
import sys
from pathlib import Path
from typing import Dict, Set

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

SRC = ROOT / "src" / "monkeygrab"

# Python's own standard library, plus the package under test itself.
# sys.stdlib_module_names (3.10+) includes private/internal names (e.g.
# "_typeshed"); that is fine here, it only widens what's *allowed*.
_ALLOWED_TOP_LEVEL_MODULES: Set[str] = set(sys.stdlib_module_names) | {"monkeygrab", "__future__"}

# Per-layer whitelist of which monkeygrab SUBPACKAGES (the segment right
# after "monkeygrab.") each layer may import from, on top of the stdlib.
# Dependencies point one way only: interfaces -> application -> domain, with
# adapters implementing ports the application layer declares -- so "adapters"
# and "application" (from domain/ports/config) never appear on the right of
# any of these sets, and "rag" never appears at all (already blocked by
# check (1) above, since "rag" is neither stdlib nor "monkeygrab").
_LAYER_ALLOWED_MONKEYGRAB_SUBPACKAGES: Dict[str, Set[str]] = {
    "domain": {"domain"},
    "ports": {"domain", "ports"},
    "config": {"config"},
    "application": {"domain", "ports", "config", "application"},
}

_LAYERS = tuple(_LAYER_ALLOWED_MONKEYGRAB_SUBPACKAGES)


def _top_level_imports(py_file: Path) -> Set[str]:
    """Return the top-level module name of every import in ``py_file``.

    Relative imports (``from . import x``, ``from .foo import y``) are
    skipped: they can only resolve to another module inside the same
    package, never to an external library.
    """
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    modules: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.module is None:
                continue  # "from . import x" -- relative, always internal
            if node.module:
                modules.add(node.module.split(".")[0])
    return modules


def _monkeygrab_import_targets(py_file: Path) -> Set[str]:
    """Return the FULL dotted module path of every ``monkeygrab.*`` import.

    Unlike ``_top_level_imports`` (which collapses every such import down to
    the single string "monkeygrab", useless for a per-layer check), this
    keeps the complete path -- e.g. "monkeygrab.adapters.vectorstore.chroma_store"
    -- so ``_subpackage_of`` can tell which layer is actually being reached
    into.
    """
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    targets: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "monkeygrab" or alias.name.startswith("monkeygrab."):
                    targets.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.module is None:
                continue  # "from . import x" -- relative, always internal
            if node.module and (node.module == "monkeygrab" or node.module.startswith("monkeygrab.")):
                targets.add(node.module)
    return targets


def _subpackage_of(dotted_monkeygrab_import: str) -> str:
    """"monkeygrab.adapters.vectorstore.chroma_store" -> "adapters".

    A bare ``import monkeygrab`` (no submodule) has no subpackage to check
    and returns ``""`` -- nothing in this codebase does that today, and a
    bare package import cannot itself drag in a specific infrastructure
    dependency the way reaching into ``monkeygrab.adapters.*`` can.
    """
    parts = dotted_monkeygrab_import.split(".")
    return parts[1] if len(parts) > 1 else ""


def _python_files(package_dir: Path):
    return sorted(package_dir.rglob("*.py"))


@pytest.mark.parametrize("package_name", _LAYERS)
def test_package_imports_nothing_outside_the_standard_library(package_name):
    package_dir = SRC / package_name
    assert package_dir.is_dir(), f"expected {package_dir} to exist"

    violations = {}
    for py_file in _python_files(package_dir):
        imported = _top_level_imports(py_file)
        infra = imported - _ALLOWED_TOP_LEVEL_MODULES
        if infra:
            violations[str(py_file.relative_to(ROOT))] = sorted(infra)

    assert not violations, (
        f"{package_name}/ must import only the standard library and "
        f"monkeygrab.* -- found infrastructure imports: {violations}"
    )


@pytest.mark.parametrize("package_name", _LAYERS)
def test_package_imports_only_allowed_monkeygrab_layers(package_name):
    """Per-layer check: catches e.g. `from monkeygrab.adapters.vectorstore.chroma_store
    import ChromaVectorStore` inside domain/, which the blanket "starts with
    monkeygrab" check above lets through (see the module docstring)."""
    package_dir = SRC / package_name
    allowed = _LAYER_ALLOWED_MONKEYGRAB_SUBPACKAGES[package_name]

    violations: Dict[str, list] = {}
    for py_file in _python_files(package_dir):
        for target in _monkeygrab_import_targets(py_file):
            sub = _subpackage_of(target)
            if sub and sub not in allowed:
                violations.setdefault(str(py_file.relative_to(ROOT)), []).append(target)

    assert not violations, (
        f"{package_name}/ may only import monkeygrab.{{{', '.join(sorted(allowed))}}} -- "
        f"found disallowed cross-layer imports: {violations}"
    )


def test_domain_ports_config_and_application_directories_are_not_accidentally_empty():
    """Guards the guard: an empty/misnamed directory would make the tests
    above vacuously pass."""
    for package_name in _LAYERS:
        assert len(_python_files(SRC / package_name)) >= 2, package_name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
