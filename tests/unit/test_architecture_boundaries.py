"""Architecture boundary test: domain/, ports/ and application/ stay infrastructure-free.

docs/design/2026-07-26-monkeygrab-v2.md, section 4: "El dominio no importa
nada de infraestructura" -- and F1's acceptance criterion makes the same
demand of ports/ ("ningun modulo de domain/ o application/ importa
chromadb, ollama, fitz ni torch"). application/ additionally may import
monkeygrab.config (use cases are constructed with an AppConfig) on top of
domain/ports, which the whitelist below already allows via the blanket
"monkeygrab" prefix -- this test does not need a separate allowance for it.
This test inspects each module's AST import statements
(not a source-text grep, which a reformatted or aliased import could dodge)
and fails if any top-level import target is not part of the Python standard
library or the ``monkeygrab`` package itself. A whitelist of "what's
allowed" (stdlib) is used rather than a denylist of "known infra libraries"
(chromadb, ollama, fitz, torch, ...) so a *new* infrastructure dependency
added later is caught automatically instead of requiring this test to be
updated in lockstep.
"""

import ast
import sys
from pathlib import Path
from typing import Set

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

SRC = ROOT / "src" / "monkeygrab"

# Python's own standard library, plus the package under test itself.
# sys.stdlib_module_names (3.10+) includes private/internal names (e.g.
# "_typeshed"); that is fine here, it only widens what's *allowed*.
_ALLOWED_TOP_LEVEL_MODULES: Set[str] = set(sys.stdlib_module_names) | {"monkeygrab", "__future__"}


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


def _python_files(package_dir: Path):
    return sorted(package_dir.rglob("*.py"))


@pytest.mark.parametrize("package_name", ["domain", "ports", "application"])
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


def test_domain_ports_and_application_directories_are_not_accidentally_empty():
    """Guards the guard: an empty/misnamed directory would make the test
    above vacuously pass."""
    assert len(_python_files(SRC / "domain")) >= 2
    assert len(_python_files(SRC / "ports")) >= 2
    assert len(_python_files(SRC / "application")) >= 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
