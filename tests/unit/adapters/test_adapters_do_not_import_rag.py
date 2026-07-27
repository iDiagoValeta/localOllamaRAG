"""Architecture boundary test: src/monkeygrab/adapters/ never imports rag.*.

Same rationale and technique as tests/unit/test_architecture_boundaries.py
(which guards domain/ and ports/): adapters must depend on config/ports/
domain plus their own third-party library, never on the legacy service
locator they are replacing (docs/design/2026-07-26-monkeygrab-v2.md,
section 1 -- "rag/engine/ modules do cfg = get_runtime(), which returns the
chat_pdfs module itself"). This inspects each adapter module's AST import
statements (not a source-text grep, which an aliased import could dodge) and
fails if any top-level import target is ``rag`` or any of its submodules
(``rag.chat_pdfs``, ``rag.engine.lexical``, ...) -- importing any of them
transitively pulls in ``rag.chat_pdfs`` via ``rag.engine.runtime.get_runtime()``.
"""

import ast
import sys
from pathlib import Path
from typing import Set

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

ADAPTERS_DIR = ROOT / "src" / "monkeygrab" / "adapters"


def _top_level_imports(py_file: Path) -> Set[str]:
    """Return the top-level module name of every import in ``py_file``.

    Relative imports (``from . import x``) are skipped: they can only
    resolve to another module inside monkeygrab itself.
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


def test_no_adapter_module_imports_the_rag_package():
    assert ADAPTERS_DIR.is_dir(), f"expected {ADAPTERS_DIR} to exist"

    violations = {}
    for py_file in _python_files(ADAPTERS_DIR):
        imported = _top_level_imports(py_file)
        if "rag" in imported:
            violations[str(py_file.relative_to(ROOT))] = "imports 'rag'"

    assert not violations, (
        "adapters/ must never import the legacy rag package (it reaches "
        f"rag.chat_pdfs via rag.engine.runtime.get_runtime()): {violations}"
    )


def test_adapters_directory_is_not_accidentally_empty():
    """Guards the guard: an empty/misnamed directory would make the test above vacuous."""
    assert len(_python_files(ADAPTERS_DIR)) >= 6  # one adapter module per port, at least


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
