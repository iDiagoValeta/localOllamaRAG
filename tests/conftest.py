"""Skip engine-dependent tests when the engine's dependencies are absent.

Some tests import ``rag.*`` by design. The equivalence tests are the clearest
case: they run the new implementation and the original engine function side by
side on the same input, so importing the engine *is* the point. Adapter tests do
the same, since wrapping a library means importing it.

Those imports fail at collection time, before any marker or ``-m`` filter gets a
say, so the fast CI gate — which deliberately installs no project dependencies —
cannot simply deselect them. It needs them not to be collected at all.

Rather than maintain a list of paths in the workflow, which drifts the moment a
test is added, each test file is inspected for an ``import rag`` and skipped when
the engine is unavailable. The same pytest command then works in both
environments: everything runs where the stack is installed, and only the pure
layers run where it is not.

Dependencies: stdlib only (ast, importlib.util) -- this file must import in an
environment with nothing but pytest available.
"""

import ast
import importlib.util
from pathlib import Path

# Imported at module scope by rag/chat_pdfs.py; any one of them missing means the
# engine cannot be imported at all.
_ENGINE_REQUIREMENTS = (
    "requests",
    "chromadb",
    "ollama",
    "fitz",
    "pymupdf4llm",
    "rank_bm25",
    "sentence_transformers",
)

_HERE = Path(__file__).parent


def _engine_importable() -> bool:
    """Return True when every module rag.chat_pdfs imports is installed."""
    for name in _ENGINE_REQUIREMENTS:
        try:
            if importlib.util.find_spec(name) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def _is_infrastructure_import(module: str) -> bool:
    """Return True for the two module paths that lead to third-party libraries.

    ``rag`` is the old engine, whose facade imports the whole stack at module
    level. ``monkeygrab.adapters`` is infrastructure by definition: an adapter
    exists to wrap a library, so importing one imports chromadb, ollama or
    pymupdf4llm even when the test doubles the object afterwards. Everything
    else under ``monkeygrab`` is pure and safe to collect anywhere.
    """
    if not module:
        return False
    root = module.split(".")[0]
    return root == "rag" or module.startswith("monkeygrab.adapters")


def _imports_engine(path: Path) -> bool:
    """Return True if ``path`` reaches infrastructure through its imports.

    Uses the syntax tree rather than a text search, so a mention in a docstring
    does not count and an import nested inside a helper function does.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(_is_infrastructure_import(alias.name) for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if _is_infrastructure_import(node.module or ""):
                return True
    return False


if _engine_importable():
    collect_ignore = []
else:
    collect_ignore = [
        str(path.relative_to(_HERE))
        for path in sorted(_HERE.rglob("test_*.py"))
        if _imports_engine(path)
    ]
