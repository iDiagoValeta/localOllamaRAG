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

Also resets a module-level leak in rag.chat_pdfs before the first test runs
(see the fixture below) -- unrelated to the skip logic above, but this is the
one conftest.py both environments load, and the leak needs to be undone
before anything else in the session observes it.

Dependencies: stdlib only (ast, importlib.util) plus pytest -- this file must
import in an environment with nothing else available.
"""

import ast
import copy
import importlib.util
from pathlib import Path

import pytest

# What importing rag.chat_pdfs pulls in, directly or through the adapters its
# pipeline entry points wire up. Any one missing means the engine cannot be
# imported at all. Erring on the strict side only skips tests that would have
# run; being too lax would let collection fail outright.
_ENGINE_REQUIREMENTS = (
    "requests",
    "faiss",
    "ollama",
    "PIL",
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
    exists to wrap a library, so importing one imports FAISS, Ollama or
    sentence-transformers even when the test doubles the object afterwards. Everything
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


def _snapshot_mutable_globals(module) -> dict[str, object]:
    """Capture data-valued module globals, including private defaults.

    Tests may monkeypatch or directly assign configuration values beyond the
    public runtime mutators. Restoring only the known mutator surface leaves
    those values alive for the rest of the process, so the isolation boundary
    snapshots the complete uppercase data namespace instead.
    """
    snapshot = {}
    for name, value in vars(module).items():
        if not (name.isupper() or name.lstrip("_").isupper()):
            continue
        try:
            snapshot[name] = copy.deepcopy(value)
        except (TypeError, ValueError):
            # Imported objects are not configuration state and cannot be
            # meaningfully copied; the mutable values in this module are data.
            continue
    return snapshot


def _restore_mutable_globals(module, snapshot: dict[str, object]) -> None:
    """Restore a snapshot and remove uppercase globals added by a test."""
    current_names = set(vars(module))
    for name in current_names - set(snapshot):
        if name.isupper() or name.lstrip("_").isupper():
            delattr(module, name)
    for name, value in snapshot.items():
        setattr(module, name, copy.deepcopy(value))


if _engine_importable():
    collect_ignore = []
else:
    collect_ignore = [
        str(path.relative_to(_HERE))
        for path in sorted(_HERE.rglob("test_*.py"))
        if _imports_engine(path)
    ]


@pytest.fixture(autouse=True, scope="session")
def _restore_chat_pdfs_settings_globals():
    """Undo import-time settings mutations before the first test runs.

    Collection imports every test module up front, before any test-scoped
    fixture gets a chance to run. Any file that imports rag.web.app triggers
    its top-level ``cargar_ajustes_persistidos()`` call, which applies the
    machine's gitignored settings.json to shared ``rag.chat_pdfs`` globals.
    The session initializer puts the known persisted choices back before the
    per-test snapshot boundary is established.
    """
    try:
        import rag.chat_pdfs as rag_engine
    except ImportError:
        return None
    rag_engine.set_docs_folder_runtime(None)
    rag_engine._restaurar_roles_y_flags_por_defecto()
    return rag_engine


@pytest.fixture(autouse=True)
def _restore_chat_pdfs_mutable_globals(_restore_chat_pdfs_settings_globals):
    """Restore every mutable chat_pdfs global around each test.

    The public mutators cover paths, roles, and pipeline flags, but tests can
    also touch lower-level constants directly. A complete snapshot prevents a
    test's direct assignments (and nested edits to configuration maps) from
    becoming an order-dependent input to the next test.
    """
    rag_engine = _restore_chat_pdfs_settings_globals
    if rag_engine is None:
        yield
        return

    snapshot = _snapshot_mutable_globals(rag_engine)
    try:
        yield
    finally:
        _restore_mutable_globals(rag_engine, snapshot)
