"""Runtime configuration access for the split RAG engine modules.

Each engine module binds ``cfg = get_runtime()`` once at import and then reads
configuration lazily as ``cfg.NAME``. This keeps rag.chat_pdfs the single owner
of runtime state while letting web/CLI/eval toggles and test monkeypatches take
effect immediately (``cfg`` is a live reference to the owning module).
"""

import sys
from types import ModuleType

_RUNTIME_MODULE = "rag.chat_pdfs"


def get_runtime() -> ModuleType:
    """Return the module that owns runtime configuration (rag.chat_pdfs).

    Handles the direct-run scenario where ``python chat_pdfs.py`` registers the
    module as ``__main__`` instead of ``rag.chat_pdfs``; it is identified by a
    distinctive constant defined before the engine imports fire.
    """
    module = sys.modules.get(_RUNTIME_MODULE)
    if module is None:
        main = sys.modules.get("__main__")
        if main is not None and hasattr(main, "MODELO_RAG"):
            return main
        import rag.chat_pdfs as module  # type: ignore[no-redef]
    return module
