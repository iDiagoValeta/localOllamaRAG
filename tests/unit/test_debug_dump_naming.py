"""Two debug dumps in the same second must not overwrite each other (#398).

``guardar_debug_rag`` named the dump with a second-resolution ``strftime``,
so two RAG queries answered within the same second produced one file: the
second write replaced the first, and its JSON sidecar along with it. These
tests freeze the clock at that worst case and assert every call leaves its
own dump.

Imports ``rag.chat_pdfs`` through ``rag.engine.debug``, so the dependency-free
fast CI gate skips this file -- see ``tests/conftest.py``.
"""

import time

import pytest

# rag.engine.debug resolves its runtime config by importing rag.chat_pdfs,
# which imports the debug module back; loading the facade first breaks that
# cycle the same way the product does.
import rag.chat_pdfs  # noqa: F401
from rag.engine import debug


@pytest.fixture
def debug_dump_dir(tmp_path, monkeypatch):
    """Enable dumping into an empty folder with a clock that never advances."""
    monkeypatch.setattr(debug.cfg, "GUARDAR_DEBUG_RAG", True)
    monkeypatch.setattr(debug.cfg, "CARPETA_DEBUG_RAG", str(tmp_path))
    monkeypatch.setattr(time, "strftime", lambda _fmt: "20260101_000000")
    monkeypatch.setattr(time, "time_ns", lambda: 0)
    return tmp_path


def test_same_question_in_the_same_second_writes_one_dump_per_query(debug_dump_dir):
    debug.guardar_debug_rag("¿Qué es X?", respuesta="primera respuesta")
    debug.guardar_debug_rag("¿Qué es X?", respuesta="segunda respuesta")

    dumps = sorted(debug_dump_dir.glob("*.txt"))
    assert len(dumps) == 2
    contenido = [dump.read_text(encoding="utf-8") for dump in dumps]
    assert any("primera respuesta" in texto for texto in contenido)
    assert any("segunda respuesta" in texto for texto in contenido)


def test_the_json_sidecar_is_unique_too(debug_dump_dir):
    debug.guardar_debug_rag("¿Qué es X?", respuesta="primera respuesta")
    debug.guardar_debug_rag("¿Qué es X?", respuesta="segunda respuesta")

    assert len(list(debug_dump_dir.glob("*.json"))) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
