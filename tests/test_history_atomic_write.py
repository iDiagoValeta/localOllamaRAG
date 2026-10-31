"""Saving the chat history must not destroy the last good copy.

Issue #322. ``guardar_historial`` opened the history file in ``"w"`` mode, so a
failure halfway through ``json.dump`` -- a full disk, the process being killed
-- left a truncated file. ``cargar_historial`` silently discarded it and the
next save consolidated the loss.

The test interposes on ``json.dump`` at the exact moment it has written part of
the new payload and then raises, which is what a disk running out of space looks
like from inside the function. The previous file must survive that byte for
byte, and the save must still honour its best-effort contract (a warning, not an
exception).

Lives outside ``tests/unit`` because it imports ``rag.engine.history``, which
binds the runtime configuration from ``rag.chat_pdfs``; the fast CI gate's
``architecture`` job installs no infrastructure and only runs ``tests/unit``.
"""

import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rag.chat_pdfs  # noqa: F401  (owns the runtime history is bound to)
from rag.engine import history

_PREVIOUS = [{"role": "user", "content": "pregunta previa"}]


@pytest.fixture
def history_path(tmp_path, monkeypatch):
    """Point the engine at a real history file holding a known good value."""
    path = tmp_path / "historial_chat.json"
    path.write_text(
        json.dumps({"chat": _PREVIOUS}, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(history.cfg, "HISTORIAL_PATH", str(path))
    monkeypatch.setattr(history.cfg, "MAX_HISTORIAL_MENSAJES", 40)
    return path


def test_failed_save_mid_dump_leaves_the_previous_history_intact(
    history_path, monkeypatch, caplog
):
    before = history_path.read_bytes()

    def fail_after_partial_write(payload, stream, **kwargs):
        # Enough bytes to leave valid-looking JSON behind if they land in the
        # real file, which is precisely the corruption being fixed.
        stream.write('[{"role": "user", "content": "nueva')
        stream.flush()
        raise OSError("No space left on device")

    monkeypatch.setattr(history.json, "dump", fail_after_partial_write)

    with caplog.at_level(logging.WARNING):
        history.guardar_historial([{"role": "user", "content": "nueva"}])

    assert history_path.read_bytes() == before
    assert history.cargar_historial() == _PREVIOUS
    assert any(
        "Error saving history" in record.getMessage() for record in caplog.records
    )
    # A partial temporary file left behind would be dead weight on every later
    # save; the failed attempt has to clean up after itself.
    assert [entry.name for entry in history_path.parent.iterdir()] == [
        history_path.name
    ]
