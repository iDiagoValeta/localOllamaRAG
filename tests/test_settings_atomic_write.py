"""settings.json is never left truncated by a failed save (issue #452).

Loose test: rag.engine.settings imports the engine runtime.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rag.chat_pdfs  # noqa: F401 -- builds the runtime the settings module reads
from rag.engine import atomic, settings


def test_a_failed_save_keeps_the_previous_settings(monkeypatch, tmp_path):
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({"active_store": "es"}), encoding="utf-8")
    monkeypatch.setattr(settings, "ruta_ajustes", lambda: str(target))

    def _dies_mid_write(data, f, **kwargs):
        f.write('{"roles": {')
        raise OSError("disk full")

    monkeypatch.setattr(atomic.json, "dump", _dies_mid_write)

    settings.guardar_ajustes_persistidos()

    assert json.loads(target.read_text(encoding="utf-8")) == {"active_store": "es"}
    assert [p.name for p in tmp_path.iterdir()] == ["settings.json"]
