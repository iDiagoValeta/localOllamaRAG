"""Unit tests for monkeygrab.adapters.embedding.jina_clip_worker wire framing.

Covers only the pipe protocol helpers (``_emit``/``_handle_request``/``_fatal``)
with a fake model -- no torch, no CUDA, no model download -- so these run in
milliseconds in the fast/architecture CI gate. ``_load_model``/``main`` are
never called here: spawning the real worker would block on a ~29s model load
and require a GPU.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from monkeygrab.adapters.embedding import jina_clip_worker as worker  # noqa: E402


class _FakeVector:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class _FakeModel:
    """Stands in for SentenceTransformer -- records encode kwargs, returns fixed vectors."""

    def __init__(self, values=None):
        self._values = list(values) if values is not None else [0.1, 0.2, 0.3]
        self.calls = []

    def encode(self, payload, **kwargs):
        self.calls.append((payload, kwargs))
        return _FakeVector(self._values)


class _FailingModel:
    def encode(self, payload, **kwargs):
        raise RuntimeError("backend boom")


def _last_message(capsys):
    out = capsys.readouterr().out
    assert out.endswith("\n"), f"framing must be one line terminated by newline: {out!r}"
    lines = out.strip().splitlines()
    assert len(lines) == 1, f"expected exactly one line-JSON message, got: {out!r}"
    return json.loads(lines[0])


def test_emit_writes_one_line_json(capsys):
    worker._emit({"id": 1, "ok": True, "vector": [0.1]})

    assert _last_message(capsys) == {"id": 1, "ok": True, "vector": [0.1]}


def test_handle_text_request_replies_ok_with_vector(capsys):
    model = _FakeModel([0.5, 0.6])

    worker._handle_request(json.dumps({"id": 7, "op": "text", "text": "hola"}), model)

    assert _last_message(capsys) == {"id": 7, "ok": True, "vector": [0.5, 0.6]}
    payload, kwargs = model.calls[0]
    assert payload == "hola"
    assert kwargs["truncate_dim"] == worker._TRUNCATE_DIM
    assert kwargs["normalize_embeddings"] is True


def test_handle_unknown_op_replies_ok_false_and_stays_alive(capsys):
    model = _FakeModel()

    worker._handle_request(json.dumps({"id": 3, "op": "audio"}), model)  # must not raise

    first = _last_message(capsys)
    assert first["id"] == 3
    assert first["ok"] is False
    assert "audio" in first["error"]

    worker._handle_request(json.dumps({"id": 4, "op": "text", "text": "sigue vivo"}), model)

    assert _last_message(capsys) == {"id": 4, "ok": True, "vector": [0.1, 0.2, 0.3]}


def test_handle_invalid_json_replies_with_id_none(capsys):
    worker._handle_request("not-json-at-all{", _FakeModel())

    message = _last_message(capsys)
    assert message["id"] is None
    assert message["ok"] is False
    assert "invalid JSON" in message["error"]


def test_handle_backend_error_replies_ok_false_without_raising(capsys):
    worker._handle_request(
        json.dumps({"id": 9, "op": "text", "text": "x"}), _FailingModel()
    )  # must not raise

    message = _last_message(capsys)
    assert message == {"id": 9, "ok": False, "error": "backend boom"}


def test_handle_image_with_missing_file_replies_ok_false(capsys, tmp_path):
    missing = str(tmp_path / "no-existe.png")

    worker._handle_request(json.dumps({"id": 5, "op": "image", "path": missing}), _FakeModel())

    message = _last_message(capsys)
    assert message["id"] == 5
    assert message["ok"] is False
    assert message["error"]


def test_fatal_emits_fatal_event_and_exits_1(capsys):
    with pytest.raises(SystemExit) as excinfo:
        worker._fatal("CUDA not available")

    assert excinfo.value.code == 1
    assert _last_message(capsys) == {"event": "fatal", "error": "CUDA not available"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
