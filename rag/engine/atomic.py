"""Atomic JSON writes for the small state files the web app keeps on disk."""
import json
import os
import tempfile
from typing import Any


def escribir_json_atomico(ruta: str, datos: Any) -> None:
    """Write ``datos`` as JSON to ``ruta`` without ever leaving it truncated.

    The payload goes to a temporary file next to the target and is swapped in
    with ``os.replace``, so a failure halfway through (full disk, killed
    process) leaves the previous file untouched. Readers of these files fall
    back to defaults on invalid JSON, so a truncated file would silently erase
    the user's state rather than fail.

    Args:
        ruta: Destination path; its directory must exist.
        datos: JSON-serializable payload.

    Raises:
        OSError: If the temporary file cannot be written or swapped in.
    """
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(ruta) or ".",
            prefix=f"{os.path.basename(ruta)}.",
            suffix=".tmp",
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)
            # Surface deferred write errors (disk full) before replacing the
            # last good copy; os.replace alone would happily swap in a file
            # whose data never reached the disk.
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, ruta)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
