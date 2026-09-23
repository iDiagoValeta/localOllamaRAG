"""``python -m rag.headless``: serve MonkeyGrab's pipeline on 127.0.0.1.

Without ``MONKEYGRAB_HEADLESS_TOKEN`` the service runs without auth on
loopback; exposing the port beyond loopback without a token is unsupported.
"""
import logging
import os
import sys
from typing import Optional

from rag.headless.app import create_app


def resolve_port(raw: Optional[str] = None) -> int:
    """Parse the headless port, failing with a usage message, not a traceback."""
    if raw is None:
        raw = os.getenv("MONKEYGRAB_HEADLESS_PORT", "5050")
    try:
        return int(raw)
    except ValueError:
        sys.exit(f"MONKEYGRAB_HEADLESS_PORT must be an integer port, got {raw!r}")


if __name__ == "__main__":
    port = resolve_port()
    token = os.getenv("MONKEYGRAB_HEADLESS_TOKEN", "") or None
    if token is None:
        # Bound to 127.0.0.1 below; the warning names the unsupported case
        # so an operator does not republish the port without auth.
        logging.warning("running without auth on loopback; exposing beyond loopback unsupported")
    create_app(token=token).run(host="127.0.0.1", port=port, threaded=True)
