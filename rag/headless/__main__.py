"""``python -m rag.headless``: serve MonkeyGrab's pipeline on 127.0.0.1."""
import os

from rag.headless.app import create_app

if __name__ == "__main__":
    port = int(os.getenv("MONKEYGRAB_HEADLESS_PORT", "5050"))
    token = os.getenv("MONKEYGRAB_HEADLESS_TOKEN", "") or None
    create_app(token=token).run(host="127.0.0.1", port=port, threaded=True)
