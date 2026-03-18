"""
MonkeyGrab CLI package.

Encapsulates the entire CLI presentation layer of the system.
Uses `rich` for visual rendering.

Modules:
    display  -- Display class with Rich (singleton `ui`)
    app      -- Main command loop and CLI orchestration
"""

from rag.cli.app import MonkeyGrabCLI
from rag.cli.display import ui

__all__ = ["MonkeyGrabCLI", "ui"]
