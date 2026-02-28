"""
MonkeyGrab CLI
=======================================================

Paquete que encapsula toda la capa de presentación CLI del sistema.
Usa `rich` para renderizado visual.

Módulos:
    display  — Clase Display con Rich (singleton `ui`)
    app      — Bucle principal de comandos y orquestación CLI
"""

from rag.cli.app import MonkeyGrabCLI
from rag.cli.display import ui

__all__ = ["MonkeyGrabCLI", "ui"]
