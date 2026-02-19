"""
MonkeyGrab CLI — Interfaz profesional de terminal
===================================================

Paquete que encapsula toda la capa de presentación CLI del sistema
RAG dual MonkeyGrab, separando la interfaz de usuario de la lógica
de recuperación documental.

Módulos:
    theme    — Paleta de colores, tipografía y constantes visuales
    renderer — Renderizado de banners, spinners, tablas y streaming
    app      — Bucle principal de comandos y orquestación CLI
"""

from rag.cli.app import MonkeyGrabCLI

__all__ = ["MonkeyGrabCLI"]
