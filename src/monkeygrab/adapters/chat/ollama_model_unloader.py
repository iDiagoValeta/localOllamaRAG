"""OllamaModelUnloader -- ModelUnloader adapter over Ollama's keep_alive=0."""

import logging
from typing import Optional, Set

import requests

from monkeygrab.config.models import ModelsConfig

# Not subclassed from monkeygrab.ports.model_unloader.ModelUnloader: Protocol
# conformance here is structural (duck typing), the same contract every other
# adapter in this package satisfies without inheriting its port.

_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaModelUnloader:
    """Best-effort VRAM reclaim: POSTs ``keep_alive=0`` to every configured
    Ollama model role except one.

    Literal port of ``liberar_modelos_ollama``/``_modelos_ollama_configurados``
    (``rag/engine/generation.py``): built once with every Ollama role's model
    name (de-duplicated -- several roles commonly
    share the same model, e.g. ``rag``/``chat``/``contextual`` all defaulting
    to ``gemma4:e4b``), so ``unload_all_except`` can ask the runtime to drop
    everything but the one about to run without needing to know model names
    at call time.
    """

    def __init__(
        self,
        models: ModelsConfig,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: int = 30,
    ):
        """Args:
            models: Model-role config; ``rag``, ``chat``, ``contextual`` and
                ``recomp`` are read.
            base_url: Ollama HTTP server base URL.
            timeout: Per-request HTTP timeout in seconds.
        """
        self._model_names: Set[str] = {
            m for m in (
                models.rag, models.chat, models.contextual, models.recomp,
            ) if m
        }
        self._base_url = base_url
        self._timeout = timeout

    def unload_all_except(self, keep: Optional[str] = None) -> None:
        """Ask Ollama to drop every configured model except ``keep``.

        Args:
            keep: Model name to leave loaded, or ``None`` to unload every
                configured model.
        """
        for name in self._model_names:
            if keep is not None and name == keep:
                continue
            try:
                requests.post(
                    f"{self._base_url}/api/generate",
                    json={"model": name, "keep_alive": 0},
                    timeout=self._timeout,
                )
            except Exception as exc:
                # Best-effort, matching liberar_modelos_ollama: an unload
                # failing (model already gone, transient network error, ...)
                # must never block the caller's own generation.
                logging.debug("Ollama unload %s: %s", name, exc)
