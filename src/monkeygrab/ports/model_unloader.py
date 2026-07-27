"""ModelUnloader -- best-effort VRAM reclaim across every configured model role."""

from typing import Optional, Protocol


class ModelUnloader(Protocol):
    """Asks the runtime to drop every configured model except one.

    Matches ``liberar_modelos_ollama(excepto=None)`` in
    ``rag/engine/generation.py``: on the target 8GB-VRAM GPU, the RAG
    generator does not reliably fit alongside whatever embedding/contextual/
    RECOMP/OCR model a previous pipeline stage left resident, so
    ``_ollama_generate_stream`` unloads every *other* configured role's
    model (``keep_alive=0``) right before it starts streaming, and again
    (this time unloading everything, including the model that just failed)
    before retrying after a 5xx response.

    This needs its own port rather than living on ``ChatModel`` because it
    requires knowing every configured role's model name at once
    (``_modelos_ollama_configurados()`` in ``rag/engine/generation.py``,
    built from all six ``ModelsConfig`` role fields) -- information a
    single-role ``ChatModel`` adapter does not have by construction (see
    ``ChatModel``'s port docstring: "one Ollama-backed model role"). A
    ``ChatModel`` adapter that wants this behavior (``OllamaChatModel``
    does, at exactly the two call sites above) takes a ``ModelUnloader`` as
    an optional constructor dependency instead.

    Failure policy: best-effort, matching the original -- a request to
    unload one model failing (that model already unloaded, a transient
    network error, ...) must never prevent the caller's own generation from
    proceeding. Individual unload failures are swallowed inside the
    adapter; this method itself never raises.
    """

    def unload_all_except(self, keep: Optional[str] = None) -> None:
        """Ask the runtime to drop every configured model except ``keep``.

        Args:
            keep: Model name to leave loaded (the one about to run), or
                ``None`` to unload every configured model, including it
                (matches ``liberar_modelos_ollama()``'s no-argument call
                before a retry).
        """
        ...
