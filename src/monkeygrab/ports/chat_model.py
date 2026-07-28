"""ChatModel -- Ollama-style text/vision generation, single-shot or streamed."""

from typing import Iterator, Optional, Protocol, Sequence

from monkeygrab.domain.generation_chunk import GenerationChunk


class ChatModel(Protocol):
    """One model role: generate text, optionally from images.

    Every pipeline role that talks to an LLM -- answer generation, chat,
    query decomposition, contextual enrichment, context synthesis, image
    description -- is a separate model name wired as its own instance. This
    port is the single capability "talk to one model"; choosing which model
    plays which role happens outside it.

    Two methods cover every use. ``generate`` returns the complete text and
    takes optional images, which only the vision role uses. ``stream`` yields
    ``GenerationChunk`` for the answer the user watches arrive. It yields
    chunks rather than plain strings because the final one carries the
    model name, stop reason, timings and token counts that the debug dump
    reports; an iterator of strings would drop all of that silently. A caller
    that only wants text reads ``chunk.text`` and ignores the rest.

    Sampling parameters -- temperature, prediction limits, context window,
    keep-alive -- are runtime tuning rather than part of what makes something
    a chat model, so they belong to the adapter and its configuration, not to
    this Protocol.

    Failure policy: hard-fail. Raise on any generation failure. Optional
    stages do degrade gracefully, but that is a decision the calling use case
    makes explicitly, visible in its own code -- never something an adapter
    does invisibly by returning empty output.
    """

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Sequence[bytes] = (),
    ) -> str:
        """Generate a complete response in one call.

        Args:
            prompt: User/task prompt.
            system: Optional system prompt.
            images: Optional raw image bytes for a vision-capable chat model.

        Returns:
            The complete generated text.

        Raises:
            Exception: On any generation failure.
        """
        ...

    def stream(self, prompt: str, *, system: Optional[str] = None) -> Iterator[GenerationChunk]:
        """Generate a response, yielding chunks as it is produced.

        Args:
            prompt: User/task prompt.
            system: Optional system prompt.

        Yields:
            One ``GenerationChunk`` per produced piece; concatenating every
            ``chunk.text`` in order is the complete response. Exactly one
            chunk (the last) has ``done=True`` and carries the generation
            metadata (``model``, ``done_reason``, timings, token counts) --
            see ``GenerationChunk``.

        Raises:
            Exception: On any generation failure.
        """
        ...
