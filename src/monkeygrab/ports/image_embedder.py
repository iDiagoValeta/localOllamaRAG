"""ImageEmbedder -- image (optionally captioned) to embedding vector."""

from typing import List, Protocol


class ImageEmbedder(Protocol):
    """Embeds one image into the same vector space as this embedder's text.

    Deliberately a SEPARATE Protocol from ``Embedder``, not an extra method
    on it. ``Embedder`` is the contract every text embedder in this project
    satisfies, including ``OllamaEmbedder`` -- and Ollama's embedding models
    are text-only, so a figure has to be captioned by a vision model first
    and only the caption text goes through ``Embedder.embed`` (see the
    ``ChatModel`` "ocr" role). Adding ``embed_image`` to ``Embedder`` would
    either force a text-only adapter to implement a method it cannot honor,
    or silently narrow what the base contract promises. Only a genuinely
    multimodal embedder (``StackConfig.is_multimodal``, see
    ``monkeygrab.config.stack``) also satisfies ``ImageEmbedder``; callers
    check that flag before deciding whether a figure needs a vision-model
    caption or can be embedded directly -- the same place they would check
    for any other capability that not every adapter behind ``Embedder`` has.

    Combining an image with its caption is part of THIS port's contract,
    not left to the caller, because how to combine two vectors from the same
    model into one (raw magnitude vs. re-normalized, sum vs. average) is
    model- and adapter-specific -- exactly the kind of detail a port exists
    to hide behind one call. A caption is used only when it is real text
    (non-empty after stripping); an image with no caption is embedded alone.

    Failure policy: hard-fail, same as every port in this project -- a
    missing image file, a backend failure, or an unexpected vector
    dimension raises. Nothing here degrades to a zero vector or silently
    skips the image.
    """

    def embed_image(self, image_path: str, *, caption: str = "") -> List[float]:
        """Embed the image at ``image_path``, optionally combined with ``caption``.

        Args:
            image_path: Path to the image file on disk.
            caption: Caption text found near the image, if any. Empty (the
                default -- also what ``ExtractedImage.caption`` is when no
                caption was found near an image) means "no caption": the
                image is embedded alone. A non-empty caption is combined
                into the returned vector by the adapter's own rule.

        Returns:
            The embedding vector, same dimensionality and space as this
            embedder's ``embed`` (text) output.

        Raises:
            Exception: On any embedding failure -- missing file, backend
                error, or unexpected vector dimension.
        """
        ...
