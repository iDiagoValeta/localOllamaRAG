"""ImageEmbedder -- image (optionally captioned) to embedding vector."""

from typing import List, Protocol


class ImageEmbedder(Protocol):
    """Embeds one image into the same vector space as this embedder's text.

    Kept separate from ``Embedder`` because text and image inputs have
    different method contracts. The production Jina CLIP adapter satisfies
    both protocols and places both modalities in one aligned vector space.

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
