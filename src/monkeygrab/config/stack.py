"""StackConfig -- which implementation stands behind each port.

Ports make an implementation replaceable in principle; this module is what makes
it replaceable in practice. Without a selector, comparing two technologies means
editing the code that wires them, which is expensive enough that the comparison
never happens and the choice stays a matter of opinion.

With it, running the same gold-case gate against pymupdf+Chroma and against
MinerU+FAISS is a change of environment variable and a second run, so "is this
actually better" becomes a measurement.

An unknown value raises at construction. Coercing it to a default is how the
previous configuration hid the fact that its MinerU path did not exist at all.
"""

from dataclasses import dataclass

from monkeygrab.config import env

# ─────────────────────────────────────────────
# SECTION 1: CHOICES
# ─────────────────────────────────────────────

# Text extraction. pymupdf is fast and loses table structure; mineru is slow and
# preserves tables as HTML and formulas as LaTeX.
EXTRACTOR_PYMUPDF = "pymupdf"
EXTRACTOR_MINERU = "mineru"
EXTRACTORS = (EXTRACTOR_PYMUPDF, EXTRACTOR_MINERU)

# Vector store. Both are exact search at this corpus size; faiss owns only the
# index while chroma also owns storage and metadata.
VECTOR_STORE_CHROMA = "chroma"
VECTOR_STORE_FAISS = "faiss"
VECTOR_STORES = (VECTOR_STORE_CHROMA, VECTOR_STORE_FAISS)

# Embeddings. ollama is text-only, so a figure must first be captioned by a
# vision model; jina_clip puts text and images in one space, which is the whole
# point of the multimodal path.
EMBEDDER_OLLAMA = "ollama"
EMBEDDER_JINA_CLIP = "jina_clip"
EMBEDDERS = (EMBEDDER_OLLAMA, EMBEDDER_JINA_CLIP)


# STACKCONFIG


@dataclass(frozen=True)
class StackConfig:
    """Which adapter implements each swappable port.

    Attributes:
        extractor: Text extraction backend, one of ``EXTRACTORS``.
        vector_store: Vector index backend, one of ``VECTOR_STORES``.
        embedder: Embedding backend, one of ``EMBEDDERS``.
    """

    extractor: str = EXTRACTOR_PYMUPDF
    vector_store: str = VECTOR_STORE_CHROMA
    embedder: str = EMBEDDER_OLLAMA

    @property
    def is_multimodal(self) -> bool:
        """True when the embedder places text and images in a shared space.

        Callers use this to decide whether figures need a vision model to
        describe them first, or can be embedded directly.
        """
        return self.embedder == EMBEDDER_JINA_CLIP

    @property
    def slug(self) -> str:
        """Short identifier of this combination, for index paths and reports.

        Two stacks produce incompatible vectors, so their indexes must never
        share a directory; embedding this in the path makes that structural
        rather than a thing to remember.
        """
        return f"{self.extractor}-{self.embedder}-{self.vector_store}"


# CONSTRUCTION


def stack_from_env() -> StackConfig:
    """Build a StackConfig from the process environment.

    Returns:
        StackConfig with the selected backends.

    Raises:
        ValueError: If any selector holds a value that is not implemented. The
            default stays the current production stack, so an unset environment
            changes nothing.
    """
    return StackConfig(
        extractor=env.read_env_choice("PDF_EXTRACTOR", EXTRACTOR_PYMUPDF, EXTRACTORS),
        vector_store=env.read_env_choice("VECTOR_STORE", VECTOR_STORE_CHROMA, VECTOR_STORES),
        embedder=env.read_env_choice("EMBEDDER", EMBEDDER_OLLAMA, EMBEDDERS),
    )
