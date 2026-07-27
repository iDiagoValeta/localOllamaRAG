"""ContextConfig -- character budget for the evidence sent to the generator.

# ─────────────────────────────────────────────
# SECTION 1: CONFIG
# ─────────────────────────────────────────────

Default is a copy of ``rag/chat_pdfs.py`` section 3.7 (``MAX_CONTEXTO_CHARS``).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextConfig:
    """Parameter ``rag.engine.generation._limitar_fragmentos_por_chars`` reads.

    Attributes:
        max_context_chars: Maximum total characters of retrieved evidence
            handed to context synthesis/formatting; fragments beyond this
            budget (consumed in ranked order) are dropped (``MAX_CONTEXTO_CHARS``).
    """

    max_context_chars: int = 24000
