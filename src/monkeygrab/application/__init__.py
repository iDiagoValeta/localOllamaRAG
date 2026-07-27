"""Application -- use cases orchestrating ports, domain and config.

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  +-- text_chunking.py       split_markdown_into_chunks, adjacent_chunk_ids
#  |                          -- moved from rag/engine/chunking.py
#  +-- rrf_fusion.py          fuse_semantic_and_keyword
#  |                          -- moved from rag/engine/retrieval.py's fusion block
#  +-- context_assembly.py    optimize_context_text, build_context_for_model, ...
#  |                          -- moved from rag/engine/context.py
#  +-- index_corpus.py        IndexCorpus  -- extract -> chunk -> contextualize? -> embed -> store
#  +-- retrieve.py            Retrieve     -- decompose? -> semantic+lexical -> RRF -> rerank? -> threshold
#  +-- answer.py              Answer       -- expand? -> char budget -> synthesize? -> generate
#
# ─────────────────────────────────────────────

This layer holds the algorithmic logic that today lives in ``rag/engine/``
entangled with the ``cfg = get_runtime()`` service locator (see
docs/design/2026-07-26-monkeygrab-v2.md, section 1). Every use case here
receives its ports and its ``AppConfig`` through the constructor and reads
config fresh at call time -- never captured in a default argument, which is
the structural fix for the bug documented in
``tests/characterization/test_stale_default_config_bug.py``.

Import boundary (enforced by ``tests/unit/test_architecture_boundaries.py``):
this package may import only the standard library, ``monkeygrab.domain``,
``monkeygrab.ports`` and ``monkeygrab.config`` -- never ``chromadb``,
``ollama``, ``fitz``, ``torch``, or any other infrastructure library, and
never a concrete adapter. Interfaces (CLI/web) are not wired to this layer
yet -- that is future work; today's CLI/web keep running on the old
``rag/chat_pdfs.py`` path unchanged.

**Equivalence.** This is a migration, not a redesign: every piece of logic
moved out of ``rag/engine/`` must produce results identical to the original
given the same inputs, including its documented surprises (see
``tests/characterization/``). ``tests/unit/application/`` proves this
piece by piece by running both the moved and the original implementation
side by side. Logic that could not be carried over 1:1 (because it depends
on infrastructure the new ports don't expose, e.g. image/OCR extraction, or
on pure-but-unlisted helper functions in modules outside this migration's
explicit scope, e.g. ``extraer_keywords``) is called out explicitly in each
use case's module docstring rather than silently invented or silently
dropped.

**Metrics.** ``rag/engine/`` mixes three different ways of surfacing
pipeline metrics: return-value tuples (``(resultado, metricas)``),
module-level ``logging.info`` calls gated by ``LOGGING_METRICAS``, and
nothing at all. This layer picks one shape and uses it everywhere: every use
case's ``run()`` returns a small ``...Result`` dataclass with the primary
output plus a ``metrics: Dict[str, Any]`` field. Pure helper functions that
used to log a side effect (e.g. ``construir_contexto_para_modelo``'s
char-savings log line) return their numbers as part of that dict instead --
a pure function in this layer never has a logging side effect. Deciding
whether/how to log or persist those metrics is left to whatever interface
layer eventually calls these use cases (mirroring how ``guardar_debug_rag``
today is a caller-side concern of ``generar_respuesta``, not of the pure
computation functions it wraps).
"""

from monkeygrab.application.answer import Answer, AnswerResult
from monkeygrab.application.index_corpus import IndexCorpus, IndexCorpusResult
from monkeygrab.application.retrieve import Retrieve, RetrieveResult

__all__ = [
    "Answer",
    "AnswerResult",
    "IndexCorpus",
    "IndexCorpusResult",
    "Retrieve",
    "RetrieveResult",
]
