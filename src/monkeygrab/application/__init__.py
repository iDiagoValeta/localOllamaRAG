"""Application use cases orchestrating domain entities, ports and configuration.

``IndexCorpus``, ``Retrieve`` and ``Answer`` are the single implementations
used by the CLI, web app, desktop app and evaluation gate. Each receives its
ports and immutable ``AppConfig`` explicitly; infrastructure and mutable
runtime state remain outside this package.

The import boundary is enforced by
``tests/unit/test_architecture_boundaries.py``: application code may depend on
the standard library, domain, ports and config, never on concrete adapters.
Every use case returns a result dataclass containing its output and metrics.
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
