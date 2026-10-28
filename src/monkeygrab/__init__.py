"""MonkeyGrab's hexagonal core: domain, ports, application use cases, config.

The pipeline runs through it. Indexing, retrieval and generation each build
their adapters through ``monkeygrab.composition`` and ``rag/engine/wiring.py``
and call a use case here (``IndexCorpus``, ``Retrieve``, ``Answer``), which is
also what the evaluation gate constructs. ``rag/chat_pdfs.py`` remains the
public facade the web layer and tests import, and it delegates to this package.

See ``docs/design/2026-07-26-monkeygrab-v2.md`` for the architecture.
"""
