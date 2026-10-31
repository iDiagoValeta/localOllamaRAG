"""Application use cases orchestrating domain entities, ports and configuration.

``IndexCorpus``, ``Retrieve`` and ``Answer`` are the single implementations
used by the CLI, web app, desktop app and evaluation gate. Each receives its
ports and immutable ``AppConfig`` explicitly; infrastructure and mutable
runtime state remain outside this package.

The import boundary is enforced by
``tests/unit/test_architecture_boundaries.py``: application code may depend on
the standard library, domain, ports and config, never on concrete adapters.
Every use case returns a result dataclass containing its output and metrics.

No names are re-exported here: import each use case or helper from its own
module (``monkeygrab.application.answer`` and so on), so the package surface
matches the files a reader can see.
"""
