"""Repo tooling that is not product and not the harness.

Exists so ``tests/unit/test_setup_environments.py`` can import
``tools.setup_environments`` by name instead of loading it by path (issue
#120). Nothing under ``src/monkeygrab/`` or ``rag/`` imports anything here,
and the architecture boundary test would fail if that changed.
"""
