"""Ports -- Protocols the application layer will depend on, adapters implement.

Every Protocol here has exactly one job and takes/returns only domain types
(``monkeygrab.domain``) and primitives -- never ``faiss.Index``,
``ollama.Client``, or any other infrastructure type. This is what lets an
adapter be replaced by rewiring dependency injection, not by touching a use case.

Hard-fail policy (project-wide, see docs/design/2026-07-26-monkeygrab-v2.md
section 3 "Politica de fallos"): every adapter implementing a port here
raises on failure instead of degrading -- no silent reranker CUDA-to-CPU-to
-disabled, no silent extraction fallback, no silent RECOMP-to-raw
-context fallback. A caller that wants a fallback chain builds it explicitly
out of two ports and a decision, not by one adapter secretly trying a second
strategy. Each Protocol's docstring restates this for its own failure modes.

No names are re-exported here: import each Protocol from its own module
(``monkeygrab.ports.embedder`` and so on), which also keeps this package's
surface equal to the files a reader can see.
"""
