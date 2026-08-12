"""harness -- the block-C configuration search harness (issue #31).

Not product: it lives outside the hexagonal core (src/monkeygrab/) and the
interface layer (rag/), and only ever reads and runs the evaluation gate
under tests/eval/ -- never redefines how it scores. See harness/README.md.
"""
