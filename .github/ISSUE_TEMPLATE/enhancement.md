---
name: Enhancement / design
about: A new capability, a behaviour change, or a measurement to make
labels: enhancement
---

<!-- Conventions: CONTRIBUTING.md. Evidence over opinion; an issue exists to
be closable with evidence -- state the closure criterion in the opening post. -->

## Problem

<!-- What a user or the project cannot do today, or what is measured and
wrong. Numbers beat adjectives. -->

## Proposed direction

<!-- The shape of the change: which layer (core use case, adapter, interface),
which ports, what it costs. If it touches product architecture or the
self-improvement loop, name the design doc it amends --
docs/design/2026-07-26-monkeygrab-v2.md or
docs/design/2026-07-28-loop-automejorable.md -- and note that a dated decision
block lands there BEFORE code moves. -->

## Constraints this repo will hold you to

- Hexagonal: new capability = use case + ports under `src/monkeygrab/`, not
  logic in the Flask/CLI layer.
- Hard-fail policy: no silent fallbacks inside adapters.
- Anything touching retrieval or generation needs the gold-case gate green.

## Closure criterion

<!-- What observable fact makes this closed: shipped behind which flag with
which tests, or measured how. -->
