---
name: Measurement / campaign finding
about: Something a gate run, probe or harness campaign measured, and what it implies
labels: eval
---

<!-- Conventions: CONTRIBUTING.md. This template exists because the two others
did not fit the work this repo mostly does: #30, #102 and #107 each report a
measurement, cite a local gitignored artifact, and propose options rather than
a direction -- and each was written free-form, so none of them look alike.

Swap the `eval` label for `loop` (or carry both) per CONTRIBUTING.md's issue
standard: `eval` is the measurement, `loop` is the optimiser built on it. -->

## What was measured

<!-- The number first, then the setup. Date, machine and stack, because a
measurement without them cannot be reproduced or aged out. A table beats a
paragraph when there is more than one figure:

| when | config | result |
|---|---|---|
| 2026-08-19 | healthy, descriptions off | 44/51 = 0.8627 |
-->

## Artifact

<!-- Which run produced it. `tests/eval/runs/` and `harness/ledger/` are
gitignored: say so explicitly and give the path anyway, so the machine that
holds it can reproduce the reading. Cited, not shipped.

Example: tests/eval/runs/20260729T020233Z_mineru-jina_clip-faiss.json (local) -->

## What it implies

<!-- The consequence, stated so it can be argued with. If the measurement
contradicts something already written down -- a design doc section, a README
claim, another issue's assumption -- name it here; that contradiction is
usually the actual finding. If it is a limit rather than a defect, say which
of the two it is. -->

## Options

<!-- Numbered, each with its cost, and say which way you lean and why. Not
deciding here is fine and often correct; deciding here without the cost of
each option is not. -->

## Closure criterion

<!-- What observable fact closes this: a measurement back in range on a named
run, a test that reproduces the effect and passes, or a written decision
including "won't do" with its reasoning.

If it CANNOT be closed autonomously -- a human audit against source PDFs, an
owner decision on a protected zone (`requirements*.txt`, flag defaults,
`tests/eval/`, `tests/characterization/`) -- state that here, in the opening
post, the way #30 does. An agent that discovers the gate halfway through has
already spent the budget. -->
