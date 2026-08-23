---
name: Bug report
about: Something the pipeline, CLI, web UI or packaging does wrong
labels: bug
---

<!-- Conventions: CONTRIBUTING.md. An issue exists to be closable with evidence:
state the closure criterion in the opening post. -->

## What happened

<!-- Exact command, what you expected, what happened instead. Paste the full
traceback if there is one -- not a summary of it. -->

## Reproduction

<!-- Minimal steps from a clean state. If it needs a specific corpus or
settings.json state, say exactly which. -->

## Evidence

<!-- Logs, run JSONs, screenshots. Cite local artifacts explicitly as local
(e.g. tests/eval/runs/ is gitignored). -->

## Environment

- OS:
- GPU (`nvidia-smi`):
- `ollama list`:
- `.venv-mineru/` present: yes / no

## Closure criterion

<!-- What observable fact will make this closed: a test that reproduces and
passes, a measured number back in range, a run artifact showing the fix. -->
