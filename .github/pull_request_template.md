<!--
Keep the order: summary first in product/system terms, then the issue link,
then how this was verified. Details and conventions: CONTRIBUTING.md.
-->
## Summary

- What changed, in product or system terms first; symbols as backup.

## Issue

Fixes #

## Test plan

- [ ] `pytest` green locally (which subsets ran)
- [ ] `ruff check .` clean
- [ ] Frontend rebuilt if `rag/web/frontend/src` changed
- [ ] Docs updated in this same change (`README.md`, architecture READMEs,
      design docs) per `docs/README.md`
- [ ] Full gate (`full-eval.yml`) run if retrieval or generation changed
- [ ] No protected-zone edits without prior agreement
      (`requirements*.txt`, flag defaults, `tests/eval/`, `tests/characterization/`)
