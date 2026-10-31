<!--
Keep the order: summary first in product/system terms, then the issue link,
then how this was verified. Details and conventions: CONTRIBUTING.md.
-->
<!-- Base branch: `develop` for topic branches. Releases and hotfixes use
?template=release.md / ?template=hotfix.md and target `main`. -->
## Summary

- What changed, in product or system terms first; symbols as backup.

## Issue

Fixes #

Labels on this PR (they feed the release notes): one type, plus `needs-gate`
if retrieval or generation changed.

## Test plan

- [ ] `pytest` green locally (which subsets ran)
- [ ] `ruff check .` clean
- [ ] Frontend rebuilt if `rag/web/frontend/src` changed
- [ ] Docs updated in this same change (`README.md`, architecture READMEs,
      design docs) per `docs/README.md`
- [ ] Full gate run locally (`python tests/eval/run_eval.py`) if retrieval or generation changed; run id ______
- [ ] No protected-zone edits without prior agreement
      (`requirements*.txt`, flag defaults, `tests/eval/`, `tests/characterization/`)
