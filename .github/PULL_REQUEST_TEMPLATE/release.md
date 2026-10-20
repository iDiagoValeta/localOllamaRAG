<!-- Release PR: release/vX.Y.Z into main. Merge with a MERGE COMMIT, not squash
(docs/standards/releases.md). Open it with ?template=release.md -->
## Release vX.Y.Z

Summary of what ships, in product terms.

## Checklist

- [ ] Version bumped where packaging reads it; `CHANGELOG.md` `Unreleased` moved under this version with its date
- [ ] Full gate (`full-eval.yml`) run on this branch: pass rate ______ (floor ______), run link ______
- [ ] `docs/model-history.md` has the row for this run
- [ ] README and architecture docs match the shipped behaviour
- [ ] Migration notes written if an index fingerprint, a setting or an API changed
- [ ] After merge: tag `vX.Y.Z`, publish release notes, back-merge into `develop`
