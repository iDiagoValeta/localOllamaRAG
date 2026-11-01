# Releases

## Normal change

- Topic branch off `develop`, squash-merge from a green PR, delete the branch.
- User-visible changes carry README or arch-doc updates in the same PR.
- Retrieval or generation changes need the full eval gate green before merge (label `needs-gate`). The gate runs locally (`python tests/eval/run_eval.py --models <model...>`), not in CI, and the run id goes in the PR.
- A PR whose label set is `bug`, `enhancement`, ... feeds the release notes automatically (`.github/release.yml` groups by label), so label the PR, not just the issue.

## Cutting a release

1. `git switch -c release/vX.Y.Z develop`. Bump the version where the packaging reads it, move `CHANGELOG.md`'s `Unreleased` block under the new version with its date.
2. Run the full gate locally on the release branch, on a GPU machine with Ollama: `python tests/eval/run_eval.py --models <model...>`. No self-hosted runner is registered, so [`full-eval.yml`](../../.github/workflows/full-eval.yml) cannot run. The person cutting the release runs it. The figure and the run id go in the release PR, the release notes and `docs/model-history.md`. A release without a gate run says so in its notes.
3. Fixes found while stabilising land on the release branch through `fix/issue-<n>-<slug>` PRs.
4. Open the PR `release/vX.Y.Z` into `main` using the release PR template. Merge with a merge commit, then tag the merge: `git tag -a vX.Y.Z -m "MonkeyGrab X.Y.Z"` and publish the GitHub release (generate notes, then edit them by hand).
5. Open the back-merge PR `release/vX.Y.Z` into `develop` and merge it with a merge commit. Delete the release branch.

## Hotfix

1. `git switch -c hotfix/vX.Y.Z main` (patch bump). The fix carries a regression test.
2. PR into `main`, merge commit, tag, release notes.
3. Back-merge PR into `develop`. If a `release/` branch is open, merge the hotfix into it too.

## Changelog

`CHANGELOG.md` follows Keep a Changelog. It is the short index; the GitHub release notes keep the measured detail (gate figures, migration notes) and the changelog links to them. Add the line to `Unreleased` in the PR that makes the change, not at release time.
