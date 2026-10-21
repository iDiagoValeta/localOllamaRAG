# Branches

Gitflow. `develop` is the default branch, so `Fixes #N` in a PR closes the issue when it merges.

| Branch | Off | Into | Merge style |
|---|---|---|---|
| `<type>/issue-<n>-<slug>` | `develop` | `develop` | squash, then delete the branch |
| `release/vX.Y.Z` | `develop` | `main`, then back into `develop` | merge commit |
| `hotfix/vX.Y.Z` | `main` | `main`, then back into `develop` | merge commit |
| `rag-lite` | long-lived variant | its own topic branches | squash |

- Topic type is `feat`, `fix`, `cleanup`, `probe` or `docs`. Use `issue-0-<slug>` only when no tracking issue exists, then open the issue first.
- Slugs are lowercase ASCII with hyphens. One concern per branch.
- `main` holds only what was released: every commit on it is a merge of a `release/` or `hotfix/` branch and carries the tag `vX.Y.Z`.
- Nothing is pushed directly to `main` or `develop`. Both are protected: a PR, the fast gate green, no force-push.
- A release branch takes only `fix/` and `docs/` branches. New features wait for the next release.
- Release and hotfix PRs use merge commits, not squash: squashing would make `main` and `develop` diverge on every release and force a conflict back-merge.
- `rag-lite` diverged from `main` on purpose (4 GB profile, AGENTS.md section 3). It is not on the release line; sync it from `develop` by PR when the shared code changes.

`tools/check_branch_policy.py` encodes this table and runs as the required check `Branch policy` on every PR. Change the table and the script together; `tests/unit/test_branch_policy.py` pins both.
