# Contributing

The operating contract for this repo — humans and agents alike — is
[`AGENTS.md`](AGENTS.md). This file covers the mechanics: how to branch, what
a change must carry, and how issues and PRs are run. The documentation
standard (what gets documented where) is [`docs/README.md`](docs/README.md).

Detail standards: [code style](docs/standards/code-style.md), [comments](docs/standards/comments.md), [commits](docs/standards/commits.md), [branches](docs/standards/branches.md), [tests](docs/standards/tests.md), [security](docs/standards/security.md), [releases](docs/standards/releases.md). Index: [Standards](docs/standards/README.md).

## Setup

```bash
python tools/setup_environments.py                     # .venv + .venv-mineru, then verify
cd rag/web/frontend && pnpm install && pnpm run build   # frontend (pnpm only)
```

`python tools/setup_environments.py --check` answers "is this machine set up?"
without installing anything, and names the component that is missing rather
than the whole setup.

Ollama must be running locally with a generator model pulled. MinerU lives in
an isolated `.venv-mineru/` interpreter; see the root `README.md`.

## Two gates, know which one you need

- **Fast gate** (`ci.yml`, every PR): lint (`ruff`), architecture dependency
  rules, unit suite against test doubles, harness tests, frontend build. No
  GPU, no models.
- **Full gate** (`tests/eval/run_eval.py`, run locally, not in CI): the real
  pipeline against every gold case. **Required before merging anything that
  touches retrieval or generation, and before every release.** The fast gate
  never exercises a real model. It needs Ollama and a GPU, which no hosted
  runner has, and no self-hosted runner is registered for this repository, so
  [`full-eval.yml`](.github/workflows/full-eval.yml) cannot run. Whoever merges
  the change or cuts the release runs it on a GPU machine:

  ```bash
  python tests/eval/run_eval.py --models <model...>
  ```

  Details: [`tests/eval/README.md`](tests/eval/README.md).

## Workflow

Gitflow, with `develop` as the default branch so `Fixes #N` closes issues when
a PR merges. Names, merge strategy and the release procedure:
[Branches](docs/standards/branches.md) and [Releases](docs/standards/releases.md).

| Branch | Role | Receives | Merge style |
|---|---|---|---|
| `main` | What was released; every commit is a tag candidate | only `release/*` and `hotfix/*` | merge commit, then tag |
| `develop` | Integration, default branch | topic branches | squash |
| `<type>/issue-<n>-<slug>` | One concern, off `develop` | nothing | deleted after merge |
| `release/vX.Y.Z` | Stabilisation of a release, off `develop` | `fix/` and `docs/` only | merge commit into `main`, back-merge into `develop` |
| `hotfix/vX.Y.Z` | Urgent fix to a release, off `main` | the fix | merge commit into `main`, back-merge into `develop` |
| `rag-lite` | Long-lived 4 GB variant, not on the release line | its own topic branches | squash |

1. Branch off `develop`. `tools/check_branch_policy.py` runs as a required
   check on every PR and rejects a wrong head/base pair (a topic branch aimed
   at `main`, a branch name that is not `<type>/issue-<n>-<slug>`).
2. One concern per branch. If your working tree mixes two concerns, commit
   and PR them separately — review and revert both depend on it.
3. Every change carries its own tests and its own doc updates in the same PR
   (rule 10 of `AGENTS.md`). A behaviour change without a test that exercises
   it does not count as done; a user-visible change without a README/arch-doc
   update likewise.
4. Open the PR early, keep it draft until local checks pass:

   ```bash
   pytest                 # full suite
   ruff check .
   ```

5. CI green, squash-merge into `develop`, delete the branch. Message shape: [Commits](docs/standards/commits.md).
6. A change to retrieval or generation also needs the full gate green. Label
   the issue and the PR `needs-gate`, run the gate locally against the topic
   branch before the squash, and put its run id in the PR. The run id is the
   timestamp in the name of the report it writes under `tests/eval/runs/`.

### PR standard

A PR description states, in this order:

1. **Summary** — what changed in product/system terms first, symbols second.
2. **Issue** — `Fixes #NNN`, or why it doesn't close one.
3. **Test plan** — how this was verified, honestly: which suites ran locally,
   whether the full gate is needed/run (with its run id), what remains as
   follow-up monitoring.

Keep PRs reviewable: under ~400 changed lines of product code when feasible,
no unrelated formatting churn, no drive-by refactors of adjacent code.

### Issue standard

An issue exists to be closable with evidence:

- **Closure criterion in the opening post.** "Fix X" is not an issue; "X is
  reproducible this way, closed when a test reproduces it and passes" is.
- **Evidence over opinion.** Measurements, run artifacts and logs beat design
  taste. Cite local artifacts explicitly as local (`tests/eval/runs/` is
  gitignored — cited, not shipped).
- **One type label minimum.** Pick the one that says what the issue *is*;
  stack modifiers, areas and a priority only where they add something.

  <!-- labels:begin -->
  <!-- Declared in .github/labels.json; both lists are checked against it by
       tests/unit/test_contribution_standard_drift.py, so a label named here
       and missing there (or the reverse) is a red test, not a discovery
       someone makes at `gh issue create` time. -->

  | Type label | Use it when |
  |---|---|
  | `bug` | The shipped product does something wrong. |
  | `enhancement` | A capability or behaviour change the product does not have. |
  | `eval` | Gold cases, grading, the measurement gate itself. |
  | `loop` | The self-improvement harness: search space, ledger, campaigns. |
  | `performance` | Latency, throughput or resource use is the subject. |
  | `research` | An exploration whose deliverable is a written conclusion. |
  | `documentation` | The docs are wrong, missing or contradict the code. |
  | `cleanup` | Removing dead weight; no behaviour change intended. |

  Modifiers stack on a type. Add them only when they say something the type
  does not:

  | Modifier | Use it when |
  |---|---|
  | `security` | Hardening, or a vulnerability already disclosed privately (see SECURITY.md). |
  | `needs-gate` | The change touches retrieval or generation; the full eval gate must be green before merge. |
  | `needs-decision` | It waits on an owner decision (a protected zone, a design choice). Say which in the issue. |
  | `blocked` | It cannot move until something external happens; the issue says what. |
  | `wontfix` | Applied when closing with the reasoning, never instead of it. |

  Area labels say where the work lives; zero or more per issue. They come from
  the prefixes the history already uses in commit and issue subjects (Web:,
  Frontend:, Headless:, Indexación:, ...).

  | Area | Covers |
  |---|---|
  | `area:core` | src/monkeygrab domain, ports, application and config. |
  | `area:adapters` | src/monkeygrab/adapters: MinerU, embedder, FAISS, Ollama, BM25, BGE. |
  | `area:indexing` | Corpus indexing, chunking, the index fingerprint. |
  | `area:retrieval` | Hybrid search, reranking, context assembly. |
  | `area:generation` | Prompts, RECOMP, streaming, citations. |
  | `area:web` | The Flask backend and its API. |
  | `area:frontend` | The React frontend in rag/web/frontend. |
  | `area:headless` | The headless HTTP service for other apps. |
  | `area:desktop` | The pywebview desktop app and its packaging. |
  | `area:study` | Study mode: quizzes and study sheets. |
  | `area:ci` | Workflows, tooling, repository configuration. |

  Priority, at most one, set by whoever triages: `priority:high` (data loss,
  wrong answers or a broken gate, do first), `priority:medium` (planned, no
  deadline), `priority:low` (when adjacent work is open). An issue with no
  priority label has not been triaged.
  <!-- labels:end -->

  `eval` and `loop` are not synonyms. `eval` is the measurement — cases,
  grading, the gate's own correctness. `loop` is the optimiser that consumes
  it. An issue about a case the grader scores wrong is `eval`; an issue about
  a campaign pairing against the wrong baseline is `loop`; one that is both
  carries both.
- **Closing comment carries the proof**: the numbers, the merged PRs, or the
  reasoned decision (including "won't do", with the reasoning). An issue
  closed without evidence will be reopened.
- Design decisions that change product architecture or the loop get a dated
  block in the relevant `docs/design/*.md` **before** code moves.

### Where the standard is enforced, not just written

Three checks, because a written convention drifted for months once already
(`docs` and `loop` were named here while neither existed on GitHub, issue
#109):

| Check | What it holds | Runs in |
|---|---|---|
| `tests/unit/test_contribution_standard_drift.py` | This document and the issue templates name only declared labels, and every type label is documented here | fast gate, `architecture` job |
| `.github/workflows/labels.yml` | `.github/labels.json` matches the labels GitHub actually holds | on changes to that file, and on demand |
| `.github/ISSUE_TEMPLATE/`, `blank_issues_enabled: false` | Every new issue starts from a template | GitHub itself |
| `tools/check_branch_policy.py` via `branch-policy.yml` | The head/base pair of every PR follows the gitflow table above | every PR, required check |

Adding a label is therefore three edits and no ambiguity: create it on
GitHub, declare it in `.github/labels.json`, name it in the tables above.
Release notes group merged PRs by these labels (`.github/release.yml`).

## Protected zones (owner agreement required)

These come from `AGENTS.md` rules 5, 6 and 9; they exist because breaking
them once cost weeks of silent drift:

- `requirements*.txt` — versions are pinned.
- Pipeline flag defaults (`USAR_*`, numeric parameters shipped to users).
- `tests/eval/` grading rules, gold cases and baseline floor.
- `tests/characterization/` pins current observed behaviour, bugs included.

## Reporting bugs

Use the bug template. Include the exact command, the full traceback, your OS
+ GPU + Ollama model list (`ollama list`) and, if indexing is involved,
whether `.venv-mineru/` exists beside the repo.

Suspected vulnerabilities or leaked secrets do not go through the bug
template: rotate first, then report privately as described in
[`SECURITY.md`](SECURITY.md).
