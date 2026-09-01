# Contributing

The operating contract for this repo — humans and agents alike — is
[`AGENTS.md`](AGENTS.md). This file covers the mechanics: how to branch, what
a change must carry, and how issues and PRs are run. The documentation
standard (what gets documented where) is [`docs/README.md`](docs/README.md).

## Setup

```bash
pip install -r rag/requirements.txt          # core
pip install -r rag/web/requirements.txt      # web UI
cd rag/web/frontend && pnpm install && pnpm run build   # frontend (pnpm only)
```

Ollama must be running locally with a generator model pulled; a CUDA GPU is
required for indexing (jina-clip hard-fails without one). MinerU and Jina CLIP
live in an isolated `.venv-mineru/` interpreter — see the root `README.md`.

## Two CI gates, know which one you need

- **Fast gate** (`ci.yml`, every PR): lint (`ruff`), architecture dependency
  rules, unit suite against test doubles, harness tests, frontend build. No
  GPU, no models.
- **Full gate** (`full-eval.yml`, manual dispatch on the self-hosted GPU
  runner): the real pipeline against every gold case. **Required before
  merging anything that touches retrieval or generation** — the fast gate
  never exercises a real model.

## Workflow

1. Branch off `main`. Name it `<type>/issue-<n>-<slug>` where type is one of
   `feat`, `fix`, `cleanup`, `probe`, `docs`; use `issue-0-<slug>` only when
   there is genuinely no tracking issue (then open one first).
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

5. CI green → squash-merge → delete the branch. The squash subject follows
   the repo's commit style: imperative mood, issue number in parentheses,
   e.g. `Wire the eval gate's query decomposer the same way the product does (#64)`.
   The body explains why, not what; durable references (`#NNN`, measured
   dates) beat references to plan steps.

### PR standard

A PR description states, in this order:

1. **Summary** — what changed in product/system terms first, symbols second.
2. **Issue** — `Fixes #NNN`, or why it doesn't close one.
3. **Test plan** — how this was verified, honestly: which suites ran locally,
   whether the full gate is needed/run, what remains as follow-up monitoring.

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
  stack a modifier only if it adds something.

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

  Modifier: `wontfix`, applied when closing with the reasoning, never instead
  of it.
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

Adding a label is therefore three edits and no ambiguity: create it on
GitHub, declare it in `.github/labels.json`, name it in the table above.

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
