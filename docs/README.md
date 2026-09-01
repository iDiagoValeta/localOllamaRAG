# Documentation standard

| Document | For |
|---|---|
| `README.md` (root) | A user installing and running MonkeyGrab. |
| `src/monkeygrab/README.md` | The layers, the dependency rule, how to add an adapter. |
| `rag/README.md` | What `rag/` is today and where the rest of the docs live. |
| `AGENTS.md` (root) | Operating contract for any agent or human working in this repo; `.claude/CLAUDE.md` and `.codex/AGENTS.md` are pointers to it and must stay pointers. |
| `CONTRIBUTING.md` | How to branch, open PRs and issues, what a change must carry, how issues get closed. |
| `harness/RUNBOOK.md` | An agent or person about to run a loop campaign: the procedure, the decision points, what each verdict means. Procedure only — the reasoning lives in `harness/README.md`. |
| `docs/model-history.md` | Which models have been measured, on what corpus and how fast — append-only, one row per run, so a later choice comes from numbers instead of memory. |
| `docs/design/*.md` | Current standing design decisions and their rationale; update them when architecture changes. |
| Directory-local `README.md` | Only where a directory's purpose isn't obvious from its code — not one per folder by default. |

## What is NOT documented

Anything the code already states unambiguously: function signatures, default
values, parameter lists, module inventories. A doc that repeats these goes
stale the first time the code changes and the doc doesn't — which is what
happened to the files this standard replaced. Document the shape of the
system and the reasoning behind a decision; point at the code for the
specifics.

## Golden rule

**If a document and the code disagree, the code is right.** Fix the
document, don't rationalize the discrepancy. This applies to every file in
this table, including this one.
