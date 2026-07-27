# Documentation standard

| Document | For |
|---|---|
| `README.md` (root) | A user installing and running MonkeyGrab. |
| `src/monkeygrab/README.md` | The layers, the dependency rule, how to add an adapter. |
| `rag/README.md` | What `rag/` is today and where the rest of the docs live. |
| `.claude/CLAUDE.md` | Operating contract for any agent working in this repo. |
| `docs/design/*.md` | Standing design decisions and their rationale — written once, not updated per change. |
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
