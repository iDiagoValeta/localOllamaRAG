# Runbook — operating the self-improvement loop

**Audience: an agent (or a person) about to run a campaign.** This file says
what to do and in what order. It says almost nothing about *why*, because
[`README.md`](README.md) next to it already does, at length, and the two
questions are not answered well by one document. Every claim here that has a
reason behind it links to where that reason lives.

Registered in [`docs/README.md`](../docs/README.md)'s table. Issue #114.

---

## 0. What this is, and what it is not

The loop searches for a pipeline configuration with a higher gold-case pass
rate, subject to a latency ceiling, recording every candidate.

**It proposes. It never integrates.** No code path here writes a winning
configuration into the product, commits anything, or touches `grade.py`,
`gold_cases.jsonl` or `baseline_min_pass_rate.txt` — structurally, not by
convention ([`README.md`](README.md), the CAUTION block). If a campaign
finds something, a human decides what happens to it.

**A green campaign is not a shipped improvement.** Every report carries a
`resolution_warning` for a reason; see §7.

---

## 1. Before launching: six checks, in order

Stop at the first one that fails. Each is cheap; the campaign is not. Check 0
and check 6 both exist because a real run on 2026-09-01 skipped them and paid
28.7 minutes to find out.

> [!IMPORTANT]
> **Run every command below with `.venv/bin/python`, not a bare `python`.**
> `--status` works under a bare interpreter and prints a confident-looking
> line, but the fingerprint it computes sees only the isolated venv — the
> product stack shows as unmeasured, and an entry written that way describes
> nothing about retrieval or generation (issue #132). The comparison now
> answers `unknown` rather than `match` in that case, so the mistake is
> visible instead of silent, but the entry is still worth nothing.

```bash
# 0. Can this machine run anything at all? Installs nothing.
.venv/bin/python tools/setup_environments.py --check
```

Six components, each reported separately: both interpreters, CUDA visible to
the isolated one, the MinerU binary, the jina-clip worker actually loading,
and which configured Ollama models are missing. A `FAIL` here means no
campaign is possible; a `warn` means something it does not build needs
attention (see `tools/setup_environments.py`).

```bash
# 1. What does the ledger hold, and how much can this launch pair against?
.venv/bin/python -m harness.cli --status
```

This is the whole precondition check for the ledger, and it evaluates
nothing. On a clean clone:

```
ledger dir: /path/to/repo/harness/ledger
reference models: rag=gemma4:e4b, chat=gemma4:e4b, contextual=gemma4:e4b, recomp=gemma4:e4b
environment: isolated:mineru=3.4.5, isolated:sentence-transformers=5.6.1, isolated:torch=2.6.0+cu124, isolated:transformers=4.57.6, product:faiss-cpu=1.15.0, product:ollama=0.6.2, product:rank-bm25=0.2.2, product:sentence-transformers=6.0.1, product:torch=2.13.0, product:transformers=5.16.1
ledger: directory does not exist yet -- a first campaign creates it
```

Read it top to bottom:

| Line | What it tells you | What to do about it |
|---|---|---|
| `reference models` | The four roles this campaign would measure under. Environment beats `settings.json` beats defaults. | If these are not what you intended, fix the environment or pass `--set models.rag=...`. A campaign A of 2026-08-23 lost hours to a `settings.json` value nobody had checked. |
| `environment` | The installed stack, per environment. `isolated:` builds the index, `product:` decides retrieval and generation. | If `isolated:mineru` is missing, `.venv-mineru` is not installed and indexing cannot run. |
| `ledger:` | What prior history exists. | See §2. |

The remaining checks:

```bash
# 2. The fast gate is green, so a failure later is the campaign's, not the tree's.
pytest -q
ruff check .

# 3. Ollama answers and holds the models the roles name.
curl -s -m 5 http://localhost:11434/api/tags -o /dev/null -w '%{http_code}\n'   # expect 200
ollama list

# 4. The isolated stack loads on this GPU. ~30 s the first time, then cached.
echo '{"id": 1, "op": "text", "text": "probe"}' \
  | .venv-mineru/bin/python src/monkeygrab/adapters/embedding/jina_clip_worker.py
# expect: {"event": "ready", "dim": 512} then {"id": 1, "ok": true, "vector": [...]}

# 5. The wiring runs end to end, with no GPU and no models, into a temp dir.
python -m harness.cli --dry-run --max-iterations 3

# 6. Is there room on the card? Anything else holding VRAM will end the run.
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

> [!WARNING]
> **On an 8 GB card, set `OLLAMA_KEEP_ALIVE=0` for the run.** Phase 1 holds
> two jina-clip workers (one per corpus, ~2.8 GiB each) plus the reranker;
> with a generator also resident, the second worker cannot start and every
> blind-set case comes back `infrastructure_error`. Measured 2026-09-01: 19 of
> 51 cases unevaluable after 28.7 minutes, the whole run discarded
> (`[gate] INCONCLUSIVE ... this run measured nothing`). Issue #123 has the
> allocator's own accounting; the keep-alive is a workaround, not a fix, and
> it makes the run's timings incomparable with ledger history that did not use
> it.
>
> Do not run anything else on the card during a campaign, the jina-clip probe
> in check 4 included. One probe alongside a live run is enough to take it
> down. Two campaigns at once will take each other down -- check `pgrep -f
> harness.cli` before launching, since a backgrounded one leaves no obvious
> trace in the terminal.

> [!TIP]
> **A campaign fits on 8 GB even though a full gate run does not.** The
> harness only ever asks for search-set ids (`source: corpus`), and
> `evaluate()` builds a corpus's stack only when the filtered case list
> needs it -- so a campaign builds one embedder where a full `run_eval.py`
> builds two, and two do not fit (#123). Measured: 32 cases, zero
> `infrastructure_error`, ~6.6 of 7.6 GiB resident.
>
> The practical consequence, plainly: **on a card this size you can run the
> loop but not the gate.** A candidate the loop accepts still needs the full
> gate before anyone believes it, and until #123 is resolved that has to
> happen somewhere with more VRAM.

> [!NOTE]
> A dry-run given `--ledger-dir <a ledger holding anything not marked demo>`
> now refuses to start (exit 2) rather than appending its synthetic iterations
> to it — which is what it used to do, indistinguishably from measured entries
> (issue #115). `--status` and `--replay` only read, so neither is affected.
> Without `--ledger-dir`, a dry-run writes to a fresh temp directory, which is
> the safe form used above.

A missing Ollama model is caught by the gate itself, at launch, with the exact
`ollama pull` line to run — verified 2026-09-01, it failed in 20 s rather than
mid-campaign. That is a real preflight; do not build another.

---

## 2. Reading `--status` on a ledger with history

```
ledger: 2 entry(ies), iterations 1-2 (accepted 2)
  last: iteration 2 accepted on the search_set -- seeded
  best search-set objective in this ledger: 27 (iteration 1) -- comparability decides whether this launch can use it
ledger history: 2 entry(ies), 1 comparable search-set state(s)
  historical high water: 25 (recovery mode arms only if the measured reference scores lower)
```

The two blocks answer different questions and the gap between them is the
point:

- `best ... in this ledger` is the best entry that exists.
- `historical high water` is the best entry **this launch could actually pair
  against**. Anything with different model roles, chunking, index-time flags,
  or a known-different stack is excluded.

When those two numbers differ, some history is not usable under this
configuration. That is normal and is not an error.

Three lines are decision points:

**`WARNING recovery-mode history mismatch -- <field>: this launch X, ledger Y`**
No prior entry is comparable. A criterion-5 recovery campaign launched in this
state runs with recovery mode silently off and rejects every candidate that
restores healthy behaviour (this is #92, and it happened). Either pin the
field with `--set` to match the ledger, or accept that this campaign starts
its own history.

**`WARNING that high-water entry was NOT measured on a stack comparable to this launch's`**
The entry your candidates will be judged against ran on a different (or
unrecorded) stack. It can hold passes the current stack cannot reach, which
reads as a regression no candidate can fix. A refresh campaign on the current
stack is what replaces it — issue #107, still open for exactly that.

**`N entry(ies) carry no stack fingerprint (written before ledger schema v3)`**
Historical entries, usable but unverifiable. Not a problem by itself; it
becomes one when the high water is among them, which the previous warning
tells you.

---

## 3. Launching

```bash
# Deterministic control: coordinate descent, reproducible from the ledger alone.
python -m harness.cli --proposer grid --max-iterations 8 --patience 3

# LLM proposer, same guardrails. Falls back to grid on any invalid output.
python -m harness.cli --proposer llm --llm-model gemma4:e4b --max-iterations 8 --patience 3

# A campaign that must be comparable with prior history: pin the fields.
python -m harness.cli \
    --ledger-dir tests/eval/runs/harness-loop \
    --set models.rag=gemma4:e4b --set models.chat=gemma4:e2b
```

`--set` is repeatable, JSON-decodes its values, and unknown keys abort at
launch rather than being silently ignored. The pins reach every measured
evaluation, the reference's included (#101/#106) — an earlier version only
pinned the harness's own bookkeeping object, and a "sabotaged" reference
scored healthy.

Always give `--max-iterations`, `--patience`, or both. The loop refuses to
start without at least one; it is what guarantees termination.

**Budget, and whose machine the number came from.** Quoting one GPU's timings
as universal is how someone budgets a night for something that takes seven.

| | reference GPU (2026-08-19) | RTX 4060 Laptop 8 GB (2026-09-01) |
|---|---|---|
| retrieval-only case | ~4.1 s | **~30 s** |
| answered case | ~28.5 s | ~9-11 s |
| full search-set evaluation | ~20 min | ~25 min (phase 1 dominates) |
| fast tier (13 cases) | ~4 min | not measured here |

The 4060 figures are with `OLLAMA_KEEP_ALIVE=0` (see the warning above), which
is part of why retrieval is slower and generation is not: phase 2 runs with
the card to itself either way. A candidate rejected at the fast tier costs the
fast tier only. A night still fits many candidates on either machine — the
design doc's original estimate was ~6x too pessimistic and has been
corrected — but size the night from the row that matches your card.

---

## 4. The five verdicts, and what each one means you should do

| Verdict | What happened | Next |
|---|---|---|
| `accepted` | Beat the ratchet, no regression, within latency. | **Not a shipped improvement.** Record it, apply §7's warning, and hand it to a human. Stage 1 is a single-field sweep from a fixed reference, so an acceptance usually ends the run shortly after (patience). |
| `rejected_no_gain` | Measured fine, did not beat the ratchet. | Nothing. This is the normal outcome and it is evidence, not failure. |
| `rejected_regression` | Lost cases against the pairing baseline. | Check *which* baseline: the entry records `regression_baseline_iteration`. If it paired against an aged high water, see §2's second warning — the rejection may be about the baseline, not the candidate. |
| `rejected_latency` | A bucket's median exceeded 1.20x the reference's. | Real. Answered and retrieval-only buckets are checked separately on purpose; a blended median would hide exactly this. |
| `inconclusive` | **The measurement failed, not the candidate.** A dead or overloaded Ollama, a retrieval exception. | Fix the infrastructure and re-run. Never read this as a regression. It still counts toward `--patience`, so a broken Ollama ends the campaign rather than filling the ledger with fiction. |

If the **reference** measurement itself carries an infrastructure error, the
loop refuses to start at all (`InconclusiveEvaluationError`). That is
deliberate: no ratchet baseline can be trusted in that state.

---

## 5. Reproducing one iteration

```bash
python -m harness.cli --replay 3 --ledger-dir <that ledger>
```

Re-runs iteration 3's exact overrides on its exact case ids and exits 0 only
if the pass/fail vector matches. This is criterion 7, and it is the check to
run before believing a surprising result.

---

## 6. What is forbidden

Not style preferences — each has a mechanism behind it and, in most cases, a
red test.

- **Never edit `tests/eval/grade.py`, `gold_cases.jsonl` or
  `baseline_min_pass_rate.txt`** to make a result look better. The harness
  cannot reach them by construction (`test_harness_boundaries.py` parses for
  it), and neither should you.
- **Never write a winning configuration into the product.** The loop proposes,
  a human integrates (design doc §5). Flipping a shipped `USAR_*` flag or a
  numeric default needs owner agreement (`AGENTS.md` rule 6).
- **Never hand-edit the ledger** to remove an inconvenient entry. It is
  append-only evidence. If the history is stale, the answer is a refresh
  campaign, not a delete (#107).
- **Never hand-write a demo result into a real ledger.** The tool now refuses
  to do it for you (§1), and the refusal exists because a demo entry is not
  inert: `GridProposer` reads history back to skip points already tried, and
  `_historical_high_water` picks a pairing baseline from it (#115).
- **Never weaken a test to make a campaign pass.** `tests/characterization/`
  pins current behaviour including its bugs; changing one is a signal to stop
  and confirm (`AGENTS.md` rule 9).

---

## 7. What is true of every result today

State these when reporting a campaign; they are not disclaimers, they are the
measurement's actual resolution.

- **The search set can win at most 5 cases**, and it registered only 3 net
  flips under a known-catastrophic sabotage — below the ~6 flips the design
  sets as the threshold for a difference not attributable to chance. An
  accepted improvement on today's corpus is **a candidate for confirmation,
  not a demonstrated result**. This is issue #30, the binding constraint on
  the whole loop, and every report carries it as `resolution_warning`.
- **Index-time knobs are not searched.** Chunking and the index-time flags
  force a full reindex per candidate, which no budget survives inside a search
  loop (#102). Declaring one is a red test, not an oversight.
- **Stage 1 is a single-field sweep from a fixed reference**, not a
  compounding hill climb. Two accepted single-field changes cannot combine
  within a run.
- **An aged baseline is visible but not fixed.** Drift is now recorded and
  reported (#107 option 2); making an aged high water reachable again needs a
  refresh campaign, which is still open work.

---

## 8. Closing a campaign

1. The ledger and `report.json` are the evidence. `tests/eval/runs/` and
   `harness/ledger/` are gitignored — cite paths explicitly as local, never
   assume a reader can open them.
2. Comment on the issue the campaign was run for, with the numbers, not a
   summary of the numbers. An issue closed without evidence gets reopened
   (`CONTRIBUTING.md`).
3. If the campaign changed a standing decision about the loop, it gets a
   dated block in `docs/design/2026-07-28-loop-automejorable.md` **before**
   code moves (`AGENTS.md` rule 2).
4. If it found a defect, open an issue with the reproduction, using the
   measurement template. If it found a limit, say which of the two it is —
   they lead to different work.
