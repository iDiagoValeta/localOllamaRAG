# `harness/` — block-C configuration search (issue #31)

An optimiser that searches for a pipeline configuration with a higher
gold-case pass rate, subject to a hard latency ceiling, recording every
candidate so a run can be reconstructed. Implements
[`docs/design/2026-07-28-loop-automejorable.md`](../docs/design/2026-07-28-loop-automejorable.md)
§4 ("Arquitectura del arnés").

> [!CAUTION]
> The harness may **read and run** the evaluation gate under `tests/eval/`.
> It never redefines how the evaluation scores. `tests/eval/grade.py`,
> `gold_cases.jsonl` and `baseline_min_pass_rate.txt` stay byte-identical —
> enforced by `harness/tests/test_harness_boundaries.py`, which parses every
> harness module's imports and write calls with `ast`, not by convention.
> Proven, not assumed: `test_adversarial_bypass_module_is_caught_by_every_check`
> parses a module doing everything a #65 PR review found the first version
> of these checks missed (`importlib.import_module("grade")`, a dotted
> `import tests.eval.grade`, `shutil.copyfile`/`os.replace`/`os.remove`
> against the three protected files, a `"grade" + ".py"` string built to
> dodge a literal match) and asserts each attack is caught.

## Why this is not product

It lives at the repo root, outside the hexagonal core (`src/monkeygrab/`)
and the interface layer (`rag/`), and consumes the evaluation gate as a
library the same way `tests/eval/run_eval.py` does. It has no product
runtime dependents: nothing under `rag/` or `src/monkeygrab/` imports it.

**Repo rule 6** ("do not flip pipeline flags without agreement") governs the
**product's** defaults — `rag/chat_pdfs.py`'s module-level flags, what ships.
The harness flips flags only inside its own candidate evaluations, run
through the gate's own config plumbing, and never writes a winner back into
the product. The loop proposes; a human integrates (design doc §5). Rule 6
is about unreviewed changes reaching users, not about an optimiser measuring
what a flag does — this is exactly the loop's job.

## Layout

```
search_space.py      Declared tunables, index-time exclusions, feasibility, reachability gate metadata
fast_tier.txt         Fixed 13-case regression filter (never declares an improvement)
unreachable_cases.txt Cases excluded from the maximised scalar — starts empty, evidence-backed
ledger.py              Append-only evidence ledger (JSON per iteration + index.jsonl)
proposers.py            GridProposer (deterministic control) + LlmProposer (Ollama-backed)
evaluator.py            Case-set selection, objective/latency math, the real-evaluator adapter
loop.py                The ratchet, the latency constraint, termination
cli.py                  Entry point (python -m harness.cli)
ledger/                 Written evidence (empty until the harness actually runs; not auto-committed)
tests/                  Unit tests — no GPU, no Ollama, no model downloads (see below)
```

## The declared search space

Written by hand as data in `search_space.py`, never inferred by
introspecting `AppConfig` — each parameter is a visible decision about what
the loop is allowed to search, independent of how expensive an evaluation
is. Validated at import time: every declared value is applied via
`AppConfig().with_overrides(**{key: value})`, which raises `ValueError` on
an unknown section or field — confirmed in this PR that it really does raise
on every unknown field, so the space cannot drift from the config it
searches without a red test.

> [!NOTE]
> **The design's original evaluation-time estimate was wrong by a factor of
> ~6.** Section 4 assumed ~2.8 h/candidate (three or four candidates a
> night); issue #27's keep-alive fix removed a per-case cold model load
> that estimate baked in. A full search-set evaluation (32 cases) now costs
> ~13 min, measured 2026-08-12 against
> `tests/eval/runs/20260812T194812Z_mineru-jina_clip-faiss.json` (local,
> gitignored) — a night now fits dozens of candidates, not three or four.
> This does not change anything about how the space is declared (see
> above); it does change the case for the LLM proposer, see "Proposers"
> below.

**Index-time keys are excluded and the exclusion is enforced,** not
remembered: `chunking.*` and `flags.usar_contextual_retrieval`/
`usar_embeddings_imagen` change what gets stored, forcing a full MinerU +
jina-clip reindex per candidate — far more expensive than the ~13 min a
retrieval/generation-only evaluation now costs, whatever the exact multiple
turns out to be once block B (#30) measures it. Declaring one is a red test
(`INDEX_TIME_KEYS`, checked by `_validate_declared_space`).

**`weight_semantic_rrf`/`weight_bm25_rrf` are one knob, not two.**
`src/monkeygrab/application/rrf_fusion.py::fuse_semantic_and_keyword` computes
`score_final = score_semantic * weight_semantic + score_keyword * weight_keyword`
and sorts by it — confirmed in this PR by reading the function: it is a
linear combination used only for ordering, so scaling both weights by any
positive constant cannot change the sort order; only their ratio matters.
Only `weight_semantic_rrf` is declared; `weight_bm25_rrf` is always derived
as `1 - weight_semantic_rrf` (`search_space.expand_overrides`).

`usar_busqueda_hibrida` and `usar_reranker` are deliberately **not**
tunable: the design doc names turning the reranker off as the known-bad
sabotage used to prove criterion 2. A knob whose "off" position is the
control the gate was validated against does not belong in the space the
same gate searches.

> [!IMPORTANT]
> This space is not "every field `AppConfig` has" — it is "every field the
> measurement can actually move", and those two sets differ today by more
> than one entry. `retrieval.min_question_length` exists on `AppConfig` but
> is absent here: nothing under `src/monkeygrab/` reads it, the real
> minimum-length check lives in the legacy `rag.chat_pdfs` globals, so
> tuning it through `evaluate()` would move nothing. See "Reachability
> gate" below for the other two ways a declared field can turn out inert.

## Reachability gate

Found 2026-08-12 while building the sibling PR (#56, the
`tests/eval/run_eval.evaluate()` library API this harness's `evaluator.py`
wraps): `evaluate()` threads its `AppConfig` into retrieval and indexing,
but generation went through `rag/engine/generation.py`'s
`generar_respuesta_silenciosa`, which did not always receive that config.
**Resolved by #56**, which audited all 48 dotted `AppConfig` fields end to
end: `context.max_context_chars` and `flags.expandir_contexto` were already
reaching generation (both are read inside `Answer.select_evidence`, which
runs under the threaded config — the original finding was overcautious
about these two); `flags.usar_recomp_synthesis` and
`flags.usar_optimizacion_contexto` were genuinely dropped (read only in
`build_user_message`, which phase 2 re-executes under a *fresh* config from
`wiring.app_config_from_runtime()`) and are now applied through the
sanctioned `rag.set_pipeline_flags(...)` runtime setter for the duration of
an evaluation, restored in a `finally` (with a test that an exception
mid-run still restores the previous globals). All four keys are honoured
today; `search_space.PENDING_REACHABILITY_KEYS` is empty.

The same audit found a *second*, different way for a declared key to be
inert: `flags.usar_llm_query_decomposition` is on `AppConfig`, but
`tests/eval/run_eval.py` builds `Retrieve(..., query_decomposer=None)`
unconditionally while the product wires it whenever the flag is on and
defaults to on — the gate measures a retrieval pipeline no user runs, so
flipping the flag inside an evaluation cannot change anything.
`evaluate()` would not raise for it either (unlike the first four, this
isn't a hard-fail case), so it is simply **removed from `SEARCH_SPACE`**
rather than relied on the gate to catch — filed as issue #64, needs
sign-off to fix because it will move the pass rate.

The gate itself stays for exactly this reason — two different failure
classes turned up in one audit, and a future added key gets the same
protection for free instead of needing someone to remember either lesson:

- `evaluator.verify_reachable` probes the injected evaluator once, at loop
  startup, with every declared key set simultaneously — before the
  reference is even measured. If `evaluate()` raises, the loop fails loudly
  with `ReachabilityError` naming what it can, instead of running a full
  night on top of a silently inert knob.
- `GridProposer` walks `search_space.proposal_order()` (currently identical
  to declaration order, since `PENDING_REACHABILITY_KEYS` is empty — but it
  would reorder known-reachable keys first the moment a future key needs
  flagging again), not necessarily `SEARCH_SPACE`'s raw order.

This is structural, the same class of protection as the boundary test: a
fake evaluator that raises on a declared key makes startup fail loudly
(`harness/tests/test_reachability_gate.py`), not silently disable the knob.

## Sets

- **Search set** (32 `source: corpus` cases) — the loop's objective.
- **Blind set** (19 `source: arxiv` cases) — never requested. Structurally
  unreachable: `evaluator.search_set_case_ids()`/`blind_set_case_ids()` both
  filter `gold_cases.jsonl` by `source`, and every case-id list `loop.py`
  ever hands an evaluator comes from one of those two functions or
  `load_fast_tier()` (itself validated as a search-set subset) — never from
  a proposer, which can only emit config overrides. Proven, not assumed:
  `harness/tests/test_evaluator.py` asserts the blind set is disjoint from
  both the search set and the fast tier.
- **Fast tier** (13 fixed ids, `fast_tier.txt`) — regression filter only, it
  **never declares an improvement**. Costs ~4 min at current measured rates
  (8 answered cases × ~28.5 s + 5 retrieval-only × ~4.1 s), against a ~13 min
  full search set — a real but much smaller saving than the design's ~32 min
  estimate assumed (see the search-space note above); its job is rejecting a
  bad candidate before it can contaminate the ratchet, not the minutes it
  happens to save. Includes all 5 of today's known search-set failures
  (three figure-retrieval failures cost ~4 s each, nearly free to include,
  and give the regression filter real signal on the retrieval side).

## Objective, constraint, noise floor

**Objective:** `objective_adjusted` = passing cases on the search set, minus
cases listed in `unreachable_cases.txt` (excluded from both numerator and
denominator, not just discounted). `unreachable_cases.txt` starts **empty**
and stays that way as of 2026-08-12: the `reason` field of all five of
today's search-set failures was read against the reference gate run and
none is the generator failing to answer over a figure it saw — three are
`figure_retrieval` cases (never call the generator) that fail because
retrieval surfaced text where an image chunk was wanted, exactly what fusion
weights/`top_k_final`/the reranker threshold move; the other two are a
number present in the abstract that did not survive into the generated
answer. All five are inside stage 1's action space. See the file's header
for the full citation. `objective_raw` (unreachable cases included) is
tracked alongside so the exclusion mechanism is auditable before it is ever
needed — today `raw == adjusted`.

**Constraint, per bucket, not blended.** Answered cases (`factual_number`/
`factual_concept`) cost ~28.5 s; retrieval-only cases (`figure_retrieval`/
`table_retrieval`) cost ~4.1 s (both measured 2026-08-12, same run cited
above; the gap was ~50x before issue #27's keep-alive fix and is ~7x now —
narrower, not gone). A single blended median is dominated by whichever
bucket is larger and would let a candidate trade retrieval quality for
fewer/cheaper generator calls without the constraint noticing. `loop.py`
checks each bucket's median against `1.20 ×` its own reference median
independently; a breach in either bucket is `rejected_latency`, even if the
candidate scored higher.

**A candidate cannot buy the objective with the retrieval-only bucket.**
The blended objective by itself has the same blind spot as a blended
latency median: a candidate that flips several `figure_retrieval`/
`table_retrieval` cases to FAIL while flipping more answered cases to PASS
would otherwise read as a plain improvement — exactly the trade design doc
§2's "Consecuencia para el arnés" warns a blended score would hide. Found
in a #65 PR review; `loop.py` now also rejects (`rejected_regression`, at
the search-set stage) any candidate whose retrieval-only bucket loses cases
net against the reference, paired by id, regardless of the blended
objective. `LedgerEntry.summary` also carries a `by_case_type` breakdown
(mirroring `run_eval.py`'s own `build_summary`) so the trade is visible in
the ledger even for an iteration this check does not reject.

**An `inconclusive` verdict, never a false "improvement" or "regression".**
`tests/eval/run_eval.py` marks a case `infrastructure_error` when it could
not be evaluated at all (dead/overloaded Ollama, a retrieval exception) and
refuses to compare such a run against the baseline. Reading those records
as ordinary failures would let a dead Ollama masquerade as a retrieval
collapse (fast tier: a good candidate wrongly `rejected_regression`) or
silently deflate the reference and inflate every later candidate into a
false `accepted` — found in the same review, and precisely the "loop sobre
una medida que miente produce basura reproducible" failure the design doc
opens with. `loop.py` gives any iteration with such a record the
`inconclusive` verdict (no ratchet move, still written to the ledger, still
counts toward `--patience`) and raises `evaluator.InconclusiveEvaluationError`
before the loop even starts if the **reference** measurement itself carries
one — no ratchet baseline can be trusted in that case.

**Noise floor: 0 cases.** Measured 2026-07-29 (design doc, criterion 1
note): two runs of the same configuration and code, same index, zero flips
across 51 gold cases. Independently re-verified in this PR restricted to
the 32-case search set: still 0 flips. A delta of 0 is not an improvement
(`loop.NOISE_FLOOR_CASES`).

## Resolution warning

The search set can win at most **5** cases today (27/32 pass on the
reference run) — below the **~6 net flips** the design doc sets as the
threshold for a paired difference not attributable to chance. Restricting
the same paired comparison to the known-catastrophic sabotage used for
criterion 2 (`RAG_TOP_K_FINAL=1`) gives only **3 net flips** on the search
set, versus 7 on the full 51-case gate — the loop's own objective set is
under-powered against the most destructive single-field change anyone has
measured, not merely against subtle ones. Every `run_loop` report carries
this as `resolution_warning`, so an accepted improvement on today's corpus
reads as a candidate for confirmation, not a demonstrated result. This is
the quantitative case for block B (#30, corpus expansion) — sharper than the
design doc's own estimate — not a reason to withhold the harness.

> [!NOTE]
> All of the numbers above come from four local, gitignored artifacts
> (`tests/eval/runs/20260729T020233Z_...json`, `...040824Z...`,
> `...081129Z...json`, `...20260812T194812Z...json`) — `tests/eval/runs/` is
> in `.gitignore`, so they are not reproducible from a fresh clone. They are
> cited, not shipped. The three 2026-07-29 runs predate issue #27's
> keep-alive fix and are cited for their PASS/FAIL evidence (noise floor,
> sabotage flips, resolution limit), which the fix did not touch
> (`compare_runs.py` against the healthy 2026-07-29 run reports 51 cases
> unchanged); the 2026-08-12 run is cited for current timing.

## Proposers

`GridProposer` is the deterministic control: coordinate descent from the
reference configuration, one field at a time, in `proposal_order()`,
skipping infeasible points and points already in the ledger. Reproducible
from the ledger alone.

> [!NOTE]
> **Why keep an LLM proposer at all, now that evaluation is cheap.** Design
> doc §4 originally argued: ~2.8 h/evaluation, three or four candidates a
> night, blind/grid search is useless at that budget, therefore an LLM
> reading actual failures is the natural fit. That argument no longer holds
> at ~13 min/evaluation (a night fits dozens of candidates, see "The
> declared search space" above) — corrected in the design doc, PR #68. This
> is **not** evidence the deterministic proposer is sufficient on its own,
> and it is not a reason to drop `LlmProposer`: it means criterion 5's
> controlled comparison between the two (design doc §2) is now cheap enough
> to settle with evidence instead of an argument from budget. Keeping both
> was the right call before this correction and is better justified after
> it, not worse.

> [!IMPORTANT]
> **Scope, stated plainly (a #65 PR review asked for this explicitly):
> stage 1 is a single-field sweep from a fixed reference, not a compounding
> hill climb.** `reference` never changes during a `run_loop` call, even
> after an acceptance — matching `GridProposer`'s own spec-given definition
> ("coordinate descent from **the reference configuration**", issue #31 spec
> §5.4). Two accepted single-field changes cannot combine into one
> two-field configuration within a run, and after the first acceptance the
> ratchet has already risen, so a further single-field change measured from
> the *original* reference is unlikely to clear it — `--patience` typically
> fires not long after the first success. This matches criterion 5 (recover
> from one deliberately worsened point) and the design doc's framing of
> block C as proving the search mechanism works, not as shipping a
> multi-step optimizer. Rebinding the reference to an accepted candidate
> (a real hill climb) is future work, not something "ratchet" already
> implies.

`LlmProposer` prompts a local Ollama model (default `gemma4:e4b`,
`http://localhost:11434`) with the declared space, the ledger history and
the currently failing search-set cases' questions, and requires strict JSON
back. **Validation is the security boundary, not the prompt:** every
returned override is checked against the declared keys, their declared
values and the feasibility predicate before it is trusted; anything else —
malformed JSON, an undeclared key, an infeasible combination, or Ollama
being unreachable — falls back to `GridProposer` after up to 3 attempts,
recorded in the ledger (`proposer_fallback`/`proposer_fallback_reason`).
Because the only thing `propose()` can return is a dict of declared dotted
keys, there is no path by which the LLM reaches `grade.py`, the gold-case
file or the baseline.

## Running it

```bash
# Smoke-test the whole loop -- proposer, feasibility, ledger, latency
# constraint, termination -- with no GPU, no Ollama, no PDFs. Writes to a
# fresh temp directory unless --ledger-dir is given.
python -m harness.cli --dry-run --max-iterations 3

# Real run (needs the sibling PR's tests/eval/run_eval.evaluate() on main,
# Ollama, and a GPU -- fails with an actionable message otherwise):
python -m harness.cli --proposer llm --max-iterations 8 --patience 3

# Criterion 7: reconstruct one ledger iteration and re-run its exact
# overrides and case ids. Exit 0 iff the pass/fail vector matches.
python -m harness.cli --replay 1 --ledger-dir /path/to/ledger
```

`--dry-run` uses `evaluator.build_demo_evaluator()`: a tiny, deterministic,
in-process synthetic landscape. It exercises the wiring, not the pipeline —
it has no opinion about which real configuration is better.

> [!NOTE]
> **The default `--ledger-dir` (`harness/ledger/`) is meant to be the same
> directory across many invocations over time** — a #65 PR review asked
> this explicitly. "Append-only, versioned, one entry per iteration" (design
> doc §4) spans the whole history a team accumulates, not one run:
> `GridProposer`'s "already tried" check and `next_iteration_number` both
> read the directory back via `ledger.read_history` to continue where the
> last invocation left off. `--dry-run` deliberately does NOT default to
> this directory (a fresh temp dir instead), precisely so demo runs never
> mix into that history. A second real invocation against a non-empty
> `--ledger-dir` — including the default, after one real run — used to
> crash (`ledger.read_history` fed `cli.py`'s own `report.json` to
> `LedgerEntry(**data)`); fixed, see `harness/tests/test_ledger.py`'s
> `test_read_history_ignores_a_report_json_in_the_same_directory`.

`evaluator.real_evaluate` maps `run_eval.evaluate()`'s actual return shape
(`{"records": [...], "config": {"dev": ..., "blind": ...}}`, confirmed
against the sibling PR's contract) — an earlier draft of this adapter
assumed `{"results": [...], "effective_config": ...}` and would have raised
`KeyError` on the very first real call, found in the same review before
`evaluate()` had even landed. It also passes `write_report=False`: the
harness ledger is the evidence, and the first real run (issue #71) wrote
five extra JSONs under `tests/eval/runs/` (including a 0-case reachability
probe) while the loop ignored `evaluate()`'s `exit_code`. `harness/tests/
test_evaluator.py` exercises the mapping against a stub `run_eval` module
carrying the real contract, and pins `write_report=False`.

## Testing

All 13 required behaviors from the implementation spec are covered under
`harness/tests/`, entirely against fake/deterministic evaluators — no GPU,
no Ollama, no model downloads, matching this repo's fast CI gate (`tests/
conftest.py`'s pattern of skipping engine-dependent tests where the stack is
absent; `harness/tests` needs nothing from that stack at all). A #65 PR
review added coverage the original 13 did not require but a real multi-hour
run depends on: `LlmProposer._real_call`'s bounded timeout and its fallback
on a connection failure/non-2xx status/non-JSON body/missing `requests`
install (all against a fake `requests` module, never a real Ollama server),
`real_evaluate`'s mapping of the sibling PR's actual return shape (against a
stub `run_eval` module), the `inconclusive` verdict, the retrieval-only
net-loss rejection, and the adversarial-module regression test for the
boundary checks.

> [!IMPORTANT]
> `harness/tests/test_criterion5_simulated.py` proves the **search logic**
> recovers from a deliberately worsened point in a synthetic, fully
> controlled landscape — for both proposers. It does **not** prove the real
> pipeline improves. The real criterion-5 run (design doc §2) still needs
> the GPU and Ollama — now at ~13 min/full-search-set-evaluation rather than
> the ~2.8 h originally assumed (see "The declared search space" above) —
> and is pending measurement; nothing in this PR claims it ran.
