# Gold eval corpus

Deterministic judge for the RAG pipeline: known-answer questions over real arXiv papers,
graded without an LLM. Built per `docs/design/2026-07-26-monkeygrab-v2.md` section 7.1-7.2.

`run_eval.py` runs the real pipeline over every case. It retrieves through the same
`Retrieve` use case the CLI and the web app go through, and generates through the same
facade entry point, so a pass rate measured here describes what ships — not a parallel
implementation that happens to live in the test tree.

## Files

| File | Role |
|---|---|
| `gold_cases.jsonl` | One JSON object per line: a question with verified accepted answers. |
| `fetch_papers.py` | Idempotent arXiv PDF downloader (id -> cached, gitignored PDF). |
| `grade.py` | Deterministic scoring: `grade_answer`, `grade_retrieval`. |
| `test_grade.py` | pytest suite for the grader + a schema check over `gold_cases.jsonl`. |
| `run_eval.py` | The gate: runs the real pipeline over every case and grades it (below). |
| `baseline_min_pass_rate.txt` | Pass-rate floor `run_eval.py` gates against. Only ever raised, never lowered. |
| `papers_cache/` | Downloaded blind-set PDFs. Gitignored — reproduced from `arxiv_id`. |
| `blind_docs/` | Blind-set PDFs staged under their `paper` slug for indexing. Gitignored. |
| `runs/` | Dated JSON results from each `run_eval.py` run. Gitignored. |
| `probe_cases_lang.jsonl` | Language-axis diagnostic probe (proposal, not part of the gate — see below). |
| `run_probe_lang.py` | Runs the probe against its own isolated FAISS collection. |
| `probe_docs_lang/` | PDFs staged for the probe, copied from `rag/docs/es\|ca/`. Gitignored. |
| `probe_runs_lang/` | Dated JSON results from each `run_probe_lang.py` run. Gitignored. |
| `probe_cases_domain.jsonl` | Domain-axis diagnostic probe (same status as the language one — see below). |
| `run_probe_domain.py` | Runs the domain probe against its own isolated FAISS collection. |
| `probe_docs_domain/`, `probe_runs_domain/`, `probe_cache_domain/` | Staged PDFs, dated results and the arXiv download cache for the domain probe. Gitignored. |

## Corpus

Four sources, mixed deliberately (the `source` field in `gold_cases.jsonl`):

- **`corpus`** -- the dev set: `rag/docs/en/`, 17 documents. Retrieved chunks and reranking
  behavior here are what the pipeline's heuristics were tuned against.
  A fresh clone does not hold them all: since #213 the PDFs are gitignored and only their
  identity is versioned (`rag/docs/corpus_manifest.json`), so run `python tools/fetch_corpus.py`
  once before the gate (`--check` says what is missing). `run_eval.py` does not fetch them itself
  and refuses to run on a partial store (`papers referenced by gold cases but not indexed`).
- **`arxiv`** (`arxiv_id` set) -- the blind set: ML papers fetched on demand by
  `fetch_papers.py`, never used to tune anything. A judge calibrated only on the dev set can't
  tell "the pipeline retrieves well" from "the pipeline overfits these PDFs"; the blind set is
  what makes that distinction possible.
- **`corpus_es`** / **`corpus_ca`** -- the product's own `rag/docs/es/` and `rag/docs/ca/`
  stores (17 documents each), evaluated the same way and for the same reason `corpus` is: they
  are what the product actually serves, not a probe of it. Each gets its own isolated FAISS
  collection (`EXTRA_DEV_CORPORA` in `run_eval.py`), so evaluating them never touches the store
  the web UI writes into.

Counted directly from `gold_cases.jsonl`: 156 cases, of which 136 reach a generator and 20 are
retrieval-only (`figure_retrieval` / `table_retrieval`); 85 `en`, 37 `es`, 34 `ca`; 75 cases
over 17 `corpus` documents, 33 over 6 `arxiv` documents, 24 over 12 `corpus_es` documents, 24
over 12 `corpus_ca` documents -- not every document in a store has a gold case yet. This is a
count, not a target: recount it (one `json.loads` per line, a `Counter` over the field you
care about) rather than trust a number written here if the file has grown since -- a stale
count transcribed by hand is exactly how this section went stale before (#221).

Some questions are deliberately cross-lingual: 10 `ca` and 13 `es` questions ask about
`corpus`/`arxiv` (English) documents, not about `corpus_es`/`corpus_ca`. `run_eval.py`'s own
comment on `EXTRA_DEV_CORPORA` explains why: "the gold set has always had Spanish and Catalan
questions asked against English papers." `lang` is the question's language, never the
document's -- `corpus_es`/`corpus_ca` cases are always asked in their own language, but
`corpus`/`arxiv` mix a majority of `en` questions with that deliberate cross-lingual minority.

Every `accepted_answers` / `expect_kind_any` value was checked by hand against the PDF (see
each case's `verified_pages` -- the physical page numbers, 1-indexed, as read directly from the
source PDF) before being added. A case whose answer could not be confirmed in the text does not
go in the file. The three `study_*` case types (see the schema below) are the exception: they
grade a whole generated artifact against a document, not a literal against a page, so they
carry no `verified_pages`.

## Schema (`gold_cases.jsonl`)

One case per line:

| Field | Meaning |
|---|---|
| `id` | Unique, `<paper-slug>-<short-name>`. |
| `paper` | Paper slug. Matches the corpus filename stem for every `source` but `"arxiv"`. |
| `source` | `"corpus"` / `"corpus_es"` / `"corpus_ca"` (already in the matching `rag/docs/<lang>/`) or `"arxiv"` (fetch first). |
| `arxiv_id` | Only when `source == "arxiv"`; passed to `fetch_papers.py`. |
| `case_type` | `factual_number` \| `factual_concept` \| `figure_retrieval` \| `table_retrieval` \| `study_summary` \| `study_outline` \| `study_quiz`. |
| `lang` | `en` \| `es` \| `ca` -- the question's language, not the document's. |
| `question` | The query text (a retrieval query for the two `*_retrieval` types). |
| `accepted_answers` | List of literals, any one of which counts as correct. Required for the two factual types. |
| `expect_kind_any` | List among `text`/`table`/`image` — the content kind expected in the top-k. Required for the two retrieval types. |
| `verified_pages` | PDF pages (1-indexed) where the fact/figure/table was confirmed. |
| `notes` | Optional: nuance worth flagging (ambiguous source values, grading caveats). |

`table` and `image` match the content taxonomy produced by the current MinerU
indexer. Tables retain their structured HTML and figures are embedded directly
with Jina CLIP.

The three `study_*` types (issue #140; graded by `grade_study` in `grade.py`) score a whole
generated artifact against its source document, not a literal against a page -- they carry no
`question` and no `verified_pages`, since the paper itself is the scope. Each carries its own
bound instead: `study_summary` needs `min_sections` (optionally `required_all`, terms every
section's text must mention); `study_outline` needs `min_nodes` (optionally
`expect_titles_any`, real section titles the outline should hit at least one of);
`study_quiz` needs `min_questions`. `test_grade.py`'s schema check is the authoritative list of
what each actually requires.

## Adding a case

1. Pick a paper already in the corpus, or a new `arxiv_id`.
2. Read the actual page with the `Read` tool's `pages` parameter (or any PDF viewer) — do not
   trust a remembered fact.
3. Write the question and the literal(s) exactly as confirmed, plus `verified_pages`.
4. Prefer literals that start and end with a letter or digit — `grade.py`'s word-boundary match
   anchors on alphanumeric transitions, so a literal starting with punctuation matches looser
   than intended (documented in `grade._contains_token`).
5. Run `pytest tests/eval/test_grade.py` — the schema test catches malformed fields.

## Running

```bash
python -m pytest tests/eval/                       # grader unit tests, no network
python tests/eval/fetch_papers.py                   # fetch every arxiv_id in gold_cases.jsonl
python tests/eval/fetch_papers.py 1512.03385         # fetch one paper by id
```

`fetch_papers.py` is idempotent: a cached, header-validated PDF is never re-downloaded.

## Running the full gate (`run_eval.py`)

Single, self-sufficient command -- no manual indexing, no machine-specific paths:

```bash
python tests/eval/run_eval.py                                 # default: gemma4:e2b
python tests/eval/run_eval.py --models gemma4:e2b gemma4:e4b   # different generator model set
python tests/eval/run_eval.py --update-baseline                # also raise the baseline if green
```

Requires a local Ollama server, every model this run needs already pulled
(`ollama pull ...` -- the runner tells you exactly which if one is missing),
and in practice a GPU: the reranker and every model role run on CPU
otherwise, which this gate does not support.

What it does, in order, failing with an actionable message the moment
something is missing:

1. Checks Ollama is reachable and every required model (`--models`, plus the
   fixed auxiliary model used for query decomposition, contextual retrieval
   and RECOMP) is installed. Jina CLIP and BGE are local Hugging Face models,
   not Ollama roles.
2. Downloads any missing blind-set arXiv papers (reusing `fetch_papers.py`)
   and stages them under `blind_docs/<paper-slug>.pdf`.
3. Indexes whatever is not already indexed -- dev-set papers for each of the three
   language stores (`rag/docs/en/`, `rag/docs/es/`, `rag/docs/ca/`) read from the
   product's own folders but stored in the eval's own isolated collections
   (`EXTRA_DEV_CORPORA` in `run_eval.py`), blind-set papers into their own collection
   under `blind_docs/` -- via the real `indexar_documentos` pipeline. A paper is reused
   only when the store's recorded index recipe (chunking, embeddings, index-time flags)
   matches the configuration this run will use; a changed recipe discards the store and
   rebuilds it.
4. Verifies every paper referenced by a gold case actually has an index
   entry before running anything.
5. Runs every case through the real retrieval + (for factual cases)
   generation pipeline, grading with `grade.py`. Retrieval wires the query
   decomposer the same way the product does whenever
   `usar_llm_query_decomposition` is on (issue #64). Retrieval for a case is
   computed once and reused across every `--models` entry -- retrieval does
   not depend on the generator, so re-running it per model would test
   nothing new. `figure_retrieval`/`table_retrieval` cases never call a
   generator at all.
6. Writes a dated JSON report to `runs/<timestamp>.json` (per-case detail:
   pass/fail, timing, and on failure the generated answer and retrieved
   fragments) plus a console summary by case type and by model. The report
   also carries a `conditions` block recording what produced that pass rate
   -- the per-corpus `AppConfig` (chunking, retrieval, flags), sampling
   parameters per Ollama role, installed stack versions, GPU, git commit and
   a hash of `gold_cases.jsonl` -- so two runs measured under different
   setups are distinguishable from the files alone (issue #222).
7. Compares the overall pass rate against `baseline_min_pass_rate.txt` and
   exits non-zero if it dropped -- that comparison is the gate. `--update-baseline`
   additionally raises the file to `pass_rate - 0.05` (rounded down to the
   nearest 0.01) *after* the gate check, and only if that is higher than the
   current value -- the baseline never moves down automatically. It also
   refuses more than one `--models` entry: the file holds one number,
   calibrated on the gate's default generator.

   That default is `run_eval.DEFAULT_MODELS` -- `Ling-3.0-tiny` unless `OLLAMA_RAG_MODEL`
   says otherwise -- and it is **not** the product's default (`gemma4:e4b`). The gate and the
   configuration harness (`harness/evaluator.real_evaluate`, which passes `DEFAULT_MODELS`)
   measure the faster model the 0.82 floor was set with, at 7.5 s per answer against 11.8 s
   for the shipped one; a no-argument `python tests/eval/run_eval.py` is therefore a statement
   about the pipeline under `Ling-3.0-tiny`, not about what a user runs. Pass
   `--models gemma4:e4b` (or export `OLLAMA_RAG_MODEL`) to measure the product; whether the
   two defaults should be made the same is issue #242.

Infrastructure failures leave the run inconclusive. Only an unfiltered gold-set
run (`case_ids is None`, the CLI) is compared against the baseline. A subset
(the harness search set / fast tier / empty reachability probe) still reports
`pass_rate` and can raise the baseline when asked, but is not the same thing as the
full, unfiltered run the baseline is calibrated against -- see Corpus above for
what "full" currently means.

## Experiment standard

`docs/model-history.md` exists so a model choice comes from numbers instead of memory, but a
row is only worth comparing to another row if both were produced under the same conditions.
Issue #218 is what happens when that goes unchecked: rows in that table were measured against
three different gold sets side by side, with nothing in the artifact saying so. This section is
what "same conditions" means for a `run_eval.py` invocation, so a later row can be trusted
instead of re-derived from memory.

### What the gate fixes for you

- `flags.usar_contextual_retrieval` is forced to `False` regardless of the running
  configuration (`_eval_app_config` in `run_eval.py`, the `with_overrides` call right after the
  full flag block) -- the only one of the nine pipeline flags the gate overrides itself.
- Generation's `keep_alive` is forced to 120 seconds for the run's duration
  (`_EVAL_GENERATION_KEEP_ALIVE_SECONDS` in `run_eval.py`), so a cold first call per model is
  not read as steady-state latency.
- Every generation call is capped at 180 s of wall clock (`GENERATION_BUDGET_SECONDS`, issue
  #229) and an exhausted cap is a failure of that (case, model) pair. The knob is
  `EVAL_GENERATION_BUDGET_SECONDS`, an environment variable of the gate rather than of the
  product, which is why `.env.example` does not list it; the value in effect is written to the
  artifact's `conditions.generation_budget_seconds`, and two runs with different budgets are not
  comparable rows.
- The `chat`, `contextual` and `recomp` roles are all pinned to `AUX_MODEL` for the whole
  evaluation (`_scoped_model_roles` around the `evaluate()` call in `run_eval.py`). A
  `--models` sweep varies only the `rag` role; the query decomposer a sweep might otherwise
  also swap along with the generator is held fixed.

### What it does not fix -- you have to, from outside

- The other eight pipeline flags inherit whatever `rag/chat_pdfs.py`'s module globals resolve
  to at import time (environment > `settings.json` > module default, per `AGENTS.md` §3). A
  flag flipped in the environment or left over in `settings.json` changes the measurement with
  nothing in the console output saying so.
- Every numeric parameter -- `RAG_CHUNK_SIZE`, `RAG_TOP_K_FINAL`, `RAG_N_RESULTADOS_SEMANTICOS`,
  the reranker threshold (`RAG_UMBRAL_SCORE_RERANKER`), the fusion weights
  (`RAG_PESO_SEMANTICO_RRF` / `RAG_PESO_BM25_RRF`), `OLLAMA_RAG_NUM_CTX` and the rest -- reads
  an `RAG_*`/`OLLAMA_*` environment variable with a module default in `rag/chat_pdfs.py`. An
  exported variable changes the run and nothing flags it.
- Sampling is fixed in code rather than read from the environment, but it still has to match
  across runs being compared: `RAG_SAMPLING_OPTIONS`, `RECOMP_SAMPLING_OPTIONS` and
  `QUERY_DECOMPOSER_SAMPLING_OPTIONS` in `rag/engine/wiring.py` set the `rag` role's
  temperature to 0.15, `recomp` to 0.1, and the decomposer (the `chat` role, the same one
  pinned to `AUX_MODEL` above) to 0.5.

### Determinism -- the part that matters most

No stage of the pipeline the gate exercises pins a random seed (issue #223); the `conditions`
block records `"seed": null` because that is what actually happens, not because the field was
left blank. Three stages sample at non-zero temperature -- the generator under test at 0.15,
RECOMP synthesis at 0.1, and query decomposition at 0.5 -- and the decomposer is the one that
matters beyond wording: it runs *before* retrieval, so a different sub-query on an otherwise
identical run changes which fragments come back, which changes what generation ever sees.
Variance is not confined to how the final answer happens to be phrased.

The practical consequence: **a row in `docs/model-history.md` is one run, and one run moves
by up to five cases.** Measured on this 156-case set on 2026-09-11 (four models run twice, see
"Measuring the noise floor" below): resampling alone flipped 28 of 536 (case, model) pairs, a
net movement of -2 to +5 cases per model. A gap that size between two models' pass rates is
not distinguishable from running one of them again. Treat it as noise, not as a finding.

### What now gets recorded

Since issue #222, every run artifact under `runs/` carries a `conditions` block: the per-corpus
`AppConfig` actually used, installed package and Ollama-server versions, GPU info, the git
commit, a sha256 of `gold_cases.jsonl`, the sampling options per role, `seed: null`, and the
`keep_alive` in effect. That block is what makes a row in `docs/model-history.md` verifiable
after the fact instead of merely asserted -- read it with `tools/diagnostics/model_history_row.py`
before trusting a row. Anything measured before this was added has no such record; treat
pre-#222 rows as rows of undocumented conditions, not as rows comparable to a later one.

Each record also says when and where it ran. `started_at` (UTC, to the second) is on every
record, so a window of a run in which every generation slowed down is visible in the artifact
instead of having to be counted off record positions (issue #234: nineteen budget exhaustions
in one run were two such windows). A `budget_exceeded` record carries `stage_at_budget` --
`context` (RECOMP synthesis on the auxiliary model, when on) or `generator` for a factual
case, `setup` or `generator` for a study case -- read off the marker
`generar_respuesta_silenciosa` sets in its `stats` dict as it goes. Every generation record
carries `vram_fraction`, the share of the model's bytes Ollama reports resident in VRAM
(`/api/ps`) right after the call: `1.0` is fully on the GPU, less is CPU offload, and the key is
absent when Ollama could not be asked (issue #235). Artifacts written before these fields
existed simply lack them; a reader must treat a missing key as "not recorded", never as zero.

### Adding a row to `docs/model-history.md`

1. Confirm the GPU is actually free. `ollama ps` reporting nothing does **not** mean it is --
   check `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` for who
   actually holds VRAM, and stop the web app and any other resident model server first.
2. Watch the console for `[index] <label>: cache hit, ...` for every corpus label, not
   `indexing N missing paper(s)` or `recipe changed ... discarding and rebuilding`. A reindex
   mid-run changes what is being measured, not just how long the run takes.
3. Measure every model that fits in one `--models` invocation, rather than one invocation per
   model. Retrieval is computed once per case and shared across every model in that invocation
   (step 5 above), which is what makes the comparison paired instead of two independent
   samples -- separate invocations give every model its own retrieval draw and lose that.
   Generation runs model-major (every pending case for one model, then the next), so a sweep
   loads each generator once rather than once per case (issue #220).
4. Read the resulting artifact with `tools/diagnostics/model_history_row.py
   tests/eval/runs/<artifact>.json`, not by re-deriving numbers from the console output.
5. Record the artifact's filename (or timestamp) in the row -- that is what lets someone else
   open its `conditions` block later and check what was actually measured.
6. Confirm `infrastructure_errors` is empty in the artifact before treating the run as
   conclusive; a run that had infrastructure failures answers a different question than the
   one the row claims to answer.

## Measuring the noise floor and the gate's sensitivity

The runs cited below are historical, measured 2026-07-29 against a 51-case gold set: six
English papers as the dev set, three arXiv papers (ResNet, BERT, ViT) as the blind set, `lang`
limited to `en`/`es`, and four `case_type` values -- not the 156-case, three-language,
four-source set `gold_cases.jsonl` holds today (see Corpus above). The fractions below
(`44/51`, `39/51`, `40/51`) belong to that older, smaller set and do not rescale to a
denominator of 156; nobody has re-run this measurement against the current file. The procedure
-- run twice, diff with `compare_runs.py` -- is unchanged and is exactly what a fresh
measurement should still follow; only the numbers already on record here are frozen at the set
they were measured on.

Both need a GPU machine with Ollama running — the fast CI gate cannot run
them. `compare_runs.py` itself is pure and is covered by the fast gate.

**Noise floor.** Run the identical configuration twice and compare:

```bash
python tests/eval/run_eval.py
python tests/eval/run_eval.py
python tests/eval/compare_runs.py tests/eval/runs/<first>.json tests/eval/runs/<second>.json
```

Every flip is noise. Record the observed number: no delta at or below it counts
as a real change, and an optimisation loop must not treat one as an improvement.

`compare_runs.py` refuses a report that carries an `infrastructure_error`, because a run
that measured nothing must not compare as noise-free. A sweep that includes a model with a
reproducible Ollama crash (Granite's `GGML_ASSERT` on the `att-study-*` cases, for one)
trips that rule on every run. Pass `--exclude-infrastructure-errors` to drop the (case,
model) pairs that never ran in either report and compare the rest; the output lists what
was excluded, so the number you record says what it rests on (issue #240).

**Measured 2026-07-29**, two full-gate runs, identical configuration and code,
same index (both logged `cache hit` on the dev and blind sets, so neither
reindexed): `tests/eval/runs/20260729T020233Z_mineru-jina_clip-faiss.json` and
`tests/eval/runs/20260729T040824Z_mineru-jina_clip-faiss.json`. These are
local, gitignored artifacts (`tests/eval/runs/`) -- nobody cloning the repo
can reproduce this check from them directly. Both 44/51 = 0.8627 overall.
`compare_runs.py` over the pair reports `identical: 51 case(s) unchanged`,
pass rate delta +0.0000, zero flips. `compare_runs.py` compares the per-case
pass/fail vector, not the generated text, so this bounds the
**classification**, not the output. Direct counterevidence sits in the same
pair: `planck-sigma8-es` fails in both runs, and a failure stores the
generated answer -- the two texts differ (one ends on "Planck lensing", the
other appends a full sentence on Planck's preferred amplitudes). The
generator runs at temperature 0.15 and varied, as expected; what's measured
is that `grade.py`'s literal-match criterion absorbs that variance, not that
the output was identical. This bounds the noise floor of the classification
below one case; it does not prove the pipeline is deterministic in general,
and two runs is a small sample, so a wider claim needs more pairs. The floor
is measured for this `grade.py`: changing the grading rules could move it
and would require re-measuring. Under this floor, a single-case flip stops
being explainable by noise, which is not the same as a difference between two
configurations being demonstrable: design doc section 3 puts that second
threshold at roughly six net flips, with a usable margin of about five cases
once figure cases are excluded. This note does not retract that count, it
only sets the floor it is interpreted against; the inference also rests on a
single pair of runs.

**Measured 2026-09-11 on the current 156-case set**, tranche 1 of the model campaign
(#218) run twice: `tests/eval/runs/20260911T032105Z_mineru-jina_clip-faiss.json` and
`tests/eval/runs/20260911T230513Z_mineru-jina_clip-faiss.json`, four generators
(`Ling-3.0-tiny`, `Llama-3.2-3B`, `Granite-4.0-H-Tiny`, `OLMoE-1B-7B`), `conditions`
blocks equal field for field except `git_commit`. `compare_runs.py
--exclude-infrastructure-errors` over the pair: 25 flipped to PASS, 12 flipped to FAIL, 521
unchanged, 6 pairs excluded (Granite's `GGML_ASSERT` crash on `att-study-*`, 2 in the first
run and 6 in the second). Two things move in those 37 flips. Nineteen are the 180 s
generation budget (#229): the first run exhausted it 24 times, in two contiguous windows
where every model ran past the cap, and the second run 5 times, all on two `study_summary`
cases that exhaust it for every small model (#234). Dropping every pair that exhausted the
budget in either run leaves 536 pairs and **28 flips, 16 up and 12 down, 4 to 11 per model,
net -2 to +5 per model** -- that is the sampling floor of one row on this set. It is a floor
for this `gold_cases.jsonl` and this `grade.py`, and it rests on one pair of runs, like the
2026-07-29 figure above; the retrieval-only cases scored 17/20 in both, so the variance is
in generation, as #223 predicted.

**Sensitivity.** Compare a healthy run (either one from the noise-floor pair
above) against a deliberately degraded one:

```bash
RAG_TOP_K_FINAL=1 python tests/eval/run_eval.py                       # POSIX (bash/zsh)
```

```powershell
$env:RAG_TOP_K_FINAL = "1"                                            # PowerShell
python tests/eval/run_eval.py
```

`$env:RAG_TOP_K_FINAL` persists for the rest of the PowerShell session --
clear it (`Remove-Item Env:RAG_TOP_K_FINAL`) or restart the shell before
running a healthy config again.

```bash
python tests/eval/compare_runs.py tests/eval/runs/<healthy>.json tests/eval/runs/<degraded>.json
```

The degraded run must flip a clearly larger number of cases to FAIL than the
noise floor. A gate that barely moves under a known degradation cannot detect
an improvement either.

**Measured 2026-07-29.** Healthy run
`tests/eval/runs/20260729T020233Z_mineru-jina_clip-faiss.json` (44/51 =
0.8627, the same run used for the noise-floor pair above) against degraded run
`tests/eval/runs/20260729T081129Z_mineru-jina_clip-faiss.json`
(`RAG_TOP_K_FINAL=1`, 39/51 = 0.7647, 121.1 min). Both logged `cache hit`,
which bounds the index, not the code: that only a docs-only commit sits
between the two runs is true, but nothing cited here supports it, and neither
report records the `RAG_TOP_K_FINAL` value each run actually used, so the
degraded configuration is asserted, not captured -- exactly the gap the
evidence ledger (design doc section 4) and acceptance criterion 7 exist to
close. These are local, gitignored artifacts (`tests/eval/runs/`)
same as the noise-floor pair -- nobody cloning the repo can reproduce this
check from them directly. `compare_runs.py` over the pair: 2 flipped to PASS,
7 flipped to FAIL, 42 unchanged, pass rate delta -0.0980. All seven flips to
FAIL are retrieval-only cases (`att-arch-figure`, `att-arch-figure-es`,
`att-bleu-table`, `dpo-pipeline-figure`, `resnet-block-figure`,
`vit-comparison-table`, `vit-overview-figure`); the two flips to PASS are
`planck-sigma8-es` and `resnet-top1-34layer`. By metric: retrieval-only fell
from 11/15 (0.7333) to 4/15 (0.2667); answer rose from 33/36 (0.9167) to 35/36
(0.9722). The degraded run also failed the baseline floor (0.7647 < 0.77),
which is the gate behaving correctly under a known degradation -- but by a
margin of 0.0053, against 0.0196 for a single case: at 40/51 = 0.7843
(retrieval-only at 5/15 instead of 4/15, still just as collapsed) the gate
would have passed the same catastrophic degradation. The aggregate fell 0.098
while retrieval fell 0.47; this is the sharpest evidence for the point the
consequence paragraph below already makes: an aggregate that barely notices a
retrieval collapse this severe is precisely what makes a loop maximising it
dangerous.

This result is only interpretable because the noise floor measured above is
zero flips -- seven flips against a floor of zero is unambiguous signal.
Retrieval and answering moved in opposite directions: retrieval collapsed
with 7 flips -- above the roughly six net flips the criterion-1 note cites
(set in design doc section 3) as the bar for a difference to be demonstrable
-- while answering improved slightly with only 2 net flips: above the
zero-flip noise floor but below that same bar, so this measurement does not
demonstrate the answer-side gain is real. A plausible reading is that less
context distracts less on narrow factual questions, but that is a
hypothesis, not a finding -- nothing here tested it. One degraded
configuration was tested
(`RAG_TOP_K_FINAL=1`), not a sweep; the design also lists disabling the
reranker as a separate sabotage, still unmeasured. This uses the
retrieval/answer split `run_eval.py` already reports, but it measures
sensitivity only -- it does not by itself close acceptance criterion 4
(separated metrics).

Consequence for the optimisation loop:
`docs/design/2026-07-28-loop-automejorable.md` section 1 defines the
objective function as a single aggregate pass rate. A loop maximising only
that aggregate could favor configurations that trade retrieval quality for
factual-answering accuracy without anyone noticing -- the aggregate alone does
not distinguish a genuine improvement from that trade. Separated metrics are
what make the trade visible; the design's objective function currently
targets the aggregate, not the split. Five of the seven flips to FAIL are
figure-retrieval cases, and design doc section 3 ("Margen inalcanzable")
already places part of those cases outside the scalar the loop maximises --
which makes the warning stronger, not weaker.

## Language-axis probe (diagnostic, not the gate)

Design doc section 3 ("El corpus, derivado del objetivo") measured on
2026-08-12 that the loop's search set -- the 32 `source: "corpus"` cases,
one of the two partitions block B still owes -- has only 5 available
failures and registers just 3 net flips against a known-catastrophic
sabotage (`RAG_TOP_K_FINAL=1`), against a ~6-flip threshold for a
demonstrable paired difference. The fix is a larger, harder search-set
corpus (~55 new cases, ~20 documents); before authoring that at scale,
section 3 ("Sonda previa") asks for a small diagnostic batch per axis that
decides whether it is worth it.

This is that batch for the **language axis**: the `es/` (Castilian) and
`ca/` (Valencian) document stores the product already ships in
`rag/docs/es/` and `rag/docs/ca/` -- two of its three fixed language
stores. Measured 2026-08-12, before #213: `gold_cases.jsonl` had zero
`corpus_es`/`corpus_ca` cases, so both stores were entirely unmeasured by
the gate, which is the gap this probe was built to diagnose. Provisioning
cost was zero, since the PDFs already shipped; the domain and form axes
are out of scope here (see the design doc's composition table).

**That gap in the gate is closed, but not by this file.** #213 gave the
gate 24 `corpus_es` and 24 `corpus_ca` cases (see Corpus above for the
current count), so "the language axis is unmeasured" is no longer true.
Those 48 cases, though, sit entirely on the twelve documents per store
that #213 added by fetching through `tools/fetch_corpus.py` (gitignored,
downloaded on demand from a versioned manifest) -- none of them touch the
five `es` and five `ca` documents that were already committed to git
before #213 existed (`git ls-files rag/docs/es rag/docs/ca`), which
include this probe's six. Those ten original documents carry zero
`gold_cases.jsonl` cases, exactly as before #213:
`Ciudad_de_las_Artes_y_las_Ciencias` and `Fallas_de_Valencia` (`es`),
`Història_de_València` and `Paella_d'arròs` (`ca`) were never in this
probe either, and remain unmeasured by both files. The probe and the gate
now both cover the language axis, on entirely disjoint documents -- which
is why none of the 18 ids below appear in `gold_cases.jsonl`, and why that
overlap check is not the thing to fix here.

The two also answer different questions, which is the more important
reason this file stays. The gate's question, since #213, is "does the
shipped product answer correctly in `es`/`ca`." This probe's question,
per the design doc's section 3 ("Sonda previa"), is "does the language
axis produce failures a self-improving loop can learn from" -- checked
against the loop's own search set (`harness/evaluator.py`'s
`search_set_case_ids()`, the complement of the blind set, which now
includes #213's `corpus_es`/`corpus_ca` cases too). The verdict recorded
below already answered that second question, on 2026-08-13, before #213:
*viable tras arreglo* -- these six documents are too easy (17/18) to
distinguish a loop improvement from a regression, not because the
pipeline mishandles `es`/`ca`. #213 does not revisit that verdict: it
added general product coverage, not the harder search-set documents the
design doc's fix calls for, so the probe's seed cases are still waiting
on that later batch (see the closing paragraph below).

- **`probe_cases_lang.jsonl`**: 18 hand-verified cases (every
  `verified_pages` checked against the actual PDF page) over 6 documents --
  `Horchata_de_chufa`, `Parque_natural_de_la_Albufera`,
  `Rodrigo_Díaz_de_Vivar` (`es`), `Llotja_de_la_Seda`, `Jaume_el_Conqueridor`,
  `Pilota_valenciana` (`ca`). Same field shape as `gold_cases.jsonl`
  (`id`/`paper`/`case_type`/`lang`/`question`/`accepted_answers` or
  `expect_kind_any`/`verified_pages`/optional `notes`), plus `"source":
  "lang_probe"` so it can never be mistaken for a `gold_cases.jsonl` row.
  **This file is a proposal, not an addition to the gate.** Per the design
  doc's assumptions (section 5, "Autoría de casos"), a batch like this needs
  human audit before any of it could be promoted into `gold_cases.jsonl` --
  nothing here does that promotion automatically, and nothing in this repo
  reads this file except `run_probe_lang.py` and its own schema test.
- **`run_probe_lang.py`**: stages the 6 PDFs into `probe_docs_lang/` (gitignored,
  copied under each case's `paper` slug) and indexes them into their own
  FAISS collection, derived from that directory's basename exactly the way
  `run_eval.py` isolates its own dev-set collection from the product's
  `docs_en` store (see the comment on `EVAL_DEV_LABEL` in `run_eval.py`).
  The result is a collection that cannot collide with `docs_es`, `docs_ca`
  (the product's live stores), `dev_docs`/`blind_docs` (this gate's own
  stores), or any other `rag/vector_db/*` collection -- verified by reading
  `derive_db_paths`: the collection name and store path are both derived
  from `os.path.basename(docs_folder)`, and `probe_docs_lang` matches none
  of those basenames. Reuses `run_eval.py`'s own indexing, retrieval and
  grading code (`ensure_indexed`, `run_retrieval_case`, `run_factual_case`)
  instead of re-implementing them, so results are directly comparable to a
  `run_eval.py` report; `run_eval.py` itself is never modified. Prints a
  pass/fail per case and a summary — no baseline, no gate, no automated
  verdict.

Run it once a GPU is free (never on the shared runner mid-search):

```bash
python tests/eval/run_probe_lang.py                      # default: gemma4:e2b
```

Reading the result is a human step: record a verdict per axis in
`docs/design/2026-07-28-loop-automejorable.md` section 3 -- *viable*,
*viable tras arreglo*, or *inviable con esta fuente* -- from the printed
summary and the JSON report under `probe_runs_lang/`. Measured 2026-08-13
(17/18, artefact gitignored): the design doc's note under "Sonda previa"
records the per-axis verdict (*viable tras arreglo*) and why these 18 cases
are not promoted into `gold_cases.jsonl`. If a later source on this axis
comes back *viable*, its cases are written to survive promotion (per
section 3's "los casos de la sonda se redactan para sobrevivir"): they are
the seed of the full batch, not throwaway material.

## Domain-axis probe (diagnostic, not the gate)

The sibling of the probe above for the **domain axis** of the same design-doc
section: does a corpus whose vocabulary is far from the ML/physics the
pipeline was tuned on produce failures a loop could learn from? Three recent
arXiv papers (two econometrics, one biomathematics), ten cases in
`probe_cases_domain.jsonl`, staged from the `fetch_papers.py` cache into their
own collection so the run can never write into a product or gate store:

```bash
python tests/eval/fetch_papers.py --dest tests/eval/probe_cache_domain 2608.18973v1 2608.18375v1 2608.17955v1
python tests/eval/run_probe_domain.py                  # or --models <generator>
```

Same shape of output as `run_eval.py`, same human step afterwards. Measured
2026-08-23 (`gemma4:e2b`, 9/10, artefact gitignored under `probe_runs_domain/`):
no lexical collapse out of domain, the one failure a generation miss on
retrieved evidence. The design doc records the verdict, *viable tras arreglo*,
and why: arXiv feeds the pipeline from any field, but a source this easy does
not give a loop enough failures to move. `test_probe_cases_domain.py` skips
itself until the three papers are in the cache.
