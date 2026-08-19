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

## Corpus

Two sets of papers, mixed deliberately:

- **Dev set** (`source: "corpus"`): the six papers already in `rag/docs/en/`. Their retrieved
  chunks and reranking behavior are what the pipeline's heuristics were tuned against.
- **Blind set** (`source: "arxiv"`, `arxiv_id` set): three papers never used to tune anything
  — ResNet (`1512.03385`), BERT (`1810.04805`), ViT (`2010.11929`). A judge calibrated only on
  the dev set can't tell "the pipeline retrieves well" from "the pipeline overfits these six
  PDFs"; the blind set is what makes that distinction possible.

Every `accepted_answers` / `expect_kind_any` value was checked by hand against the PDF (see
each case's `verified_pages` — the physical page numbers, 1-indexed, as read directly from the
source PDF) before being added. A case whose answer could not be confirmed in the text does not
go in the file.

## Schema (`gold_cases.jsonl`)

One case per line:

| Field | Meaning |
|---|---|
| `id` | Unique, `<paper-slug>-<short-name>`. |
| `paper` | Paper slug. Matches the corpus filename stem for `source: "corpus"`. |
| `source` | `"corpus"` (already in `rag/docs/en/`) or `"arxiv"` (fetch first). |
| `arxiv_id` | Only when `source == "arxiv"`; passed to `fetch_papers.py`. |
| `case_type` | `factual_number` \| `factual_concept` \| `figure_retrieval` \| `table_retrieval`. |
| `lang` | `en` \| `es` — the question's language, not the document's. |
| `question` | The query text (a retrieval query for the two `*_retrieval` types). |
| `accepted_answers` | List of literals, any one of which counts as correct. Required for the two factual types. |
| `expect_kind_any` | List among `text`/`table`/`image` — the content kind expected in the top-k. Required for the two retrieval types. |
| `verified_pages` | PDF pages (1-indexed) where the fact/figure/table was confirmed. |
| `notes` | Optional: nuance worth flagging (ambiguous source values, grading caveats). |

`table` and `image` match the content taxonomy produced by the current MinerU
indexer. Tables retain their structured HTML and figures are embedded directly
with Jina CLIP.

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
3. Indexes whatever is not already indexed -- dev-set papers read from
   `rag/docs/en/` but stored in the eval's own isolated collection, blind-set
   papers into their own collection under `blind_docs/` -- via the real
   `indexar_documentos` pipeline. A paper is reused only when the store's
   recorded index recipe (chunking, embeddings, index-time flags) matches
   the configuration this run will use; a changed recipe discards the store
   and rebuilds it.
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
   fragments) plus a console summary by case type and by model.
7. Compares the overall pass rate against `baseline_min_pass_rate.txt` and
   exits non-zero if it dropped -- that comparison is the gate. `--update-baseline`
   additionally raises the file to `pass_rate - 0.05` (rounded down to the
   nearest 0.01) *after* the gate check, and only if that is higher than the
   current value -- the baseline never moves down automatically. It also
   refuses more than one `--models` entry: the file holds one number,
   calibrated on the default generator.

Infrastructure failures leave the run inconclusive. Only an unfiltered gold-set
run (`case_ids is None`, the CLI) is compared against the baseline. A subset
(the harness search set / fast tier / empty reachability probe) still reports
`pass_rate` and can raise the baseline when asked, but is not the 51-case gate.

## Measuring the noise floor and the gate's sensitivity

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
`rag/docs/es/` and `rag/docs/ca/` -- two of its three fixed language stores
-- which `gold_cases.jsonl` never exercises today. Provisioning cost is
zero, since the PDFs already ship; the domain and form axes are out of
scope here (see the design doc's composition table).

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
