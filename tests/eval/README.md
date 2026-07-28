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
   generation pipeline, grading with `grade.py`. Retrieval for a case is
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
   current value -- the baseline never moves down automatically.

Infrastructure failures leave the run inconclusive. Only a complete run with
the calibrated generator model is compared against the baseline.

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
