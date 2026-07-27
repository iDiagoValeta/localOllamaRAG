# Handoff: wire the multimodal stack and run the A/B

Paste this whole file as the opening prompt. It is self-contained: everything you
need is either here or in the repository.

---

## What this project is

A local RAG system over PDFs: hybrid retrieval (semantic + BM25 fused with RRF),
cross-encoder reranking, and generation through a local Ollama model. No cloud,
no API keys. Interfaces: a CLI, a Flask + React web app, and a packaged Windows
desktop app.

The owner's goals, in his words and in priority order:

1. **Better at tables and images.** This is the point of the whole exercise.
2. **More efficient** retrieval.
3. **Swappable everything** — any technology or model replaceable by
   configuration, so future branches can compare them automatically.
4. **Clean architecture, no half measures.** No silent fallbacks.
5. **CI that verifies each stage** rather than looking green.
6. **Minimal documentation.** He is explicit that documentation bloat annoys him.

## Where the work stands

A hexagonal core exists under `src/monkeygrab/`: domain entities, ports,
application use cases, adapters, immutable config. `rag/engine/*` delegates its
pure logic there while keeping its public API, and `rag/` holds the interfaces.
The dependency rule is enforced per layer by an AST test.

Three new adapters exist and are individually verified against real inputs:

| Adapter | Verified |
|---|---|
| MinerU extractor | 15 pages in 83 s: 4 tables with HTML intact, 5 figures with captions |
| jina-clip-v2 embedder | Figure vs. text describing it: 0.357. Vs. unrelated text: 0.055 |
| FAISS store | Same vectors as Chroma return the same ids in the same order |

Backends are selected by environment variable: `PDF_EXTRACTOR`
(`pymupdf`|`mineru`), `VECTOR_STORE` (`chroma`|`faiss`), `EMBEDDER`
(`ollama`|`jina_clip`). Unknown values raise at startup. `build_stack()` in
`src/monkeygrab/composition.py` returns the three built adapters.

**Nothing is wired.** The indexing use case still receives whatever its caller
constructs by hand, and the current stack remains the default.

## The measurement you are working against

The gate runs 51 gold cases whose every answer was verified by reading the cited
page of the source PDF. Current stack, with `gemma4:e2b`:

| Case type | Pass |
|---|---|
| `factual_number` | 21/25 (84%) |
| `factual_concept` | 8/11 (73%) |
| `figure_retrieval` | 5/10 (50%) |
| `table_retrieval` | **0/5 (0%)** |

Overall 34/51 (66.7%). Baseline threshold: 0.61.

Three of the papers (ResNet, BERT, ViT) are a **blind set**: the retrieval
heuristics were tuned on "Attention is all you need", and the figure failures
cluster in the blind set. Treat a gain that appears only on the development
papers as suspect.

## Environment

- **Python interpreter for everything**: `C:\Users\nadiv\anaconda3\python.exe`.
  Not the one on PATH — that one lacks the dependencies.
- Tests: `C:\Users\nadiv\anaconda3\python.exe -m pytest -q --no-header`.
  Currently **324 passed, 1 skipped**.
- Lint: `C:\Users\nadiv\anaconda3\python.exe -m ruff check .` — currently clean.
- Ollama runs locally with `gemma4:e4b`, `gemma4:e2b`, `qwen3.5:0.8b`,
  `embeddinggemma:latest`. GPU: RTX 4060 Laptop, **8 GB** — one model at a time,
  two generators at once will make Ollama return HTTP 500.
- `.venv-mineru/Scripts/python.exe` is an isolated venv holding MinerU's CLI and
  the only environment where jina-clip loads. The embedder adapter drives a
  worker process there.

## Hard constraints — do not violate these

**Files you must not modify.** They are the only reason a green result means
anything:

- `tests/characterization/**` — pins the pipeline's current behaviour, bugs
  included. If one fails, your change altered behaviour: fix your change.
- `tests/eval/gold_cases.jsonl` — the questions and their verified answers.
- `tests/eval/baseline_min_pass_rate.txt` — the regression threshold.

If the gate fails, the answer is never to adjust the judge, lower the threshold,
or drop hard cases. Report the number you got.

**Decisions already made. Do not revisit them:**

- **jina-clip-v2 cannot run in the main environment.** Under transformers 5.1 it
  fails initializing weights; disabling lazy init and forcing CPU do not help. It
  works under transformers 4.57, which is why it runs out of process. **Do not
  fix this by downgrading transformers in the main environment** — it would
  degrade every other project sharing that interpreter.
- **MinerU is an external CLI**, never a Python dependency. Its pins conflict
  with the product's.
- **CUDA is required** for the multimodal embedder, deliberately: CPU embedding
  measured ~100 s per document, so failing loudly beats degrading silently.
- **Hard-fail everywhere.** No adapter degrades to a substitute. Two documented
  exceptions exist (image extraction, and one explicit CUDA→CPU reranker retry);
  do not add a third.
- **You do not decide the default stack.** Run the A/B, report the numbers, stop.
  The owner chooses.

## Tasks

Do them in order. Each has an acceptance criterion checkable by a command, not by
your judgement.

### 1. Point the indexing use case at the selected backends

`IndexCorpus` receives its ports by constructor. Make the caller build them from
configuration via `build_stack()` instead of hardcoding them. Do not change the
public API of `rag/chat_pdfs.py`: the web layer imports from it, including three
private symbols.

**Accept when**: with no environment variables set, `pytest` still reports 324
passed / 1 skipped and `tests/characterization/` shows an empty `git diff`.

### 2. Index a corpus with the new stack

```
PDF_EXTRACTOR=mineru VECTOR_STORE=faiss EMBEDDER=jina_clip
```

The index path already carries a stack slug, so the new index cannot collide with
the existing one. Expect this to be slow: MinerU is ~80 s per paper and the
embedder needs ~28 s to load once.

**Accept when**: the index exists and contains chunks whose format is `table` and
whose format is `image`. If there are zero table chunks, stop and diagnose —
that is the single most important signal in this whole task. Note that
`_text_chunk_format` in `index_corpus.py` marks a chunk as a table when it
contains HTML table markup; if MinerU's tables are not reaching it, the wiring is
wrong.

### 3. Run the A/B

Run `tests/eval/run_eval.py` twice against the same 51 cases: once with the
current stack, once with the new one. Same model (`gemma4:e2b`) both times.

**Accept when**: two result JSONs exist under `tests/eval/runs/` and you can
state, per case type, how each stack scored. A run reporting infrastructure
errors is inconclusive and does not count — rerun it with nothing else using the
GPU.

### 4. Report

Write the comparison into the pull request: the table by case type, both stacks
side by side, and for every case that changed verdict, which one and why you
think so. Say plainly if the new stack loses. Do not change the default.

## How to work

- Read the code before changing it. `src/monkeygrab/README.md` explains the
  layers and how to add an adapter; `docs/design/2026-07-26-monkeygrab-v2.md` is
  the design of record and outranks any other document.
- Comments explain **why**, never mechanics, and never reference plan steps or
  task numbers.
- Work on a branch off `feat/multimodal-stack`. Do not commit to `main`.
- If something contradicts these instructions, say so instead of guessing. One
  instruction in this project was already wrong — it asked a store to raise on
  absent ids, contradicting its port — and catching it was worth more than
  complying.

## Traps that have already cost time

- **A perfectly extracted table indexed as prose is invisible.** Extraction was
  only half the value; the chunk has to be marked as a table. This was the actual
  cause of the 0/5, not MinerU's absence.
- **Two Ollama models at once on 8 GB** kills a 36-minute eval run with an HTTP
  500 halfway through.
- **Local success does not imply CI success.** A dependency installed by hand
  while developing will not exist on a runner, and some failures only reproduce
  on Linux.
